# src/training/model_loader.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.training.train_config import LoRAConfig


MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


def get_bnb_config() -> BitsAndBytesConfig:
    """
    Build the 4-bit quantization configuration.

    load_in_4bit: Compress base model weights to 4-bit integers.
    Reduces model memory from ~8GB to ~2.5GB.

    bnb_4bit_quant_type="nf4": Normal Float 4 — a 4-bit format
    designed specifically for normally-distributed neural network
    weights. More accurate than standard int4 quantization.

    bnb_4bit_compute_dtype=bfloat16: Even though weights are stored
    in 4-bit, actual matrix math happens in BF16. The GPU dequantizes
    on the fly for each computation. BF16 is faster than FP32 and
    your GPU supports it natively.

    bnb_4bit_use_double_quant: Quantizes the quantization constants
    themselves. Saves another ~0.4GB. Negligible quality impact.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(cfg: LoRAConfig) -> LoraConfig:
    """
    Build the PEFT LoRA configuration from our dataclass.
    This tells PEFT which layers to inject adapters into
    and how large those adapters should be.
    """
    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias=cfg.bias,
        task_type=cfg.task_type,
    )


def load_model_and_tokenizer(lora_cfg: LoRAConfig = None):
    """
    Load Phi-3 Mini in 4-bit and wrap it with LoRA adapters.

    Returns:
        model: Phi-3 Mini with frozen 4-bit base + trainable adapters
        tokenizer: Phi-3 tokenizer configured for training
    """
    if lora_cfg is None:
        lora_cfg = LoRAConfig()

    print(f"Loading {MODEL_NAME} in 4-bit quantization...")
    print(f"VRAM before loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Step 1: Load the base model in 4-bit
    # This downloads ~2.5GB on first run (cached after that)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=get_bnb_config(),
        device_map="cuda:0",        # Put entire model on your GPU
        trust_remote_code=True,   # Phi-3 has custom model code
        torch_dtype=torch.bfloat16,
    )

    print(f"VRAM after base model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Step 2: Prepare model for k-bit training
    # This does two things:
    # - Casts layer norms to FP32 for training stability
    # - Enables gradient checkpointing to save memory
    #   (recomputes activations during backward pass instead
    #    of storing them — trades compute for memory)
    model = prepare_model_for_kbit_training(model)

    # Step 3: Inject LoRA adapters into the target layers
    # After this call, only the adapter parameters require gradients.
    # The base model parameters are frozen.
    model = get_peft_model(model, get_lora_config(lora_cfg))

    # Print how many parameters are actually being trained
    model.print_trainable_parameters()

    print(f"VRAM after LoRA adapters: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Step 4: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
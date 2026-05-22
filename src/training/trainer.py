# src/training/trainer.py

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from src.data.sql_loader import load_sql_data
from src.data.prompt_formatter import format_training_example
from src.training.model_loader import load_model_and_tokenizer
from src.training.train_config import LoRAConfig, TrainingConfig


def format_for_sft(example: dict) -> str:
    """
    Formatting function passed to SFTTrainer.
    SFTTrainer calls this on every example to get the
    training string. Must return a plain string.
    """
    return format_training_example(example)


def run_training(
    lora_cfg: LoRAConfig = None,
    train_cfg: TrainingConfig = None,
):
    """
    Full training pipeline:
    1. Load data
    2. Load model + tokenizer
    3. Configure SFTTrainer
    4. Train
    5. Save adapter
    """
    if lora_cfg is None:
        lora_cfg = LoRAConfig()
    if train_cfg is None:
        train_cfg = TrainingConfig()

    # --- 1. Load data ---
    print("Loading dataset...")
    train_examples, val_examples = load_sql_data(
        val_size=2000,
        max_train_samples=None,  # Use full 76K training set
    )

    # SFTTrainer requires HuggingFace Dataset objects, not plain lists
    train_dataset = Dataset.from_list(train_examples)
    val_dataset   = Dataset.from_list(val_examples)

    print(f"Train: {len(train_dataset)} examples")
    print(f"Val:   {len(val_dataset)} examples")

    # --- 2. Load model ---
    model, tokenizer = load_model_and_tokenizer(lora_cfg)

    # --- 3. Configure SFTTrainer ---
    sft_config = SFTConfig(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_ratio=train_cfg.warmup_ratio,
        bf16=train_cfg.bf16,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        max_seq_length=train_cfg.max_seq_length,
        packing=train_cfg.packing,
        evaluation_strategy="steps",
        eval_steps=train_cfg.save_steps,
        load_best_model_at_end=False,
        report_to="none",         # No wandb/tensorboard for now
        dataset_text_field=None,  # We use formatting_func instead
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=format_for_sft,
        tokenizer=tokenizer,
    )

    # --- 4. Train ---
    print("\nStarting training...")
    print(f"Effective batch size: "
          f"{train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps}")

    trainer.train()

    # --- 5. Save adapter ---
    print("\nSaving LoRA adapter...")
    model.save_pretrained(train_cfg.output_dir)
    tokenizer.save_pretrained(train_cfg.output_dir)
    print(f"Adapter saved to {train_cfg.output_dir}")
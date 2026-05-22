# src/serving/inference.py

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.data.prompt_formatter import format_inference_prompt
from src.training.model_loader import MODEL_NAME, get_bnb_config


# Read from environment variables so Docker can configure these
ADAPTER_REPO = os.getenv(
    "ADAPTER_REPO",
    "Sid9797/querycraft-phi3-sql"
)
MOCK_MODEL = os.getenv("MOCK_MODEL", "false").lower() == "true"


class SQLGenerator:
    """
    Wraps model loading and SQL generation in a single class.

    Designed to be instantiated once at server startup and
    reused for every request — loading the model is expensive
    (~8 seconds), generating SQL is fast (~2 seconds).

    When MOCK_MODEL=true (used in CI/CD where there's no GPU),
    returns a hardcoded SQL string instead of loading the model.
    This lets us test the API layer without GPU hardware.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.ready = False

    def load(self):
        """Load model and tokenizer. Call once at startup."""
        if MOCK_MODEL:
            print("MOCK_MODEL=true — skipping real model load")
            self.ready = True
            return

        print(f"Loading base model: {MODEL_NAME}")
        print(f"Loading adapter from: {ADAPTER_REPO}")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=get_bnb_config(),
            device_map="cuda:0",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Load LoRA adapter from HuggingFace Hub
        model = PeftModel.from_pretrained(model, ADAPTER_REPO)
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.ready = True
        print("Model ready")

    def generate(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Generate SQL from a natural language question and schema.

        Args:
            question:       Natural language question
            context:        CREATE TABLE schema string
            max_new_tokens: Max tokens to generate

        Returns:
            Generated SQL string
        """
        if MOCK_MODEL:
            return f"SELECT * FROM table WHERE question = '{question}'"

        prompt = format_inference_prompt(question, context)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return generated.strip().split("\n")[0].strip()


# Single global instance — created once, reused per request
generator = SQLGenerator()
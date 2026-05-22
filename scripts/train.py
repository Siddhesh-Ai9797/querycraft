# scripts/train.py

"""
Launch training from the querycraft root:
    python scripts/train.py

Optional quick test on 500 examples (verifies setup in ~5 min):
    python scripts/train.py --smoke-test
"""

import argparse
from src.training.trainer import run_training
from src.training.train_config import LoRAConfig, TrainingConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on 500 examples for 1 epoch to verify setup"
    )
    args = parser.parse_args()

    lora_cfg  = LoRAConfig()
    train_cfg = TrainingConfig()

    if args.smoke_test:
        print("=== SMOKE TEST MODE (500 examples) ===")
        train_cfg.num_train_epochs = 1
        train_cfg.logging_steps    = 10
        train_cfg.save_steps       = 9999  # Don't save checkpoints in smoke test
        train_cfg.output_dir       = "models/smoke-test"

        # Override the loader to use 500 examples
        from src.data.sql_loader import load_sql_data
        from datasets import Dataset
        from src.training.model_loader import load_model_and_tokenizer
        from src.training.trainer import format_for_sft
        from trl import SFTTrainer, SFTConfig

        train_examples, val_examples = load_sql_data(
            val_size=100,
            max_train_samples=500,
        )
        train_dataset = Dataset.from_list(train_examples)
        val_dataset   = Dataset.from_list(val_examples)

        model, tokenizer = load_model_and_tokenizer(lora_cfg)

        sft_config = SFTConfig(
            output_dir=train_cfg.output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.05,
            bf16=True,
            logging_steps=10,
            save_steps=9999,
            max_seq_length=256,
            packing=True,
            report_to="none",
            dataset_text_field=None,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            formatting_func=format_for_sft,
            tokenizer=tokenizer,
        )

        trainer.train()
        print("\nSmoke test complete. Setup is working.")
        return

    run_training(lora_cfg, train_cfg)


if __name__ == "__main__":
    main()
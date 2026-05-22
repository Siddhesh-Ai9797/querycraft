# src/training/train_config.py

from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """
    Controls the adapter architecture.

    r (rank): Size of the adapter matrices. Higher = more capacity
    to learn but more memory and risk of overfitting.
    r=16 is the standard starting point for instruction fine-tuning.

    lora_alpha: Scaling factor for the adapter's contribution.
    Rule of thumb: set to 2x rank. So r=16 → alpha=32.
    Higher alpha = adapter has stronger influence on outputs.

    lora_dropout: Randomly zeros out some adapter values during
    training. Prevents the adapter from over-relying on any single
    path. 0.05 = 5% dropout, light regularization.

    target_modules: Which layers inside Phi-3 Mini get adapters.
    These are the attention and feed-forward projection layers —
    the parts most responsible for language understanding.
    Adding adapters here covers the model's reasoning capacity.

    bias: Whether to train bias terms. "none" means we only
    train the adapter matrices, not any bias parameters.
    Standard choice for QLoRA.
    """
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.target_modules is None:
            # These are the exact layer names inside Phi-3 Mini.
            # Different models have different names — this list is
            # specific to Phi-3 architecture.
            self.target_modules = [
                "q_proj",    # query projection  (attention)
                "k_proj",    # key projection     (attention)
                "v_proj",    # value projection   (attention)
                "o_proj",    # output projection  (attention)
                "gate_proj", # gate projection    (feed-forward)
                "up_proj",   # up projection      (feed-forward)
                "down_proj", # down projection    (feed-forward)
            ]


@dataclass
class TrainingConfig:
    """
    Controls the training loop.

    output_dir: Where checkpoints are saved locally during training.

    num_train_epochs: How many times the model sees the full dataset.
    1 epoch on 76K examples is substantial. Start here.

    per_device_train_batch_size: Examples processed per GPU per step.
    4 is safe for 8.55GB VRAM with MAX_LENGTH=256.

    gradient_accumulation_steps: Accumulate gradients from this many
    batches before updating weights. Effective batch size = 4 × 4 = 16.

    learning_rate: How much to adjust weights each step.
    2e-4 is the standard for LoRA fine-tuning.

    warmup_ratio: Fraction of training steps where learning rate
    ramps up from near-zero to full learning_rate. Prevents unstable
    updates at the very start of training. 0.05 = first 5% of steps.

    bf16: Use BF16 precision for the adapter training.
    Your RTX 5060 Ti supports this natively — confirmed in Step 2.2.

    logging_steps: Print loss to console every N steps.
    50 steps lets you watch training progress without spam.

    save_steps: Save a checkpoint every N steps. 500 steps gives
    you recovery points without filling your disk.

    max_seq_length: Maximum token length per example. Matches the
    MAX_LENGTH we set in tokenizer_utils.py based on our data stats.

    packing: Combines multiple short examples into one sequence
    up to max_seq_length. Since our examples average 124 tokens
    and max_seq_length is 256, roughly 2 examples fit per sequence.
    This roughly doubles training efficiency — less padding wasted.
    """
    output_dir: str = "models/phi3-sql"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    bf16: bool = True
    logging_steps: int = 50
    save_steps: int = 500
    save_total_limit: int = 2
    max_seq_length: int = 256
    packing: bool = True
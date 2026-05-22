# src/data/tokenizer_utils.py

from transformers import AutoTokenizer
from src.data.prompt_formatter import format_training_example


MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MAX_LENGTH = 512


def load_tokenizer() -> AutoTokenizer:
    """
    Load the Phi-3 Mini tokenizer.

    padding_side='right' is required for SFTTrainer.
    We assign eos_token as pad_token because Phi-3 has no
    dedicated pad token — this is the standard workaround.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def tokenize_example(example: dict, tokenizer: AutoTokenizer) -> dict:
    """
    Tokenize a single example for training.

    truncation=True  — cuts sequences longer than MAX_LENGTH.
    padding='max_length' — pads shorter ones up to MAX_LENGTH.
    return_tensors=None — returns plain Python lists, not PyTorch
                          tensors. SFTTrainer handles tensor
                          conversion internally.
    """
    full_text = format_training_example(example)

    tokenized = tokenizer(
        full_text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors=None,
    )

    tokenized["text"] = full_text
    return tokenized


def get_token_length_stats(
    examples: list[dict],
    tokenizer: AutoTokenizer,
) -> dict:
    """
    Compute sequence length statistics across examples.

    Run this before committing to MAX_LENGTH. The key number
    to look at is p90 — if p90 is 380, MAX_LENGTH=512 safely
    covers 90% of examples with headroom. If p90 is 600,
    you're truncating heavily and should increase MAX_LENGTH.
    """
    lengths = []
    for ex in examples:
        tokens = tokenizer(
            format_training_example(ex),
            truncation=False,
        )
        lengths.append(len(tokens["input_ids"]))

    lengths.sort()
    n = len(lengths)

    return {
        "min":          lengths[0],
        "max":          lengths[-1],
        "mean":         round(sum(lengths) / n),
        "p50":          lengths[n // 2],
        "p90":          lengths[int(n * 0.90)],
        "p95":          lengths[int(n * 0.95)],
        "over_512":     sum(1 for length in lengths if length > 512),
        "over_512_pct": round(sum(1 for length in lengths if length > 512) / n * 100, 1),
    }
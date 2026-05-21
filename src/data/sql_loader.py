# src/data/sql_loader.py

from datasets import load_dataset, Dataset
from typing import Optional


def load_sql_data(
    val_size: int = 2000,
    max_train_samples: Optional[int] = None,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Load the sql-create-context dataset and split into train/validation.

    The dataset only ships with a train split, so we carve out
    val_size examples for validation ourselves. This is standard
    practice whenever a dataset doesn't provide an official val split.

    Args:
        val_size:          Number of examples to reserve for validation
        max_train_samples: If set, cap training examples at this number.
                           Use small values (e.g. 500) during dev to
                           avoid waiting on tokenization every run.
        seed:              Random seed for reproducible splits

    Returns:
        (train_examples, val_examples) as lists of dicts with keys:
            question  — natural language question
            context   — CREATE TABLE SQL schema string
            answer    — gold SQL query
    """
    raw = load_dataset("b-mc2/sql-create-context", split="train")

    # Shuffle before splitting so val isn't just the last 2000 examples
    raw = raw.shuffle(seed=seed)

    val_dataset: Dataset = raw.select(range(val_size))
    train_dataset: Dataset = raw.select(range(val_size, len(raw)))

    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(max_train_samples))

    def to_dicts(dataset: Dataset) -> list[dict]:
        return [
            {
                "question": ex["question"],
                "context":  ex["context"],
                "answer":   ex["answer"],
            }
            for ex in dataset
        ]

    return to_dicts(train_dataset), to_dicts(val_dataset)
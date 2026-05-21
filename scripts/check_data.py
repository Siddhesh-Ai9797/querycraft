# scripts/check_data.py
"""
Phase 1 sanity check.
Run from the querycraft root:
    python scripts/check_data.py
"""

from src.data.sql_loader import load_sql_data
from src.data.prompt_formatter import format_training_example, format_inference_prompt
from src.data.tokenizer_utils import load_tokenizer, tokenize_example, get_token_length_stats


def main():
    print("=" * 60)
    print("STEP 1: Loading dataset")
    print("=" * 60)

    train, val = load_sql_data(val_size=2000, max_train_samples=500)

    print(f"Train examples : {len(train)}")
    print(f"Val examples   : {len(val)}")

    ex = train[0]
    print(f"\nSample example:")
    print(f"  Question : {ex['question']}")
    print(f"  Context  : {ex['context'][:80]}...")
    print(f"  Answer   : {ex['answer']}")

    print("\n" + "=" * 60)
    print("STEP 2: Checking prompt format")
    print("=" * 60)

    train_str    = format_training_example(ex)
    inference_str = format_inference_prompt(ex["question"], ex["context"])

    print(f"\nTraining string ends with SQL : {train_str.strip().endswith(ex['answer'].strip())}")
    print(f"Inference prompt ends at marker : {inference_str.rstrip().endswith('### SQL:')}")
    print(f"\nFull training string preview:")
    print(train_str[:300] + "...")

    print("\n" + "=" * 60)
    print("STEP 3: Loading tokenizer (downloads ~600MB on first run)")
    print("=" * 60)

    tokenizer = load_tokenizer()
    print(f"Vocab size     : {tokenizer.vocab_size}")
    print(f"Pad token      : {tokenizer.pad_token!r}")
    print(f"EOS token      : {tokenizer.eos_token!r}")
    print(f"Padding side   : {tokenizer.padding_side}")

    print("\n" + "=" * 60)
    print("STEP 4: Tokenizing one example")
    print("=" * 60)

    tokenized = tokenize_example(ex, tokenizer)
    print(f"input_ids length      : {len(tokenized['input_ids'])}")
    print(f"attention_mask length : {len(tokenized['attention_mask'])}")
    print(f"Pad token count       : {tokenized['input_ids'].count(tokenizer.pad_token_id)}")

    print("\n" + "=" * 60)
    print("STEP 5: Token length statistics (500 examples)")
    print("=" * 60)

    stats = get_token_length_stats(train[:500], tokenizer)
    for key, val_ in stats.items():
        print(f"  {key:>15} : {val_}")

    print("\n" + "=" * 60)
    print("Phase 1 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
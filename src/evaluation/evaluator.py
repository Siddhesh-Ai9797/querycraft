# src/evaluation/evaluator.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from src.data.sql_loader import load_sql_data
from src.data.prompt_formatter import format_inference_prompt
from src.evaluation.metrics import evaluate_single, aggregate_metrics
from src.training.model_loader import MODEL_NAME, get_bnb_config


def generate_sql(
    model,
    tokenizer,
    question: str,
    schema: str,
    max_new_tokens: int = 128,
) -> str:
    """
    Generate a SQL query from a question and schema.

    This is inference mode — we feed the prompt and let
    the model complete it. We stop at the first newline
    after the SQL starts, since SQL queries are single lines.

    Args:
        model:          The loaded model (base or fine-tuned)
        tokenizer:      The tokenizer
        question:       Natural language question
        schema:         CREATE TABLE context
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated SQL string, cleaned up
    """
    prompt = format_inference_prompt(question, schema)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # Greedy decoding — deterministic output
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the prompt)
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # Clean up — take only the first line (the SQL query)
    sql = generated.strip().split("\n")[0].strip()
    return sql


def load_base_model(tokenizer_only: bool = False):
    """Load the base Phi-3 Mini without any adapter."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="left",   # Left padding for inference batching
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer_only:
        return None, tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=get_bnb_config(),
        device_map="cuda:0",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(adapter_path: str = "models/phi3-sql"):
    """
    Load the fine-tuned model by stacking the LoRA adapter
    on top of the base model.

    The adapter file is tiny (~150MB). The base model is shared —
    we load it once and apply the adapter on top.
    """
    model, tokenizer = load_base_model()

    # Load and apply the LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def run_evaluation(
    n_samples: int = 200,
    adapter_path: str = "models/phi3-sql",
):
    """
    Evaluate both base and fine-tuned models on n_samples
    from the validation set.

    Prints a side-by-side comparison table and returns
    both result sets.

    Args:
        n_samples:    Number of validation examples to evaluate
        adapter_path: Path to the saved LoRA adapter
    """
    # Load validation data
    _, val_examples = load_sql_data(val_size=2000)
    eval_examples = val_examples[:n_samples]

    print(f"\nEvaluating on {n_samples} validation examples...")
    print("=" * 60)

    results = {}

    for model_name, loader in [
        ("base",       load_base_model),
        ("finetuned",  lambda: load_finetuned_model(adapter_path)),
    ]:
        print(f"\nLoading {model_name} model...")
        model, tokenizer = loader()

        model_results = []
        for ex in tqdm(eval_examples, desc=f"Evaluating {model_name}"):
            predicted = generate_sql(
                model, tokenizer,
                ex["question"], ex["context"],
            )
            metrics = evaluate_single(
                predicted=predicted,
                gold=ex["answer"],
                schema=ex["context"],
            )
            metrics["predicted"] = predicted
            metrics["gold"]      = ex["answer"]
            metrics["question"]  = ex["question"]
            model_results.append(metrics)

        results[model_name] = model_results

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    return results


def print_report(results: dict):
    """Print a formatted comparison report."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for model_name, model_results in results.items():
        agg = aggregate_metrics(model_results)
        print(f"\n{model_name.upper()} MODEL:")
        print(f"  Exact Match Accuracy : {agg['exact_match_pct']}%")
        print(f"  Execution Accuracy   : {agg['execution_accuracy_pct']}%")
        print(f"  Mean BLEU Score      : {agg['mean_bleu']}")
        print(f"  Examples evaluated   : {agg['n_examples']}")

    # Print a few side-by-side examples
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS (first 5 examples)")
    print("=" * 60)
    for i in range(min(5, len(results["base"]))):
        base = results["base"][i]
        ft   = results["finetuned"][i]
        print(f"\nExample {i+1}:")
        print(f"  Question  : {base['question']}")
        print(f"  Gold SQL  : {base['gold']}")
        print(f"  Base      : {base['predicted']}")
        print(f"  Finetuned : {ft['predicted']}")
        print(f"  EM  base/ft: {base['exact_match']} / {ft['exact_match']}")
        print(f"  Exec base/ft: {base['execution_accuracy']} / {ft['execution_accuracy']}")
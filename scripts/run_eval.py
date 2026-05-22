# scripts/evaluate.py

"""
Run evaluation from the querycraft root:
    python scripts/evaluate.py

For a quick test on 50 examples:
    python scripts/evaluate.py --n-samples 50
"""

import argparse
import json
from pathlib import Path
from src.evaluation.evaluator import run_evaluation, print_report
from src.evaluation.metrics import aggregate_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of validation examples to evaluate (default 200)"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="models/phi3-sql",
        help="Path to saved LoRA adapter"
    )
    args = parser.parse_args()

    results = run_evaluation(
        n_samples=args.n_samples,
        adapter_path=args.adapter_path,
    )

    print_report(results)

    # Save results to JSON for your README
    output = {}
    for model_name, model_results in results.items():
        output[model_name] = aggregate_metrics(model_results)

    out_path = Path("evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
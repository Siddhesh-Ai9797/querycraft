# scripts/upload_model.py

"""
Upload the trained LoRA adapter to HuggingFace Hub.

Run from querycraft root:
    python scripts/upload_model.py --repo-name querycraft-phi3-sql
"""

import argparse
from huggingface_hub import HfApi, create_repo
from pathlib import Path


def upload_adapter(repo_name: str, adapter_path: str = "models/phi3-sql"):
    """
    Upload LoRA adapter files to HuggingFace Hub.

    Args:
        repo_name:    Name for your HF repository
        adapter_path: Local path to the saved adapter
    """
    api = HfApi()

    # Get your username automatically from the logged-in token
    user = api.whoami()["name"]
    repo_id = f"{user}/{repo_name}"

    print(f"Creating repository: {repo_id}")
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,       # Don't error if repo already exists
        private=False,       # Public so recruiters can see it
    )

    print(f"Uploading adapter from {adapter_path}...")
    api.upload_folder(
        folder_path=adapter_path,
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"\nAdapter uploaded successfully!")
    print(f"View at: https://huggingface.co/{repo_id}")
    return repo_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-name",
        type=str,
        default="querycraft-phi3-sql",
        help="HuggingFace repo name"
    )
    args = parser.parse_args()
    upload_adapter(args.repo_name)


if __name__ == "__main__":
    main()
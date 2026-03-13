#!/usr/bin/env python3
"""Upload CLI-Bench dataset to HuggingFace Hub."""

import argparse
from pathlib import Path


def collect_task_files(tasks_dir: Path) -> list[Path]:
    """Collect all YAML task files from a directory."""
    return sorted(tasks_dir.glob("*.yaml"))


def build_repo_id(org: str, benchmark: str) -> str:
    """Build HuggingFace repo ID from org and benchmark name."""
    slug = benchmark.replace("_", "-")
    return f"{org}/{slug}"


def upload_benchmark(repo_id: str, token: str) -> None:
    """Upload CLI-Bench dataset to HF Hub."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    # Create repo if needed
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)

    tasks_dir = Path("data/tasks")

    # Upload all YAML task files
    task_files = collect_task_files(tasks_dir)
    for yaml_file in task_files:
        api.upload_file(
            path_or_fileobj=str(yaml_file),
            path_in_repo=f"tasks/{yaml_file.name}",
            repo_id=repo_id,
            repo_type="dataset",
        )

    # Upload metadata
    metadata_path = Path("data/metadata.yaml")
    if metadata_path.exists():
        api.upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo="metadata.yaml",
            repo_id=repo_id,
            repo_type="dataset",
        )

    print(f"Uploaded CLI-Bench to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload CLI-Bench to HuggingFace")
    parser.add_argument(
        "--org",
        type=str,
        default="ChengyiX",
        help="HF organization",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="HuggingFace token",
    )
    args = parser.parse_args()

    repo_id = build_repo_id(args.org, "CLI-Bench")
    upload_benchmark(repo_id, args.token)


if __name__ == "__main__":
    main()

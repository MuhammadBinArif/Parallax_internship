#!/usr/bin/env python3
"""
Download a real-world text dataset and save raw records to data/raw/.

Default dataset:
- wikimedia/wikipedia
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from datasets import load_dataset


def load_streaming_dataset(dataset: str, config: str | None, split: str):
    """
    Load a Hugging Face dataset in streaming mode.

    Notes:
    - Do NOT pass trust_remote_code here.
    - Some newer versions of `datasets` no longer support it.
    """
    if config:
        return load_dataset(dataset, config, split=split, streaming=True)
    return load_dataset(dataset, split=split, streaming=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw text dataset")
    parser.add_argument(
        "--dataset",
        default="wikimedia/wikipedia",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--config",
        default="20231101.en",
        help="Dataset config name. Use empty string for datasets without config.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=50000,
        help="Number of raw documents to download",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/wikipedia_raw.jsonl"),
        help="Output raw JSONL path",
    )
    parser.add_argument(
        "--id-prefix",
        default=None,
        help="Prefix for generated IDs. Defaults to dataset name.",
    )
    args = parser.parse_args()

    if args.num_docs < 5000:
        raise ValueError("--num-docs must be at least 5000 for this task.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    config = args.config.strip() or None
    prefix = args.id_prefix or args.dataset.replace("/", "-")

    print(f"Loading dataset: {args.dataset}")
    if config:
        print(f"Using config: {config}")
    print(f"Split: {args.split}")
    print("Mode: streaming")

    try:
        ds = load_streaming_dataset(
            dataset=args.dataset,
            config=config,
            split=args.split,
        )
    except Exception as exc:
        print("\nFailed to load dataset.")
        print("Please verify the dataset name and config on Hugging Face:")
        print("https://huggingface.co/datasets")
        raise exc

    written = 0

    with args.output.open("w", encoding="utf-8") as f:
        for example in itertools.islice(ds, args.num_docs):
            record = dict(example)

            # Preserve original id if present
            if "id" in record and record["id"] is not None:
                record["_source_id"] = record["id"]

            # Generate a stable id for our pipeline
            record["_id"] = f"{prefix}-{written}"

            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            written += 1

            if written % 5000 == 0:
                print(f"Written {written} documents...")

    print(f"\nSaved {written} raw documents to {args.output}")

    if written < 5000:
        raise AssertionError(
            f"Only {written} documents were downloaded. "
            "At least 5000 are required."
        )


if __name__ == "__main__":
    main()

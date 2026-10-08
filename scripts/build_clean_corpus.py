#!/usr/bin/env python3
"""
Build a cleaned corpus from raw JSONL data.

Outputs:
- data/processed/clean_corpus.parquet
- data/processed/clean_corpus.jsonl
- reports/validation_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Allow imports from repo root when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import iter_clean_documents


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean corpus")
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw/wikipedia_raw.jsonl"),
        help="Path to raw JSONL file",
    )
    parser.add_argument(
        "--source",
        default="wikimedia/wikipedia",
        help="Dataset source name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed corpus",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/validation_report.json"),
        help="Path to validation report",
    )
    parser.add_argument(
        "--min-docs",
        type=int,
        default=5000,
        help="Minimum number of clean documents required",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=120,
        help="Minimum character count for a valid document",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="Maximum character count after truncation",
    )
    parser.add_argument(
        "--allow-non-english",
        action="store_true",
        help="Disable English language filtering",
    )
    args = parser.parse_args()

    if not args.raw.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {args.raw}\n"
            "Run scripts/download_dataset.py first."
        )

    print(f"Reading raw dataset from {args.raw}...")
    raw_docs = list(read_jsonl(args.raw))
    print(f"Loaded {len(raw_docs)} raw documents.")

    print("Cleaning documents...")
    cleaned_docs = list(
        iter_clean_documents(
            raw_docs,
            source=args.source,
            text_fields=(
                "text",
                "title",
                "content",
                "abstract",
                "article",
                "body",
                "intro",
                "life",
                "career",
                "works",
                "personal_life",
                "awards",
            ),
            id_fields=("_id", "id", "idx", "uid"),
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            require_english=not args.allow_non_english,
        )
    )

    print(f"Produced {len(cleaned_docs)} clean documents.")

    if len(cleaned_docs) < args.min_docs:
        raise AssertionError(
            f"Only {len(cleaned_docs)} clean documents were produced. "
            f"At least {args.min_docs} are required.\n"
            "Try downloading more raw documents or lowering --min-chars slightly."
        )

    df = pd.DataFrame(cleaned_docs)

    # Validation checks
    assert not df.empty, "Cleaned dataframe is empty"
    assert df["id"].is_unique, "Document IDs are not unique"
    assert df["text"].notna().all(), "Some documents have null text"
    assert (df["char_count"] >= args.min_chars).all(), "Some documents are shorter than min_chars"
    assert (df["char_count"] <= args.max_chars).all(), "Some documents exceed max_chars"
    assert (df["char_count"] == df["text"].str.len()).all(), "char_count does not match text length"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = args.output_dir / "clean_corpus.parquet"
    jsonl_path = args.output_dir / "clean_corpus.jsonl"

    print(f"Writing parquet to {parquet_path}...")
    df.to_parquet(parquet_path, index=False)

    print(f"Writing JSONL to {jsonl_path}...")
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    sample_df = df.head(3).copy()
    sample_df["text_preview"] = sample_df["text"].str.slice(0, 200)
    samples = sample_df.drop(columns=["text"]).to_dict(orient="records")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(args.raw),
        "source": args.source,
        "raw_document_count": len(raw_docs),
        "clean_document_count": len(df),
        "min_documents_required": args.min_docs,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "language_filter_enabled": not args.allow_non_english,
        "unique_ids": bool(df["id"].is_unique),
        "null_text_count": int(df["text"].isna().sum()),
        "char_count_min": int(df["char_count"].min()),
        "char_count_max": int(df["char_count"].max()),
        "word_count_mean": float(df["word_count"].mean()),
        "outputs": {
            "parquet": str(parquet_path),
            "jsonl": str(jsonl_path),
        },
        "samples": samples,
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nValidation summary:")
    print(f"- Raw documents: {len(raw_docs)}")
    print(f"- Clean documents: {len(df)}")
    print(f"- Min chars: {args.min_chars}")
    print(f"- Max chars: {args.max_chars}")
    print(f"- Language filter enabled: {not args.allow_non_english}")
    print(f"- Report written to: {args.report_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.week3.retriever_adapter import build_retriever


def main() -> None:
    retriever = build_retriever()

    queries = [
        "What is anarchism?",
        "Who was Pierre-Joseph Proudhon?",
        "Write a poem about black holes.",
    ]

    for query in queries:
        print("\n" + "=" * 80)
        print("QUERY:", query)
        print("=" * 80)

        chunks = retriever.retrieve(query, top_k=5)

        if not chunks:
            print("No chunks returned.")
            continue

        for chunk in chunks:
            print(
                f"id={chunk.id} | "
                f"score={chunk.score} | "
                f"text={chunk.text[:140]!r}"
            )


if __name__ == "__main__":
    main()

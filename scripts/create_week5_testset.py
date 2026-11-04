from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.week3.retriever_adapter import build_retriever

WORD_RE = re.compile(r"[a-z0-9]+")

QUERIES = [
    "What is anarchism?",
    "Who was Pierre-Joseph Proudhon?",
    "What is the etymology of the word anarchy?",
    "How did modern anarchism emerge from the Enlightenment?",
    "What are revolutionary and evolutionary anarchist strategies?",
    "What is libertarian socialism?",
    "How did the Spanish Civil War affect the anarchist movement?",
    "What is a stateless society?",
    "Who were William Godwin and Max Stirner?",
    "What is collectivist anarchism?",
]

REQUIRED_TERMS = {
    "What is anarchism?": ["anarchism"],
    "Who was Pierre-Joseph Proudhon?": ["proudhon"],
    "What is the etymology of the word anarchy?": ["anarchy", "greek", "anarkhia"],
    "How did modern anarchism emerge from the Enlightenment?": ["enlightenment"],
    "What are revolutionary and evolutionary anarchist strategies?": ["revolutionary", "evolutionary"],
    "What is libertarian socialism?": ["libertarian", "socialism"],
    "How did the Spanish Civil War affect the anarchist movement?": ["spanish", "civil", "war"],
    "What is a stateless society?": ["stateless"],
    "Who were William Godwin and Max Stirner?": ["godwin", "stirner"],
    "What is collectivist anarchism?": ["collectivist"],
}


def text_tokens(text: str) -> set[str]:
    return set(WORD_RE.findall((text or "").lower()))


def is_relevant(chunk_text: str, required_terms: list[str], min_hits: int = 1) -> bool:
    tokens = text_tokens(chunk_text)
    hits = sum(1 for term in required_terms if term.lower() in tokens)
    return hits >= min_hits


def main() -> None:
    retriever = build_retriever()
    try:
        total = retriever.collection.count()
    except Exception:
        total = 0

    print(f"Collection count: {total}")

    testset = []
    for query in QUERIES:
        # Retrieve a wide pool so the relevant set is meaningful
        pool = retriever.retrieve(query, top_k=min(50, max(10, total)))
        required = REQUIRED_TERMS.get(query, [])

        relevant_ids = [
            c.id for c in pool
            if is_relevant(c.text, required, min_hits=1)
        ]

        testset.append({
            "query": query,
            "required_terms": required,
            "retrieved_pool_size": len(pool),
            "relevant_ids": relevant_ids,
            "relevant_count": len(relevant_ids),
        })
        print(f"{query[:50]:50s} → relevant: {len(relevant_ids)} / {len(pool)}")

    out = ROOT / "data" / "eval" / "week5" / "retrieval_testset.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(testset, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")

    skipped = [t["query"] for t in testset if t["relevant_count"] == 0]
    if skipped:
        print(f"\n⚠️  {len(skipped)} queries have no relevant chunks. "
              f"Widen REQUIRED_TERMS or increase pool size.")


if __name__ == "__main__":
    main()

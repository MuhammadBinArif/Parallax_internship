from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.week4.vector_db_enrichment import EnrichedVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Filtered semantic search with NLP metadata")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--topic", type=int, default=None)
    parser.add_argument("--sentiment", type=str, default=None, choices=["positive", "negative", "neutral"])
    parser.add_argument("--has-person", action="store_true", default=None)
    parser.add_argument("--collection", default="week4_enriched")
    args = parser.parse_args()

    store = EnrichedVectorStore(
        persist_directory="chroma_db",
        collection_name=args.collection,
    )

    # For demo, use a simple embedding
    # In production, use the same embedding model as Week 2
    from src.embeddings import SentenceTransformerEmbedder
    embedder = SentenceTransformerEmbedder()
    result = embedder.embed([args.query])
    query_embedding = result.vectors[0].tolist()

    hits = store.query_with_filter(
        query_embedding=query_embedding,
        n_results=args.top_k,
        topic_filter=args.topic,
        sentiment_filter=args.sentiment,
        has_person_filter=args.has_person if args.has_person else None,
    )

    print(f"\nQuery: {args.query}")
    print(f"Filters: topic={args.topic}, sentiment={args.sentiment}, has_person={args.has_person}")
    print(f"Results: {len(hits)}")
    print("-" * 80)

    for rank, hit in enumerate(hits, 1):
        print(f"[{rank}] id={hit['id']}")
        print(f"    score={hit['score']:.4f}")
        print(f"    metadata={json.dumps(hit['metadata'], indent=6)[:200]}")
        print(f"    text={hit['text'][:150]}")
        print()


if __name__ == "__main__":
    main()

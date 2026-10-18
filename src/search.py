from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from src.embeddings import SentenceTransformerEmbedder
from src.vector_store import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic search against ChromaDB.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--persist-dir", default="chroma_db")
    parser.add_argument("--collection", default="week2_chunks")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder(model_name=args.model)
    store = ChromaVectorStore(
        persist_directory=args.persist_dir,
        collection_name=args.collection,
    )
    query_embedding, embedding_seconds = embedder.embed_one(args.query)
    hits = store.query(query_embedding, n_results=args.top_k)

    if args.json:
        print(json.dumps({
            "query": args.query,
            "embedding_seconds": embedding_seconds,
            "results": [asdict(h) for h in hits],
        }, indent=2))
        return

    print(f"Query: {args.query}")
    print(f"Embedding latency: {embedding_seconds * 1000:.2f} ms")
    print(f"Results: {len(hits)}")
    print("-" * 80)
    for rank, hit in enumerate(hits, start=1):
        preview = hit.document.replace("\n", " ")
        if len(preview) > 240:
            preview = preview[:237] + "..."
        print(f"[{rank}] id={hit.id}")
        print(f"    score={hit.score:.4f} distance={hit.distance:.4f}")
        print(f"    metadata={hit.metadata}")
        print(f"    text={preview}")
        print()


if __name__ == "__main__":
    main()

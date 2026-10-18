from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from src.embeddings import SentenceTransformerEmbedder
from src.vector_store import ChromaVectorStore

DEFAULT_QUERIES = [
    "What is text chunking?",
    "Why are embeddings useful for search?",
    "How does ChromaDB store vectors?",
    "What is cosine similarity?",
    "How do I benchmark retrieval latency?",
]


def load_queries(path: Path) -> List[str]:
    if not path.exists():
        print(f"Query file not found: {path}. Using default queries.")
        return DEFAULT_QUERIES[:]
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "queries" in data:
            data = data["queries"]
        if not isinstance(data, list):
            raise ValueError("JSON query file must contain a list or {'queries': [...]}")
        queries = []
        for item in data:
            queries.append(str(item.get("query", "")) if isinstance(item, dict) else str(item))
        return [q.strip() for q in queries if q.strip()]
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] * (c - k) + s[c] * (k - f))


def mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def run_benchmark(queries, persist_dir, collection, model, top_k, repeats,
                  warmup_repeats, output_csv: Path, output_json: Path) -> Dict[str, Any]:
    embedder = SentenceTransformerEmbedder(model_name=model)
    store = ChromaVectorStore(persist_directory=persist_dir, collection_name=collection)

    if warmup_repeats > 0 and queries:
        print(f"Running {warmup_repeats} warm-up iteration(s)...")
        for _ in range(warmup_repeats):
            qe, _ = embedder.embed_one(queries[0])
            store.query(qe, n_results=top_k)

    rows: List[Dict[str, Any]] = []
    per_query: List[Dict[str, Any]] = []
    all_total, all_embed, all_search, all_hits = [], [], [], []

    for query in queries:
        qrows = []
        for r in range(repeats):
            start = time.perf_counter()
            qe, es = embedder.embed_one(query)
            mid = time.perf_counter()
            hits = store.query(qe, n_results=top_k)
            end = time.perf_counter()
            row = {
                "query": query, "repeat": r,
                "embed_ms": es * 1000.0,
                "search_ms": (end - mid) * 1000.0,
                "total_ms": (end - start) * 1000.0,
                "hits": len(hits),
                "top_distance": hits[0].distance if hits else None,
                "top_score": hits[0].score if hits else None,
            }
            rows.append(row); qrows.append(row)
            all_total.append(row["total_ms"]); all_embed.append(row["embed_ms"])
            all_search.append(row["search_ms"]); all_hits.append(row["hits"])
        totals = [x["total_ms"] for x in qrows]
        embeds = [x["embed_ms"] for x in qrows]
        searches = [x["search_ms"] for x in qrows]
        hits = [x["hits"] for x in qrows]
        per_query.append({
            "query": query, "repeats": repeats,
            "mean_total_ms": mean(totals),
            "p50_total_ms": percentile(totals, 50),
            "p95_total_ms": percentile(totals, 95),
            "mean_embed_ms": mean(embeds),
            "mean_search_ms": mean(searches),
            "mean_hits": mean([float(h) for h in hits]),
        })

    overall = {
        "num_queries": len(queries),
        "repeats_per_query": repeats,
        "total_measurements": len(rows),
        "mean_total_ms": mean(all_total),
        "p50_total_ms": percentile(all_total, 50),
        "p95_total_ms": percentile(all_total, 95),
        "mean_embed_ms": mean(all_embed),
        "mean_search_ms": mean(all_search),
        "mean_hits": mean([float(h) for h in all_hits]),
    }
    summary = {
        "config": {
            "persist_dir": persist_dir, "collection": collection, "model": model,
            "top_k": top_k, "repeats": repeats, "warmup_repeats": warmup_repeats,
        },
        "overall": overall,
        "per_query": per_query,
    }
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval latency.")
    parser.add_argument("--queries", default="data/queries/benchmark_queries.json")
    parser.add_argument("--persist-dir", default="chroma_db")
    parser.add_argument("--collection", default="week2_chunks")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--output-csv", default="reports/week2/benchmark.csv")
    parser.add_argument("--output-json", default="reports/week2/benchmark_summary.json")
    args = parser.parse_args()

    queries = load_queries(Path(args.queries))
    if not queries:
        raise SystemExit("No queries provided.")

    summary = run_benchmark(
        queries=queries, persist_dir=args.persist_dir, collection=args.collection,
        model=args.model, top_k=args.top_k, repeats=args.repeats,
        warmup_repeats=args.warmup_repeats, output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
    )
    print("Benchmark complete.")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()

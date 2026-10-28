from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.week4.nlp_pipeline import NLPPipeline


def load_corpus(path: Path, limit: int = 0) -> list[dict]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
            if limit > 0 and len(docs) >= limit:
                break
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 4 NLP Analysis Pipeline")
    parser.add_argument("--input", default="data/processed/clean_corpus.jsonl")
    parser.add_argument("--limit-docs", type=int, default=500)
    parser.add_argument("--n-topics", type=int, default=10)
    parser.add_argument("--use-bertopic", action="store_true")
    parser.add_argument("--output", default="reports/week4/nlp_metadata.jsonl")
    parser.add_argument("--log", default="reports/week4/nlp_pipeline_log.json")
    args = parser.parse_args()

    input_path = ROOT / args.input
    output_path = ROOT / args.output
    log_path = ROOT / args.log

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {input_path}...")
    docs = load_corpus(input_path, limit=args.limit_docs)
    print(f"Loaded {len(docs)} documents")

    texts = [d["text"] for d in docs]

    # Initialize pipeline
    pipeline = NLPPipeline(
        n_topics=args.n_topics,
        use_bertopic=args.use_bertopic,
        sentiment_method="vader",
    )

    # Fit topics
    print("\n=== Topic Modeling ===")
    pipeline.fit_topics(texts)

    # Analyze all documents
    print("\n=== NLP Analysis ===")
    start = time.perf_counter()
    results = pipeline.analyze_batch(texts, batch_size=50)
    elapsed = time.perf_counter() - start
    print(f"Analysis complete in {elapsed:.2f}s")

    # Save results
    with output_path.open("w", encoding="utf-8") as f:
        for doc, result in zip(docs, results):
            enriched = {
                "id": doc["id"],
                "text": doc["text"][:200],
                **{k: v for k, v in result.items() if k != "entities"},
                "entity_count": result.get("entity_count", 0),
            }
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    # Statistics
    topic_counts = {}
    sentiment_counts = {}
    edge_cases = {}

    for r in results:
        tid = r["topic_id"]
        topic_counts[tid] = topic_counts.get(tid, 0) + 1

        sl = r["sentiment_label"]
        sentiment_counts[sl] = sentiment_counts.get(sl, 0) + 1

        ec = r.get("edge_case")
        if ec:
            edge_cases[ec] = edge_cases.get(ec, 0) + 1

    log_data = {
        "documents_processed": len(docs),
        "n_topics": args.n_topics,
        "use_bertopic": args.use_bertopic,
        "sentiment_method": "vader",
        "total_seconds": round(elapsed, 2),
        "avg_seconds_per_doc": round(elapsed / max(1, len(docs)), 4),
        "topic_distribution": dict(sorted(topic_counts.items())),
        "sentiment_distribution": sentiment_counts,
        "edge_cases": edge_cases,
        "topic_labels": pipeline.topic_modeler.get_topic_labels() if pipeline.topic_modeler else [],
    }

    log_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")

    print(f"\n=== Results ===")
    print(f"Documents processed: {len(docs)}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Topics: {len(topic_counts)}")
    print(f"Sentiment distribution: {sentiment_counts}")
    print(f"Edge cases: {edge_cases}")
    print(f"\nSaved to: {output_path}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()

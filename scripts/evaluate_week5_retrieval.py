from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.week3.retriever_adapter import build_retriever

KS = (1, 3, 5, 10)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for rid in top_k if rid in relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for rid in top_k if rid in relevant) / len(relevant)


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, rid in enumerate(retrieved, 1):
        if rid in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    for i, rid in enumerate(retrieved, 1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    dcg = 0.0
    for i, rid in enumerate(top_k, 1):
        rel = 1.0 if rid in relevant else 0.0
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    ideal_rels = [1.0] * min(len(relevant), k)
    idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal_rels, 1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / idcg


def main() -> None:
    testset_path = ROOT / "data" / "eval" / "week5" / "retrieval_testset.json"
    if not testset_path.exists():
        raise SystemExit(f"Missing test set: {testset_path}. Run create_week5_testset.py first.")

    testset = json.loads(testset_path.read_text(encoding="utf-8"))
    retriever = build_retriever()

    max_k = max(KS)
    results = []

    print(f"Evaluating {len(testset)} queries at K={KS}\n")

    for item in testset:
        query = item["query"]
        relevant = set(item["relevant_ids"])
        if not relevant:
            print(f"  SKIP (no relevant chunks): {query[:60]}")
            continue

        start = time.perf_counter()
        retrieved = retriever.retrieve(query, top_k=max_k)
        latency_ms = (time.perf_counter() - start) * 1000.0

        retrieved_ids = [c.id for c in retrieved]

        row = {"query": query, "latency_ms": round(latency_ms, 2),
               "relevant_count": len(relevant), "retrieved_count": len(retrieved_ids)}

        for k in KS:
            row[f"precision@{k}"] = round(precision_at_k(retrieved_ids, relevant, k), 4)
            row[f"recall@{k}"] = round(recall_at_k(retrieved_ids, relevant, k), 4)

        row["ap"] = round(average_precision(retrieved_ids, relevant), 4)
        row["mrr"] = round(mrr(retrieved_ids, relevant), 4)
        row["ndcg@10"] = round(ndcg_at_k(retrieved_ids, relevant, 10), 4)

        results.append(row)
        print(f"  {query[:55]:55s}  P@5={row['precision@5']:.3f}  R@5={row['recall@5']:.3f}  MRR={row['mrr']:.3f}")

    if not results:
        raise SystemExit("No valid queries evaluated. Fix test set.")

    summary = {
        "num_queries": len(results),
        "ks": list(KS),
    }
    for k in KS:
        summary[f"mean_precision@{k}"] = round(mean(r[f"precision@{k}"] for r in results), 4)
        summary[f"mean_recall@{k}"] = round(mean(r[f"recall@{k}"] for r in results), 4)
    summary["mean_map"] = round(mean(r["ap"] for r in results), 4)
    summary["mean_mrr"] = round(mean(r["mrr"] for r in results), 4)
    summary["mean_ndcg@10"] = round(mean(r["ndcg@10"] for r in results), 4)
    summary["mean_latency_ms"] = round(mean(r["latency_ms"] for r in results), 2)

    out_dir = ROOT / "reports" / "week5"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "retrieval_eval.json").write_text(
        json.dumps({"summary": summary, "per_query": results}, indent=2),
        encoding="utf-8",
    )

    print("\n=== Retrieval Evaluation Summary ===")
    for k in KS:
        print(f"Precision@{k}: {summary[f'mean_precision@{k}']:.4f}   "
              f"Recall@{k}: {summary[f'mean_recall@{k}']:.4f}")
    print(f"MAP:        {summary['mean_map']:.4f}")
    print(f"MRR:        {summary['mean_mrr']:.4f}")
    print(f"NDCG@10:    {summary['mean_ndcg@10']:.4f}")
    print(f"Latency:    {summary['mean_latency_ms']:.2f} ms")
    print(f"\nSaved: reports/week5/retrieval_eval.json")


if __name__ == "__main__":
    main()

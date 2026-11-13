from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def safe_load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    consolidated: dict = {}

    # --- Week 1: validation report ---
    v = safe_load(REPORTS / "week1" / "validation_report.json")
    if v:
        consolidated["week1"] = {
            "raw_documents": v.get("raw_document_count"),
            "clean_documents": v.get("clean_document_count"),
            "unique_ids": v.get("unique_ids"),
            "null_text_count": v.get("null_text_count"),
        }

    # --- Week 2: ingest + benchmark ---
    ingest = safe_load(REPORTS / "week2" / "ingest_log.json")
    bench = safe_load(REPORTS / "week2" / "benchmark_summary.json")
    if ingest or bench:
        overall = bench.get("overall", {}) if bench else {}
        consolidated["week2"] = {
            "documents_processed": ingest.get("documents_processed"),
            "chunks_created": ingest.get("chunks_created"),
            "upserted": ingest.get("upserted"),
            "skipped": ingest.get("skipped"),
            "collection_count": ingest.get("collection_count_after_ingest"),
            "embedding_model": ingest.get("embedding_model"),
            "embedding_dimension": ingest.get("embedding_dimension"),
            "mean_total_ms": overall.get("mean_total_ms"),
            "mean_embed_ms": overall.get("mean_embed_ms"),
            "mean_search_ms": overall.get("mean_search_ms"),
            "p95_total_ms": overall.get("p95_total_ms"),
            "num_queries": overall.get("num_queries"),
            "repeats_per_query": overall.get("repeats_per_query"),
        }

    # --- Week 3: latency logs (aggregate) ---
    w3_log = REPORTS / "week3" / "latency_logs.jsonl"
    if w3_log.exists():
        rows = []
        for line in w3_log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        if rows:
            def m(key: str) -> float:
                vals = [
                    r["latency_ms"][key] for r in rows
                    if r.get("latency_ms", {}).get(key) is not None
                ]
                return round(sum(vals) / len(vals), 2) if vals else 0.0

            categories: dict = {}
            for r in rows:
                categories[r.get("category", "unknown")] = categories.get(r.get("category", "unknown"), 0) + 1

            consolidated["week3"] = {
                "num_requests": len(rows),
                "mean_retrieval_ms": m("retrieval_ms"),
                "mean_prompt_build_ms": m("prompt_build_ms"),
                "mean_generation_ms": m("generation_ms"),
                "mean_validation_ms": m("validation_ms"),
                "mean_total_ms": m("total_ms"),
                "category_distribution": categories,
            }

    # --- Week 4: NLP pipeline + sentiment eval + enrichment ---
    w4_pipe = safe_load(REPORTS / "week4" / "nlp_pipeline_log.json")
    w4_sent = safe_load(REPORTS / "week4" / "sentiment_evaluation.json")
    w4_enr = safe_load(REPORTS / "week4" / "enrichment_log.json")
    if w4_pipe or w4_sent or w4_enr:
        consolidated["week4"] = {
            "documents_analyzed": w4_pipe.get("documents_processed"),
            "n_topics": w4_pipe.get("n_topics"),
            "sentiment_method": w4_pipe.get("sentiment_method"),
            "avg_seconds_per_doc": w4_pipe.get("avg_seconds_per_doc"),
            "topic_distribution": w4_pipe.get("topic_distribution"),
            "sentiment_distribution": w4_pipe.get("sentiment_distribution"),
            "edge_cases": w4_pipe.get("edge_cases"),
            "sentiment_accuracy": w4_sent.get("accuracy"),
            "sentiment_macro_f1": w4_sent.get("macro_f1"),
            "enriched_upserted": w4_enr.get("upserted"),
            "enriched_collection_count": w4_enr.get("collection_count"),
        }

    # --- Week 5: retrieval + e2e ---
    w5_ret = safe_load(REPORTS / "week5" / "retrieval_eval.json")
    w5_e2e = safe_load(REPORTS / "week5" / "e2e_eval.json")
    if w5_ret or w5_e2e:
        r_sum = w5_ret.get("summary", {}) if w5_ret else {}
        e_sum = w5_e2e.get("summary", {}) if w5_e2e else {}
        consolidated["week5"] = {
            "retrieval_num_queries": r_sum.get("num_queries"),
            "mean_precision@1": r_sum.get("mean_precision@1"),
            "mean_precision@3": r_sum.get("mean_precision@3"),
            "mean_precision@5": r_sum.get("mean_precision@5"),
            "mean_precision@10": r_sum.get("mean_precision@10"),
            "mean_recall@1": r_sum.get("mean_recall@1"),
            "mean_recall@3": r_sum.get("mean_recall@3"),
            "mean_recall@5": r_sum.get("mean_recall@5"),
            "mean_recall@10": r_sum.get("mean_recall@10"),
            "mean_map": r_sum.get("mean_map"),
            "mean_mrr": r_sum.get("mean_mrr"),
            "mean_ndcg@10": r_sum.get("mean_ndcg@10"),
            "retrieval_mean_latency_ms": r_sum.get("mean_latency_ms"),
            "e2e_num_queries": e_sum.get("num_queries"),
            "e2e_category_accuracy": e_sum.get("category_accuracy"),
            "e2e_citation_presence_rate": e_sum.get("citation_presence_rate"),
            "e2e_mean_pipeline_ms": e_sum.get("mean_pipeline_total_ms"),
            "e2e_p95_pipeline_ms": e_sum.get("p95_total_ms"),
            "e2e_mean_tokens": e_sum.get("mean_total_tokens"),
        }

    out = REPORTS / "consolidated_benchmarks.json"
    out.write_text(json.dumps(consolidated, indent=2), encoding="utf-8")

    # --- Emit markdown ---
    print("| Area | Metric | Value |")
    print("|---|---|---:|")

    def emit(week: str, label: str, key: str) -> None:
        val = consolidated.get(week, {}).get(key)
        if val is not None:
            print(f"| {label} | {key} | {val} |")

    for k in ["raw_documents", "clean_documents", "unique_ids", "null_text_count"]:
        emit("week1", "Week 1", k)
    for k in ["documents_processed", "chunks_created", "upserted", "skipped", "collection_count",
              "embedding_model", "embedding_dimension", "mean_total_ms", "mean_embed_ms",
              "mean_search_ms", "p95_total_ms", "num_queries", "repeats_per_query"]:
        emit("week2", "Week 2", k)
    for k in ["num_requests", "mean_retrieval_ms", "mean_generation_ms",
              "mean_validation_ms", "mean_total_ms"]:
        emit("week3", "Week 3", k)
    for k in ["documents_analyzed", "n_topics", "sentiment_method", "sentiment_accuracy",
              "sentiment_macro_f1", "enriched_upserted", "enriched_collection_count"]:
        emit("week4", "Week 4", k)
    for k in ["retrieval_num_queries", "mean_precision@5", "mean_recall@5", "mean_map",
              "mean_mrr", "mean_ndcg@10", "retrieval_mean_latency_ms",
              "e2e_category_accuracy", "e2e_citation_presence_rate",
              "e2e_mean_pipeline_ms", "e2e_p95_pipeline_ms", "e2e_mean_tokens"]:
        emit("week5", "Week 5", k)

    print(f"\nSaved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

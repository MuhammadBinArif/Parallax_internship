"""
Compile all evaluation metrics from weeks 1-5 into one JSON file.

Sources:
    reports/week1/validation_report.json
    reports/week1/verify_env.json
    reports/week2/ingest_log.json
    reports/week2/benchmark_summary.json
    reports/week3/latency_logs.jsonl
    reports/week3/eval_results.json
    reports/week4/nlp_pipeline_log.json
    reports/week4/sentiment_evaluation.json
    reports/week4/enrichment_log.json
    reports/week5/retrieval_eval.json
    reports/week5/e2e_eval.json

Output:
    reports/benchmarks.json
    reports/benchmarks.md
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        print(f"  SKIP (missing): {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  SKIP (bad json): {path}  ({exc})")
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"  SKIP (missing): {path}")
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main() -> None:
    print("Compiling benchmarks from reports/week1 ... reports/week5")

    benchmarks: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "weeks": {},
    }

    # ---- Week 1 ----
    w1_validation = load_json(REPORTS / "week1" / "validation_report.json")
    w1_env = load_json(REPORTS / "week1" / "verify_env.json")
    benchmarks["weeks"]["week1"] = {
        "raw_documents": (w1_validation or {}).get("raw_document_count"),
        "clean_documents": (w1_validation or {}).get("clean_document_count"),
        "unique_ids": (w1_validation or {}).get("unique_ids"),
        "null_text_count": (w1_validation or {}).get("null_text_count"),
        "language_filter": (w1_validation or {}).get("language_filter_enabled"),
        "env_passed": (w1_env or {}).get("passed"),
    }

    # ---- Week 2 ----
    w2_ingest = load_json(REPORTS / "week2" / "ingest_log.json")
    w2_bench = load_json(REPORTS / "week2" / "benchmark_summary.json")
    w2_overall = (w2_bench or {}).get("overall", {})
    benchmarks["weeks"]["week2"] = {
        "documents_processed": (w2_ingest or {}).get("documents_processed"),
        "chunks_created": (w2_ingest or {}).get("chunks_created"),
        "upserted": (w2_ingest or {}).get("upserted"),
        "skipped": (w2_ingest or {}).get("skipped"),
        "embedding_model": (w2_ingest or {}).get("embedding_model"),
        "embedding_dimension": (w2_ingest or {}).get("embedding_dimension"),
        "collection_count": (w2_ingest or {}).get("collection_count_after_ingest"),
        "mean_retrieval_ms": w2_overall.get("mean_total_ms"),
        "mean_embed_ms": w2_overall.get("mean_embed_ms"),
        "mean_search_ms": w2_overall.get("mean_search_ms"),
        "p50_total_ms": w2_overall.get("p50_total_ms"),
        "p95_total_ms": w2_overall.get("p95_total_ms"),
        "queries_benchmarked": w2_overall.get("num_queries"),
    }

    # ---- Week 3 ----
    w3_logs = load_jsonl(REPORTS / "week3" / "latency_logs.jsonl")
    w3_eval = load_json(REPORTS / "week3" / "eval_results.json") or []
    w3_categories: dict[str, int] = {}
    w3_statuses: dict[str, int] = {}
    for row in w3_logs:
        cat = row.get("category", "unknown")
        st = row.get("status", "unknown")
        w3_categories[cat] = w3_categories.get(cat, 0) + 1
        w3_statuses[st] = w3_statuses.get(st, 0) + 1

    latency_vals = [r.get("latency_ms", {}).get("total_ms", 0) for r in w3_logs if isinstance(r, dict)]
    gen_vals = [r.get("latency_ms", {}).get("generation_ms", 0) for r in w3_logs if isinstance(r, dict)]
    ret_vals = [r.get("latency_ms", {}).get("retrieval_ms", 0) for r in w3_logs if isinstance(r, dict)]
    token_vals = [r.get("tokens", {}).get("total_tokens", 0) for r in w3_logs if isinstance(r, dict)]

    benchmarks["weeks"]["week3"] = {
        "queries_evaluated": len(w3_eval) if isinstance(w3_eval, list) else len(w3_logs),
        "categories": w3_categories,
        "statuses": w3_statuses,
        "mean_total_ms": round(statistics.mean(latency_vals), 2) if latency_vals else None,
        "mean_generation_ms": round(statistics.mean(gen_vals), 2) if gen_vals else None,
        "mean_retrieval_ms": round(statistics.mean(ret_vals), 2) if ret_vals else None,
        "mean_total_tokens": round(statistics.mean(token_vals), 1) if token_vals else None,
        "llm_provider": (w3_logs[0].get("provider") if w3_logs else None),
        "llm_model": (w3_logs[0].get("model") if w3_logs else None),
    }

    # ---- Week 4 ----
    w4_pipeline = load_json(REPORTS / "week4" / "nlp_pipeline_log.json")
    w4_sentiment = load_json(REPORTS / "week4" / "sentiment_evaluation.json")
    w4_enrich = load_json(REPORTS / "week4" / "enrichment_log.json")
    benchmarks["weeks"]["week4"] = {
        "documents_analyzed": (w4_pipeline or {}).get("documents_processed"),
        "n_topics": (w4_pipeline or {}).get("n_topics"),
        "sentiment_distribution": (w4_pipeline or {}).get("sentiment_distribution"),
        "edge_cases": (w4_pipeline or {}).get("edge_cases"),
        "sentiment_accuracy": (w4_sentiment or {}).get("accuracy"),
        "sentiment_macro_f1": (w4_sentiment or {}).get("macro_f1"),
        "enriched_chunks": (w4_enrich or {}).get("upserted"),
        "enriched_collection_count": (w4_enrich or {}).get("collection_count"),
    }

    # ---- Week 5 ----
    w5_retrieval = load_json(REPORTS / "week5" / "retrieval_eval.json")
    w5_e2e = load_json(REPORTS / "week5" / "e2e_eval.json")
    w5_api_logs = load_jsonl(REPORTS / "week5" / "api_request_logs.jsonl")

    ret_summary = (w5_retrieval or {}).get("summary", {})
    e2e_summary = (w5_e2e or {}).get("summary", {})

    benchmarks["weeks"]["week5"] = {
        "retrieval": {
            "num_queries": ret_summary.get("num_queries"),
            "precision@1": ret_summary.get("mean_precision@1"),
            "precision@5": ret_summary.get("mean_precision@5"),
            "precision@10": ret_summary.get("mean_precision@10"),
            "recall@5": ret_summary.get("mean_recall@5"),
            "recall@10": ret_summary.get("mean_recall@10"),
            "map": ret_summary.get("mean_map"),
            "mrr": ret_summary.get("mean_mrr"),
            "ndcg@10": ret_summary.get("mean_ndcg@10"),
            "mean_latency_ms": ret_summary.get("mean_latency_ms"),
        },
        "e2e": {
            "num_queries": e2e_summary.get("num_queries"),
            "category_accuracy": e2e_summary.get("category_accuracy"),
            "citation_presence_rate": e2e_summary.get("citation_presence_rate"),
            "mean_pipeline_ms": e2e_summary.get("mean_pipeline_total_ms"),
            "p95_pipeline_ms": e2e_summary.get("p95_total_ms"),
            "mean_total_tokens": e2e_summary.get("mean_total_tokens"),
        },
        "api_requests_logged": len(w5_api_logs),
    }

    # ---- Aggregate ----
    benchmarks["aggregate"] = {
        "total_pipeline_weeks": 5,
        "total_documents": benchmarks["weeks"]["week1"].get("clean_documents"),
        "total_chunks_indexed": benchmarks["weeks"]["week2"].get("collection_count"),
        "end_to_end_mean_latency_ms": benchmarks["weeks"]["week5"]["e2e"].get("mean_pipeline_ms"),
        "retrieval_mean_latency_ms": benchmarks["weeks"]["week2"].get("mean_retrieval_ms"),
    }

    # Write
    out_json = REPORTS / "benchmarks.json"
    out_json.write_text(json.dumps(benchmarks, indent=2), encoding="utf-8")
    print(f"\n✅  Wrote {out_json}")

    # ---- Markdown summary ----
    md = ["# Benchmark Summary", ""]
    md.append(f"_Generated: {benchmarks['generated_at']}_")
    md.append("")

    md.append("## Week 1 — Data Foundation")
    w1 = benchmarks["weeks"]["week1"]
    md.append(f"- Raw documents: **{w1.get('raw_documents'):,}**" if w1.get('raw_documents') else "- Raw documents: n/a")
    md.append(f"- Clean documents: **{w1.get('clean_documents'):,}**" if w1.get('clean_documents') else "- Clean documents: n/a")
    md.append(f"- Environment checks passed: {w1.get('env_passed')}")
    md.append("")

    md.append("## Week 2 — Retrieval")
    w2 = benchmarks["weeks"]["week2"]
    md.append(f"- Chunks indexed: **{w2.get('collection_count')}**")
    md.append(f"- Embedding model: `{w2.get('embedding_model')}` ({w2.get('embedding_dimension')}-dim)")
    md.append(f"- Mean retrieval latency: **{w2.get('mean_retrieval_ms')} ms**")
    md.append(f"- Mean embed / search split: {w2.get('mean_embed_ms')} ms / {w2.get('mean_search_ms')} ms")
    md.append(f"- P95 latency: {w2.get('p95_total_ms')} ms")
    md.append("")

    md.append("## Week 3 — Generation")
    w3 = benchmarks["weeks"]["week3"]
    md.append(f"- Provider / model: `{w3.get('llm_provider')}` / `{w3.get('llm_model')}`")
    md.append(f"- Mean total latency: **{w3.get('mean_total_ms')} ms**")
    md.append(f"- Mean generation latency: {w3.get('mean_generation_ms')} ms")
    md.append(f"- Mean retrieval latency: {w3.get('mean_retrieval_ms')} ms")
    md.append(f"- Mean tokens: {w3.get('mean_total_tokens')}")
    md.append(f"- Category distribution: {w3.get('categories')}")
    md.append("")

    md.append("## Week 4 — NLP Enrichment")
    w4 = benchmarks["weeks"]["week4"]
    md.append(f"- Documents analyzed: {w4.get('documents_analyzed')}")
    md.append(f"- Topics discovered: **{w4.get('n_topics')}**")
    md.append(f"- Sentiment accuracy: **{w4.get('sentiment_accuracy')}**")
    md.append(f"- Sentiment macro F1: {w4.get('sentiment_macro_f1')}")
    md.append(f"- Enriched chunks: {w4.get('enriched_chunks')}")
    md.append(f"- Sentiment distribution: {w4.get('sentiment_distribution')}")
    md.append("")

    md.append("## Week 5 — API + Evaluation")
    w5 = benchmarks["weeks"]["week5"]
    r5 = w5["retrieval"]
    e5 = w5["e2e"]
    md.append("### Retrieval")
    md.append(f"- Queries: {r5.get('num_queries')}")
    md.append(f"- **Precision@1: {r5.get('precision@1')}**")
    md.append(f"- Precision@5: {r5.get('precision@5')}")
    md.append(f"- Recall@10: {r5.get('recall@10')}")
    md.append(f"- MAP: {r5.get('map')}")
    md.append(f"- MRR: {r5.get('mrr')}")
    md.append(f"- nDCG@10: {r5.get('ndcg@10')}")
    md.append("")
    md.append("### End-to-End Generation")
    md.append(f"- Queries: {e5.get('num_queries')}")
    md.append(f"- Category accuracy: **{e5.get('category_accuracy')}**")
    md.append(f"- Citation presence: {e5.get('citation_presence_rate')}")
    md.append(f"- Mean pipeline latency: {e5.get('mean_pipeline_ms')} ms")
    md.append(f"- P95 latency: {e5.get('p95_pipeline_ms')} ms")
    md.append("")
    md.append(f"- Total API requests logged: {w5.get('api_requests_logged')}")

    md.append("")
    md.append("## Aggregate")
    agg = benchmarks["aggregate"]
    md.append(f"- Total clean documents: **{agg.get('total_documents'):,}**" if agg.get('total_documents') else "- Total clean documents: n/a")
    md.append(f"- Total chunks indexed: **{agg.get('total_chunks_indexed')}**")
    md.append(f"- End-to-end mean latency: **{agg.get('end_to_end_mean_latency_ms')} ms**")
    md.append(f"- Retrieval mean latency: {agg.get('retrieval_mean_latency_ms')} ms")

    out_md = REPORTS / "benchmarks.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"✅  Wrote {out_md}")


if __name__ == "__main__":
    main()

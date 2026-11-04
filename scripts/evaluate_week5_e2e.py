from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.week3.llm_client import LLMClient
from src.week3.rag_service import RAGService
from src.week3.retriever_adapter import build_retriever
from src.week3.settings import load_llm_config, load_rag_config

E2E_TEST_QUERIES = [
    {"query": "What is anarchism?",                                     "expected_category": "answered"},
    {"query": "Who was Pierre-Joseph Proudhon?",                        "expected_category": "answered"},
    {"query": "Write a poem about black holes.",                        "expected_category": "out_of_domain"},
    {"query": "Who won the FIFA World Cup in 2014?",                    "expected_category": "out_of_domain"},
    {"query": "What is the current leadership of anarchism in 2026?",   "expected_category": "insufficient_context"},
    {"query": "What is the price of SolarFlex panels in Germany?",      "expected_category": "insufficient_context"},
]


def main() -> None:
    llm_config = load_llm_config()
    rag_config = load_rag_config()

    llm_client = LLMClient(
        provider=llm_config.provider,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        model=llm_config.model,
        timeout=llm_config.timeout,
        max_retries=llm_config.max_retries,
    )
    retriever = build_retriever()
    service = RAGService(retriever=retriever, llm_client=llm_client, settings=rag_config)

    results = []
    total_start = time.perf_counter()

    print(f"Running E2E evaluation on {len(E2E_TEST_QUERIES)} queries\n")

    for item in E2E_TEST_QUERIES:
        query = item["query"]
        expected = item["expected_category"]

        t0 = time.perf_counter()
        try:
            result = service.answer(query)
            error = None
        except Exception as exc:
            result = {
                "query": query, "status": "error", "category": "error",
                "answer": "", "citations": [],
                "latency": {"total_ms": 0.0, "retrieval_ms": 0.0,
                            "generation_ms": 0.0, "validation_ms": 0.0},
                "token_usage": {"total_tokens": 0},
            }
            error = type(exc).__name__
        wall_ms = (time.perf_counter() - t0) * 1000.0

        actual = result["category"]
        correct = actual == expected
        check = "✓" if correct else "✗"
        print(f"  {check} {query[:50]:50s}  expected={expected:22s}  actual={actual}")

        results.append({
            "query": query,
            "expected_category": expected,
            "actual_category": actual,
            "status": result["status"],
            "correct": correct,
            "wall_ms": round(wall_ms, 2),
            "pipeline_total_ms": result["latency"]["total_ms"],
            "retrieval_ms": result["latency"]["retrieval_ms"],
            "generation_ms": result["latency"]["generation_ms"],
            "validation_ms": result["latency"]["validation_ms"],
            "total_tokens": result["token_usage"]["total_tokens"],
            "citation_count": len(result["citations"]),
            "error_type": error,
        })
        time.sleep(1)  # rate-limit friendly

    wall_total_s = time.perf_counter() - total_start

    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results) if results else 0.0

    answered = [r for r in results if r["actual_category"] == "answered"]
    answered_cited = [r for r in answered if r["citation_count"] > 0]
    citation_rate = (len(answered_cited) / len(answered)) if answered else 0.0

    totals = sorted(r["pipeline_total_ms"] for r in results)
    p95_idx = int(round(0.95 * (len(totals) - 1)))

    summary = {
        "num_queries": len(results),
        "category_accuracy": round(accuracy, 4),
        "correct": correct_count,
        "citation_presence_rate": round(citation_rate, 4),
        "answered_queries": len(answered),
        "mean_wall_ms": round(mean(r["wall_ms"] for r in results), 2),
        "mean_pipeline_total_ms": round(mean(r["pipeline_total_ms"] for r in results), 2),
        "mean_retrieval_ms": round(mean(r["retrieval_ms"] for r in results), 2),
        "mean_generation_ms": round(mean(r["generation_ms"] for r in results), 2),
        "mean_validation_ms": round(mean(r["validation_ms"] for r in results), 2),
        "p50_total_ms": round(totals[len(totals) // 2], 2) if totals else 0.0,
        "p95_total_ms": round(totals[p95_idx], 2) if totals else 0.0,
        "mean_total_tokens": round(mean(r["total_tokens"] for r in results), 1),
        "total_wall_seconds": round(wall_total_s, 2),
    }

    out_dir = ROOT / "reports" / "week5"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e2e_eval.json").write_text(
        json.dumps({"summary": summary, "per_query": results}, indent=2),
        encoding="utf-8",
    )

    print("\n=== E2E Evaluation Summary ===")
    print(f"Category accuracy:      {summary['category_accuracy']:.2%} ({correct_count}/{len(results)})")
    print(f"Citation presence rate: {summary['citation_presence_rate']:.2%}")
    print(f"Mean pipeline latency:  {summary['mean_pipeline_total_ms']:.2f} ms")
    print(f"P95 pipeline latency:   {summary['p95_total_ms']:.2f} ms")
    print(f"Mean total tokens:      {summary['mean_total_tokens']:.0f}")
    print(f"\nSaved: reports/week5/e2e_eval.json")


if __name__ == "__main__":
    main()

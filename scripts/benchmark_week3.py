from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.week3.llm_client import LLMClient
from src.week3.rag_service import RAGService
from src.week3.retriever_adapter import build_retriever
from src.week3.settings import load_llm_config, load_rag_config


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

    service = RAGService(
        retriever=retriever,
        llm_client=llm_client,
        settings=rag_config,
    )

    eval_path = ROOT / "data" / "eval" / "week3_queries.json"

    if not eval_path.exists():
        raise SystemExit(f"Missing evaluation file: {eval_path}")

    queries = json.loads(eval_path.read_text(encoding="utf-8"))

    results = []

    for item in queries:
        query = item["query"]
        expected = item.get("expected_category", "")

        print("\nRunning:", query)

        try:
            result = service.answer(query)
        except Exception as exc:
            result = {
                "query": query,
                "status": "error",
                "category": "error",
                "answer": str(exc),
                "citations": [],
                "latency": {
                    "retrieval_ms": 0,
                    "prompt_build_ms": 0,
                    "generation_ms": 0,
                    "validation_ms": 0,
                    "total_ms": 0,
                },
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "debug": {
                    "error_type": type(exc).__name__,
                },
            }

        results.append({
            "query": query,
            "expected_category": expected,
            "actual_category": result["category"],
            "status": result["status"],
            "answer": result["answer"],
            "citations": result["citations"],
            "latency": result["latency"],
            "tokens": result["token_usage"],
            "debug": result["debug"],
        })

        print("Category:", result["category"])
        print("Status:", result["status"])
        print("Total ms:", result["latency"]["total_ms"])

        # Gentle delay to reduce rate-limit risk.
        time.sleep(1)

    out_dir = ROOT / "reports" / "week3"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "eval_results.json"
    out_csv = out_dir / "eval_summary.csv"

    out_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query",
            "expected_category",
            "actual_category",
            "status",
            "total_ms",
            "retrieval_ms",
            "generation_ms",
            "validation_ms",
            "total_tokens",
            "citation_count",
            "answer_preview",
        ])

        for r in results:
            writer.writerow([
                r["query"][:160],
                r["expected_category"],
                r["actual_category"],
                r["status"],
                r["latency"]["total_ms"],
                r["latency"]["retrieval_ms"],
                r["latency"]["generation_ms"],
                r["latency"]["validation_ms"],
                r["tokens"]["total_tokens"],
                len(r["citations"]),
                r["answer"][:160],
            ])

    print("\nSaved:")
    print(out_json)
    print(out_csv)


if __name__ == "__main__":
    main()

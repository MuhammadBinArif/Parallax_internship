from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.week3.llm_client import LLMClient
from src.week3.rag_service import RAGService
from src.week3.retriever_adapter import build_retriever
from src.week3.settings import load_llm_config, load_rag_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Week 3 hallucination-resistant RAG demo."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query to answer. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override RAG_TOP_K for this run.",
    )
    args = parser.parse_args()

    llm_config = load_llm_config()
    rag_config = load_rag_config()

    if args.top_k is not None:
        rag_config.top_k = args.top_k

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

    query = args.query or input("Enter query: ").strip()

    if not query:
        raise SystemExit("Query cannot be empty.")

    result = service.answer(query)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

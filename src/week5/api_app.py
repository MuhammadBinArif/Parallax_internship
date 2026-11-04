from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import FastAPI, Request

from src.week3.llm_client import LLMClient, LLMError
from src.week3.rag_service import RAGService
from src.week3.retriever_adapter import build_retriever
from src.week3.settings import load_llm_config, load_rag_config
from src.week5.api_errors import APIError, install_error_handlers
from src.week5.api_logging import (
    RequestLogger,
    attach_request_context,
    build_log_record,
)
from src.week5.api_models import (
    ChunkResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
)

API_VERSION = "week5-1.0.0"


def _looks_like_llm_error(err_type: str) -> bool:
    if not err_type:
        return False
    markers = ("LLM", "Rate", "Timeout", "Token", "Truncat")
    return any(marker in err_type for marker in markers)


def create_app(
    llm_client: Optional[LLMClient] = None,
    retriever: Optional[Any] = None,
    rag_service: Optional[RAGService] = None,
    log_path: str = "reports/week5/api_request_logs.jsonl",
) -> FastAPI:
    """
    App factory. Tests inject a fake LLM client and fake retriever.
    """
    app = FastAPI(
        title="Week 5 RAG API",
        version=API_VERSION,
        description="FastAPI wrapper around the Week 3 RAG + Week 4 NLP metadata.",
    )

    logger = RequestLogger(log_path)

    # Startup config (needed for /health, even when fakes are injected).
    llm_config = load_llm_config()
    rag_config = load_rag_config()

    if retriever is None:
        retriever = build_retriever()

    if llm_client is None:
        llm_client = LLMClient(
            provider=llm_config.provider,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            model=llm_config.model,
            timeout=llm_config.timeout,
            max_retries=llm_config.max_retries,
        )

    if rag_service is None:
        rag_service = RAGService(
            retriever=retriever,
            llm_client=llm_client,
            settings=rag_config,
        )

    # ---- App state ----
    app.state.retriever = retriever
    app.state.rag_service = rag_service
    app.state.logger = logger
    app.state.llm_client = llm_client

    app.state.collection_name = getattr(
        getattr(retriever, "collection", None), "name", "unknown"
    )
    app.state.collection_count = (
        retriever.collection.count()
        if hasattr(retriever, "collection")
        and hasattr(retriever.collection, "count")
        else 0
    )

    install_error_handlers(app, logger)

    # ---- Middleware: attach request_id + start time ----
    @app.middleware("http")
    async def add_request_context(request: Request, call_next):
        await attach_request_context(request)
        return await call_next(request)

    # ---- GET /health ----
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=API_VERSION,
            chroma_collection=app.state.collection_name,
            chroma_count=app.state.collection_count,
            llm_provider=llm_config.provider,
            llm_model=llm_config.model,
        )

    # ---- POST /retrieve ----
    @app.post("/retrieve", response_model=RetrieveResponse)
    async def retrieve_endpoint(request: Request, body: RetrieveRequest):
        start = time.perf_counter()

        try:
            chunks = app.state.retriever.retrieve(body.query, top_k=body.top_k)
        except Exception as exc:
            raise APIError(
                status_code=500,
                error="retrieval_failed",
                message="Retrieval pipeline failed.",
                details={"error_type": type(exc).__name__},
            )

        total_ms = round((time.perf_counter() - start) * 1000.0, 2)

        record = build_log_record(
            request,
            status_code=200,
            error_type=None,
            extra={
                "endpoint": "retrieve",
                "top_k": body.top_k,
                "count": len(chunks),
                "query_len": len(body.query),
            },
        )
        logger.log(record)

        return RetrieveResponse(
            request_id=getattr(request.state, "request_id", "unknown"),
            query=body.query,
            count=len(chunks),
            chunks=[
                ChunkResponse(
                    id=c.id,
                    text=c.text[:400],
                    score=c.score,
                    metadata=dict(c.metadata or {}),
                )
                for c in chunks
            ],
            latency_ms={"retrieve_ms": total_ms},
        )

    # ---- POST /query ----
    @app.post("/query", response_model=QueryResponse)
    async def query_endpoint(request: Request, body: QueryRequest):
        start = time.perf_counter()

        try:
            result = app.state.rag_service.answer(body.query)
        except LLMError as exc:
            raise APIError(
                status_code=503,
                error="llm_unavailable",
                message="The LLM provider is currently unavailable.",
                details={"error_type": type(exc).__name__},
            )
        except Exception as exc:
            raise APIError(
                status_code=500,
                error="internal_server_error",
                message="RAG pipeline failed.",
                details={
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:300],
                },
            )

        if result.get("status") == "error":
            err_type = (result.get("debug") or {}).get("error_type") or "UnknownError"
            if _looks_like_llm_error(err_type):
                raise APIError(
                    status_code=503,
                    error="llm_unavailable",
                    message="The LLM provider is currently unavailable.",
                    details={"error_type": err_type},
                )
            raise APIError(
                status_code=500,
                error="pipeline_error",
                message="RAG pipeline failed to produce a valid response.",
                details={"error_type": err_type},
            )

        total_ms = round((time.perf_counter() - start) * 1000.0, 2)

        record = build_log_record(
            request,
            status_code=200,
            error_type=None,
            extra={
                "endpoint": "query",
                "category": result.get("category"),
                "status": result.get("status"),
                "total_ms": total_ms,
                "query_len": len(body.query),
            },
        )
        logger.log(record)

        return QueryResponse(
            request_id=result.get(
                "request_id", getattr(request.state, "request_id", "unknown")
            ),
            query=result["query"],
            status=result["status"],
            category=result["category"],
            answer=result["answer"],
            citations=result["citations"],
            latency={**result["latency"], "http_total_ms": total_ms},
            token_usage=result["token_usage"],
            debug=result.get("debug", {}),
        )

    # Do NOT define @app.exception_handler(ValueError) here.
    # That handler previously raised APIError inside another handler,
    # which Starlette does not chain reliably, causing all 9 tests to fail.

    return app

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.week3.rag_service import RAGService, REFUSAL_MESSAGES
from src.week3.settings import RAGConfig
from src.week3.types import Chunk
from src.week5.api_app import create_app


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, top_k=5):
        return self.chunks[:top_k]


class FakeLLMClient:
    provider = "fake"
    model = "fake-model"

    def __init__(self, content):
        self.content = content

    def chat(self, messages, max_tokens=1500, temperature=0.0):
        return {
            "content": self.content,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "finish_reason": "stop",
            "latency_ms": 1.0,
            "attempts": 1,
        }


def build_test_app(tmp_path: Path, llm_content: str) -> TestClient:
    chunks = [Chunk(
        id="chunk-1",
        text="Anarchism is a political philosophy that opposes hierarchical authority.",
        score=0.8,
    )]
    llm = FakeLLMClient(llm_content)
    retriever = FakeRetriever(chunks)

    settings = RAGConfig(
        top_k=5, min_chunk_score=0.1, support_threshold=0.2,
        max_context_chars=1000, max_output_tokens=200,
        temperature=0.0, log_path=str(tmp_path / "rag_logs.jsonl"),
    )
    service = RAGService(retriever=retriever, llm_client=llm, settings=settings)

    app = create_app(
        llm_client=llm,
        retriever=retriever,
        rag_service=service,
        log_path=str(tmp_path / "api_logs.jsonl"),
    )
    return TestClient(app)


def test_health(tmp_path):
    client = build_test_app(tmp_path, "{}")
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_retrieve_ok(tmp_path):
    client = build_test_app(tmp_path, "{}")
    r = client.post("/retrieve", json={"query": "What is anarchism?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["chunks"][0]["id"] == "chunk-1"


def test_query_answered(tmp_path):
    llm_content = (
        '{"category": "answered", '
        '"answer": "Anarchism is a political philosophy that opposes all forms of hierarchical authority [chunk-1].", '
        '"citations": ["chunk-1"]}'
    )
    client = build_test_app(tmp_path, llm_content)
    r = client.post("/query", json={"query": "What is anarchism?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["category"] == "answered"
    assert "chunk-1" in body["citations"]


def test_query_empty_returns_422(tmp_path):
    client = build_test_app(tmp_path, "{}")
    r = client.post("/query", json={"query": ""})
    assert r.status_code == 422
    body = r.json()
    assert "error" in body
    assert "request_id" in body


def test_query_whitespace_returns_422(tmp_path):
    client = build_test_app(tmp_path, "{}")
    r = client.post("/query", json={"query": "    "})
    assert r.status_code == 422


def test_query_missing_body_returns_422(tmp_path):
    client = build_test_app(tmp_path, "{}")
    r = client.post("/query", json={})
    assert r.status_code == 422


def test_query_invalid_topk_returns_422(tmp_path):
    client = build_test_app(tmp_path, "{}")
    r = client.post("/query", json={"query": "test", "top_k": 999})
    assert r.status_code == 422


def test_query_llm_failure_returns_5xx(tmp_path):
    """LLM crashes → 500 or 503, not a crash."""
    from src.week3.llm_client import LLMError

    class BoomLLM:
        provider = "boom"
        model = "boom"
        def chat(self, *a, **kw):
            raise LLMError("simulated outage")

    chunks = [Chunk(id="c1", text="some text", score=0.9)]
    retriever = FakeRetriever(chunks)
    settings = RAGConfig(
        top_k=5, min_chunk_score=0.1, support_threshold=0.2,
        max_context_chars=1000, max_output_tokens=200,
        temperature=0.0, log_path=str(tmp_path / "r.jsonl"),
    )
    service = RAGService(retriever=retriever, llm_client=BoomLLM(), settings=settings)
    app = create_app(llm_client=BoomLLM(), retriever=retriever, rag_service=service,
                     log_path=str(tmp_path / "api.jsonl"))
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/query", json={"query": "test"})
    assert r.status_code in (500, 503)
    assert r.json()["error"] in {"internal_server_error", "llm_unavailable"}


def test_request_logs_written(tmp_path):
    client = build_test_app(tmp_path, "{}")
    client.get("/health")
    client.post("/retrieve", json={"query": "x"})
    log_file = tmp_path / "api_logs.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    import json as _j
    records = [_j.loads(l) for l in lines]
    assert any(r.get("path") == "/retrieve" for r in records)

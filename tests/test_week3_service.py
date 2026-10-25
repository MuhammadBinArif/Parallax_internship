from src.week3.rag_service import RAGService
from src.week3.settings import RAGConfig
from src.week3.types import Chunk


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
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
            "finish_reason": "stop",
            "latency_ms": 1.0,
            "attempts": 1,
        }


def test_answered_query(tmp_path):
    chunks = [
        Chunk(
            id="chunk-1",
            text="Anarchism is a political philosophy that opposes hierarchical authority.",
            score=0.8,
        )
    ]

    llm = FakeLLMClient(
        '{"category": "answered", "answer": "Anarchism is a political philosophy that opposes hierarchical authority [chunk-1].", "citations": ["chunk-1"]}'
    )

    settings = RAGConfig(
        top_k=5,
        min_chunk_score=0.1,
        support_threshold=0.2,
        max_context_chars=1000,
        max_output_tokens=100,
        temperature=0.0,
        log_path=str(tmp_path / "latency.jsonl"),
    )

    service = RAGService(
        retriever=FakeRetriever(chunks),
        llm_client=llm,
        settings=settings,
    )

    result = service.answer("What is anarchism?")

    assert result["status"] == "ok"
    assert result["category"] == "answered"
    assert result["citations"] == ["chunk-1"]
    assert (tmp_path / "latency.jsonl").exists()


def test_out_of_domain_when_no_chunks(tmp_path):
    llm = FakeLLMClient(
        '{"category": "out_of_domain", "answer": "", "citations": []}'
    )

    settings = RAGConfig(
        top_k=5,
        min_chunk_score=0.1,
        support_threshold=0.2,
        max_context_chars=1000,
        max_output_tokens=100,
        temperature=0.0,
        log_path=str(tmp_path / "latency.jsonl"),
    )

    service = RAGService(
        retriever=FakeRetriever([]),
        llm_client=llm,
        settings=settings,
    )

    result = service.answer("Write a poem about black holes.")

    assert result["status"] == "refusal"
    assert result["category"] == "out_of_domain"


def test_insufficient_context_from_model(tmp_path):
    chunks = [
        Chunk(
            id="chunk-1",
            text="Anarchism is a political philosophy.",
            score=0.8,
        )
    ]

    llm = FakeLLMClient(
        '{"category": "insufficient_context", "answer": "", "citations": []}'
    )

    settings = RAGConfig(
        top_k=5,
        min_chunk_score=0.1,
        support_threshold=0.2,
        max_context_chars=1000,
        max_output_tokens=100,
        temperature=0.0,
        log_path=str(tmp_path / "latency.jsonl"),
    )

    service = RAGService(
        retriever=FakeRetriever(chunks),
        llm_client=llm,
        settings=settings,
    )

    result = service.answer("What is the exact population of anarchists in 2026?")

    assert result["status"] == "refusal"
    assert result["category"] == "insufficient_context"

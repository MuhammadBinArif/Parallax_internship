from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class ChunkResponse(BaseModel):
    id: str
    text: str
    score: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    request_id: str
    query: str
    count: int
    chunks: list[ChunkResponse]
    latency_ms: dict[str, float]


class QueryResponse(BaseModel):
    request_id: str
    query: str
    status: str
    category: str
    answer: str
    citations: list[str]
    latency: dict[str, float]
    token_usage: dict[str, int]
    debug: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str
    chroma_collection: str
    chroma_count: int
    llm_provider: str
    llm_model: str


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    message: str
    details: Optional[dict[str, Any]] = None

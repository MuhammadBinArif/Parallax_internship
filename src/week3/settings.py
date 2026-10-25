from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    load_dotenv(ROOT / ".env", override=True)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    timeout: float
    max_retries: int


@dataclass
class RAGConfig:
    top_k: int
    min_chunk_score: Optional[float]
    support_threshold: float
    max_context_chars: int
    max_output_tokens: int
    temperature: float
    log_path: str


def load_llm_config() -> LLMConfig:
    load_env()

    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        base_url = os.getenv(
            "LLM_BASE_URL",
            "https://api.groq.com/openai/v1",
        ).strip().rstrip("/")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        base_url = os.getenv(
            "LLM_BASE_URL",
            "https://openrouter.ai/api/v1",
        ).strip().rstrip("/")
    else:
        raise ValueError("LLM_PROVIDER must be 'groq' or 'openrouter'.")

    model = os.getenv("LLM_MODEL_ID", "").strip()

    if not api_key:
        raise ValueError(
            "API key is missing. "
            "Set GROQ_API_KEY or OPENROUTER_API_KEY in .env."
        )

    if not model or model == "replace-with-active-model-id":
        raise ValueError(
            "LLM_MODEL_ID is missing or invalid. "
            "Set an active model ID from the provider's live model list."
        )

    return LLMConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=_get_float("LLM_TIMEOUT_SECONDS", 60.0),
        max_retries=_get_int("LLM_MAX_RETRIES", 4),
    )


def load_rag_config() -> RAGConfig:
    load_env()

    min_score_raw = os.getenv("RAG_MIN_CHUNK_SCORE", "0.25").strip().lower()

    if min_score_raw in {"", "none", "null"}:
        min_chunk_score: Optional[float] = None
    else:
        try:
            min_chunk_score = float(min_score_raw)
        except ValueError:
            min_chunk_score = 0.25

    log_path = os.getenv(
        "LATENCY_LOG_PATH",
        "reports/week3/latency_logs.jsonl",
    ).strip()

    log_path_obj = Path(log_path)
    if not log_path_obj.is_absolute():
        log_path_obj = ROOT / log_path_obj

    return RAGConfig(
        top_k=_get_int("RAG_TOP_K", 5),
        min_chunk_score=min_chunk_score,
        support_threshold=_get_float("RAG_SUPPORT_THRESHOLD", 0.35),
        max_context_chars=_get_int("RAG_MAX_CONTEXT_CHARS", 12000),
        max_output_tokens=_get_int("LLM_MAX_OUTPUT_TOKENS", 1500),
        temperature=_get_float("LLM_TEMPERATURE", 0.0),
        log_path=str(log_path_obj),
    )

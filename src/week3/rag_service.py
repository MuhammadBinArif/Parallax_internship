from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.week3.llm_client import (
    InputTokenLimitError,
    LLMClient,
    LLMError,
    OutputTruncatedError,
)
from src.week3.prompts import (
    STRICT_REVISION_INSTRUCTION,
    build_messages,
)
from src.week3.settings import RAGConfig
from src.week3.types import Chunk
from src.week3.validators import (
    extract_citations,
    normalize_category,
    parse_json_object,
    unique_preserve_order,
    validate_groundedness,
)


REFUSAL_MESSAGES = {
    "out_of_domain": (
        "I can only answer questions related to the indexed documents. "
        "This request appears to be outside that scope."
    ),
    "insufficient_context": (
        "The retrieved documents do not contain enough information to answer "
        "this question confidently."
    ),
    "unsupported": (
        "I could not verify that the generated answer is fully supported by "
        "the retrieved documents, so I am not providing it."
    ),
    "error": (
        "I encountered an issue while generating a reliable answer. "
        "Please try again later."
    ),
}


class RAGService:
    OFF_TOPIC_PATTERNS = (
        r"write\s+a\s+poem",
        r"\bpoem\b",
        r"\bpoetry\b",
        r"\bhaiku\b",
        r"\bsonnet\b",
        r"\bjoke\b",
        r"\bsong\b",
        r"\blyrics\b",
        r"\bfiction\b",
        r"\bblack\s+holes?\b",
        r"\bfifa\b",
        r"\bworld\s+cup\b",
        r"\bolympics\b",
        r"\bweather\b",
        r"\bstock\s+price\b",
        r"\bbitcoin\s+price\b",
        r"\bcapital\s+of\b",
    )

    OFF_TOPIC_RE = re.compile("|".join(OFF_TOPIC_PATTERNS), re.IGNORECASE)

    def __init__(
        self,
        retriever: Any,
        llm_client: LLMClient,
        settings: RAGConfig,
    ):
        self.retriever = retriever
        self.llm = llm_client
        self.settings = settings

    @classmethod
    def _looks_off_topic(cls, query: str) -> bool:
        return bool(cls.OFF_TOPIC_RE.search(query or ""))

    def _filter_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        if self.settings.min_chunk_score is None:
            return chunks

        return [
            chunk
            for chunk in chunks
            if chunk.score is None
            or chunk.score >= self.settings.min_chunk_score
        ]

    def _pack_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        selected: list[Chunk] = []
        used_chars = 0

        for chunk in chunks:
            chunk_size = len(chunk.text) + 50

            if used_chars + chunk_size <= self.settings.max_context_chars:
                selected.append(chunk)
                used_chars += chunk_size
            elif not selected:
                truncated = Chunk(
                    id=chunk.id,
                    text=chunk.text[: self.settings.max_context_chars],
                    score=chunk.score,
                    metadata=chunk.metadata,
                )
                selected.append(truncated)
                break
            else:
                break

        return selected

    def _call_json(
        self,
        query: str,
        chunks: list[Chunk],
        max_output_tokens: int,
        extra_instruction: Optional[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        last_error: Optional[Exception] = None

        for json_attempt in range(2):
            instruction = extra_instruction

            if json_attempt == 1:
                repair = (
                    "Your previous response was not valid JSON. "
                    "Return exactly one JSON object and nothing else."
                )
                instruction = f"{instruction}\n{repair}" if instruction else repair

            messages = build_messages(
                query=query,
                chunks=chunks,
                extra_instruction=instruction,
            )

            response = self.llm.chat(
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=self.settings.temperature,
            )

            try:
                parsed = parse_json_object(response["content"])
                return parsed, response
            except Exception as exc:
                last_error = exc

        raise LLMError(
            f"Model did not return valid JSON after retry. Last error: {last_error}"
        )

    def _generate_with_recovery(
        self,
        query: str,
        chunks: list[Chunk],
        extra_instruction: Optional[str],
    ) -> tuple[dict[str, Any], dict[str, Any], list[Chunk], int]:
        current_chunks = chunks
        max_output_tokens = self.settings.max_output_tokens
        current_extra_instruction = extra_instruction

        input_trimmed = False
        output_recovery = 0

        while True:
            try:
                parsed, response = self._call_json(
                    query=query,
                    chunks=current_chunks,
                    max_output_tokens=max_output_tokens,
                    extra_instruction=current_extra_instruction,
                )

                return parsed, response, current_chunks, max_output_tokens

            except InputTokenLimitError:
                if input_trimmed or len(current_chunks) <= 1:
                    raise

                current_chunks = current_chunks[:-1]
                input_trimmed = True

            except OutputTruncatedError:
                if output_recovery == 0 and max_output_tokens < 4096:
                    max_output_tokens = min(
                        4096,
                        max(2000, int(max_output_tokens * 1.5)),
                    )
                    output_recovery = 1
                    continue

                if output_recovery < 3:
                    output_recovery += 1
                    suffix = "Be concise and return only the JSON object."

                    if current_extra_instruction:
                        current_extra_instruction = (
                            f"{current_extra_instruction}\n{suffix}"
                        )
                    else:
                        current_extra_instruction = suffix

                    continue

                raise

    def answer(self, query: str) -> dict[str, Any]:
        request_id = uuid.uuid4().hex

        total_start = time.perf_counter()

        retrieval_ms = 0.0
        prompt_build_ms = 0.0
        generation_ms = 0.0
        validation_ms = 0.0

        attempts = 0
        prompt_tokens = 0
        completion_tokens = 0

        raw_chunk_count = 0
        filtered_chunk_count = 0
        used_chunks: list[Chunk] = []

        category = "error"
        status = "error"
        answer_text = REFUSAL_MESSAGES["error"]
        citations: list[str] = []
        issues: list[str] = []
        error_type: Optional[str] = None
        off_topic_guard = False

        try:
            retrieval_start = time.perf_counter()
            raw_chunks = self.retriever.retrieve(
                query,
                top_k=self.settings.top_k,
            )
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            raw_chunk_count = len(raw_chunks)

            filtered_chunks = self._filter_chunks(raw_chunks)
            filtered_chunk_count = len(filtered_chunks)

            prompt_build_start = time.perf_counter()
            used_chunks = self._pack_chunks(filtered_chunks)
            prompt_build_ms = (time.perf_counter() - prompt_build_start) * 1000

            max_score: Optional[float] = None
            if used_chunks:
                scores = [chunk.score for chunk in used_chunks if chunk.score is not None]
                if scores:
                    max_score = max(scores)

            off_topic_guard = self._looks_off_topic(query) and (
                not used_chunks
                or (max_score is not None and max_score < 0.70)
            )

            if not used_chunks or off_topic_guard:
                category = "out_of_domain"
                status = "refusal"
                answer_text = REFUSAL_MESSAGES["out_of_domain"]

            else:
                finalized = False

                for revision in range(2):
                    extra_instruction = (
                        None if revision == 0 else STRICT_REVISION_INSTRUCTION
                    )

                    generation_start = time.perf_counter()

                    try:
                        parsed, response, used_chunks, max_out_used = (
                            self._generate_with_recovery(
                                query=query,
                                chunks=used_chunks,
                                extra_instruction=extra_instruction,
                            )
                        )
                    except InputTokenLimitError:
                        error_type = "InputTokenLimitError"
                        break
                    except OutputTruncatedError:
                        error_type = "OutputTruncatedError"
                        break
                    except LLMError as exc:
                        error_type = type(exc).__name__
                        break

                    generation_ms += (time.perf_counter() - generation_start) * 1000

                    attempts += int(response.get("attempts", 1))

                    usage = response.get("usage") or {}
                    prompt_tokens += int(usage.get("prompt_tokens") or 0)
                    completion_tokens += int(usage.get("completion_tokens") or 0)

                    validation_start = time.perf_counter()

                    model_category = normalize_category(parsed.get("category"))

                    if model_category == "invalid":
                        model_category = "insufficient_context"

                    if model_category != "answered":
                        category = model_category
                        status = "refusal"
                        answer_text = REFUSAL_MESSAGES.get(
                            category,
                            REFUSAL_MESSAGES["insufficient_context"],
                        )
                        citations = []
                        finalized = True
                        validation_ms += (time.perf_counter() - validation_start) * 1000
                        break

                    proposed_answer = str(parsed.get("answer") or "")

                    proposed_citations = parsed.get("citations") or []
                    if not isinstance(proposed_citations, list):
                        proposed_citations = []

                    proposed_citations = [
                        str(c).strip()
                        for c in proposed_citations
                        if str(c).strip()
                    ]

                    proposed_citations += extract_citations(proposed_answer)
                    proposed_citations = unique_preserve_order(proposed_citations)

                    ok, current_issues = validate_groundedness(
                        answer=proposed_answer,
                        chunks=used_chunks,
                        threshold=self.settings.support_threshold,
                        declared_citations=proposed_citations,
                    )

                    validation_ms += (time.perf_counter() - validation_start) * 1000
                    issues = current_issues

                    if ok:
                        category = "answered"
                        status = "ok"
                        answer_text = proposed_answer
                        citations = proposed_citations
                        issues = []
                        finalized = True
                        break

                    if revision == 1:
                        category = "unsupported"
                        status = "refusal"
                        answer_text = REFUSAL_MESSAGES["unsupported"]
                        citations = []
                        finalized = True
                        break

                if not finalized and error_type is None:
                    category = "error"
                    status = "error"
                    answer_text = REFUSAL_MESSAGES["error"]

        except Exception as exc:
            error_type = type(exc).__name__
            category = "error"
            status = "error"
            answer_text = REFUSAL_MESSAGES["error"]

        total_ms = (time.perf_counter() - total_start) * 1000

        latency = {
            "retrieval_ms": round(retrieval_ms, 2),
            "prompt_build_ms": round(prompt_build_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "validation_ms": round(validation_ms, 2),
            "total_ms": round(total_ms, 2),
        }

        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        debug = {
            "chunks_retrieved": raw_chunk_count,
            "chunks_filtered": filtered_chunk_count,
            "chunks_used": len(used_chunks),
            "chunk_ids_used": [chunk.id for chunk in used_chunks],
            "max_chunk_score": max_score,
            "off_topic_guard": off_topic_guard,
            "llm_attempts": attempts,
            "verification_issue_count": len(issues),
            "verification_issues_preview": issues[:5],
            "error_type": error_type,
        }

        record = {
            "request_id": request_id,
            "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest()[:16],
            "query_length": len(query),
            "provider": getattr(self.llm, "provider", "unknown"),
            "model": getattr(self.llm, "model", "unknown"),
            "status": status,
            "category": category,
            "latency_ms": latency,
            "tokens": token_usage,
            "retrieval": {
                "raw_chunks": raw_chunk_count,
                "filtered_chunks": filtered_chunk_count,
                "used_chunks": len(used_chunks),
                "max_chunk_score": max_score,
            },
            "off_topic_guard": off_topic_guard,
            "llm_attempts": attempts,
            "error_type": error_type,
            "issue_count": len(issues),
            "issues_preview": issues[:3],
        }

        self._append_log(record)

        return {
            "request_id": request_id,
            "query": query,
            "status": status,
            "category": category,
            "answer": answer_text,
            "citations": citations,
            "latency": latency,
            "token_usage": token_usage,
            "debug": debug,
        }

    def _append_log(self, record: dict[str, Any]) -> None:
        record["timestamp"] = (
            datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

        path = Path(self.settings.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

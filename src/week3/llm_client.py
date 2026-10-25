from __future__ import annotations

import os
import random
import time
from typing import Any, Optional

import requests


class LLMError(Exception):
    """Base class for LLM-related errors."""


class InputTokenLimitError(LLMError):
    """Raised when the prompt/context exceeds the model token limit."""


class OutputTruncatedError(LLMError):
    """Raised when model output is truncated or empty due to token budget."""


class InvalidResponseError(LLMError):
    """Raised when provider response cannot be parsed as a valid chat completion."""


class LLMClient:
    """
    Minimal OpenAI-compatible chat client for Groq or OpenRouter.

    Handles:
    - timeouts
    - rate limits
    - server errors
    - network errors
    - model not found
    - authentication errors
    - input token limit errors
    - output truncation errors
    - empty-content reasoning-model failures
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 4,
    ):
        self.provider = provider.strip().lower()
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)

        if not api_key:
            raise ValueError("API key is required.")

        if not self.model:
            raise ValueError("Model ID is required.")

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if self.provider == "openrouter":
            self.headers["HTTP-Referer"] = os.getenv(
                "OPENROUTER_APP_URL",
                "https://localhost",
            )
            self.headers["X-Title"] = os.getenv(
                "OPENROUTER_APP_NAME",
                "Week 3 RAG System",
            )

    def _retry_delay(
        self,
        attempt: int,
        retry_after: Optional[float] = None,
    ) -> float:
        if retry_after is not None and retry_after > 0:
            return min(float(retry_after), 60.0)

        return min(30.0, (2 ** attempt) * (0.5 + random.random()))

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 2):
            start = time.perf_counter()

            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )

                latency_ms = (time.perf_counter() - start) * 1000

                if response.status_code == 429:
                    retry_after = None
                    retry_after_header = response.headers.get("Retry-After")

                    if retry_after_header:
                        try:
                            retry_after = float(retry_after_header)
                        except ValueError:
                            retry_after = None

                    time.sleep(self._retry_delay(attempt, retry_after))
                    continue

                if response.status_code >= 500:
                    time.sleep(self._retry_delay(attempt))
                    continue

                if response.status_code == 404:
                    raise LLMError(
                        "Model not found (404). "
                        "Check the provider's live model list and update LLM_MODEL_ID. "
                        f"Response: {response.text[:500]}"
                    )

                if response.status_code == 400:
                    body = response.text.lower()
                    token_markers = (
                        "token",
                        "context length",
                        "maximum context",
                        "too many tokens",
                        "context window",
                        "input is too long",
                    )

                    if any(marker in body for marker in token_markers):
                        raise InputTokenLimitError(response.text[:1000])

                    raise LLMError(
                        f"LLM 400 Bad Request: {response.text[:1000]}"
                    )

                if response.status_code in (401, 403):
                    raise LLMError(
                        f"LLM authentication/permission error "
                        f"({response.status_code}): {response.text[:500]}"
                    )

                if 400 <= response.status_code < 500:
                    raise LLMError(
                        f"LLM HTTP {response.status_code}: {response.text[:500]}"
                    )

                response.raise_for_status()

                try:
                    data = response.json()
                except ValueError as exc:
                    last_error = exc
                    time.sleep(self._retry_delay(attempt))
                    continue

                choices = data.get("choices") or []

                if not choices:
                    raise InvalidResponseError(
                        f"LLM response contains no choices: {str(data)[:500]}"
                    )

                choice = choices[0]
                message = choice.get("message") or {}
                content = message.get("content")

                if content is None:
                    content = choice.get("text")

                finish_reason = choice.get("finish_reason")
                usage = data.get("usage") or {}

                if content is None:
                    if finish_reason == "length":
                        raise OutputTruncatedError(
                            "Model returned no content because max_tokens was reached."
                        )

                    raise InvalidResponseError(
                        f"LLM response contains no content: {str(choice)[:500]}"
                    )

                content_str = str(content).strip()

                if not content_str:
                    if finish_reason == "length":
                        raise OutputTruncatedError(
                            "Model returned empty content because max_tokens was too small "
                            "or reasoning consumed the token budget."
                        )

                    reasoning = message.get("reasoning")

                    raise InvalidResponseError(
                        "Model returned empty content. "
                        f"finish_reason={finish_reason}. "
                        f"reasoning_preview={str(reasoning)[:200]}"
                    )

                if finish_reason == "length":
                    raise OutputTruncatedError(
                        "Model output was truncated because max_tokens was reached. "
                        f"content_preview={content_str[:200]}"
                    )

                return {
                    "content": content_str,
                    "usage": usage,
                    "finish_reason": finish_reason,
                    "latency_ms": latency_ms,
                    "attempts": attempt,
                }

            except (
                InputTokenLimitError,
                OutputTruncatedError,
                InvalidResponseError,
                LLMError,
            ):
                raise

            except requests.exceptions.Timeout as exc:
                last_error = exc
                time.sleep(self._retry_delay(attempt))
                continue

            except requests.exceptions.RequestException as exc:
                last_error = exc
                time.sleep(self._retry_delay(attempt))
                continue

            except Exception as exc:
                last_error = exc
                time.sleep(self._retry_delay(attempt))
                continue

        raise LLMError(
            f"LLM request failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

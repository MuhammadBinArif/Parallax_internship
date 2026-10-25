from __future__ import annotations

import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.week3.llm_client import (
    InputTokenLimitError,
    InvalidResponseError,
    LLMClient,
    LLMError,
    OutputTruncatedError,
)


def load_provider_config() -> tuple[str, str, str, str]:
    env_path = ROOT / ".env"

    print("Loading .env from:", env_path)
    print("Exists:", env_path.exists())

    load_dotenv(dotenv_path=env_path, override=True)

    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        base_url = "https://api.groq.com/openai/v1"
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        base_url = "https://openrouter.ai/api/v1"
    else:
        raise ValueError("LLM_PROVIDER must be 'groq' or 'openrouter'.")

    model = os.getenv("LLM_MODEL_ID", "").strip()

    if not api_key:
        raise ValueError("API key is missing. Set GROQ_API_KEY or OPENROUTER_API_KEY in .env.")

    if not model or model == "replace-with-active-model-id":
        raise ValueError("LLM_MODEL_ID is missing or invalid. Set an active model ID in .env.")

    return provider, api_key, base_url, model


def main() -> None:
    provider, api_key, base_url, model = load_provider_config()

    print("Provider:", provider)
    print("Model ID:", model)
    print("API key loaded:", bool(api_key))

    client = LLMClient(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", 60)),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", 4)),
    )

    messages = [
        {
            "role": "system",
            "content": "You are a reliable test assistant. Follow instructions exactly.",
        },
        {
            "role": "user",
            "content": "Reply with exactly: OK",
        },
    ]

    try:
        response = client.chat(
            messages=messages,
            max_tokens=int(os.getenv("LLM_TEST_MAX_TOKENS", 200)),
            temperature=float(os.getenv("LLM_TEMPERATURE", 0)),
        )

        print("\nLLM client test succeeded.")
        print("Content:", response["content"])
        print("Finish reason:", response["finish_reason"])
        print("Latency ms:", round(response["latency_ms"], 2))
        print("Attempts:", response["attempts"])
        print("Usage:", json.dumps(response["usage"], indent=2))

    except InputTokenLimitError as exc:
        print("\nInput token limit error:")
        print(str(exc)[:1000])

    except OutputTruncatedError as exc:
        print("\nOutput truncated error:")
        print(str(exc)[:1000])
        print("\nSuggestion: increase LLM_TEST_MAX_TOKENS or LLM_MAX_OUTPUT_TOKENS.")

    except InvalidResponseError as exc:
        print("\nInvalid response error:")
        print(str(exc)[:1000])

    except LLMError as exc:
        print("\nLLM error:")
        print(str(exc)[:1000])


if __name__ == "__main__":
    main()

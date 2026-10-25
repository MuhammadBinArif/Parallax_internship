import os
import json
from pathlib import Path

import requests
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[1] / ".env"

print("Loading .env from:", env_path)
print("Exists:", env_path.exists())

load_dotenv(dotenv_path=env_path, override=True)

key = os.getenv("GROQ_API_KEY")
model = os.getenv("LLM_MODEL_ID")
provider = os.getenv("LLM_PROVIDER")

print("Provider:", provider)
print("Model ID:", model)
print("API key loaded:", bool(key))

if not key:
    raise SystemExit("GROQ_API_KEY is missing")

if not model or model == "replace-with-active-model-id":
    raise SystemExit("LLM_MODEL_ID is still missing or invalid")

payload = {
    "model": model,
    "messages": [
        {"role": "user", "content": "Reply with exactly: OK"}
    ],
    "temperature": 0,
    "max_tokens": 200,
}

r = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {key}"},
    json=payload,
    timeout=30,
)

print("Status code:", r.status_code)

try:
    data = r.json()
except Exception:
    print("Non-JSON response:")
    print(r.text[:1000])
    r.raise_for_status()
    raise

print("Raw response preview:")
print(json.dumps(data, indent=2)[:2000])

r.raise_for_status()

choice = data["choices"][0]
message = choice.get("message") or {}
content = message.get("content")
finish_reason = choice.get("finish_reason")
usage = data.get("usage") or {}

print("Finish reason:", finish_reason)
print("Usage:", usage)

if not content or not str(content).strip():
    raise SystemExit(
        "Model returned empty content. "
        "Increase max_tokens or try another text-generation model."
    )

print("Model replied:", content.strip())

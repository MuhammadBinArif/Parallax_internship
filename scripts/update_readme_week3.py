from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

START = "<!-- WEEK3_SECTION_START -->"
END = "<!-- WEEK3_SECTION_END -->"


def read_eval_rows() -> list[dict[str, str]]:
    csv_path = ROOT / "reports" / "week3" / "eval_summary.csv"

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    log_path = ROOT / "reports" / "week3" / "latency_logs.jsonl"

    if not log_path.exists():
        return []

    rows = []

    for line in log_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue

        record = json.loads(line)
        latency = record.get("latency_ms", {})
        tokens = record.get("tokens", {})

        rows.append(
            {
                "query": record.get("query_hash", ""),
                "expected_category": "",
                "actual_category": record.get("category", ""),
                "status": record.get("status", ""),
                "total_ms": latency.get("total_ms", 0),
                "retrieval_ms": latency.get("retrieval_ms", 0),
                "generation_ms": latency.get("generation_ms", 0),
                "validation_ms": latency.get("validation_ms", 0),
                "total_tokens": tokens.get("total_tokens", 0),
                "citation_count": 0,
                "answer_preview": "",
            }
        )

    return rows


def mean_value(rows: list[dict[str, str]], key: str) -> float:
    values = []

    for row in rows:
        try:
            values.append(float(row.get(key, 0) or 0))
        except Exception:
            continue

    if not values:
        return 0.0

    return round(sum(values) / len(values), 2)


def count_category(rows: list[dict[str, str]], category: str) -> int:
    return sum(1 for row in rows if row.get("actual_category") == category)


def count_status(rows: list[dict[str, str]], status: str) -> int:
    return sum(1 for row in rows if row.get("status") == status)


def build_section() -> str:
    rows = read_eval_rows()

    provider = os.getenv("LLM_PROVIDER", "groq")
    model = os.getenv("LLM_MODEL_ID", "unknown")

    queries = len(rows)
    answered = sum(
        1
        for row in rows
        if row.get("actual_category") == "answered"
        and row.get("status") == "ok"
    )
    out_of_domain = count_category(rows, "out_of_domain")
    insufficient = count_category(rows, "insufficient_context")
    unsupported = count_category(rows, "unsupported")
    errors = count_status(rows, "error")

    mean_total_ms = mean_value(rows, "total_ms")
    mean_retrieval_ms = mean_value(rows, "retrieval_ms")
    mean_generation_ms = mean_value(rows, "generation_ms")
    mean_validation_ms = mean_value(rows, "validation_ms")
    mean_tokens = mean_value(rows, "total_tokens")

    lines = []

    lines.append(START)
    lines.append("")
    lines.append("## Week 3 — LLM Integration & Prompt Engineering")
    lines.append("")
    lines.append("### Objective")
    lines.append("")
    lines.append(
        "Integrate an LLM API with the Week 2 retrieval system to generate grounded answers "
        "from retrieved chunks, while handling hallucinations, out-of-domain queries, API errors, "
        "and end-to-end latency logging."
    )
    lines.append("")
    lines.append("### LLM Provider")
    lines.append("")
    lines.append(f"- Provider: `{provider}`")
    lines.append(f"- Model ID: `{model}`")
    lines.append("- API key: loaded securely from `.env` and not committed")
    lines.append("")
    lines.append("### What I Built")
    lines.append("")
    lines.append("The Week 3 system extends the Week 2 ChromaDB retriever with a hallucination-resistant RAG pipeline.")
    lines.append("")
    lines.append("Pipeline:")
    lines.append("")
    lines.append("1. User submits a query.")
    lines.append("2. Week 2 retriever retrieves top-k chunks from ChromaDB.")
    lines.append("3. Weak chunks are filtered using `RAG_MIN_CHUNK_SCORE`.")
    lines.append("4. Remaining chunks are packed into a bounded context.")
    lines.append("5. A strict system prompt instructs the LLM to answer only from context.")
    lines.append("6. Retrieved chunks are injected as XML-style context blocks with chunk IDs.")
    lines.append("7. The LLM returns structured JSON with `category`, `answer`, and `citations`.")
    lines.append("8. The system validates citation IDs against retrieved chunks.")
    lines.append("9. The system checks sentence-level support using lexical overlap and fuzzy matching.")
    lines.append("10. If unsupported, the system retries once with a stricter revision prompt.")
    lines.append("11. If still unsafe, it returns a refusal.")
    lines.append("12. Obvious off-topic requests are guarded and refused as `out_of_domain`.")
    lines.append("13. End-to-end latency and token usage are logged.")
    lines.append("")
    lines.append("### Architecture")
    lines.append("")
    lines.append("```text")
    lines.append("User Query")
    lines.append("   |")
    lines.append("   v")
    lines.append("Week 2 ChromaDB Retriever")
    lines.append("   |")
    lines.append("   v")
    lines.append("Chunk Filtering and Packing")
    lines.append("   |")
    lines.append("   v")
    lines.append("Prompt Builder")
    lines.append("   |")
    lines.append("   v")
    lines.append("Groq LLM API Client")
    lines.append("   |")
    lines.append("   v")
    lines.append("JSON Parser")
    lines.append("   |")
    lines.append("   v")
    lines.append("Citation and Support Validator")
    lines.append("   |")
    lines.append("   v")
    lines.append("Final Answer or Safe Refusal")
    lines.append("   |")
    lines.append("   v")
    lines.append("Latency Logger")
    lines.append("```")
    lines.append("")
    lines.append("### How To Run Locally")
    lines.append("")
    lines.append("```bash")
    lines.append("cd ~/internship-repo")
    lines.append("conda activate ParallaxLabsInternship")
    lines.append("")
    lines.append("python scripts/check_week3_llm_client.py")
    lines.append("python scripts/check_week3_retriever.py")
    lines.append("")
    lines.append('python scripts/run_week3_demo.py --query "What is anarchism?"')
    lines.append('python scripts/run_week3_demo.py --query "Write a poem about black holes."')
    lines.append('python scripts/run_week3_demo.py --query "What is the current leadership of the anarchist movement in 2026?"')
    lines.append("")
    lines.append("python scripts/benchmark_week3.py")
    lines.append("python -m pytest -v")
    lines.append("```")
    lines.append("")
    lines.append("### Example Outputs")
    lines.append("")
    lines.append("#### Grounded answer")
    lines.append("")
    lines.append("```json")
    lines.append("{")
    lines.append('  "status": "ok",')
    lines.append('  "category": "answered",')
    lines.append('  "answer": "Anarchism is a political philosophy that rejects hierarchical authority [chunk-id].",')
    lines.append('  "citations": ["chunk-id"]')
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("#### Out-of-domain refusal")
    lines.append("")
    lines.append("```json")
    lines.append("{")
    lines.append('  "status": "refusal",')
    lines.append('  "category": "out_of_domain",')
    lines.append('  "answer": "I can only answer questions related to the indexed documents. This request appears to be outside that scope."')
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("#### Insufficient-context refusal")
    lines.append("")
    lines.append("```json")
    lines.append("{")
    lines.append('  "status": "refusal",')
    lines.append('  "category": "insufficient_context",')
    lines.append('  "answer": "The retrieved documents do not contain enough information to answer this question confidently."')
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("### Hallucination Resistance Strategy")
    lines.append("")
    lines.append("The system uses:")
    lines.append("")
    lines.append("- strict system prompt")
    lines.append("- context-only answering")
    lines.append("- structured JSON output")
    lines.append("- required citations")
    lines.append("- citation ID validation")
    lines.append("- sentence-level support checking")
    lines.append("- one strict revision attempt")
    lines.append("- safe refusal if unsupported")
    lines.append("- deterministic off-topic guard for obvious creative/trivia requests")
    lines.append("")
    lines.append("### Error Handling Strategy")
    lines.append("")
    lines.append("The LLM client handles:")
    lines.append("")
    lines.append("- timeouts")
    lines.append("- 429 rate limits")
    lines.append("- 5xx server errors")
    lines.append("- network errors")
    lines.append("- 401/403 authentication errors")
    lines.append("- 404 model-not-found errors")
    lines.append("- input token limit errors")
    lines.append("- output truncation errors")
    lines.append("- invalid JSON responses")
    lines.append("- empty reasoning-model content caused by small token budgets")
    lines.append("")
    lines.append("### Latency Logging")
    lines.append("")
    lines.append("Logs are stored in:")
    lines.append("")
    lines.append("```text")
    lines.append("reports/week3/latency_logs.jsonl")
    lines.append("```")
    lines.append("")
    lines.append("Each log includes:")
    lines.append("")
    lines.append("- retrieval_ms")
    lines.append("- prompt_build_ms")
    lines.append("- generation_ms")
    lines.append("- validation_ms")
    lines.append("- total_ms")
    lines.append("- prompt_tokens")
    lines.append("- completion_tokens")
    lines.append("- total_tokens")
    lines.append("- category")
    lines.append("- status")
    lines.append("- error_type")
    lines.append("")
    lines.append("### Results")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Queries evaluated | {queries} |")
    lines.append(f"| Answered successfully | {answered} |")
    lines.append(f"| Out-of-domain refusals | {out_of_domain} |")
    lines.append(f"| Insufficient-context refusals | {insufficient} |")
    lines.append(f"| Unsupported refusals | {unsupported} |")
    lines.append(f"| Errors | {errors} |")
    lines.append(f"| Mean total latency | {mean_total_ms} ms |")
    lines.append(f"| Mean retrieval latency | {mean_retrieval_ms} ms |")
    lines.append(f"| Mean generation latency | {mean_generation_ms} ms |")
    lines.append(f"| Mean validation latency | {mean_validation_ms} ms |")
    lines.append(f"| Mean total tokens | {mean_tokens} |")
    lines.append("")
    lines.append("### Known Limitations")
    lines.append("")
    lines.append("- Heuristic support checking can produce false positives or false negatives.")
    lines.append("- Token estimation is approximate unless replaced with a provider-specific tokenizer.")
    lines.append("- Answer quality depends on Week 2 retrieval quality.")
    lines.append("- If Week 2 indexed only a subset of the corpus, Week 3 can only answer questions covered by that subset.")
    lines.append("- Free model availability and model IDs may change frequently.")
    lines.append("")
    lines.append("### Submission Checklist")
    lines.append("")
    lines.append("- [x] LLM API integrated")
    lines.append("- [x] System prompt implemented")
    lines.append("- [x] Context injection implemented")
    lines.append("- [x] Robust error handling implemented")
    lines.append("- [x] Rate limit handling implemented")
    lines.append("- [x] Timeout handling implemented")
    lines.append("- [x] Token limit handling implemented")
    lines.append("- [x] Hallucination checks implemented")
    lines.append("- [x] Out-of-domain handling implemented")
    lines.append("- [x] Insufficient-context handling implemented")
    lines.append("- [x] End-to-end latency logging implemented")
    lines.append("- [x] README.md included")
    lines.append("- [x] Instructions to run code included")
    lines.append("")
    lines.append(END)
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    readme_path = ROOT / "README.md"

    if readme_path.exists():
        text = readme_path.read_text(encoding="utf-8")
    else:
        text = "# Internship Repository\n"

    section = build_section()

    if START in text and END in text:
        before = text.split(START, 1)[0]
        after = text.split(END, 1)[1]
        new_text = before + section + after
        print("Replaced existing marked Week 3 section in README.md.")
    elif "## Week 3" in text:
        new_text = text.rstrip() + "\n\n" + section
        print(
            "Warning: found an existing unmarked Week 3 section. "
            "Appended a new marked section. Remove the old section manually if needed."
        )
    else:
        new_text = text.rstrip() + "\n\n" + section
        print("Appended Week 3 section to README.md.")

    readme_path.write_text(new_text, encoding="utf-8")


if __name__ == "__main__":
    main()

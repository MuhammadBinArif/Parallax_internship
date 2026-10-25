from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Optional


CITATION_RE = re.compile(r"\[([^\]]+)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model response.")

        obj = json.loads(cleaned[start:end + 1])

    if not isinstance(obj, dict):
        raise ValueError("Model response JSON root is not an object.")

    return obj


def extract_citations(text: str) -> list[str]:
    citations = []

    for match in CITATION_RE.finditer(text or ""):
        raw = match.group(1)

        for part in re.split(r"[,;]", raw):
            part = part.strip()
            if part:
                citations.append(part)

    return citations


def remove_citations(text: str) -> str:
    return CITATION_RE.sub(" ", text or "")


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []

    for item in items:
        item = str(item).strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)

    return result


def normalize_category(value: Any) -> str:
    raw = str(value or "").strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")

    if raw in {"answered", "answer", "ok", "success", "supported"}:
        return "answered"

    if raw in {
        "insufficient_context",
        "insufficient",
        "not_enough_context",
        "unknown",
        "missing_context",
    }:
        return "insufficient_context"

    if raw in {
        "out_of_domain",
        "off_topic",
        "not_in_domain",
        "unrelated",
        "irrelevant",
    }:
        return "out_of_domain"

    return "invalid"


def sentence_support(sentence: str, evidence: str) -> float:
    sentence_tokens = tokenize(sentence)
    evidence_tokens = tokenize(evidence)

    if not sentence_tokens:
        return 1.0

    if not evidence_tokens:
        return 0.0

    sentence_set = set(sentence_tokens)
    evidence_set = set(evidence_tokens)

    overlap = len(sentence_set & evidence_set) / max(1, len(sentence_set))

    ratio = SequenceMatcher(
        None,
        " ".join(sentence_tokens),
        " ".join(evidence_tokens),
    ).ratio()

    return max(overlap, ratio * 0.9)


def validate_groundedness(
    answer: str,
    chunks: list[Any],
    threshold: float,
    declared_citations: Optional[list[str]] = None,
) -> tuple[bool, list[str]]:
    if not answer or not answer.strip():
        return False, ["Empty answer."]

    inline_citations = extract_citations(answer)
    declared = [
        str(c).strip()
        for c in (declared_citations or [])
        if str(c).strip()
    ]

    citations = unique_preserve_order(declared + inline_citations)

    if not citations:
        return False, ["No citations found."]

    chunks_by_id = {chunk.id: chunk.text for chunk in chunks}

    unknown = [cid for cid in citations if cid not in chunks_by_id]
    issues = []

    if unknown:
        issues.append("Unknown citations: " + ", ".join(unknown))

    valid_citations = [cid for cid in citations if cid in chunks_by_id]

    if not valid_citations:
        return False, issues or ["No valid citations."]

    cleaned_answer = remove_citations(answer)
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_RE.split(cleaned_answer)
        if sentence.strip()
    ]

    substantive_sentences = [
        sentence
        for sentence in sentences
        if len(tokenize(sentence)) >= 5
    ]

    if not substantive_sentences:
        return False, ["No substantive sentences to verify."]

    for sentence in substantive_sentences:
        best_score = 0.0

        for cid in valid_citations:
            score = sentence_support(sentence, chunks_by_id[cid])
            if score > best_score:
                best_score = score

        if best_score < threshold:
            issues.append(
                f"Low support ({best_score:.2f}): {sentence[:120]}"
            )

    return len(issues) == 0, issues

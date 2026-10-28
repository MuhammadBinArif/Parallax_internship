from __future__ import annotations

import re
from typing import Optional


def is_too_short(text: str, min_chars: int = 50, min_words: int = 10) -> bool:
    """Check if document is too short for meaningful NLP analysis."""
    if not text or len(text.strip()) < min_chars:
        return True
    word_count = len(text.split())
    return word_count < min_words


def is_mostly_jargon(text: str, threshold: float = 0.4) -> bool:
    """
    Detect if text is mostly jargon/technical terms.
    Uses heuristic: high ratio of capitalized words, numbers, or special tokens.
    """
    words = text.split()
    if not words:
        return False

    jargon_count = 0
    for word in words:
        # All caps (acronyms)
        if word.isupper() and len(word) > 1:
            jargon_count += 1
        # Contains many numbers
        elif re.search(r"\d{3,}", word):
            jargon_count += 1
        # Very long compound words (technical terms)
        elif len(word) > 20:
            jargon_count += 1
        # Contains special characters typical of code/jargon
        elif re.search(r"[{}\[\]<>=$#@]", word):
            jargon_count += 1

    ratio = jargon_count / len(words)
    return ratio > threshold


def preprocess_for_nlp(text: str) -> str:
    """
    Clean text specifically for NLP analysis.
    Removes noise while preserving meaning.
    """
    if not text:
        return ""

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)
    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove very long sequences of punctuation
    text = re.sub(r"[.!?,;:]{3,}", " ", text)

    return text.strip()


def handle_edge_case(text: str) -> Optional[str]:
    """
    Returns a special category if text is an edge case.
    Returns None if text is normal.
    """
    if not text or not text.strip():
        return "empty"
    if is_too_short(text):
        return "too_short"
    if is_mostly_jargon(text):
        return "jargon_heavy"
    if len(text) > 50000:
        return "very_long"
    return None

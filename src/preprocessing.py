from __future__ import annotations

import re
import unicodedata
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Iterator

try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except Exception:
    _NLP = None


DEFAULT_TEXT_FIELDS = (
    "text",
    "title",
    "content",
    "abstract",
    "article",
    "body",
    "intro",
    "life",
    "career",
    "works",
    "personal_life",
    "awards",
)

DEFAULT_ID_FIELDS = ("_id", "id", "idx", "uid")

_SPECIAL_SPACES = {
    "\u00a0": " ",   # non-breaking space
    "\u2007": " ",   # figure space
    "\u202f": " ",   # narrow no-break space
    "\u205f": " ",   # medium mathematical space
    "\u3000": " ",   # ideographic space
    "\u200b": "",    # zero width space
    "\u200c": "",    # zero width non-joiner
    "\u200d": "",    # zero width joiner
    "\ufeff": "",    # byte order mark
}

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TRAILING_WHITESPACE_BEFORE_NEWLINE_RE = re.compile(r"[ \t]+\n")
_LEADING_WHITESPACE_AFTER_NEWLINE_RE = re.compile(r"\n[ \t]+")

_ENGLISH_STOPWORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get",
    "which", "go", "me", "when", "make", "can", "like", "time", "no",
    "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first",
    "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us",
}


def strip_html(text: str) -> str:
    """
    Remove HTML tags and extract readable text.

    Purpose:
    - Many web-scraped documents contain HTML markup.
    - Embedding models work better on plain text.
    """
    if not isinstance(text, str):
        return ""

    if "<" not in text or ">" not in text:
        return text

    if BeautifulSoup is not None:
        soup = BeautifulSoup(text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        return soup.get_text(separator=" ")

    # Fallback if BeautifulSoup is not installed.
    return _HTML_TAG_RE.sub(" ", text)


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters and remove problematic invisible/control chars.

    Purpose:
    - Converts visually similar characters to canonical forms.
    - Removes zero-width spaces and control characters.
    - Replaces unusual space characters with normal spaces.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)

    for src, dst in _SPECIAL_SPACES.items():
        text = text.replace(src, dst)

    text = _CONTROL_RE.sub("", text)
    return text


def clean_whitespace(text: str) -> str:
    """
    Collapse excessive spaces and newlines.

    Purpose:
    - Produces consistent formatting.
    - Reduces noise before embedding/tokenization.
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WHITESPACE_BEFORE_NEWLINE_RE.sub("\n", text)
    text = _LEADING_WHITESPACE_AFTER_NEWLINE_RE.sub("\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    return text.strip()


def remove_urls_and_emails(text: str) -> str:
    """
    Remove URLs and email addresses.

    Purpose:
    - These are often noise for semantic text representation.
    """
    if not isinstance(text, str):
        return ""

    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    return text


def clean_text(
    text: str,
    *,
    remove_urls: bool = True,
) -> str:
    """
    Full text cleaning pipeline.

    Steps:
    1. Unicode normalization
    2. HTML stripping
    3. Optional URL/email removal
    4. Whitespace cleanup
    """
    text = normalize_unicode(text)
    text = strip_html(text)

    if remove_urls:
        text = remove_urls_and_emails(text)

    text = clean_whitespace(text)
    return text


def is_mostly_english(
    text: str,
    *,
    min_tokens: int = 20,
    threshold: float = 0.10,
    latin_threshold: float = 0.70,
    max_chars: int = 5000,
) -> bool:
    """
    Lightweight English-language heuristic.

    Purpose:
    - Filter out obviously non-English documents.
    - This is not a production-grade language detector,
      but it is sufficient for a beginner Week 1 pipeline.

    Notes:
    - Uses spaCy tokenizer if available.
    - Falls back to regex tokenization if spaCy model is missing.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    sample = text[:max_chars]

    letters = [ch for ch in sample if ch.isalpha()]
    if not letters:
        return False

    ascii_letters = sum(
        1 for ch in letters
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z")
    )

    if ascii_letters / len(letters) < latin_threshold:
        return False

    if _NLP is not None:
        doc = _NLP(sample)
        tokens = [token.text.lower() for token in doc if token.is_alpha]
    else:
        tokens = re.findall(r"[a-z']+", sample.lower())

    if len(tokens) < min_tokens:
        return False

    stopword_hits = sum(1 for token in tokens if token in _ENGLISH_STOPWORDS)
    return (stopword_hits / len(tokens)) >= threshold


def extract_text(
    doc: Mapping[str, Any],
    text_fields: Sequence[str],
) -> str:
    """
    Extract and concatenate text from selected fields of a document.

    Purpose:
    - Different datasets store text in different fields.
    - This normalizes them into one text string.
    """
    parts: list[str] = []

    for field in text_fields:
        value = doc.get(field)

        if value is None:
            continue

        if isinstance(value, Mapping):
            nested = extract_text(value, text_fields)
            if nested:
                parts.append(nested)

        elif isinstance(value, str):
            if value.strip():
                parts.append(value.strip())

        elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            for item in value:
                if isinstance(item, Mapping):
                    nested = extract_text(item, text_fields)
                    if nested:
                        parts.append(nested)
                elif isinstance(item, str) and item.strip():
                    parts.append(item.strip())

    return "\n\n".join(parts)


def clean_document(
    doc: Mapping[str, Any],
    *,
    doc_id: str,
    source: str,
    text_fields: Sequence[str],
    min_chars: int,
    max_chars: int,
    require_english: bool,
) -> dict[str, Any] | None:
    """
    Clean a single document and return a standardized record.

    Returns:
        None if the document should be filtered out.
    """
    raw_text = extract_text(doc, text_fields)
    text = clean_text(raw_text)

    if len(text) < min_chars:
        return None

    if len(text) > max_chars:
        text = text[:max_chars].rstrip()

        if len(text) < min_chars:
            return None

    if require_english and not is_mostly_english(text):
        return None

    word_count = len(re.findall(r"\S+", text))

    return {
        "id": str(doc_id),
        "text": text,
        "char_count": len(text),
        "word_count": word_count,
        "source": source,
        "language": "en_heuristic" if require_english else "unknown",
    }


def iter_clean_documents(
    raw_docs: Iterable[Any],
    *,
    source: str = "unknown",
    text_fields: Sequence[str] = DEFAULT_TEXT_FIELDS,
    id_fields: Sequence[str] = DEFAULT_ID_FIELDS,
    min_chars: int = 100,
    max_chars: int = 100000,
    require_english: bool = True,
) -> Iterator[dict[str, Any]]:
    """
    Iterate over raw documents and yield cleaned, standardized documents.

    Purpose:
    - Provides a reusable pipeline function.
    - Handles missing IDs by generating deterministic ones.
    """
    safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "-", source).strip("-")
    if not safe_source:
        safe_source = "doc"

    seen_ids: set[str] = set()

    for index, doc in enumerate(raw_docs):
        if not isinstance(doc, Mapping):
            continue

        base_id = None

        for field in id_fields:
            if field in doc and doc[field] is not None:
                candidate = str(doc[field]).strip()
                if candidate:
                    base_id = candidate
                    break

        if base_id is None:
            base_id = f"{safe_source}-{index}"

        if base_id in seen_ids:
            base_id = f"{base_id}-{index}"

        suffix = 0
        while base_id in seen_ids:
            suffix += 1
            base_id = f"{base_id}-{suffix}"

        seen_ids.add(base_id)

        cleaned = clean_document(
            doc,
            doc_id=base_id,
            source=source,
            text_fields=text_fields,
            min_chars=min_chars,
            max_chars=max_chars,
            require_english=require_english,
        )

        if cleaned is not None:
            yield cleaned

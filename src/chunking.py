from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
]


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def clean_text(text: Optional[str]) -> str:
    """
    Normalize whitespace while preserving paragraph boundaries.
    """
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces immediately before newlines.
    text = re.sub(r" *\n", "\n", text)

    return text.strip()


def _chunk_namespace(metadata: Optional[Dict[str, Any]]) -> str:
    if not metadata:
        return ""
    for key in ("id", "doc_id", "document_id", "line_number", "index", "source"):
        value = metadata.get(key)
        if value is not None:
            return str(value)
    return ""


def make_chunk_id(text: str, index: int, source: Optional[str] = None, namespace: str = "") -> str:
    """
    Deterministic chunk ID based on source, namespace, index, and content.
    """
    payload = f"{source or ''}|{namespace or ''}|{index}|{text}"
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:32]


def _hard_split(text: str, chunk_size: int) -> List[str]:
    """
    Fallback splitter for text with no usable separators.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _split_to_max(
    text: str,
    separators: Sequence[str],
    chunk_size: int,
) -> List[str]:
    """
    Recursively split text until every piece is <= chunk_size.
    """
    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    if not separators:
        return _hard_split(text, chunk_size)

    sep = separators[0]

    if not sep:
        return _hard_split(text, chunk_size)

    parts = [part.strip() for part in text.split(sep) if part.strip()]

    # If this separator does not help, try the next one.
    if len(parts) <= 1:
        return _split_to_max(text, separators[1:], chunk_size)

    pieces: List[str] = []

    for part in parts:
        if len(part) <= chunk_size:
            pieces.append(part)
        else:
            pieces.extend(_split_to_max(part, separators[1:], chunk_size))

    return pieces


def chunk_text(
    text: Optional[str],
    *,
    source: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """
    Recursive character-based chunking with overlap.

    Strategy:
    1. Clean text.
    2. Recursively split by natural boundaries:
       paragraphs -> lines -> sentences -> clauses -> words.
    3. Merge small pieces back together up to chunk_size.
    4. Add overlap between adjacent chunks when possible.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")

    # Overlap must leave room for at least one character of new content.
    chunk_overlap = min(chunk_overlap, max(0, chunk_size - 1))

    cleaned = clean_text(text)
    base_metadata = dict(metadata or {})

    if source:
        base_metadata["source"] = str(source)

    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [
            Chunk(
                id=make_chunk_id(cleaned, 0, source, _chunk_namespace(base_metadata)),
                text=cleaned,
                metadata={**base_metadata, "chunk_index": 0},
            )
        ]

    pieces = _split_to_max(cleaned, list(separators), chunk_size)

    if not pieces:
        return []

    merged: List[str] = []
    current = ""

    for piece in pieces:
        if not current:
            current = piece
            continue

        candidate = f"{current} {piece}".strip()

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        merged.append(current)

        # Build next chunk using overlap from the previous chunk.
        # Reserve 1 character for the joining space if overlap is used.
        max_overlap = max(0, chunk_size - len(piece) - 1)
        overlap_len = min(chunk_overlap, max_overlap)
        overlap = current[-overlap_len:] if overlap_len > 0 else ""

        if overlap:
            current = f"{overlap} {piece}".strip()
        else:
            current = piece

    if current:
        merged.append(current)

    chunks: List[Chunk] = []

    for idx, chunk_str in enumerate(merged):
        chunk_str = chunk_str.strip()

        if not chunk_str:
            continue

        chunks.append(
            Chunk(
                id=make_chunk_id(chunk_str, idx, source, _chunk_namespace(base_metadata)),
                text=chunk_str,
                metadata={**base_metadata, "chunk_index": idx},
            )
        )

    return chunks

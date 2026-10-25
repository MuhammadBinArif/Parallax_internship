from __future__ import annotations

import numbers
import os
import sys
from pathlib import Path
from typing import Any, Optional

import chromadb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embeddings import SentenceTransformerEmbedder
from src.week3.types import Chunk


EMBED_METHOD_NAMES = (
    "embed_query",
    "encode_query",
    "get_embedding",
    "embed_one",
    "embed",
    "encode",
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, numbers.Real))


def _is_number_sequence(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(_is_number(x) for x in value)
    )


def _as_vector(raw: Any) -> list[float]:
    """
    Convert common embedding return types into a flat list[float].

    Supports:
    - list[float]
    - tuple[float]
    - numpy array
    - torch tensor
    - EmbeddingResult(vectors=...)
    - objects with .vector / .embedding / .vectors
    - tuple results like (vector, seconds)
    """
    if raw is None:
        raise TypeError("Embedding method returned None")

    # Dataclass/object wrappers: EmbeddingResult(vectors=...), etc.
    for attr in ("vectors", "vector", "embedding", "embeddings", "values"):
        if hasattr(raw, attr):
            nested = getattr(raw, attr)
            if nested is not None:
                return _as_vector(nested)

    # Tuple results like (vector, seconds)
    if isinstance(raw, tuple):
        if not raw:
            raise ValueError("Embedding method returned empty tuple")
        return _as_vector(raw[0])

    # NumPy arrays / PyTorch tensors
    if hasattr(raw, "shape") and hasattr(raw, "ndim"):
        if getattr(raw, "size", 0) == 0:
            raise ValueError("Embedding method returned empty array")

        arr = raw

        if int(arr.ndim) > 1:
            arr = arr[0]

        if int(arr.ndim) == 0:
            try:
                return [float(arr.tolist())]
            except Exception:
                return [float(arr.item())]

        try:
            values = arr.tolist()
        except AttributeError:
            values = list(arr)

        return [float(x) for x in values]

    # Lists / tuples
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise ValueError("Embedding method returned empty sequence")

        if _is_number_sequence(raw):
            return [float(x) for x in raw]

        return _as_vector(raw[0])

    # Last-resort iterable of numbers
    try:
        return [float(x) for x in raw]
    except TypeError as exc:
        raise TypeError(
            f"Cannot convert embedding object of type {type(raw)!r} to a float vector"
        ) from exc


def _try_method(embedder: Any, method_name: str, query: str) -> Optional[list[float]]:
    """
    Try a candidate embedding method with several common call signatures.
    """
    fn = getattr(embedder, method_name, None)
    if not callable(fn):
        return None

    variants = (
        (([query],), {"normalize": True}),
        (([query],), {}),
        ((query,), {"normalize": True}),
        ((query,), {}),
    )

    for args, kwargs in variants:
        try:
            raw = fn(*args, **kwargs)
            vec = _as_vector(raw)
            if vec:
                return vec
        except Exception:
            continue

    return None


def _embed_query(embedder: Any, query: str) -> list[float]:
    """
    Produce a query embedding using the Week 2 embedder.
    """
    errors: list[str] = []

    for method_name in EMBED_METHOD_NAMES:
        fn = getattr(embedder, method_name, None)
        if not callable(fn):
            continue

        try:
            vec = _try_method(embedder, method_name, query)
            if vec is not None:
                return vec
            errors.append(f"{method_name}: returned unusable value")
        except Exception as exc:
            errors.append(f"{method_name}: {type(exc).__name__}: {exc}")

    # Direct fallback to underlying SentenceTransformer model.
    model = getattr(embedder, "_model", None) or getattr(embedder, "model", None)
    if model is not None and hasattr(model, "encode"):
        for kwargs in ({"normalize_embeddings": True}, {}):
            try:
                raw = model.encode([query], **kwargs)
                vec = _as_vector(raw)
                if vec:
                    return vec
            except Exception as exc:
                errors.append(
                    f"model.encode{kwargs}: {type(exc).__name__}: {exc}"
                )

    public_methods = [
        name
        for name in dir(embedder)
        if not name.startswith("_") and callable(getattr(embedder, name, None))
    ]

    raise AttributeError(
        "Could not produce a query embedding from SentenceTransformerEmbedder. "
        f"Tried methods: {EMBED_METHOD_NAMES}. "
        f"Available public methods: {public_methods}. "
        f"Errors: {errors}"
    )


def _score_from_distance(distance: Any, metric: str) -> Optional[float]:
    """
    Convert Chroma distance into a higher-is-better score.

    For cosine distance, Chroma usually returns:
        distance = 1 - cosine_similarity

    So:
        score = 1 - distance
    """
    if distance is None:
        return None

    try:
        d = float(distance)
    except Exception:
        return None

    metric = (metric or "cosine").strip().lower()

    if metric in {"cosine", "cos"}:
        return max(0.0, min(1.0, 1.0 - d))

    # Fallback for L2 / generic distance.
    return 1.0 / (1.0 + max(0.0, d))


class Week2ChromaRetriever:
    """
    Adapter that connects Week 3 to the Week 2 ChromaDB index.
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: Optional[str] = None,
        embedder: Optional[SentenceTransformerEmbedder] = None,
        distance_metric: Optional[str] = None,
    ):
        path = Path(persist_directory)
        if not path.is_absolute():
            path = ROOT / path

        if not path.exists():
            raise RuntimeError(
                f"ChromaDB path does not exist: {path}. "
                "Run Week 2 ingestion first."
            )

        self.embedder = embedder or SentenceTransformerEmbedder()
        self.distance_metric = distance_metric or os.getenv(
            "CHROMA_DISTANCE_METRIC",
            "cosine",
        )

        self.client = chromadb.PersistentClient(path=str(path))

        collections = self.client.list_collections()

        if not collections:
            raise RuntimeError(
                "No ChromaDB collections found. "
                "Run Week 2 ingestion first."
            )

        self.collection = None

        if collection_name:
            try:
                self.collection = self.client.get_collection(collection_name)
            except Exception:
                self.collection = None

        if self.collection is None:
            first = collections[0]
            name = getattr(first, "name", first)
            self.collection = self.client.get_collection(str(name))

        count = self.collection.count()

        if count == 0:
            raise RuntimeError(
                f"ChromaDB collection '{getattr(self.collection, 'name', 'unknown')}' is empty. "
                "Run Week 2 ingestion first."
            )

    def _query_collection(
        self,
        query_embedding: list[float],
        n_results: int,
    ) -> dict[str, Any]:
        """
        Query ChromaDB, preferring distances when available.
        """
        try:
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas"],
            )

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        if top_k <= 0:
            return []

        query_embedding = _embed_query(self.embedder, query)

        count = self.collection.count()
        n_results = min(top_k, count)

        results = self._query_collection(query_embedding, n_results)

        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        distances = results.get("distances") or []

        # Chroma usually returns nested lists for batch queries.
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if documents and isinstance(documents[0], list):
            documents = documents[0]
        if metadatas and isinstance(metadatas[0], list):
            metadatas = metadatas[0]
        if distances and isinstance(distances[0], list):
            distances = distances[0]

        chunks: list[Chunk] = []

        for i, chunk_id in enumerate(ids):
            text = ""

            if i < len(documents) and documents[i] is not None:
                text = str(documents[i])

            metadata: dict[str, Any] = {}

            if i < len(metadatas) and metadatas[i]:
                metadata = dict(metadatas[i])

            if not text.strip():
                text = str(metadata.get("text", ""))

            if not text.strip():
                continue

            distance = distances[i] if i < len(distances) else None
            score = _score_from_distance(distance, self.distance_metric)

            chunks.append(
                Chunk(
                    id=str(chunk_id),
                    text=text,
                    score=score,
                    metadata=metadata,
                )
            )

        return chunks


def build_retriever(collection_name: Optional[str] = None) -> Week2ChromaRetriever:
    if collection_name is None:
        collection_name = os.getenv("CHROMA_COLLECTION", "").strip() or None

    return Week2ChromaRetriever(collection_name=collection_name)

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import chromadb
import numpy as np


@dataclass
class SearchHit:
    id: str
    document: str
    metadata: Dict[str, Any]
    distance: float
    score: float


def _sanitize_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    ChromaDB metadata values must be primitive types.

    This function:
    - removes None values
    - converts unsupported values to strings
    - returns None for empty metadata, because ChromaDB rejects empty dicts
    """
    if not metadata:
        return None

    clean: Dict[str, Any] = {}

    for key, value in metadata.items():
        if value is None:
            continue

        str_key = str(key)

        if isinstance(value, (str, bool, int, float)):
            clean[str_key] = value
        elif isinstance(value, (list, tuple, set)):
            clean[str_key] = ", ".join(str(item) for item in value)
        else:
            clean[str_key] = str(value)

    return clean or None


class ChromaVectorStore:
    """
    Persistent ChromaDB wrapper for chunk ingestion and semantic search.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "week2_chunks",
        distance_function: str = "cosine",
    ):
        os.makedirs(persist_directory, exist_ok=True)

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.distance_function = distance_function

        self.client = chromadb.PersistentClient(path=persist_directory)

        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance_function},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to get/create Chroma collection '{collection_name}': {exc}"
            ) from exc

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """
        Delete and recreate the collection.

        Useful when switching embedding models because embedding dimension
        may change.
        """
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_function},
        )

    def _prepare_batch(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ):
        requested = len(ids)

        if len(documents) != requested:
            raise ValueError("ids and documents must have the same length")

        if len(embeddings) != requested:
            raise ValueError("ids and embeddings must have the same length")

        if metadatas is None:
            metadatas = [None for _ in range(requested)]

        if len(metadatas) != requested:
            raise ValueError("ids and metadatas must have the same length")

        valid_ids: List[str] = []
        valid_docs: List[str] = []
        valid_embeddings: List[List[float]] = []
        valid_metadatas: List[Optional[Dict[str, Any]]] = []
        skipped: List[Dict[str, Any]] = []

        expected_dim: Optional[int] = None

        for index, (chunk_id, document, embedding, metadata) in enumerate(
            zip(ids, documents, embeddings, metadatas)
        ):
            try:
                id_str = str(chunk_id).strip()

                if not id_str:
                    raise ValueError("empty id")

                doc_str = str(document).strip() or " "

                arr = np.asarray(embedding, dtype=np.float32).reshape(-1)

                if arr.size == 0:
                    raise ValueError("empty embedding")

                if not np.all(np.isfinite(arr)):
                    raise ValueError("embedding contains NaN or Inf")

                if expected_dim is None:
                    expected_dim = arr.size
                elif arr.size != expected_dim:
                    raise ValueError(
                        f"embedding dimension mismatch: got {arr.size}, expected {expected_dim}"
                    )

                valid_ids.append(id_str)
                valid_docs.append(doc_str)
                valid_embeddings.append(arr.tolist())
                valid_metadatas.append(_sanitize_metadata(metadata))

            except Exception as exc:
                skipped.append(
                    {
                        "index": index,
                        "id": str(chunk_id),
                        "reason": str(exc),
                    }
                )

        return (
            valid_ids,
            valid_docs,
            valid_embeddings,
            valid_metadatas,
            skipped,
        )

    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        if batch_size <= 0:
            batch_size = 100

        requested = len(ids)

        (
            valid_ids,
            valid_docs,
            valid_embeddings,
            valid_metadatas,
            skipped,
        ) = self._prepare_batch(ids, documents, embeddings, metadatas)

        # Defensive deduplication:
        # ChromaDB rejects duplicate IDs inside one upsert batch.
        # If duplicates occur, keep the latest occurrence, matching upsert semantics.
        seen: Dict[str, int] = {}
        unique_ids: List[str] = []
        unique_docs: List[str] = []
        unique_embeddings: List[List[float]] = []
        unique_metadatas: List[Optional[Dict[str, Any]]] = []
        duplicate_count = 0

        for cid, doc, emb, meta in zip(
            valid_ids,
            valid_docs,
            valid_embeddings,
            valid_metadatas,
        ):
            if cid in seen:
                pos = seen[cid]
                unique_ids[pos] = cid
                unique_docs[pos] = doc
                unique_embeddings[pos] = emb
                unique_metadatas[pos] = meta
                duplicate_count += 1
            else:
                seen[cid] = len(unique_ids)
                unique_ids.append(cid)
                unique_docs.append(doc)
                unique_embeddings.append(emb)
                unique_metadatas.append(meta)

        upserted = 0

        for start in range(0, len(unique_ids), batch_size):
            end = min(start + batch_size, len(unique_ids))

            batch_metadatas = unique_metadatas[start:end]

            # If every metadata item is None, pass metadatas=None.
            # ChromaDB accepts None metadata entries, but this is safer.
            metadatas_arg = (
                batch_metadatas
                if any(m is not None for m in batch_metadatas)
                else None
            )

            try:
                self.collection.upsert(
                    ids=unique_ids[start:end],
                    documents=unique_docs[start:end],
                    embeddings=unique_embeddings[start:end],
                    metadatas=metadatas_arg,
                )
                upserted += end - start
            except Exception as exc:
                raise RuntimeError(
                    "ChromaDB upsert failed. "
                    "If you changed embedding models or dimensions, use a new collection name "
                    "or call reset() before ingesting. "
                    f"Original error: {exc}"
                ) from exc

        return {
            "requested": requested,
            "upserted": upserted,
            "skipped": len(skipped),
            "duplicates": duplicate_count,
            "skipped_details": skipped[:20],
        }

    def query(
        self,
        query_embedding: Sequence[float] | np.ndarray,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        arr = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

        if arr.size == 0:
            return []

        if not np.all(np.isfinite(arr)):
            raise ValueError("query embedding contains NaN or Inf")

        current_count = self.count()

        if current_count == 0:
            return []

        n_results = min(max(1, int(n_results)), current_count)

        result = self.collection.query(
            query_embeddings=[arr.tolist()],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [])
        ids = ids[0] if ids else []

        docs = result.get("documents", [])
        docs = docs[0] if docs else []

        metadatas = result.get("metadatas", [])
        metadatas = metadatas[0] if metadatas else []

        distances = result.get("distances", [])
        distances = distances[0] if distances else []

        hits: List[SearchHit] = []

        for i, chunk_id in enumerate(ids):
            document = docs[i] if i < len(docs) and docs[i] is not None else ""
            metadata = (
                metadatas[i]
                if i < len(metadatas) and metadatas[i] is not None
                else {}
            )
            distance = (
                distances[i]
                if i < len(distances) and distances[i] is not None
                else float("nan")
            )

            try:
                distance_float = float(distance)
            except Exception:
                distance_float = float("nan")

            if not math.isfinite(distance_float):
                score = 0.0
            elif self.distance_function == "cosine":
                # Chroma cosine distance is typically 1 - cosine_similarity.
                score = 1.0 - distance_float
                score = max(0.0, min(1.0, score))
            elif self.distance_function == "l2":
                score = 1.0 / (1.0 + distance_float)
            else:
                score = -distance_float

            hits.append(
                SearchHit(
                    id=str(chunk_id),
                    document=str(document),
                    metadata=metadata,
                    distance=distance_float,
                    score=score,
                )
            )

        return hits

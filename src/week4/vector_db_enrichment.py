from __future__ import annotations

import time
from typing import Any, Optional

import chromadb


class EnrichedVectorStore:
    """
    Extends Week 2 ChromaDB with NLP metadata for filtered retrieval.
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "week4_enriched",
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name

        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_enriched_chunks(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """Upsert chunks with enriched NLP metadata."""
        total = len(ids)
        upserted = 0
        skipped = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_ids = ids[start:end]
            batch_docs = documents[start:end]
            batch_embs = embeddings[start:end]
            batch_metas = metadatas[start:end]

            # Sanitize metadata for ChromaDB
            clean_metas = []
            for meta in batch_metas:
                clean = {}
                for k, v in meta.items():
                    if v is None:
                        continue
                    if isinstance(v, (str, int, float, bool)):
                        clean[k] = v
                    elif isinstance(v, (list, tuple)):
                        clean[k] = str(v)
                    else:
                        clean[k] = str(v)
                clean_metas.append(clean if clean else {"_placeholder": True})

            try:
                self.collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embs,
                    metadatas=clean_metas,
                )
                upserted += end - start
            except Exception as e:
                print(f"Error upserting batch {start}-{end}: {e}")
                skipped += end - start

        return {
            "total": total,
            "upserted": upserted,
            "skipped": skipped,
            "collection_count": self.collection.count(),
        }

    def query_with_filter(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        topic_filter: Optional[int] = None,
        sentiment_filter: Optional[str] = None,
        has_person_filter: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        """Query with optional NLP metadata filters."""
        where_clauses = []

        if topic_filter is not None:
            where_clauses.append({"topic_id": {"$eq": topic_filter}})
        if sentiment_filter is not None:
            where_clauses.append({"sentiment_label": {"$eq": sentiment_filter}})
        if has_person_filter is not None:
            where_clauses.append({"has_person": {"$eq": has_person_filter}})

        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"Filtered query failed: {e}. Falling back to unfiltered.")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

        hits = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, chunk_id in enumerate(ids):
            hits.append({
                "id": chunk_id,
                "text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
                "score": 1.0 - dists[i] if i < len(dists) and dists[i] is not None else 0.0,
            })

        return hits

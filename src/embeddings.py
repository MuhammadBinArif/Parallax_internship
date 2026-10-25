from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    seconds: float
    model: str
    count: int
    dimension: int


class SentenceTransformerEmbedder:
    """
    Thin wrapper around sentence-transformers.

    Uses normalized embeddings by default because ChromaDB cosine search
    works best when vectors are unit length.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        try:
            self.dimension = int(self._model.get_embedding_dimension())
        except AttributeError:
            try:
                self.dimension = int(self._model.get_embedding_dimension())
            except AttributeError:
                self.dimension = int(self._model.get_sentence_embedding_dimension())
    def embed(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(
                vectors=np.empty((0, self.dimension), dtype=np.float32),
                seconds=0.0,
                model=self.model_name,
                count=0,
                dimension=self.dimension,
            )

        # ChromaDB does not like completely empty documents.
        # Replace empty strings with a single space to keep indexing stable.
        sanitized = [str(text).strip() or " " for text in texts]

        start = time.perf_counter()

        vectors = self._model.encode(
            sanitized,
            batch_size=batch_size or self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
        )

        seconds = time.perf_counter() - start
        arr = np.asarray(vectors, dtype=np.float32)

        return EmbeddingResult(
            vectors=arr,
            seconds=seconds,
            model=self.model_name,
            count=len(sanitized),
            dimension=int(arr.shape[1]) if arr.ndim > 1 else self.dimension,
        )

    def embed_one(
        self,
        text: str,
        *,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, float]:
        result = self.embed([text], normalize=normalize, show_progress=False)
        return result.vectors[0], result.seconds

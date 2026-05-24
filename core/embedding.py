"""Sentence embedding generation backed by sentence-transformers."""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING


class EmbeddingGenerator:
    """Wraps a SentenceTransformer model with cached batch encoding.

    The model is loaded eagerly so the first query doesn't pay the cold-start
    cost (~1s) under user-facing latency.
    """

    def __init__(self):
        self._model = SentenceTransformer(
            EMBEDDING.model_name,
            device=EMBEDDING.device,
            cache_folder=EMBEDDING.cache_dir,
        )
        self._batch_size = EMBEDDING.batch_size
        self._normalize = EMBEDDING.normalize_embeddings

    @property
    def model(self) -> SentenceTransformer:
        return self._model

    def encode(self, text: str) -> np.ndarray:
        """Encode a single string. Returns a 1-D array."""
        return self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
        )

    def encode_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Encode a batch of strings. Returns a 2-D array (N, dim).

        Returns an empty array (shape (0,)) when given no input — callers must
        check before treating the result as 2-D.
        """
        if not texts:
            return np.array([])

        return self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=show_progress,
        )

    def similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between two texts in [-1, 1]."""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)

        norm1 = float(np.linalg.norm(emb1))
        norm2 = float(np.linalg.norm(emb2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

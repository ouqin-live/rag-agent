"""Embedding wrappers with offline fallback."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Encode texts into dense vectors."""

    @abstractmethod
    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        """Return an array of shape (len(texts), dim)."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class FallbackEmbedding(BaseEmbedder):
    """Deterministic offline embedding fallback based on character random projection.

    Used when the network is unavailable or the local model cache is missing.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim
        self._rng = np.random.default_rng(seed=42)
        self._proj = self._rng.standard_normal((65536, dim))

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        vecs = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            if not text:
                continue
            idx = np.array([ord(c) % 65536 for c in text], dtype=np.int64)
            weights = np.ones(len(text), dtype=np.float32)
            vec = (self._proj[idx].T @ weights) / max(len(text), 1)
            vecs[i] = vec
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            vecs = vecs / norms
        return vecs


class SentenceTransformerEmbedder(BaseEmbedder):
    """Wrapper around sentence-transformers with graceful offline fallback."""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", fallback_dim: int = 384):
        self.model_name = model_name
        self._model = None
        self._dim_value = fallback_dim

        # Honor offline setting explicitly so users behind firewalls do not time out.
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            # Prefer the new method name; fall back to the legacy one.
            if hasattr(self._model, "get_embedding_dimension"):
                model_dim = self._model.get_embedding_dimension()
            else:
                model_dim = self._model.get_sentence_embedding_dimension()
            self._dim_value = model_dim or fallback_dim
            # Ensure the offline fallback matches the loaded model dimension.
            fallback_dim = self._dim_value
            logger.info("Loaded embedding model: %s (dim=%d)", model_name, self._dim_value)
        except Exception as exc:  # pragma: no cover - network/cache failures
            logger.warning(
                "Failed to load embedding model %s (%s). Using fallback embedder.",
                model_name,
                exc,
            )
            self._model = None

        self._fallback = FallbackEmbedding(dim=fallback_dim)

    @property
    def dim(self) -> int:
        return self._dim_value

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        if self._model is None:
            return self._fallback.encode(texts, normalize_embeddings=normalize_embeddings)
        return self._model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        )


def get_embedder(model_name: str | None = None) -> BaseEmbedder:
    """Factory that returns a SentenceTransformer embedder with fallback."""
    name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    return SentenceTransformerEmbedder(model_name=name)

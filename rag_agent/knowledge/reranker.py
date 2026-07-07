"""Rerankers for improving retrieval precision after coarse retrieval."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from rag_agent.embedder import BaseEmbedder
from rag_agent.knowledge.base import RetrievalResult

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Re-rank a list of retrieval results by computing relevance scores per pair."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return the top_k results re-ordered by relevance."""
        ...


class CrossEncoderReranker(BaseReranker):
    """Reranker powered by a sentence-transformers CrossEncoder.

    Falls back to no-op (identity) when the model cannot be loaded.
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-base"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            logger.info("Loaded reranker model: %s", self.model_name)
        except Exception as exc:
            logger.warning(
                "Failed to load reranker model %s (%s). Reranking disabled.",
                self.model_name,
                exc,
            )
            self._model = None

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if self._model is None or not results:
            return results[:top_k]

        # 构建 (query, chunk_text) 对
        pairs = [(query, r.text) for r in results]
        scores = self._model.predict(pairs)

        # 按 Cross-Encoder 分数排序
        ranked = sorted(
            zip(results, scores), key=lambda x: x[1], reverse=True
        )
        return [
            RetrievalResult(chunk=r.chunk, score=float(score))
            for r, score in ranked[:top_k]
        ]


class EmbeddingReranker(BaseReranker):
    """Reranker using the existing bi-encoder for pair-wise similarity.

    Reuses the project's embedder (e.g. BAAI/bge-small-zh-v1.5) to compute
    cosine similarity between the query and each candidate chunk.  This is
    less precise than a Cross-Encoder, but requires NO extra model download.
    """

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        texts = [r.text for r in results]
        # 同时编码 query 和候选文本
        all_embeddings = self.embedder.encode([query] + texts, normalize_embeddings=True)
        query_vec = all_embeddings[0]
        chunk_vecs = all_embeddings[1:]

        # 余弦相似度（已归一化，点积即余弦相似度）
        scores = chunk_vecs @ query_vec

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            RetrievalResult(chunk=results[i].chunk, score=float(score))
            for i, score in ranked[:top_k]
        ]

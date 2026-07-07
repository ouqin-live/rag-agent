"""Cross-Encoder reranker for improving retrieval precision."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

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

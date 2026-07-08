"""Semantic cache: reuse previous answers when the question intent is similar."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rag_agent.embedder import BaseEmbedder
from rag_agent.evaluation.base import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    """A single semantic-cache entry."""

    query: str
    query_embedding: np.ndarray
    answer: str
    contexts: list[str]
    long_term_facts: list[str]
    evaluation: EvaluationResult | None
    created_at: float = field(default_factory=time.time)


class SemanticCache:
    """In-memory semantic cache keyed by query embedding similarity.

    When a new query arrives, its embedding is compared with cached query
    embeddings. If the cosine similarity is above ``threshold`` and the entry
    has not expired, the cached response is returned, skipping retrieval and
    LLM generation entirely.

    The cache is user-scoped so that facts/preferences of different users do
    not leak between sessions.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        threshold: float = 0.92,
        ttl_seconds: float | None = None,
        max_entries_per_user: int = 100,
    ):
        self.embedder = embedder
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries_per_user = max_entries_per_user
        # {user_id: [CacheEntry, ...]}
        self._store: dict[str, list[_CacheEntry]] = {}

    def lookup(
        self,
        query: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        """Return a cached response if a semantically similar query exists."""
        entries = self._store.get(user_id, [])
        if not entries:
            return None

        query_embedding = self.embedder.encode([query], normalize_embeddings=True)[0]
        best_entry: _CacheEntry | None = None
        best_score = -1.0

        now = time.time()
        valid_entries: list[_CacheEntry] = []
        for entry in entries:
            if self._is_expired(entry, now):
                continue
            valid_entries.append(entry)
            score = float(query_embedding @ entry.query_embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        # Keep only non-expired entries
        self._store[user_id] = valid_entries

        if best_entry is None or best_score < self.threshold:
            return None

        logger.info(
            "Semantic cache hit for user %s (score=%.4f): %r",
            user_id,
            best_score,
            best_entry.query,
        )
        return {
            "answer": best_entry.answer,
            "contexts": best_entry.contexts,
            "long_term_facts": best_entry.long_term_facts,
            "evaluation": best_entry.evaluation,
            "cached_query": best_entry.query,
            "similarity": best_score,
        }

    def store(
        self,
        query: str,
        user_id: str,
        answer: str,
        contexts: list[str],
        long_term_facts: list[str],
        evaluation: EvaluationResult | None,
    ) -> None:
        """Store a response in the semantic cache."""
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)[0]
        entry = _CacheEntry(
            query=query,
            query_embedding=np.asarray(query_embedding, dtype=np.float32),
            answer=answer,
            contexts=list(contexts),
            long_term_facts=list(long_term_facts),
            evaluation=evaluation,
        )
        self._store.setdefault(user_id, []).append(entry)
        self._enforce_capacity(user_id)
        logger.debug("Semantic cache stored for user %s: %r", user_id, query)

    def clear(self, user_id: str | None = None) -> None:
        """Clear cache for a specific user or all users."""
        if user_id is None:
            self._store.clear()
        else:
            self._store.pop(user_id, None)

    def _is_expired(self, entry: _CacheEntry, now: float) -> bool:
        if self.ttl_seconds is None:
            return False
        return (now - entry.created_at) > self.ttl_seconds

    def _enforce_capacity(self, user_id: str) -> None:
        entries = self._store.get(user_id, [])
        if len(entries) > self.max_entries_per_user:
            # Remove oldest entries
            self._store[user_id] = entries[-self.max_entries_per_user :]

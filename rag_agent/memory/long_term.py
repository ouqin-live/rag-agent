"""Long-term user memory backed by a vector store."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rag_agent.embedder import BaseEmbedder, get_embedder
from rag_agent.knowledge.base import Chunk, VectorStore
from rag_agent.knowledge.chroma_store import ChromaVectorStore
from rag_agent.knowledge.store import LocalVectorStore

logger = logging.getLogger(__name__)


@dataclass
class MemoryFact:
    """A single long-term fact about a user."""

    id: str
    user_id: str
    content: str
    created_at: datetime
    updated_at: datetime


def _make_fact_id(user_id: str, content: str) -> str:
    payload = f"{user_id}:{content}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class LongTermMemory:
    """Stores and retrieves user-specific facts/preferences across sessions."""

    def __init__(
        self,
        store: VectorStore,
        embedder: BaseEmbedder,
        dedup_threshold: float = 0.92,
        merge_threshold: float = 0.80,
        max_facts_per_user: int = 100,
    ):
        self.store = store
        self.embedder = embedder
        self.dedup_threshold = dedup_threshold
        self.merge_threshold = merge_threshold
        self.max_facts_per_user = max_facts_per_user

    @classmethod
    def from_local_store(
        cls,
        store_path: str | Path,
        embedder: BaseEmbedder | None = None,
        dedup_threshold: float = 0.92,
        merge_threshold: float = 0.80,
        max_facts_per_user: int = 100,
    ) -> "LongTermMemory":
        """使用 LocalVectorStore（SQLite + numpy）创建长期记忆。"""
        embedder = embedder or get_embedder()
        store = LocalVectorStore(store_path, dim=embedder.dim)
        return cls(
            store=store,
            embedder=embedder,
            dedup_threshold=dedup_threshold,
            merge_threshold=merge_threshold,
            max_facts_per_user=max_facts_per_user,
        )

    @classmethod
    def from_chroma_store(
        cls,
        store_path: str | Path,
        embedder: BaseEmbedder | None = None,
        dedup_threshold: float = 0.92,
        merge_threshold: float = 0.80,
        max_facts_per_user: int = 100,
    ) -> "LongTermMemory":
        """使用 ChromaVectorStore（HNSW 索引 + 自动持久化）创建长期记忆。推荐方式。"""
        embedder = embedder or get_embedder()
        store = ChromaVectorStore(store_path, collection_name="memory_facts")
        return cls(
            store=store,
            embedder=embedder,
            dedup_threshold=dedup_threshold,
            merge_threshold=merge_threshold,
            max_facts_per_user=max_facts_per_user,
        )

    def remember(self, user_id: str, fact: str) -> str | None:
        """Store a fact about the user, deduplicating/merging when appropriate.

        Returns the fact id if stored, or None if skipped as duplicate.
        """
        fact = fact.strip()
        if not fact or len(fact) < 3:
            return None

        fact_id = _make_fact_id(user_id, fact)
        embedding = self.embedder.encode([fact], normalize_embeddings=True)[0]

        # Check for duplicates or near-duplicates for this user
        existing = self._recall_raw(user_id, fact, top_k=5)

        for result in existing:
            old_chunk = result.chunk
            old_content = old_chunk.metadata.get("fact", old_chunk.text)
            similarity = float(result.score)

            if similarity >= self.dedup_threshold:
                logger.debug("Skipping duplicate fact for user %s: %s", user_id, fact)
                return old_chunk.id

            if similarity >= self.merge_threshold:
                # Replace with the more detailed/longer fact
                merged = self._merge_facts(old_content, fact)
                new_id = _make_fact_id(user_id, merged)
                self.forget(user_id, old_chunk.id)
                return self._insert_fact(new_id, user_id, merged)

        # Enforce per-user capacity limit
        self._enforce_capacity(user_id)
        return self._insert_fact(fact_id, user_id, fact)

    def recall(self, user_id: str, query: str, top_k: int = 3) -> list[MemoryFact]:
        """Retrieve facts relevant to ``query`` for a specific user."""
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)[0]
        results = self.store.search(
            query_embedding,
            top_k=top_k,
            filters={"user_id": user_id},
        )
        facts: list[MemoryFact] = []
        for result in results:
            chunk = result.chunk
            content = chunk.metadata.get("fact", chunk.text)
            created_at = self._parse_datetime(chunk.metadata.get("created_at"))
            updated_at = self._parse_datetime(chunk.metadata.get("updated_at"))
            facts.append(
                MemoryFact(
                    id=chunk.id,
                    user_id=user_id,
                    content=content,
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return facts

    def forget(self, user_id: str, fact_id: str) -> None:
        """Remove a specific fact by id."""
        # Because each fact is stored under its own doc_id, delete_by_doc removes exactly one fact.
        self.store.delete_by_doc(fact_id)
        self.store.persist()
        logger.info("Forgot fact %s for user %s", fact_id, user_id)

    def _recall_raw(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list:
        """Internal recall returning store results (used for dedup)."""
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)[0]
        return self.store.search(
            query_embedding,
            top_k=top_k,
            filters={"user_id": user_id},
        )

    def _insert_fact(self, fact_id: str, user_id: str, fact: str) -> str:
        now = datetime.now(timezone.utc).isoformat()
        chunk = Chunk(
            id=fact_id,
            text=fact,
            doc_id=fact_id,
            metadata={
                "user_id": user_id,
                "fact": fact,
                "created_at": now,
                "updated_at": now,
            },
            embedding=np.asarray(
                self.embedder.encode([fact], normalize_embeddings=True)[0],
                dtype=np.float32,
            ),
        )
        self.store.add([chunk])
        self.store.persist()
        logger.info("Remembered fact for user %s: %s", user_id, fact[:60])
        return fact_id

    def _enforce_capacity(self, user_id: str) -> None:
        """If the user has too many facts, delete the oldest ones."""
        # Collect all facts for the user by scanning with a broad query
        all_facts = self._recall_raw(user_id, "", top_k=self.max_facts_per_user * 2)
        user_facts = [r for r in all_facts if r.chunk.metadata.get("user_id") == user_id]

        if len(user_facts) < self.max_facts_per_user:
            return

        # Sort by created_at ascending and remove oldest
        sorted_facts = sorted(
            user_facts,
            key=lambda r: self._parse_datetime(r.chunk.metadata.get("created_at")),
        )
        to_remove = len(sorted_facts) - self.max_facts_per_user + 1
        for result in sorted_facts[:to_remove]:
            self.forget(user_id, result.chunk.id)

    @staticmethod
    def _merge_facts(old: str, new: str) -> str:
        """Merge two similar facts, preferring the longer/more specific one."""
        return new if len(new) >= len(old) else old

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                # Strip 'Z' suffix for fromisoformat compatibility
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    def __len__(self) -> int:
        return len(self.store)

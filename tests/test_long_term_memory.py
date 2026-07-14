"""Tests for long-term memory storage and recall."""

from __future__ import annotations

from pathlib import Path

from rag_agent.embedder import BaseEmbedder
from rag_agent.memory.long_term import LongTermMemory


def test_remember_and_recall(temp_dir: Path, fact_embedder: BaseEmbedder) -> None:
    ltm = LongTermMemory.from_local_store(
        temp_dir,
        embedder=fact_embedder,
        dedup_threshold=0.99,
        merge_threshold=0.99,
        max_facts_per_user=100,
    )

    fact_id = ltm.remember("user-1", "User likes blue color")
    assert fact_id is not None

    facts = ltm.recall("user-1", "What does the user like?", top_k=3)
    assert len(facts) == 1
    assert "blue" in facts[0].content.lower()


def test_deduplication(temp_dir: Path, fact_embedder: BaseEmbedder) -> None:
    ltm = LongTermMemory.from_local_store(
        temp_dir,
        embedder=fact_embedder,
        dedup_threshold=0.95,
        merge_threshold=0.99,
        max_facts_per_user=100,
    )

    id1 = ltm.remember("user-1", "User likes blue color")
    id2 = ltm.remember("user-1", "User likes blue color")

    # Duplicate should return the existing fact id, not create a new one.
    assert id1 is not None
    assert id2 == id1

    facts = ltm.recall("user-1", "User preferences", top_k=5)
    assert len(facts) == 1


def test_user_isolation(temp_dir: Path, fact_embedder: BaseEmbedder) -> None:
    ltm = LongTermMemory.from_local_store(
        temp_dir,
        embedder=fact_embedder,
        dedup_threshold=0.99,
        merge_threshold=0.99,
        max_facts_per_user=100,
    )

    ltm.remember("user-a", "User likes coffee")
    ltm.remember("user-b", "User likes tea")

    facts_a = ltm.recall("user-a", "What does user like?", top_k=3)
    assert len(facts_a) == 1
    assert "coffee" in facts_a[0].content.lower()

    facts_b = ltm.recall("user-b", "What does user like?", top_k=3)
    assert len(facts_b) == 1
    assert "tea" in facts_b[0].content.lower()


def test_capacity_limit(temp_dir: Path, fact_embedder: BaseEmbedder) -> None:
    ltm = LongTermMemory.from_local_store(
        temp_dir,
        embedder=fact_embedder,
        dedup_threshold=0.99,
        merge_threshold=0.99,
        max_facts_per_user=2,
    )

    ltm.remember("user-1", "Fact one about red")
    ltm.remember("user-1", "Fact two about blue")
    ltm.remember("user-1", "Fact three about coffee")

    facts = ltm.recall("user-1", "All facts", top_k=10)
    assert len(facts) <= 2

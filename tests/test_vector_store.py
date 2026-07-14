"""Tests for vector store implementations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rag_agent.embedder import BaseEmbedder
from rag_agent.knowledge.base import Chunk
from rag_agent.knowledge.store import LocalVectorStore


def test_local_vector_store_add_and_search(
    temp_dir: Path,
    sample_chunks: list[Chunk],
    mock_embedder: BaseEmbedder,
) -> None:
    store = LocalVectorStore(temp_dir, dim=mock_embedder.dim)
    store.add(sample_chunks)

    query = "What does RAG stand for?"
    query_embedding = mock_embedder.encode([query])[0]
    results = store.search(query_embedding, top_k=2)

    assert len(results) == 2
    assert all(r.chunk.text for r in results)
    # The closest chunk should mention RAG.
    assert "RAG" in results[0].chunk.text or "retrieval" in results[0].chunk.text


def test_local_vector_store_delete_by_doc(
    temp_dir: Path,
    sample_chunks: list[Chunk],
) -> None:
    store = LocalVectorStore(temp_dir, dim=8)
    store.add(sample_chunks)
    assert len(store) == 3

    store.delete_by_doc("doc-1")
    assert len(store) == 0


def test_local_vector_store_persistence(
    temp_dir: Path,
    sample_chunks: list[Chunk],
    mock_embedder: BaseEmbedder,
) -> None:
    store1 = LocalVectorStore(temp_dir, dim=mock_embedder.dim)
    store1.add(sample_chunks)
    store1.persist()

    store2 = LocalVectorStore(temp_dir, dim=mock_embedder.dim)
    assert len(store2) == len(sample_chunks)

    query_embedding = mock_embedder.encode(["RAG meaning"])[0]
    results = store2.search(query_embedding, top_k=1)
    assert len(results) == 1


def test_local_vector_store_filters_by_metadata(
    temp_dir: Path,
    mock_embedder: BaseEmbedder,
) -> None:
    texts = ["alpha", "beta", "gamma"]
    embeddings = mock_embedder.encode(texts)
    chunks = [
        Chunk(
            id=f"c-{i}",
            text=text,
            doc_id="doc-filter",
            metadata={"group": "even" if i % 2 == 0 else "odd"},
            embedding=embeddings[i],
        )
        for i, text in enumerate(texts)
    ]

    store = LocalVectorStore(temp_dir, dim=mock_embedder.dim)
    store.add(chunks)

    query_embedding = mock_embedder.encode(["alpha"])[0]
    results = store.search(query_embedding, top_k=10, filters={"group": "even"})

    assert len(results) == 2
    assert all(r.chunk.metadata.get("group") == "even" for r in results)

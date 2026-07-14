"""Shared fixtures for the test suite.

All tests use lightweight, deterministic components so the suite can run
offline without downloading embedding models or calling external LLMs.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rag_agent.embedder import BaseEmbedder
from rag_agent.knowledge.base import Chunk
from rag_agent.llm import MockLLMClient


class MockEmbedder(BaseEmbedder):
    """Deterministic embedder for tests.

    Produces a unique normalized vector for each text by hashing its content.
    """

    def __init__(self, dim: int = 8):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        rng = np.random.default_rng(seed=42)
        base = rng.standard_normal((len(texts), self._dim)).astype(np.float32)

        for i, text in enumerate(texts):
            # Mix the text hash into the vector so different texts differ.
            h = hash(text) % (2**16)
            base[i] += np.sin(np.arange(self._dim) + h) * 0.5

        if normalize_embeddings:
            norms = np.linalg.norm(base, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            base = base / norms
        return base


class FactEmbedder(BaseEmbedder):
    """Tiny embedder that makes similar facts collide for dedup tests."""

    def __init__(self, dim: int = 4):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        vecs = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Simple keyword-based embedding for deterministic tests.
            lower = text.lower()
            if "blue" in lower or "蓝色" in lower:
                vecs[i, 0] = 1.0
            if "red" in lower or "红色" in lower:
                vecs[i, 1] = 1.0
            if "coffee" in lower or "咖啡" in lower:
                vecs[i, 2] = 1.0
            if "tea" in lower or "茶" in lower:
                vecs[i, 3] = 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            vecs = vecs / norms
        return vecs


@pytest.fixture
def temp_dir() -> Path:
    """Provide a temporary directory that is cleaned up after the test."""
    path = Path(tempfile.mkdtemp(prefix="rag_agent_test_"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def mock_embedder() -> BaseEmbedder:
    return MockEmbedder(dim=8)


@pytest.fixture
def fact_embedder() -> BaseEmbedder:
    return FactEmbedder(dim=4)


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def sample_chunks(mock_embedder: BaseEmbedder) -> list[Chunk]:
    texts = [
        "RAG stands for retrieval augmented generation.",
        "It retrieves documents before generating an answer.",
        "Vector stores are used to search relevant chunks.",
    ]
    embeddings = mock_embedder.encode(texts)
    return [
        Chunk(
            id=f"chunk-{i}",
            text=text,
            doc_id="doc-1",
            metadata={"index": i},
            embedding=embeddings[i],
        )
        for i, text in enumerate(texts)
    ]

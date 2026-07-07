"""Base abstractions for the knowledge module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rag_agent.embedder import BaseEmbedder


@dataclass
class Document:
    """A source document."""

    id: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A chunk sliced from a document, optionally with an embedding."""

    id: str
    text: str
    doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None


@dataclass
class RetrievalResult:
    """A single retrieval result."""

    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text


class BaseLoader(ABC):
    """Load raw documents from a source (file path or URL)."""

    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """Return a list of documents loaded from ``source``."""
        ...


class BaseChunker(ABC):
    """Split a document into chunks."""

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        ...


class VectorStore(ABC):
    """Persistent vector store for chunks."""

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Add or overwrite chunks."""
        ...

    @abstractmethod
    def delete_by_doc(self, doc_id: str) -> None:
        """Remove all chunks belonging to ``doc_id``."""
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search by a pre-computed query embedding."""
        ...

    @abstractmethod
    def persist(self) -> None:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

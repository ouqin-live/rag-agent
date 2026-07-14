"""Tests for document chunkers."""

from __future__ import annotations

import pytest

from rag_agent.knowledge.base import Document
from rag_agent.knowledge.chunker import FixedSizeChunker, RecursiveChunker


def test_fixed_size_chunker_splits_document() -> None:
    doc = Document(
        id="doc-1",
        content="This is sentence one. This is sentence two. This is sentence three.",
        source="test.txt",
    )
    chunker = FixedSizeChunker(chunk_size=40, overlap=5)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    assert all(c.doc_id == "doc-1" for c in chunks)
    assert all(c.text for c in chunks)
    # IDs should be deterministic.
    assert len({c.id for c in chunks}) == len(chunks)


def test_fixed_size_chunker_empty_document() -> None:
    doc = Document(id="doc-empty", content="   ", source="test.txt")
    chunker = FixedSizeChunker(chunk_size=50, overlap=5)
    assert chunker.chunk(doc) == []


def test_fixed_size_chunker_overlap_invalid() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=50, overlap=50)


def test_fixed_size_chunker_chinese_sentences() -> None:
    # Sentence splitter expects whitespace after terminators.
    doc = Document(
        id="doc-zh",
        content="这是第一句。 这是第二句。 这是第三句。 这是第四句。",
        source="test.txt",
    )
    chunker = FixedSizeChunker(chunk_size=8, overlap=2)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    assert any("第一句" in c.text or "第二句" in c.text for c in chunks)


def test_recursive_chunker_fallback() -> None:
    doc = Document(
        id="doc-rec",
        content="Hello world. This is a test.",
        source="test.txt",
    )
    chunker = RecursiveChunker(chunk_size=20, overlap=0)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 1
    assert all(c.text for c in chunks)

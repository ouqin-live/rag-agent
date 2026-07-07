"""Chunkers that split documents into retrieval units."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

import numpy as np

from rag_agent.knowledge.base import BaseChunker, Chunk, Document

if TYPE_CHECKING:
    from rag_agent.embedder import BaseEmbedder


def _make_chunk_id(doc_id: str, text: str, index: int) -> str:
    payload = f"{doc_id}:{index}:{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class FixedSizeChunker(BaseChunker):
    """Split text into fixed-size chunks with optional overlap.

    Splits on sentence boundaries when possible to improve readability.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content.strip()
        if not text:
            return []

        # Prefer sentence-level boundaries
        sentences = self._split_sentences(text)
        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_len + sentence_len > self.chunk_size and current_parts:
                chunks.append(self._build_chunk(doc, current_parts, len(chunks)))
                # Keep trailing text for overlap
                overlap_text = self._build_overlap(current_parts)
                current_parts = [overlap_text, sentence] if overlap_text else [sentence]
                current_len = sum(len(p) for p in current_parts)
            else:
                current_parts.append(sentence)
                current_len += sentence_len

        if current_parts:
            chunks.append(self._build_chunk(doc, current_parts, len(chunks)))

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        # Split on Chinese/Japanese full stop or Latin sentence terminators
        raw = re.split(r"(?<=[。！？.!?])\s+", text)
        return [s.strip() for s in raw if s.strip()]

    def _build_overlap(self, parts: list[str]) -> str:
        """Return a prefix of the previous chunk up to ``overlap`` chars."""
        text = "".join(parts)
        if len(text) <= self.overlap:
            return text
        return text[-self.overlap :]

    def _build_chunk(self, doc: Document, parts: list[str], index: int) -> Chunk:
        text = "".join(parts)
        return Chunk(
            id=_make_chunk_id(doc.id, text, index),
            text=text,
            doc_id=doc.id,
            metadata={"chunk_index": index, "source": doc.source},
        )


class SemanticChunker(BaseChunker):
    """Split text at topic boundaries by detecting drops in semantic similarity.

    Computes embeddings for each sentence, then places chunk boundaries
    where the cosine similarity between adjacent sentences falls below
    ``similarity_threshold``.
    """

    def __init__(
        self,
        embedder: "BaseEmbedder",
        similarity_threshold: float = 0.3,
        min_chunk_size: int = 50,
        max_chunk_size: int = 800,
    ):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content.strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return self._single_chunk(doc, sentences)

        # 计算每个句子的 embedding
        embeddings = self.embedder.encode(sentences, normalize_embeddings=True)

        # 找到语义断点：相邻句相似度低于阈值的位置
        breakpoints = self._find_breakpoints(sentences, embeddings)

        # 在断点处切分，并按大小合并
        return self._build_chunks(doc, sentences, breakpoints)

    def _split_sentences(self, text: str) -> list[str]:
        raw = re.split(r"(?<=[。！？.!?])\s+", text)
        return [s.strip() for s in raw if s.strip()]

    def _find_breakpoints(
        self,
        sentences: list[str],
        embeddings: np.ndarray,
    ) -> list[int]:
        """Return indices of sentences that should start a new chunk."""
        breakpoints: list[int] = [0]  # first sentence always starts a chunk
        for i in range(len(sentences) - 1):
            sim = float(embeddings[i] @ embeddings[i + 1])
            if sim < self.similarity_threshold:
                breakpoints.append(i + 1)
        return breakpoints

    def _build_chunks(
        self,
        doc: Document,
        sentences: list[str],
        breakpoints: list[int],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_parts: list[str] = []
        chunk_len = 0

        last_bp = 0
        for bp in breakpoints[1:] + [len(sentences)]:
            segment = sentences[last_bp:bp]
            segment_text = "".join(segment)
            segment_len = len(segment_text)

            if chunk_len + segment_len <= self.max_chunk_size:
                # 合入当前 chunk
                chunk_parts.extend(segment)
                chunk_len += segment_len
            else:
                # 当前 chunk 满了，保存并开启新 chunk
                if chunk_parts:
                    chunks.append(self._build_chunk(doc, chunk_parts, len(chunks)))
                chunk_parts = list(segment)
                chunk_len = segment_len

            last_bp = bp

        if chunk_parts:
            chunks.append(self._build_chunk(doc, chunk_parts, len(chunks)))

        return chunks

    def _build_chunk(self, doc: Document, parts: list[str], index: int) -> Chunk:
        text = "".join(parts)
        return Chunk(
            id=_make_chunk_id(doc.id, text, index),
            text=text,
            doc_id=doc.id,
            metadata={"chunk_index": index, "source": doc.source, "chunker": "semantic"},
        )

    def _single_chunk(self, doc: Document, sentences: list[str]) -> list[Chunk]:
        text = "".join(sentences)
        return [
            Chunk(
                id=_make_chunk_id(doc.id, text, 0),
                text=text,
                doc_id=doc.id,
                metadata={"chunk_index": 0, "source": doc.source, "chunker": "semantic"},
            )
        ]


class RecursiveChunker(BaseChunker):
    """Recursively split text by separators of decreasing granularity.

    Order: paragraphs -> sentences -> words -> characters.
    Each final chunk respects ``chunk_size``.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", "。", "！", "？", ". ", " ", ""]

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content.strip()
        if not text:
            return []
        splits = self._recursive_split(text, self.separators)
        return self._merge_splits(doc, splits)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]
        sep = separators[0]
        rest = separators[1:]
        if sep == "":
            return list(text)
        parts = [p.strip() for p in text.split(sep) if p.strip()]
        result: list[str] = []
        for part in parts:
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._recursive_split(part, rest))
        return result

    def _merge_splits(self, doc: Document, splits: list[str]) -> list[Chunk]:
        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0

        for part in splits:
            part_len = len(part)
            if current_len + part_len > self.chunk_size and current:
                chunks.append(self._build_chunk(doc, current, len(chunks)))
                overlap_text = self._build_overlap(current)
                current = [overlap_text, part] if overlap_text else [part]
                current_len = sum(len(p) for p in current)
            else:
                current.append(part)
                current_len += part_len

        if current:
            chunks.append(self._build_chunk(doc, current, len(chunks)))
        return chunks

    def _build_overlap(self, parts: list[str]) -> str:
        text = "".join(parts)
        if len(text) <= self.overlap:
            return text
        return text[-self.overlap :]

    def _build_chunk(self, doc: Document, parts: list[str], index: int) -> Chunk:
        text = "".join(parts)
        return Chunk(
            id=_make_chunk_id(doc.id, text, index),
            text=text,
            doc_id=doc.id,
            metadata={"chunk_index": index, "source": doc.source},
        )

"""Chunkers that split documents into retrieval units."""

from __future__ import annotations

import hashlib
import re

from rag_agent.knowledge.base import BaseChunker, Chunk, Document


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

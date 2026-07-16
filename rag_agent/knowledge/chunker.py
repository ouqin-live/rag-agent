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


class MarkdownStructureChunker(BaseChunker):
    """按 Markdown 标题层级分块，保持文档结构完整性。

    核心策略：
    - 使用 mistune 解析 Markdown AST，精确识别标题、代码块、表格、列表
    - 按 h2（##）切大段，h3（###）切小段
    - 每个 chunk 带上父标题作为上下文前缀
    - 代码块和表格作为独立原子单元，不被打散
    """

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 800,
        heading_split_level: int = 2,
        merge_small: bool = True,
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.heading_split_level = heading_split_level
        self.merge_small = merge_small
        # 惰性初始化 mistune AST 渲染器
        self._md = None

    def _get_ast(self, text: str) -> list[dict]:
        """用 mistune 解析 Markdown 为 AST token 列表。"""
        if self._md is None:
            import mistune
            self._md = mistune.create_markdown(renderer="ast")
        return self._md(text)  # type: ignore[no-any-return]

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content.strip()
        if not text:
            return []

        sections = self._parse_sections(text)
        if not sections:
            return []

        raw_chunks = self._split_by_headings(sections)
        merged = self._merge_small_chunks(raw_chunks) if self.merge_small else raw_chunks
        return self._finalize(merged, doc)

    def _parse_sections(self, text: str) -> list[dict]:
        """用 mistune AST 解析 Markdown 结构为段列表。

        每个段包含 level（标题级数，0=正文）、heading（标题文本）、body（内容行列表）。
        mistune 自动处理代码块、表格、列表的边界，不需要手动状态追踪。
        """
        tokens = self._get_ast(text)
        sections: list[dict] = []
        current: dict | None = None

        for token in tokens:
            token_type = token.get("type", "")

            if token_type == "heading":
                level = token.get("attrs", {}).get("level", 1)
                heading_text = self._extract_text(token)
                sections.append({
                    "level": level,
                    "heading": heading_text,
                    "body": [],
                })
                current = sections[-1]

            elif token_type in ("blank_line", "thematic_break"):
                # 空行和分隔线：保留到 body 以维持段落间距
                if current is not None:
                    current["body"].append("")

            else:
                # 段落、代码块、表格、列表等：提取文本并加入 body
                token_text = self._extract_text(token)
                if not token_text and token_type not in ("block_code", "table", "list"):
                    continue

                if current is None:
                    current = {"level": 0, "heading": "", "body": []}
                    sections.append(current)

                # 代码块：保留围栏标记
                if token_type == "block_code":
                    lang = token.get("attrs", {}).get("info", "")
                    current["body"].append(f"```{lang}")
                    current["body"].append(token_text)
                    current["body"].append("```")
                else:
                    current["body"].append(token_text)

        return sections

    def _extract_text(self, token: dict) -> str:
        """从 mistune AST token 中递归提取纯文本。"""
        token_type = token.get("type", "")

        # 叶子节点：直接返回 text 或 raw
        if token_type == "text":
            return token.get("text", "") or token.get("raw", "")
        if token_type == "code":
            return token.get("text", "") or token.get("raw", "")
        if token_type == "softbreak":
            return "\n"
        if token_type == "linebreak":
            return "\n"

        # 有 raw 字段的 token（block_code, block_html 等）
        raw = token.get("raw", "")
        if raw:
            return raw

        # 递归处理 children / tokens
        children = token.get("children") or token.get("tokens") or []
        # 列表项之间加换行
        if token_type in ("list_item",):
            text = "".join(self._extract_text(child) for child in children)
            return text + "\n"

        if children:
            return "".join(self._extract_text(child) for child in children)

        return ""

    def _split_by_headings(self, sections: list[dict]) -> list[dict]:
        """在指定标题层级处切分，并为每个 chunk 构建上下文前缀。"""
        chunks: list[dict] = []
        current_lines: list[str] = []
        heading_stack: list[tuple[int, str]] = []

        for section in sections:
            level = section["level"]
            heading = section["heading"]
            body = section["body"]
            body_text = "\n".join(body).strip()

            if not body_text:
                if level > 0:
                    heading_stack = [(l, h) for l, h in heading_stack if l < level]
                    heading_stack.append((level, heading))
                continue

            should_split = level > 0 and level <= self.heading_split_level

            if should_split and current_lines:
                context = self._build_context(heading_stack)
                chunks.append({"context": context, "text": "\n".join(current_lines)})
                current_lines = []

            if level > 0:
                heading_stack = [(l, h) for l, h in heading_stack if l < level]
                heading_stack.append((level, heading))

            context = self._build_context(heading_stack)
            if context:
                current_lines.append(context)
            current_lines.append(body_text)

        if current_lines:
            context = self._build_context(heading_stack)
            chunks.append({"context": context, "text": "\n".join(current_lines)})

        return chunks

    @staticmethod
    def _build_context(heading_stack: list[tuple[int, str]]) -> str:
        """从标题栈构建上下文，如 '## 评估指标 > ### Faithfulness'"""
        if not heading_stack:
            return ""
        parts = []
        for level, heading in heading_stack:
            parts.append(f"{'#' * level} {heading}")
        return " > ".join(parts)

    def _merge_small_chunks(self, chunks: list[dict]) -> list[dict]:
        """合并过小的 chunk（< min_chunk_size）到相邻 chunk。"""
        if not chunks or len(chunks) == 1:
            return chunks

        merged: list[dict] = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            if len(current["text"]) < self.min_chunk_size and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                combined = len(current["text"]) + len(next_chunk["text"])
                if combined <= self.max_chunk_size:
                    merged.append({
                        "context": current["context"] or next_chunk["context"],
                        "text": current["text"] + "\n\n" + next_chunk["text"],
                    })
                    i += 2
                    continue
            merged.append(current)
            i += 1

        return merged

    def _finalize(self, chunks: list[dict], doc: Document) -> list[Chunk]:
        """构建最终的 Chunk 对象，超大 chunk 二次切分。"""
        result: list[Chunk] = []
        for raw in chunks:
            text = raw["text"]
            context = raw["context"]
            if len(text) <= self.max_chunk_size:
                result.append(self._make_chunk(doc, text, context, len(result)))
            else:
                for sub_text, sub_context in self._split_large_chunk(text, context):
                    result.append(self._make_chunk(doc, sub_text, sub_context, len(result)))
        return result

    def _split_large_chunk(
        self, text: str, context: str
    ) -> list[tuple[str, str]]:
        """按句子边界二次切分超大 chunk，每个子块保留上下文。"""
        sentences = re.split(r"(?<=[。！？.!?])\s+", text)
        if len(sentences) <= 1:
            return [(text, context)]

        result: list[tuple[str, str]] = []
        current_parts: list[str] = []
        current_len = 0

        for sentence in sentences:
            s_len = len(sentence)
            if current_len + s_len > self.max_chunk_size and current_parts:
                sub_text = "".join(current_parts)
                full = f"{context}\n\n{sub_text}" if context else sub_text
                result.append((full, context))
                current_parts = [sentence]
                current_len = s_len
            else:
                current_parts.append(sentence)
                current_len += s_len

        if current_parts:
            sub_text = "".join(current_parts)
            full = f"{context}\n\n{sub_text}" if context else sub_text
            result.append((full, context))

        return result

    def _make_chunk(
        self, doc: Document, text: str, context: str, index: int
    ) -> Chunk:
        full_text = f"{context}\n\n{text}" if context else text
        return Chunk(
            id=_make_chunk_id(doc.id, full_text, index),
            text=full_text,
            doc_id=doc.id,
            metadata={
                "chunk_index": index,
                "source": doc.source,
                "chunker": "markdown_structure",
                "heading_context": context,
            },
        )

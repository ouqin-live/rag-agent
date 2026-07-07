"""High-level KnowledgeBase that orchestrates loaders, chunkers, embedders and stores."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from rag_agent.embedder import BaseEmbedder, get_embedder
from rag_agent.knowledge.base import BaseChunker, BaseLoader, Chunk, RetrievalResult, VectorStore
from rag_agent.knowledge.chunker import FixedSizeChunker
from rag_agent.knowledge.chroma_store import ChromaVectorStore
from rag_agent.knowledge.loader import AutoLoader
from rag_agent.knowledge.reranker import BaseReranker, CrossEncoderReranker
from rag_agent.knowledge.store import LocalVectorStore

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Facade for document ingestion and semantic search."""

    def __init__(
        self,
        store: VectorStore,
        chunker: BaseChunker | None = None,
        embedder: BaseEmbedder | None = None,
        loader: BaseLoader | None = None,
        reranker: BaseReranker | None = None,
    ):
        self.store = store
        self.chunker = chunker or FixedSizeChunker()
        self.embedder = embedder or get_embedder()
        self.loader = loader or AutoLoader()
        # Reranker：Cross-Encoder 精排，默认不启用
        self.reranker = reranker

        # BM25 关键词检索索引（内部分词后重建）
        self._bm25_index: BM25Okapi | None = None
        self._bm25_chunks: list[Chunk] = []

    @classmethod
    def from_local_store(
        cls,
        store_path: str | Path,
        chunker: BaseChunker | None = None,
        embedder: BaseEmbedder | None = None,
        loader: BaseLoader | None = None,
    ) -> "KnowledgeBase":
        """使用 LocalVectorStore（SQLite + numpy）创建知识库。"""
        embedder = embedder or get_embedder()
        store = LocalVectorStore(store_path, dim=embedder.dim)
        return cls(store=store, chunker=chunker, embedder=embedder, loader=loader)

    @classmethod
    def from_chroma_store(
        cls,
        store_path: str | Path,
        chunker: BaseChunker | None = None,
        embedder: BaseEmbedder | None = None,
        loader: BaseLoader | None = None,
    ) -> "KnowledgeBase":
        """使用 ChromaVectorStore（HNSW 索引 + 自动持久化）创建知识库。推荐方式。"""
        embedder = embedder or get_embedder()
        store = ChromaVectorStore(store_path)
        return cls(store=store, chunker=chunker, embedder=embedder, loader=loader)

    # 文档入库入口：加载 → 分块 → 向量化 → 增量更新入库 → 持久化
    def add_document(
        self,
        source: str,
        loader: str | BaseLoader | None = "auto",
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """加载文档、切分、计算 embedding 并存储到向量库。

        如果该文档已存在，会先删除旧 chunks 再插入新 chunks（增量更新）。
        返回新增 chunk 的 id 列表。
        """
        # 合并用户传入的元数据，并确定使用哪个加载器
        user_metadata = metadata or {}
        chosen_loader = self._resolve_loader(loader)

        # 1. 加载文档：根据 source 类型（txt/md/pdf/url）读取原始内容
        docs = chosen_loader.load(source)
        if not docs:
            logger.warning("No documents loaded from source: %s", source)
            return []

        # 2. 分块：把每个文档切分成适合检索的 chunk，并继承文档元数据
        all_chunks: list[Chunk] = []
        for doc in docs:
            doc.metadata.update(user_metadata)
            chunks = self.chunker.chunk(doc)
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
                chunk.metadata.setdefault("source", doc.source)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks produced from source: %s", source)
            return []

        # 3. 向量化：批量计算所有 chunk 的 embedding，并归一化
        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        for chunk, vec in zip(all_chunks, embeddings):
            chunk.embedding = np.asarray(vec, dtype=np.float32)

        # 4. 增量更新：先删除该文档已有的旧 chunks，避免重复入库
        doc_ids = {chunk.doc_id for chunk in all_chunks}
        for doc_id in doc_ids:
            self.store.delete_by_doc(doc_id)

        # 5. 入库并持久化：把新 chunks 写入向量库（默认 Chroma，自动持久化）
        self.store.add(all_chunks)
        self.store.persist()

        logger.info(
            "Added %d chunks from %s (docs=%d)",
            len(all_chunks),
            source,
            len(docs),
        )
        # 6. 重建 BM25 关键词索引
        self._rebuild_bm25()
        return [chunk.id for chunk in all_chunks]

    def remove_document(self, doc_id: str) -> None:
        """Remove a document and all its chunks from the knowledge base."""
        self.store.delete_by_doc(doc_id)
        self.store.persist()
        logger.info("Removed document %s", doc_id)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """语义检索（纯 Dense 向量检索）。"""
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)[0]
        return self.store.search(query_embedding, top_k=top_k, filters=filters)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        rrf_k: int = 60,
    ) -> list[RetrievalResult]:
        """混合检索：Dense（向量）+ BM25（关键词）+ RRF 融合。

        RRF (Reciprocal Rank Fusion) 将两种排序融合为最终结果。
        """
        # Dense 检索
        dense_results = self.search(query, top_k=top_k * 2, filters=filters)

        # BM25 检索
        if self._bm25_index is None:
            self._rebuild_bm25()

        bm25_results: list[RetrievalResult] = []
        if self._bm25_index is not None and self._bm25_chunks:
            tokenized = _tokenize_bm25(query)
            bm25_scores = self._bm25_index.get_scores(tokenized)
            # 归一化 BM25 分数到 0~1
            max_score = float(bm25_scores.max()) if bm25_scores.max() > 0 else 1.0
            top_indices = np.argsort(bm25_scores)[-top_k * 2 :][::-1]
            for idx in top_indices:
                if bm25_scores[idx] > 0:
                    bm25_results.append(
                        RetrievalResult(
                            chunk=self._bm25_chunks[idx],
                            score=float(bm25_scores[idx]) / max_score,
                        )
                    )

        if not dense_results and not bm25_results:
            return []

        # RRF 融合（粗排）
        fused = _rrf_fusion(dense_results, bm25_results, top_k=top_k * 2, k=rrf_k)

        # Cross-Encoder 精排（可选）
        if self.reranker is not None:
            fused = self.reranker.rerank(query, fused, top_k=top_k)

        return fused[:top_k]

    def _rebuild_bm25(self) -> None:
        """从向量库重建 BM25 关键词索引。"""
        try:
            # 用空向量做一次全量查询来获取所有 chunks
            dim = self.embedder.dim
            all_results = self.store.search(
                np.zeros(dim, dtype=np.float32), top_k=len(self.store)
            )
            self._bm25_chunks = [r.chunk for r in all_results if r.chunk.text.strip()]
        except Exception:
            self._bm25_chunks = []

        if not self._bm25_chunks:
            self._bm25_index = None
            return

        tokenized_corpus = [_tokenize_bm25(c.text) for c in self._bm25_chunks]
        self._bm25_index = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index rebuilt: %d chunks", len(self._bm25_chunks))

    def _resolve_loader(self, loader: str | BaseLoader | None) -> BaseLoader:
        if loader is None or loader == "auto":
            return self.loader
        if isinstance(loader, BaseLoader):
            return loader
        raise ValueError(f"loader must be 'auto' or a BaseLoader instance, got {type(loader)}")

    def __len__(self) -> int:
        return len(self.store)


# ------------------------------------------------------------------
# 内部辅助：分词 & RRF 融合
# ------------------------------------------------------------------
def _tokenize_bm25(text: str) -> list[str]:
    """中英文混合分词：中文按 bigram 切，英文按空格切。"""
    # 中文部分：取中文字符，生成 bigram
    chinese_chars = re.findall(r"[\u4e00-\u9fff]+", text)
    tokens: list[str] = []
    for seg in chinese_chars:
        tokens.append(seg)  # 完整词
        for i in range(len(seg) - 1):
            tokens.append(seg[i : i + 2])  # bigram
    # 英文/数字部分
    english_part = re.sub(r"[\u4e00-\u9fff]+", " ", text)
    tokens.extend(re.findall(r"[a-zA-Z0-9]+", english_part.lower()))
    return [t for t in tokens if len(t) > 0]


def _rrf_fusion(
    dense: list[RetrievalResult],
    bm25: list[RetrievalResult],
    top_k: int = 5,
    k: int = 60,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion：融合 Dense 和 BM25 检索结果。"""
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievalResult] = {}

    for rank, result in enumerate(dense, 1):
        cid = result.chunk.id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_map[cid] = result

    for rank, result in enumerate(bm25, 1):
        cid = result.chunk.id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        RetrievalResult(chunk=chunk_map[cid], score=score)
        for cid, score in ranked[:top_k]
    ]

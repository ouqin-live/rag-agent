"""High-level KnowledgeBase that orchestrates loaders, chunkers, embedders and stores."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from rag_agent.embedder import BaseEmbedder, get_embedder
from rag_agent.knowledge.base import BaseChunker, BaseLoader, Chunk, RetrievalResult, VectorStore
from rag_agent.knowledge.chunker import FixedSizeChunker
from rag_agent.knowledge.chroma_store import ChromaVectorStore
from rag_agent.knowledge.loader import AutoLoader
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
    ):
        self.store = store
        self.chunker = chunker or FixedSizeChunker()
        self.embedder = embedder or get_embedder()
        self.loader = loader or AutoLoader()

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
        # 6. 返回新增 chunk 的 id 列表，方便调用方追踪
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
        """Embed the query and search the vector store."""
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)[0]
        return self.store.search(query_embedding, top_k=top_k, filters=filters)

    def _resolve_loader(self, loader: str | BaseLoader | None) -> BaseLoader:
        if loader is None or loader == "auto":
            return self.loader
        if isinstance(loader, BaseLoader):
            return loader
        raise ValueError(f"loader must be 'auto' or a BaseLoader instance, got {type(loader)}")

    def __len__(self) -> int:
        return len(self.store)

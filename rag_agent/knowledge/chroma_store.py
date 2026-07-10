"""Chroma-based vector store with HNSW indexing and auto-persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

import chromadb
from chromadb.api.types import EmbeddingFunction

from rag_agent.knowledge.base import Chunk, RetrievalResult, VectorStore

logger = logging.getLogger(__name__)


class _PassThroughEmbedding(EmbeddingFunction):
    """A no-op embedding function: embeddings are pre-computed and passed directly."""

    def __call__(self, input: list[dict] | list[str]) -> list[list[float]]:  # type: ignore[override]
        raise NotImplementedError("ChromaVectorStore expects pre-computed embeddings via add()")


class ChromaVectorStore(VectorStore):
    """A vector store backed by Chroma with HNSW indexing.

    Faster and more scalable than ``LocalVectorStore``, while still running
    locally with zero external dependencies beyond ``chromadb``.
    """

    def __init__(self, store_path: str | Path, collection_name: str = "rag_agent_kb"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(path=str(self.store_path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=_PassThroughEmbedding(),
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaVectorStore ready: %d chunks in collection '%s'",
            self._collection.count(),
            collection_name,
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add(self, chunks: list[Chunk]) -> None:
        """Add or overwrite chunks in Chroma."""
        if not chunks:
            return

        # Chroma's upsert replaces existing ids, so we use chunk.id directly
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [_sanitize_metadata(chunk.metadata) for chunk in chunks]
        embeddings = _to_embeddings([chunk.embedding for chunk in chunks])

        # Upsert：如果 id 已存在则更新，否则新增
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def delete_by_doc(self, doc_id: str) -> None:
        """Remove all chunks whose metadata contains ``doc_id``."""
        self._collection.delete(where={"doc_id": doc_id})

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        raw_count = self._collection.count()
        if raw_count == 0:
            return []

        query_vec = _to_embedding(query_embedding)
        where_filter = _build_where(filters) if filters else None

        try:
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=min(top_k, raw_count),
                where=where_filter,
                include=["metadatas", "documents", "embeddings", "distances"],
            )
        except Exception as exc:
            logger.warning("Chroma query failed (collection may be inconsistent): %s", exc)
            return []

        retrieved: list[RetrievalResult] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, chunk_id in enumerate(ids):
            chunk = Chunk(
                id=chunk_id,
                text=documents[i] if i < len(documents) else "",
                doc_id=metadatas[i].get("doc_id", "") if i < len(metadatas) else "",
                metadata=_deserialize_metadata(metadatas[i]) if i < len(metadatas) else {},
                embedding=None,
            )
            distance = distances[i] if i < len(distances) else 1.0
            # Chroma returns cosine distance; convert to similarity score
            score = 1.0 - min(max(float(distance), 0.0), 2.0)
            retrieved.append(RetrievalResult(chunk=chunk, score=score))

        return retrieved

    # ------------------------------------------------------------------
    # Persistence (Chroma auto-persists – this is a no-op for interface compatibility)
    # ------------------------------------------------------------------
    def persist(self) -> None:
        """Chroma auto-persists on every write; explicit call kept for compatibility."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._collection.count()


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------
def _to_embedding(vec: np.ndarray) -> list[float]:
    v = np.asarray(vec, dtype=np.float32).ravel()
    return v.tolist()


def _to_embeddings(vecs: list[np.ndarray | None]) -> list[list[float]]:
    result: list[list[float]] = []
    for v in vecs:
        if v is None:
            result.append([])
        else:
            result.append(_to_embedding(v))
    return result


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts str | int | float | bool metadata values.

    Complex types (list, dict) are JSON-serialised to strings.
    """
    cleaned: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif v is not None:
            cleaned[k] = json.dumps(v, ensure_ascii=False)
    return cleaned


def _deserialize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Reverse the sanitisation: attempt to json-load string values."""
    result: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                result[k] = v
        else:
            result[k] = v
    return result


def _build_where(filters: dict[str, Any]) -> dict[str, Any]:
    """Convert flat filters dict into Chroma's where syntax.

    Currently supports exact-match on top-level keys only.
    """
    if len(filters) == 1:
        k, v = next(iter(filters.items()))
        # Serialize complex values the same way _sanitize_metadata does
        if not isinstance(v, (str, int, float, bool)):
            v = json.dumps(v, ensure_ascii=False)
        return {k: {"$eq": v}}

    # Multiple filters: combine with $and
    conditions: list[dict[str, Any]] = []
    for k, v in filters.items():
        if not isinstance(v, (str, int, float, bool)):
            v = json.dumps(v, ensure_ascii=False)
        conditions.append({k: {"$eq": v}})
    return {"$and": conditions}

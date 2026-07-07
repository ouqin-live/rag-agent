"""Local persistent vector store backed by SQLite + numpy."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from rag_agent.knowledge.base import Chunk, RetrievalResult, VectorStore

logger = logging.getLogger(__name__)


class LocalVectorStore(VectorStore):
    """A disk-persistent vector store using SQLite for metadata and numpy for vectors.

    The store is self-contained and works fully offline.  It expects chunks to arrive
    with pre-computed embeddings (filled by ``KnowledgeBase``).
    """

    def __init__(self, store_path: str | Path, dim: int | None = None):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.store_path / "kb.db"
        self.vectors_path = self.store_path / "vectors.npy"

        self._dim = dim
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._doc_id_to_indices: dict[str, set[int]] = {}

        self._init_db()
        self._load()

    # ------------------------------------------------------------------
    # Schema & persistence
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    metadata TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT,
                    vector_idx INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
                    ON chunks(doc_id);
                """
            )

    def _load(self) -> None:
        """Load chunks from SQLite and vectors from numpy file."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, doc_id, text, metadata, vector_idx FROM chunks"
            ).fetchall()

        self._chunks = []
        self._doc_id_to_indices = {}
        for row in rows:
            chunk = Chunk(
                id=row["id"],
                text=row["text"],
                doc_id=row["doc_id"],
                metadata=json.loads(row["metadata"] or "{}"),
                embedding=None,
            )
            idx = len(self._chunks)
            self._chunks.append(chunk)
            self._doc_id_to_indices.setdefault(row["doc_id"], set()).add(idx)

        if self.vectors_path.exists() and self._chunks:
            try:
                self._vectors = np.load(self.vectors_path, allow_pickle=False)
                if self._vectors.shape[0] != len(self._chunks):
                    logger.warning(
                        "Vector count (%d) does not match chunk count (%d). Rebuilding empty vectors.",
                        self._vectors.shape[0],
                        len(self._chunks),
                    )
                    self._vectors = None
                else:
                    for i, chunk in enumerate(self._chunks):
                        chunk.embedding = self._vectors[i]
                    self._dim = int(self._vectors.shape[1])
            except Exception as exc:
                logger.warning("Failed to load vectors: %s", exc)
                self._vectors = None

        if self._vectors is None and self._chunks and self._dim is not None:
            self._vectors = np.zeros((len(self._chunks), self._dim), dtype=np.float32)

        logger.info(
            "LocalVectorStore loaded: %d chunks, dim=%s",
            len(self._chunks),
            self._dim,
        )

    def persist(self) -> None:
        """Persist metadata to SQLite and vectors to numpy file."""
        if self._vectors is not None and len(self._chunks) > 0:
            np.save(self.vectors_path, self._vectors)
        elif self.vectors_path.exists():
            self.vectors_path.unlink()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks. Existing chunks with the same id are overwritten."""
        if not chunks:
            return

        dim = self._resolve_dim(chunks)
        if dim is None:
            raise ValueError("Chunks must have embeddings or store must have a known dim")

        # Grow vector matrix if needed
        start_idx = len(self._chunks)
        new_count = len(chunks)

        if self._vectors is None or self._vectors.shape[0] == 0:
            self._vectors = np.zeros((0, dim), dtype=np.float32)
        else:
            if self._vectors.shape[1] != dim:
                raise ValueError(
                    f"Embedding dimension mismatch: store={self._vectors.shape[1]}, chunks={dim}"
                )

        # Append placeholder rows and fill in below
        self._vectors = np.vstack(
            [self._vectors, np.zeros((new_count, dim), dtype=np.float32)]
        )

        with sqlite3.connect(self.db_path) as conn:
            for offset, chunk in enumerate(chunks):
                idx = start_idx + offset
                self._chunks.append(chunk)
                self._doc_id_to_indices.setdefault(chunk.doc_id, set()).add(idx)

                assert chunk.embedding is not None
                self._vectors[idx] = chunk.embedding

                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks (id, doc_id, text, metadata, vector_idx)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.id,
                        chunk.doc_id,
                        chunk.text,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                        idx,
                    ),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents (id, source, metadata)
                    VALUES (?, ?, ?)
                    """,
                    (
                        chunk.doc_id,
                        chunk.metadata.get("source", chunk.doc_id),
                        json.dumps({}, ensure_ascii=False),
                    ),
                )

        self._dim = dim

    def delete_by_doc(self, doc_id: str) -> None:
        """Remove all chunks belonging to ``doc_id``.

        Implementation rebuilds the vector matrix and rewrites chunk rows to keep
        indices contiguous. This is simple and correct for small-to-medium stores.
        """
        indices = sorted(self._doc_id_to_indices.get(doc_id, set()))
        if not indices:
            return

        keep_mask = np.ones(len(self._chunks), dtype=bool)
        keep_mask[indices] = False

        self._chunks = [c for i, c in enumerate(self._chunks) if keep_mask[i]]
        if self._vectors is not None:
            self._vectors = self._vectors[keep_mask]

        self._doc_id_to_indices.pop(doc_id, None)
        self._rebuild_indices()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def _rebuild_indices(self) -> None:
        """Recalculate doc_id -> index mapping after deletions."""
        self._doc_id_to_indices = {}
        for idx, chunk in enumerate(self._chunks):
            self._doc_id_to_indices.setdefault(chunk.doc_id, set()).add(idx)

        # Update vector_idx in SQLite
        with sqlite3.connect(self.db_path) as conn:
            for idx, chunk in enumerate(self._chunks):
                conn.execute(
                    "UPDATE chunks SET vector_idx = ? WHERE id = ?",
                    (idx, chunk.id),
                )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if self._vectors is None or len(self._chunks) == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"Query dim {query.shape[0]} != store dim {self._vectors.shape[1]}"
            )

        # Apply metadata filters
        candidates = list(range(len(self._chunks)))
        if filters:
            candidates = [
                i
                for i in candidates
                if self._matches_filters(self._chunks[i], filters)
            ]

        if not candidates:
            return []

        candidate_vectors = self._vectors[candidates]
        scores = candidate_vectors @ query
        top_local = np.argsort(scores)[-top_k:][::-1]

        results: list[RetrievalResult] = []
        for local_idx in top_local:
            global_idx = candidates[local_idx]
            results.append(
                RetrievalResult(chunk=self._chunks[global_idx], score=float(scores[local_idx]))
            )
        return results

    @staticmethod
    def _matches_filters(chunk: Chunk, filters: dict[str, Any]) -> bool:
        for key, value in filters.items():
            if chunk.metadata.get(key) != value:
                return False
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_dim(self, chunks: list[Chunk]) -> int | None:
        if self._dim is not None:
            return self._dim
        for chunk in chunks:
            if chunk.embedding is not None:
                return int(chunk.embedding.shape[0])
        return None

    def __len__(self) -> int:
        return len(self._chunks)

"""Centralised application settings via Pydantic Settings.

All tunables can be controlled through environment variables or a ``.env``
file at the project root. Explicit constructor arguments still take precedence
where supported.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------
    project_name: str = "rag-agent"
    debug: bool = Field(default=False, description="Enable debug logging")
    log_level: str = Field(default="INFO", description="Logging level")

    # ------------------------------------------------------------------
    # Storage paths
    # ------------------------------------------------------------------
    kb_store_path: Path = Field(
        default=Path("data/kb"),
        description="Directory for the knowledge-base vector store",
    )
    memory_store_path: Path = Field(
        default=Path("data/memory"),
        description="Directory for the long-term memory vector store",
    )
    eval_db_path: Path = Field(
        default=Path("data/eval/evaluations.db"),
        description="SQLite path for evaluation results",
    )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    embedding_model: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        description="Sentence-transformers model name",
    )
    embedding_fallback_dim: int = Field(
        default=384,
        description="Dimension of the offline fallback embedder",
    )

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    llm_base_url: str | None = Field(
        default=None,
        description="OpenAI-compatible base URL",
    )
    llm_api_key: str | None = Field(
        default=None,
        description="API key for the LLM service",
    )
    llm_model: str | None = Field(
        default=None,
        description="Default model name",
    )
    llm_timeout: float = Field(default=60.0, description="LLM request timeout")
    llm_temperature: float = Field(
        default=0.3,
        description="Default generation temperature",
    )

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------
    agent_top_k: int = Field(default=5, description="Default retrieval top-k")
    agent_max_turns: int = Field(
        default=6,
        description="Default short-term memory turn limit",
    )
    query_transform_enabled: bool = Field(
        default=True,
        description="Enable query rewriting before retrieval",
    )
    query_transform_max_history_turns: int = Field(
        default=3,
        description="Max conversation turns used for query rewriting context",
    )
    agent_system_prompt: str = Field(
        default=(
            "你是一个严谨的 RAG 助手。请仅根据提供的参考资料和已知用户信息回答问题，"
            "不要编造参考资料之外的信息。如果参考资料不足，请明确说明。"
        ),
        description="Default system prompt",
    )

    # ------------------------------------------------------------------
    # Long-term memory
    # ------------------------------------------------------------------
    memory_dedup_threshold: float = Field(
        default=0.92,
        description="Cosine similarity above which a fact is considered duplicate",
    )
    memory_merge_threshold: float = Field(
        default=0.80,
        description="Cosine similarity above which two facts are merged",
    )
    memory_max_facts_per_user: int = Field(
        default=100,
        description="Maximum number of long-term facts stored per user",
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    eval_failure_threshold: float = Field(
        default=0.6,
        description="Overall score below which a record is marked as failure",
    )
    eval_answer_min_length: int = Field(
        default=5,
        description="Minimum answer length in characters",
    )
    eval_answer_max_length: int = Field(
        default=2000,
        description="Maximum answer length in characters",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()

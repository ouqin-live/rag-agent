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
    llm_max_retries: int = Field(
        default=3,
        description="Max retry attempts for transient LLM failures",
    )
    llm_retry_backoff: float = Field(
        default=2.0,
        description="Exponential backoff factor for LLM retries",
    )
    llm_fallback_models: str = Field(
        default="",
        description="Comma-separated fallback model names for LLM degradation",
    )
    llm_health_failure_threshold: int = Field(
        default=3,
        description="Consecutive failures before a model is marked unhealthy",
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
    semantic_cache_enabled: bool = Field(
        default=True,
        description="Enable semantic cache for query/answer reuse",
    )
    semantic_cache_threshold: float = Field(
        default=0.92,
        description="Cosine similarity threshold for semantic cache hits",
    )
    semantic_cache_ttl_seconds: float | None = Field(
        default=None,
        description="TTL for semantic cache entries; None means no expiration",
    )

    # ------------------------------------------------------------------
    # Agentic RAG (P2-1)
    # ------------------------------------------------------------------
    agentic_enabled: bool = Field(
        default=False,
        description="Enable Agentic RAG with self-correction loop",
    )
    agentic_max_iterations: int = Field(
        default=2,
        description="Max ReAct / self-correction iterations per turn",
    )
    agentic_faithfulness_threshold: float = Field(
        default=0.5,
        description="Faithfulness threshold below which to trigger correction",
    )
    agentic_use_llm_router: bool = Field(
        default=False,
        description="Use LLM-based query routing instead of rule-based routing",
    )

    # ------------------------------------------------------------------
    # Guardrails (P2-3)
    # ------------------------------------------------------------------
    guardrails_enabled: bool = Field(
        default=True,
        description="Enable safety guardrails for input and output",
    )
    guardrails_prompt_injection_enabled: bool = Field(
        default=True,
        description="Detect prompt injection patterns in user input",
    )
    guardrails_prompt_injection_hard_block: bool = Field(
        default=False,
        description="Block requests with high-risk injection patterns",
    )
    guardrails_pii_enabled: bool = Field(
        default=True,
        description="Detect PII (phone, email, ID card, etc.) in user input",
    )
    guardrails_pii_hard_block: bool = Field(
        default=False,
        description="Block requests containing high-risk PII",
    )
    guardrails_output_toxicity_enabled: bool = Field(
        default=True,
        description="Detect toxic/harmful content in LLM output",
    )
    guardrails_output_toxicity_hard_block: bool = Field(
        default=False,
        description="Block toxic/harmful output responses",
    )
    guardrails_confidence_enabled: bool = Field(
        default=True,
        description="Warn when retrieval confidence is too low",
    )
    guardrails_confidence_threshold: float = Field(
        default=0.3,
        description="Retrieval confidence threshold below which to warn",
    )
    guardrails_raise_on_block: bool = Field(
        default=False,
        description="Raise exception when a guardrail hard-blocks (else return safe fallback)",
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

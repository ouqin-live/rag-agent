"""LangGraph state definition for the agentic RAG workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    """Shared state that flows through the LangGraph computation.

    Every node reads from this state and returns a partial dict of fields
    to update. LangGraph merges the returned dict into the state before
    passing it to the next node.
    """

    # ---- Input ----
    question: str
    user_id: str
    chat_history: list[dict[str, str]]

    # ---- Routing ----
    use_knowledge_base: bool
    use_long_term_memory: bool
    use_calculator: bool
    use_datetime: bool
    use_web_search: bool
    route_reasoning: str

    # ---- Query transformation ----
    search_query: str

    # ---- Retrieval results ----
    contexts: list[str]
    long_term_facts: list[str]
    tool_results: list[dict[str, Any]]

    # ---- Generation ----
    answer: str

    # ---- Self-correction ----
    iteration: int
    correction_score: float
    needs_correction: bool
    correction_message: str

    # ---- Guardrails ----
    guardrail_blocked: bool
    guardrail_message: str

    # ---- Cache ----
    cache_hit: bool
    cached_answer: str

    # ---- Evaluation ----
    evaluation: dict[str, Any] | None

    # ---- Meta ----
    reasoning: str

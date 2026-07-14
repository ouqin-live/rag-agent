"""Agentic RAG capabilities: routing, ReAct loop, and self-correction."""

from __future__ import annotations

from rag_agent.agentic.base import AgenticContext, BaseTool
from rag_agent.agentic.react import ReactLoop
from rag_agent.agentic.router import QueryRouter, RouteDecision, RuleBasedRouter
from rag_agent.agentic.self_correction import SelfCorrector
from rag_agent.agentic.tools import CalculatorTool, DatetimeTool

__all__ = [
    "AgenticContext",
    "BaseTool",
    "CalculatorTool",
    "DatetimeTool",
    "QueryRouter",
    "ReactLoop",
    "RouteDecision",
    "RuleBasedRouter",
    "SelfCorrector",
]

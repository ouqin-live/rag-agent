"""LangGraph-based agent that wraps the compiled graph."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph

from rag_agent.graph.state import GraphState

logger = logging.getLogger(__name__)


class LangGraphAgent:
    """Agent that delegates to a compiled LangGraph for the agentic workflow.

    Wraps ``graph.invoke`` / ``graph.ainvoke`` so it can be used as a drop-in
    replacement for the existing ``ReactLoop``-based agent.
    """

    def __init__(self, graph: StateGraph):
        self.graph = graph

    def chat(self, user_id: str, question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Run one turn synchronously through the LangGraph.

        Returns a dict with keys matching ``GraphState`` fields.
        """
        initial: GraphState = {
            "question": question,
            "user_id": user_id,
            "chat_history": history or [],
        }
        try:
            result = self.graph.invoke(initial)
            return result
        except Exception as exc:
            logger.error("LangGraph chat failed: %s", exc)
            return {
                "question": question,
                "answer": "当前无法生成回答，请稍后再试。",
                "contexts": [],
            }

    async def achat(self, user_id: str, question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Run one turn asynchronously through the LangGraph."""
        initial: GraphState = {
            "question": question,
            "user_id": user_id,
            "chat_history": history or [],
        }
        try:
            result = await self.graph.ainvoke(initial)
            return result
        except Exception as exc:
            logger.error("LangGraph async chat failed: %s", exc)
            return {
                "question": question,
                "answer": "当前无法生成回答，请稍后再试。",
                "contexts": [],
            }

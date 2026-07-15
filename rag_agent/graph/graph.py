"""Build the agentic LangGraph workflow."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from rag_agent.graph.nodes import decide_after_correction, decide_after_route, make_nodes
from rag_agent.graph.state import GraphState

logger = logging.getLogger(__name__)


def build_agentic_graph(
    knowledge_base: Any,
    llm_client: Any,
    long_term_memory: Any | None = None,
    query_transformer: Any | None = None,
    self_corrector: Any | None = None,
    tools: dict[str, Any] | None = None,
    router: Any | None = None,
    system_prompt: str = "",
    top_k: int = 5,
    temperature: float = 0.3,
) -> StateGraph:
    """Build a compiled LangGraph for the agentic RAG workflow.

    The graph implements: route → transform_query → retrieve → generate →
    self_correction, with a conditional loop back to retrieve when correction
    is needed.
    """
    nodes = make_nodes(
        knowledge_base=knowledge_base,
        llm_client=llm_client,
        long_term_memory=long_term_memory,
        query_transformer=query_transformer,
        self_corrector=self_corrector,
        tools=tools,
        router=router,
        system_prompt=system_prompt,
        top_k=top_k,
        temperature=temperature,
    )

    builder = StateGraph(GraphState)

    builder.add_node("route", nodes["route"])
    builder.add_node("transform_query", nodes["transform_query"])
    builder.add_node("retrieve", nodes["retrieve"])
    builder.add_node("generate", nodes["generate"])
    builder.add_node("self_correction", nodes["self_correction"])

    # Entry
    builder.add_edge(START, "route")

    # Route → transform
    builder.add_edge("route", "transform_query")

    # Transform → retrieve
    builder.add_edge("transform_query", "retrieve")

    # Retrieve → generate
    builder.add_edge("retrieve", "generate")

    # Generate → self_correction
    builder.add_edge("generate", "self_correction")

    # Self-correction → retrieve (loop) or END
    builder.add_conditional_edges(
        "self_correction",
        decide_after_correction,
        {
            "retrieve": "retrieve",
            END: END,
        },
    )

    return builder.compile()

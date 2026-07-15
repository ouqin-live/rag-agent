"""Build the agentic LangGraph workflow."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from rag_agent.graph.nodes import (
    decide_after_cache,
    decide_after_correction,
    decide_after_input_guardrail,
    make_nodes,
)
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
    semantic_cache: Any | None = None,
    guardrails: Any | None = None,
    evaluator: Any | None = None,
    memory_extractor: Any | None = None,
) -> StateGraph:
    """Build a compiled LangGraph for the agentic RAG workflow.

    The graph implements:
    input_guardrail → cache_lookup → route → transform_query → retrieve →
    generate → output_guardrail → self_correction → remember → evaluate,
    with conditional short-circuits for guardrail blocks and cache hits,
    and a loop back from self_correction to retrieve when needed.
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
        semantic_cache=semantic_cache,
        guardrails=guardrails,
        evaluator=evaluator,
        memory_extractor=memory_extractor,
    )

    builder = StateGraph(GraphState)

    # Add all nodes
    builder.add_node("input_guardrail", nodes["input_guardrail"])
    builder.add_node("cache_lookup", nodes["cache_lookup"])
    builder.add_node("route", nodes["route"])
    builder.add_node("transform_query", nodes["transform_query"])
    builder.add_node("retrieve", nodes["retrieve"])
    builder.add_node("generate", nodes["generate"])
    builder.add_node("output_guardrail", nodes["output_guardrail"])
    builder.add_node("self_correction", nodes["self_correction"])
    builder.add_node("remember", nodes["remember"])
    builder.add_node("evaluate", nodes["evaluate"])

    # Entry → input_guardrail
    builder.add_edge(START, "input_guardrail")

    # input_guardrail → cache_lookup (or END if blocked)
    builder.add_conditional_edges(
        "input_guardrail",
        decide_after_input_guardrail,
        {
            "cache_lookup": "cache_lookup",
            END: END,
        },
    )

    # cache_lookup → route (or END if hit)
    builder.add_conditional_edges(
        "cache_lookup",
        decide_after_cache,
        {
            "route": "route",
            END: END,
        },
    )

    # route → transform_query
    builder.add_edge("route", "transform_query")

    # transform_query → retrieve
    builder.add_edge("transform_query", "retrieve")

    # retrieve → generate
    builder.add_edge("retrieve", "generate")

    # generate → output_guardrail
    builder.add_edge("generate", "output_guardrail")

    # output_guardrail → self_correction
    builder.add_edge("output_guardrail", "self_correction")

    # self_correction → retrieve (loop) or remember
    builder.add_conditional_edges(
        "self_correction",
        decide_after_correction,
        {
            "retrieve": "retrieve",
            "remember": "remember",
        },
    )

    # remember → evaluate
    builder.add_edge("remember", "evaluate")

    # evaluate → END
    builder.add_edge("evaluate", END)

    return builder.compile()


if __name__ == "__main__":
    """Quick view: `uv run python -m rag_agent.graph.graph`"""
    import tempfile

    from rag_agent.embedder import FallbackEmbedding
    from rag_agent.knowledge import KnowledgeBase
    from rag_agent.llm import MockLLMClient

    kb = KnowledgeBase.from_local_store(tempfile.mkdtemp(), embedder=FallbackEmbedding())
    llm = MockLLMClient()
    graph = build_agentic_graph(knowledge_base=kb, llm_client=llm)
    # 同时输出 ASCII 和 Mermaid 两种视图
    print("=== ASCII ===")
    print(graph.get_graph().draw_ascii())
    print("\n=== Mermaid ===")
    print(graph.get_graph().draw_mermaid())

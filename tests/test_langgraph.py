"""Tests for the LangGraph agentic workflow."""

from __future__ import annotations

from pathlib import Path

from rag_agent.agent import Agent, AgentConfig
from rag_agent.embedder import BaseEmbedder
from rag_agent.graph import LangGraphAgent, build_agentic_graph
from rag_agent.knowledge import KnowledgeBase
from rag_agent.llm import MockLLMClient
from rag_agent.memory import ShortTermMemory


def _build_kb(temp_dir: Path, embedder: BaseEmbedder) -> KnowledgeBase:
    kb = KnowledgeBase.from_local_store(str(temp_dir / "kb"), embedder=embedder)
    doc_path = temp_dir / "rag.txt"
    doc_path.write_text(
        "RAG is retrieval augmented generation. It retrieves documents before generating answers.",
        encoding="utf-8",
    )
    kb.add_document(source=str(doc_path))
    return kb


def test_langgraph_agent_chat(
    temp_dir: Path,
    mock_embedder: BaseEmbedder,
) -> None:
    kb = _build_kb(temp_dir, mock_embedder)
    llm = MockLLMClient({"RAG": "RAG is retrieval augmented generation."})

    from rag_agent.agentic import CalculatorTool

    graph = build_agentic_graph(
        knowledge_base=kb,
        llm_client=llm,
        tools={"calculator": CalculatorTool()},
        system_prompt="你是一个 RAG 助手。",
    )
    agent = LangGraphAgent(graph)

    result = agent.chat("user-1", "What is RAG?")
    assert result["answer"]
    assert "RAG" in result["answer"] or "retrieval" in result["answer"]


def test_langgraph_agent_chat_async(
    temp_dir: Path,
    mock_embedder: BaseEmbedder,
) -> None:
    kb = _build_kb(temp_dir, mock_embedder)
    llm = MockLLMClient()

    graph = build_agentic_graph(
        knowledge_base=kb,
        llm_client=llm,
        system_prompt="你是一个 RAG 助手。",
    )
    agent = LangGraphAgent(graph)

    import asyncio

    result = asyncio.run(agent.achat("user-1", "What is RAG?"))
    assert result["answer"]


def test_agent_langgraph_via_agent(
    temp_dir: Path,
    mock_embedder: BaseEmbedder,
) -> None:
    """Test that the main Agent class works with LangGraph enabled."""
    kb = _build_kb(temp_dir, mock_embedder)
    stm = ShortTermMemory()
    llm = MockLLMClient({"RAG": "RAG stands for retrieval augmented generation."})

    from rag_agent.agentic import CalculatorTool

    agent = Agent(
        AgentConfig(
            knowledge_base=kb,
            short_term_memory=stm,
            llm_client=llm,
            tools={"calculator": CalculatorTool()},
        )
    )

    response = agent.chat("user-1", "What is RAG?")
    assert response.answer
    assert "RAG" in response.answer


def test_langgraph_with_calculator_tool(
    temp_dir: Path,
    mock_embedder: BaseEmbedder,
) -> None:
    kb = _build_kb(temp_dir, mock_embedder)
    llm = MockLLMClient({"1 + 2": "3"})

    from rag_agent.agentic import CalculatorTool

    graph = build_agentic_graph(
        knowledge_base=kb,
        llm_client=llm,
        tools={"calculator": CalculatorTool()},
        system_prompt="你是一个 RAG 助手。",
    )
    agent = LangGraphAgent(graph)

    result = agent.chat("user-1", "1 + 2 等于多少")
    assert result["answer"]

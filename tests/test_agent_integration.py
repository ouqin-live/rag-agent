"""Integration tests for the Agent end-to-end chat flow.

These tests use MockLLMClient and a LocalVectorStore so they run fully offline.
"""

from __future__ import annotations

from pathlib import Path

from rag_agent.agent import Agent, AgentConfig
from rag_agent.embedder import BaseEmbedder
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


def test_agent_chat_with_mock_llm(
    temp_dir: Path,
    mock_embedder: BaseEmbedder,
    mock_llm: MockLLMClient,
) -> None:
    kb = _build_kb(temp_dir, mock_embedder)
    stm = ShortTermMemory()

    responses = {
        "RAG": "RAG is retrieval augmented generation.",
    }
    llm = MockLLMClient(responses=responses)

    agent = Agent(
        AgentConfig(
            knowledge_base=kb,
            short_term_memory=stm,
            llm_client=llm,
        )
    )

    response = agent.chat("user-1", "What is RAG?")
    assert response.answer
    assert isinstance(response.contexts, list)


def test_agent_chat_async(
    temp_dir: Path,
    mock_embedder: BaseEmbedder,
) -> None:
    kb = _build_kb(temp_dir, mock_embedder)
    stm = ShortTermMemory()
    llm = MockLLMClient()

    agent = Agent(
        AgentConfig(
            knowledge_base=kb,
            short_term_memory=stm,
            llm_client=llm,
        )
    )

    import asyncio

    response = asyncio.run(agent.achat("user-1", "What is RAG?"))
    assert response.answer


def test_agentic_mode_uses_tools(
    temp_dir: Path,
    mock_embedder: BaseEmbedder,
) -> None:
    from rag_agent.agentic import CalculatorTool

    kb = _build_kb(temp_dir, mock_embedder)
    stm = ShortTermMemory()
    llm = MockLLMClient(responses={"1 + 2": "3"})

    agent = Agent(
        AgentConfig(
            knowledge_base=kb,
            short_term_memory=stm,
            llm_client=llm,
            agentic_enabled=True,
            tools={"calculator": CalculatorTool()},
        )
    )

    response = agent.chat("user-1", "1 + 2 等于多少")
    assert response.answer
    # The answer should include the calculator result or reference.
    assert "3" in response.answer or "计算" in response.answer

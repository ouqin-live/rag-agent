"""验证 Phase 4：Agent 端到端编排（记忆 + 知识库 + 生成 + 评估 + 报告）。"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from rag_agent.agent import Agent, AgentConfig, ChatResponse
from rag_agent.embedder import get_embedder
from rag_agent.evaluation import Evaluator
from rag_agent.knowledge import FixedSizeChunker, KnowledgeBase, SemanticChunker
from rag_agent.knowledge.reranker import EmbeddingReranker
from rag_agent.llm import MockLLMClient, OpenAICompatibleClient
from rag_agent.memory import LongTermMemory, MediumTermMemory, ShortTermMemory


def setup_test_docs(tmpdir: Path) -> Path:
    md_path = tmpdir / "rag_guide.md"
    md_path.write_text(
        "# RAG 指南\n\n"
        "RAG（检索增强生成）将外部知识库与大型语言模型结合。\n"
        "它通过检索相关文档片段，并将其作为上下文输入语言模型，从而减少幻觉。\n"
        "RAGAS 框架中的 Faithfulness 指标用于衡量生成答案是否忠于检索上下文。\n"
        "Answer Relevance 指标衡量答案与用户问题的相关程度。\n",
        encoding="utf-8",
    )
    return md_path


_METRIC_NAMES = {
    "faithfulness": "忠实度 (Faithfulness)",
    "answer_relevance": "回答相关性 (Answer Relevance)",
    "context_precision": "上下文精确率 (Context Precision)",
}


def print_response(label: str, resp: ChatResponse) -> None:
    print(f"\n{label}")
    if resp.search_query:
        print(f"  检索query: {resp.search_query}")
    print(f"  回答: {resp.answer[:100]}...")
    if resp.long_term_facts:
        print("  召回长期记忆:")
        for fact in resp.long_term_facts:
            print(f"    - {fact}")
    if resp.evaluation:
        print(f"  综合评分: {resp.evaluation.overall_score:.4f}")
        for name, score in resp.evaluation.scores.items():
            display_name = _METRIC_NAMES.get(name, name)
            print(f"    - {display_name}: {score:.4f}")


def main() -> None:
    tmpdir = Path(tempfile.gettempdir()) / "rag_agent_phase4_test"
    tmpdir.mkdir(parents=True, exist_ok=True)
    md_path = setup_test_docs(tmpdir)

    data_dir = Path("data/phase4")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    embedder = get_embedder()

    # 知识库（Chroma：HNSW 索引 + 自动持久化 + 语义分块）
    kb = KnowledgeBase.from_chroma_store(
        store_path=data_dir / "kb",
        chunker=SemanticChunker(embedder=embedder, similarity_threshold=0.3),
        embedder=embedder,
    )
    kb.add_document(str(md_path), metadata={"tag": "rag"})
    print(f"知识库已加载: {len(kb)} chunks")

    # 启用 Embedding 重排序（复用已有 embedder，零额外下载）
    kb.reranker = EmbeddingReranker(embedder)

    # 混合检索快速验证
    hybrid_results = kb.hybrid_search("RAG 减少幻觉", top_k=2)
    print(f"混合检索（BM25 + Dense）Top-2:")
    for i, r in enumerate(hybrid_results, 1):
        print(f"  {i}. [{r.score:.4f}] {r.text[:60]}...")

    # 长期记忆（Chroma：HNSW 索引 + 自动持久化）
    ltm = LongTermMemory.from_chroma_store(
        store_path=data_dir / "memory",
        embedder=embedder,
        max_facts_per_user=20,
    )

    # 尝试接入真实 LLM，失败则降级为 Mock
    mock_responses = {
        "RAG 是什么": "RAG 是检索增强生成，它通过检索外部文档片段并将其作为上下文输入语言模型来生成回答。",
        "减少幻觉": "RAG 通过引入外部知识作为上下文，让模型基于参考资料生成回答，从而减少幻觉。",
        "Faithfulness": "Faithfulness 是 RAGAS 框架的指标，用于衡量生成答案是否忠于检索到的上下文。",
        "用中文": "好的，我会用中文回答您的问题。",
    }
    try:
        llm = OpenAICompatibleClient()
        print("✅ 使用真实 LLM 生成回答")
    except Exception:
        llm = MockLLMClient(responses=mock_responses)
        print("⚠️  降级为 Mock 模式（LLM 不可用）")

    # 评估器（使用真实 LLM 打分，失败自动降级为离线 fallback）
    evaluator = Evaluator.with_llm(llm=llm, db_path=data_dir / "eval" / "evaluations.db")

    agent = Agent(
        AgentConfig(
            knowledge_base=kb,
            short_term_memory=ShortTermMemory(max_turns=4),
            medium_term_memory=MediumTermMemory(llm_client=llm),
            long_term_memory=ltm,
            evaluator=evaluator,
            llm_client=llm,
        )
    )

    user_id = "u_phase4"

    # 1. 用户表达偏好
    print("\n=== 1. 第一轮：用户表达偏好 ===")
    resp = agent.chat(user_id, "请用中文回答我，我喜欢简洁的解释。")
    print_response("User: 请用中文回答我，我喜欢简洁的解释。", resp)

    # 2. 用户提问
    print("\n=== 2. 第二轮：知识库问答 ===")
    resp = agent.chat(user_id, "RAG 是什么？")
    print_response("User: RAG 是什么？", resp)

    # 3. 多轮对话（验证短期记忆）
    print("\n=== 3. 第三轮：基于上下文的追问 ===")
    resp = agent.chat(user_id, "那它怎么减少幻觉？")
    print_response("User: 那它怎么减少幻觉？", resp)

    # 4. 新会话，验证长期记忆召回
    print("\n=== 4. 新会话：验证长期记忆召回 ===")
    agent2 = Agent(
        AgentConfig(
            knowledge_base=kb,
            short_term_memory=ShortTermMemory(max_turns=4),
            long_term_memory=ltm,
            evaluator=evaluator,
            llm_client=llm,
        )
    )
    resp = agent2.chat(user_id, "请解释 Faithfulness 指标，尽量简洁。")
    print_response("User: 请解释 Faithfulness 指标，尽量简洁。", resp)

    # 5. 查看长期记忆库
    print("\n=== 5. 长期记忆库内容 ===")
    print(f"  内部 chunk 数: {len(ltm.store)}")
    all_facts = ltm.recall(user_id, "", top_k=20)
    for i, fact in enumerate(all_facts, 1):
        print(f"  {i}. {fact.content}")

    # 6. 评估报告
    print("\n=== 6. 失败案例报告 ===")
    report = agent.generate_failure_report(threshold=0.6, limit=10)
    if report:
        print(report)
    else:
        print("未生成报告（无评估器）")

    print("\n✅ Phase 4 端到端验证完成")


if __name__ == "__main__":
    main()

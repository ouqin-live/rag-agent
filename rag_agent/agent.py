"""Agent orchestration: memory + knowledge base + generation + evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rag_agent.evaluation import Evaluator
from rag_agent.evaluation.base import EvaluationResult
from rag_agent.knowledge import KnowledgeBase, RetrievalResult
from rag_agent.llm import BaseLLMClient, MockLLMClient
from rag_agent.memory import LongTermMemory, RuleBasedMemoryExtractor, ShortTermMemory

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Response from the Agent for a single user turn."""

    answer: str
    contexts: list[RetrievalResult] = field(default_factory=list)
    long_term_facts: list[str] = field(default_factory=list)
    evaluation: EvaluationResult | None = None


@dataclass
class AgentConfig:
    """Configuration for the Agent."""

    knowledge_base: KnowledgeBase
    short_term_memory: ShortTermMemory
    long_term_memory: LongTermMemory | None = None
    evaluator: Evaluator | None = None
    llm_client: BaseLLMClient | None = None
    system_prompt: str = "你是一个严谨的 RAG 助手。请仅根据提供的参考资料和已知用户信息回答问题，不要编造参考资料之外的信息。如果参考资料不足，请明确说明。"
    fallback_enabled: bool = True


class Agent:
    """End-to-end RAG Agent with memory and auto-evaluation."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.extractor = RuleBasedMemoryExtractor()
        if self.config.llm_client is None:
            self.config.llm_client = MockLLMClient()

    def chat(self, user_id: str, question: str) -> ChatResponse:
        """Run a full question-answering turn."""
        # 1. Long-term memory recall
        long_term_facts: list[str] = []
        if self.config.long_term_memory:
            facts = self.config.long_term_memory.recall(user_id, question, top_k=3)
            long_term_facts = [f.content for f in facts]

        # 2. Knowledge base retrieval
        kb_results = self.config.knowledge_base.search(question, top_k=5)
        contexts = [r.text for r in kb_results]

        # 3. Build prompt
        messages = self._build_messages(
            question=question,
            long_term_facts=long_term_facts,
            contexts=contexts,
        )

        # 4. LLM generation with fallback
        try:
            answer = self.config.llm_client.generate(messages, temperature=0.3)
        except Exception as exc:
            logger.warning("LLM generation failed: %s", exc)
            if self.config.fallback_enabled:
                answer = self._fallback_generate(question, contexts)
            else:
                raise

        # 5. Update short-term memory
        self.config.short_term_memory.add("user", question)
        self.config.short_term_memory.add("assistant", answer)

        # 6. Extract and store long-term facts
        if self.config.long_term_memory is not None:
            facts = self.extractor.extract(question, answer)
            for fact in facts:
                self.config.long_term_memory.remember(user_id, fact)

        # 7. Evaluate
        evaluation = None
        if self.config.evaluator:
            try:
                evaluation = self.config.evaluator.evaluate(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                )
            except Exception as exc:
                logger.warning("Evaluation failed: %s", exc)

        return ChatResponse(
            answer=answer,
            contexts=kb_results,
            long_term_facts=long_term_facts,
            evaluation=evaluation,
        )

    def reset_session(self) -> None:
        """Clear the current short-term conversation history."""
        self.config.short_term_memory.clear()

    def generate_failure_report(
        self,
        threshold: float = 0.6,
        limit: int = 20,
    ) -> str | None:
        """Generate a text report of recent failure cases."""
        if self.config.evaluator is None:
            return None
        from rag_agent.evaluation.report import ReportGenerator

        report_gen = ReportGenerator(db_path=self.config.evaluator.db_path)
        return report_gen.generate_text_report(threshold=threshold, limit=limit)

    def _build_messages(
        self,
        question: str,
        long_term_facts: list[str],
        contexts: list[str],
    ) -> list[dict[str, str]]:
        system_parts = [self.config.system_prompt]

        if long_term_facts:
            fact_block = "\n".join(f"- {f}" for f in long_term_facts)
            system_parts.append(f"\n[用户相关信息]\n{fact_block}")

        if contexts:
            system_parts.append("\n[参考资料]\n" + "\n".join(f"- {c}" for c in contexts))

        messages: list[dict[str, str]] = [
            {"role": "system", "content": "\n".join(system_parts)},
        ]

        # Inject recent short-term history before the current question
        history = self.config.short_term_memory.get_history_text()
        if history:
            messages.append({"role": "user", "content": f"[对话历史]\n{history}"})
            messages.append(
                {"role": "assistant", "content": "我会结合以上历史回答您的问题。"}
            )

        messages.append({"role": "user", "content": question})
        return messages

    def _fallback_generate(self, question: str, contexts: list[str]) -> str:
        """Template-based fallback when the LLM is unavailable."""
        if contexts:
            summary = "；".join(contexts[:2])
            return f"基于参考资料：{summary}。"
        return "当前无法生成回答，请稍后再试。"

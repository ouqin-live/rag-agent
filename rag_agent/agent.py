"""Agent orchestration: memory + knowledge base + generation + evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rag_agent.cache import SemanticCache
from rag_agent.config import get_settings
from rag_agent.embedder import get_embedder
from rag_agent.evaluation import Evaluator
from rag_agent.evaluation.base import EvaluationResult
from rag_agent.knowledge import KnowledgeBase, RetrievalResult
from rag_agent.llm import BaseLLMClient, MockLLMClient
from rag_agent.memory import (
    LLMMemoryExtractor,
    LongTermMemory,
    MediumTermMemory,
    RuleBasedMemoryExtractor,
    ShortTermMemory,
)
from rag_agent.retrieval import (
    IdentityTransformer,
    QueryTransformer,
    RewritingTransformer,
)

logger = logging.getLogger(__name__)


def _build_query_transformer(llm_client: BaseLLMClient) -> QueryTransformer:
    """Build the default query transformer based on application settings."""
    settings = get_settings()
    if not settings.query_transform_enabled:
        return IdentityTransformer()
    return RewritingTransformer(
        llm_client=llm_client,
        max_history_turns=settings.query_transform_max_history_turns,
    )


def _build_semantic_cache() -> SemanticCache | None:
    """Build the default semantic cache based on application settings."""
    settings = get_settings()
    if not settings.semantic_cache_enabled:
        return None
    return SemanticCache(
        embedder=get_embedder(),
        threshold=settings.semantic_cache_threshold,
        ttl_seconds=settings.semantic_cache_ttl_seconds,
    )


def _default_system_prompt() -> str:
    return get_settings().agent_system_prompt


@dataclass
class ChatResponse:
    """Response from the Agent for a single user turn."""

    answer: str
    contexts: list[RetrievalResult] = field(default_factory=list)
    long_term_facts: list[str] = field(default_factory=list)
    search_query: str = ""
    cache_hit: bool = False
    cached_query: str | None = None
    evaluation: EvaluationResult | None = None


@dataclass
class AgentConfig:
    """Configuration for the Agent."""

    knowledge_base: KnowledgeBase
    short_term_memory: ShortTermMemory
    medium_term_memory: MediumTermMemory | None = None
    long_term_memory: LongTermMemory | None = None
    evaluator: Evaluator | None = None
    llm_client: BaseLLMClient | None = None
    query_transformer: QueryTransformer | None = None
    semantic_cache: SemanticCache | None = None
    system_prompt: str = field(default_factory=_default_system_prompt)
    fallback_enabled: bool = True


class Agent:
    """End-to-end RAG Agent with memory and auto-evaluation."""

    def __init__(self, config: AgentConfig):
        self.config = config
        if self.config.llm_client is None:
            self.config.llm_client = MockLLMClient()
        # 默认初始化 query transformer（如未显式传入）
        if self.config.query_transformer is None:
            self.config.query_transformer = _build_query_transformer(
                self.config.llm_client
            )
        # 默认初始化 semantic cache（如未显式传入）
        if self.config.semantic_cache is None:
            self.config.semantic_cache = _build_semantic_cache()
        # 有 LLM 时优先用 LLM 提取事实，失败自动降级到规则提取
        self.extractor = LLMMemoryExtractor(self.config.llm_client)

    def chat(self, user_id: str, question: str) -> ChatResponse:
        """Run a full question-answering turn (synchronous)."""
        return self._run_turn(user_id, question, sync=True)

    async def achat(self, user_id: str, question: str) -> ChatResponse:
        """Run a full question-answering turn (asynchronous)."""
        return await self._run_turn_async(user_id, question)

    def reset_session(self) -> None:
        """Clear the current short-term conversation history."""
        self.config.short_term_memory.clear()

    def generate_failure_report(
        self,
        threshold: float | None = None,
        limit: int = 20,
    ) -> str | None:
        """Generate a text report of recent failure cases."""
        if self.config.evaluator is None:
            return None
        from rag_agent.evaluation.report import ReportGenerator

        threshold = threshold or self.config.evaluator.failure_threshold
        report_gen = ReportGenerator(db_path=self.config.evaluator.db_path)
        return report_gen.generate_text_report(threshold=threshold, limit=limit)

    # ------------------------------------------------------------------
    # Internal turn logic
    # ------------------------------------------------------------------
    def _run_turn(self, user_id: str, question: str, sync: bool) -> ChatResponse:
        # -1. Semantic cache lookup
        cached = self._lookup_semantic_cache(user_id, question)
        if cached is not None:
            self.config.short_term_memory.add("user", question)
            self.config.short_term_memory.add("assistant", cached["answer"])
            self._archive_to_medium_term()
            return ChatResponse(
                answer=cached["answer"],
                contexts=[],
                long_term_facts=cached.get("long_term_facts", []),
                search_query="",
                cache_hit=True,
                cached_query=cached.get("cached_query"),
                evaluation=cached.get("evaluation"),
            )

        # 0. Query transformation before retrieval
        search_queries = self.config.query_transformer.transform(
            question, self.config.short_term_memory.get_messages()
        )
        search_query = search_queries[0] if search_queries else question

        # 1. Long-term memory recall
        long_term_facts: list[str] = []
        if self.config.long_term_memory:
            facts = self.config.long_term_memory.recall(user_id, search_query, top_k=3)
            long_term_facts = [f.content for f in facts]

        settings = get_settings()
        temperature = settings.llm_temperature
        top_k = settings.agent_top_k

        # 2. 混合检索
        kb_results = self.config.knowledge_base.hybrid_search(search_query, top_k=top_k)
        contexts = [r.text for r in kb_results]

        # 3. Build prompt
        messages = self._build_messages(
            question=question,
            long_term_facts=long_term_facts,
            contexts=contexts,
        )

        # 4. LLM generation with fallback
        try:
            answer = self.config.llm_client.generate(messages, temperature=temperature)
        except Exception as exc:
            logger.warning("LLM generation failed: %s", exc)
            if self.config.fallback_enabled:
                answer = self._fallback_generate(question, contexts)
            else:
                raise

        # 5. Update short-term memory and archive
        self.config.short_term_memory.add("user", question)
        self.config.short_term_memory.add("assistant", answer)
        self._archive_to_medium_term()

        # 6. Extract and store long-term facts
        if self.config.long_term_memory is not None:
            facts = self.extractor.extract(question, answer)
            for fact in facts:
                self.config.long_term_memory.remember(user_id, fact)

        # 7. Evaluate
        evaluation = self._evaluate(question, answer, contexts)

        # 8. Store in semantic cache
        self._store_semantic_cache(
            user_id=user_id,
            question=question,
            answer=answer,
            contexts=contexts,
            long_term_facts=long_term_facts,
            evaluation=evaluation,
        )

        return ChatResponse(
            answer=answer,
            contexts=kb_results,
            long_term_facts=long_term_facts,
            search_query=search_query,
            evaluation=evaluation,
        )

    async def _run_turn_async(self, user_id: str, question: str) -> ChatResponse:
        # -1. Semantic cache lookup
        cached = self._lookup_semantic_cache(user_id, question)
        if cached is not None:
            self.config.short_term_memory.add("user", question)
            self.config.short_term_memory.add("assistant", cached["answer"])
            self._archive_to_medium_term()
            return ChatResponse(
                answer=cached["answer"],
                contexts=[],
                long_term_facts=cached.get("long_term_facts", []),
                search_query="",
                cache_hit=True,
                cached_query=cached.get("cached_query"),
                evaluation=cached.get("evaluation"),
            )

        # 0. Query transformation before retrieval
        search_queries = await self.config.query_transformer.atransform(
            question, self.config.short_term_memory.get_messages()
        )
        search_query = search_queries[0] if search_queries else question

        # 1. Long-term memory recall (sync vector search is fast; keep sync)
        long_term_facts: list[str] = []
        if self.config.long_term_memory:
            facts = self.config.long_term_memory.recall(user_id, search_query, top_k=3)
            long_term_facts = [f.content for f in facts]

        settings = get_settings()
        temperature = settings.llm_temperature
        top_k = settings.agent_top_k

        # 2. 混合检索
        kb_results = self.config.knowledge_base.hybrid_search(search_query, top_k=top_k)
        contexts = [r.text for r in kb_results]

        # 3. Build prompt
        messages = self._build_messages(
            question=question,
            long_term_facts=long_term_facts,
            contexts=contexts,
        )

        # 4. Async LLM generation with fallback
        try:
            answer = await self.config.llm_client.agenerate(messages, temperature=temperature)
        except Exception as exc:
            logger.warning("LLM generation failed: %s", exc)
            if self.config.fallback_enabled:
                answer = self._fallback_generate(question, contexts)
            else:
                raise

        # 5. Update short-term memory and archive
        self.config.short_term_memory.add("user", question)
        self.config.short_term_memory.add("assistant", answer)
        self._archive_to_medium_term()

        # 6. Extract and store long-term facts
        if self.config.long_term_memory is not None:
            facts = self.extractor.extract(question, answer)
            for fact in facts:
                self.config.long_term_memory.remember(user_id, fact)

        # 7. Evaluate
        evaluation = self._evaluate(question, answer, contexts)

        # 8. Store in semantic cache
        self._store_semantic_cache(
            user_id=user_id,
            question=question,
            answer=answer,
            contexts=contexts,
            long_term_facts=long_term_facts,
            evaluation=evaluation,
        )

        return ChatResponse(
            answer=answer,
            contexts=kb_results,
            long_term_facts=long_term_facts,
            search_query=search_query,
            evaluation=evaluation,
        )

    async def achat_stream(
        self,
        user_id: str,
        question: str,
    ):
        """Stream the answer token by token.

        Yields answer chunks (strings). Short-term memory and evaluation are
        updated after the full answer has been produced.
        """
        # -1. Semantic cache lookup
        cached = self._lookup_semantic_cache(user_id, question)
        if cached is not None:
            answer = cached["answer"]
            self.config.short_term_memory.add("user", question)
            self.config.short_term_memory.add("assistant", answer)
            self._archive_to_medium_term()
            yield answer
            return

        # 0. Query transformation before retrieval
        search_queries = await self.config.query_transformer.atransform(
            question, self.config.short_term_memory.get_messages()
        )
        search_query = search_queries[0] if search_queries else question

        long_term_facts: list[str] = []
        if self.config.long_term_memory:
            facts = self.config.long_term_memory.recall(user_id, search_query, top_k=3)
            long_term_facts = [f.content for f in facts]

        settings = get_settings()
        temperature = settings.llm_temperature
        top_k = settings.agent_top_k

        kb_results = self.config.knowledge_base.hybrid_search(search_query, top_k=top_k)
        contexts = [r.text for r in kb_results]

        messages = self._build_messages(
            question=question,
            long_term_facts=long_term_facts,
            contexts=contexts,
        )

        full_answer = ""
        try:
            async for chunk in self.config.llm_client.agenerate_stream(
                messages, temperature=temperature
            ):
                full_answer += chunk
                yield chunk
        except Exception as exc:
            logger.warning("LLM streaming failed: %s", exc)
            if self.config.fallback_enabled:
                fallback = self._fallback_generate(question, contexts)
                full_answer = fallback
                yield fallback
            else:
                raise

        # Post-turn bookkeeping after streaming completes
        self.config.short_term_memory.add("user", question)
        self.config.short_term_memory.add("assistant", full_answer)
        self._archive_to_medium_term()

        if self.config.long_term_memory is not None:
            facts = self.extractor.extract(question, full_answer)
            for fact in facts:
                self.config.long_term_memory.remember(user_id, fact)

        self._evaluate(question, full_answer, contexts)

        # Store in semantic cache
        self._store_semantic_cache(
            user_id=user_id,
            question=question,
            answer=full_answer,
            contexts=contexts,
            long_term_facts=long_term_facts,
            evaluation=None,  # streaming path does not keep evaluation object
        )

    def _evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> EvaluationResult | None:
        if not self.config.evaluator:
            return None
        try:
            return self.config.evaluator.evaluate(
                question=question,
                answer=answer,
                contexts=contexts,
            )
        except Exception as exc:
            logger.warning("Evaluation failed: %s", exc)
            return None

    def _lookup_semantic_cache(
        self, user_id: str, question: str
    ) -> dict[str, Any] | None:
        if not self.config.semantic_cache:
            return None
        return self.config.semantic_cache.lookup(question, user_id)

    def _store_semantic_cache(
        self,
        user_id: str,
        question: str,
        answer: str,
        contexts: list[str],
        long_term_facts: list[str],
        evaluation: EvaluationResult | None,
    ) -> None:
        if not self.config.semantic_cache:
            return
        self.config.semantic_cache.store(
            query=question,
            user_id=user_id,
            answer=answer,
            contexts=contexts,
            long_term_facts=long_term_facts,
            evaluation=evaluation,
        )

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

        # 注入中期记忆（本次会话摘要）
        if self.config.medium_term_memory:
            summary = self.config.medium_term_memory.get_summary()
            if summary:
                system_parts.append(f"\n[本次会话摘要]\n{summary}")

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

    def _archive_to_medium_term(self) -> None:
        """当短期记忆超过限制时，将旧轮次归档到中期记忆。"""
        if self.config.medium_term_memory is None:
            return
        # ShortTermMemory._enforce_limit 已自动裁剪，这里只取最旧的一轮做归档
        messages = self.config.short_term_memory.get_messages()
        # 如果消息数超过 2*max_turns，说明有旧消息被裁剪了
        # 简化处理：每轮后检查，把超出部分的第一轮归档
        cut_messages = self._extract_overflow_messages()
        if cut_messages:
            self.config.medium_term_memory.update(cut_messages)

    def _extract_overflow_messages(self) -> list:
        """Return messages that would be removed from short-term memory."""
        from rag_agent.memory.short_term import Message

        messages = self.config.short_term_memory.get_messages()
        max_msgs = self.config.short_term_memory.max_turns * 2
        if len(messages) > max_msgs:
            overflow = len(messages) - max_msgs
            # 取最旧的完整轮次（user + assistant 成对）
            to_archive = messages[:overflow]
            # 如果最后一条是 user，也加上（保证成对）
            result: list = []
            for msg in to_archive:
                result.append(msg)
            return result
        return []

    def _fallback_generate(self, question: str, contexts: list[str]) -> str:
        """Template-based fallback when the LLM is unavailable."""
        if contexts:
            summary = "；".join(contexts[:2])
            return f"基于参考资料：{summary}。"
        return "当前无法生成回答，请稍后再试。"

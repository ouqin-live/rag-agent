"""Agent orchestration: memory + knowledge base + generation + evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rag_agent.agentic import (
    BaseTool,
    QueryRouter,
    ReactLoop,
    RuleBasedRouter,
    SelfCorrector,
)
from rag_agent.cache import SemanticCache
from rag_agent.config import get_settings
from rag_agent.embedder import get_embedder
from rag_agent.evaluation import Evaluator
from rag_agent.evaluation.base import EvaluationResult
from rag_agent.guardrails import Guardrails, GuardrailsConfig
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
    # Agentic RAG (P2-1)
    agentic_enabled: bool = False
    router: QueryRouter | None = None
    tools: dict[str, BaseTool] | None = None
    self_corrector: SelfCorrector | None = None
    # Guardrails (P2-3)
    guardrails: Guardrails | None = None


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
        # 默认初始化 agentic 组件
        self._init_agentic()
        # 默认初始化护栏
        self._init_guardrails()

    def _init_agentic(self) -> None:
        """Initialize the ReAct loop when agentic mode is enabled."""
        self._react_loop: ReactLoop | None = None
        self._langgraph_agent: Any = None
        if not self.config.agentic_enabled:
            return

        settings = get_settings()

        # LangGraph 模式
        if settings.agentic_use_langgraph:
            from rag_agent.graph import LangGraphAgent, build_agentic_graph

            router = self.config.router or RuleBasedRouter()
            self_corrector = self.config.self_corrector or SelfCorrector(
                llm_client=self.config.llm_client,
                threshold=settings.agentic_faithfulness_threshold,
                max_iterations=settings.agentic_max_iterations,
            )

            graph = build_agentic_graph(
                knowledge_base=self.config.knowledge_base,
                llm_client=self.config.llm_client,
                long_term_memory=self.config.long_term_memory,
                query_transformer=self.config.query_transformer,
                self_corrector=self_corrector,
                tools=self.config.tools or {},
                router=router,
                system_prompt=self.config.system_prompt,
                top_k=settings.agent_top_k,
                temperature=settings.llm_temperature,
            )
            self._langgraph_agent = LangGraphAgent(graph)
            return

        # 原有 ReactLoop 模式
        router = self.config.router
        if router is None and settings.agentic_use_llm_router:
            from rag_agent.agentic.router import LLMQueryRouter

            router = LLMQueryRouter(llm_client=self.config.llm_client)
        router = router or RuleBasedRouter()

        self_corrector = self.config.self_corrector or SelfCorrector(
            llm_client=self.config.llm_client,
            threshold=settings.agentic_faithfulness_threshold,
            max_iterations=settings.agentic_max_iterations,
        )

        self._react_loop = ReactLoop(
            llm_client=self.config.llm_client,
            knowledge_base=self.config.knowledge_base,
            long_term_memory=self.config.long_term_memory,
            short_term_memory=self.config.short_term_memory,
            medium_term_memory=self.config.medium_term_memory,
            router=router,
            self_corrector=self_corrector,
            query_transformer=self.config.query_transformer,
            evaluator=self.config.evaluator,
            tools=self.config.tools or {},
            system_prompt=self.config.system_prompt,
            max_iterations=settings.agentic_max_iterations,
            top_k=settings.agent_top_k,
            temperature=settings.llm_temperature,
        )

    def _init_guardrails(self) -> None:
        """初始化安全护栏。"""
        if self.config.guardrails is not None:
            self._guardrails = self.config.guardrails
            return
        settings = get_settings()
        gc = GuardrailsConfig(
            enabled=settings.guardrails_enabled,
            prompt_injection_enabled=settings.guardrails_prompt_injection_enabled,
            prompt_injection_hard_block=settings.guardrails_prompt_injection_hard_block,
            pii_detection_enabled=settings.guardrails_pii_enabled,
            pii_hard_block=settings.guardrails_pii_hard_block,
            output_toxicity_enabled=settings.guardrails_output_toxicity_enabled,
            output_toxicity_hard_block=settings.guardrails_output_toxicity_hard_block,
            confidence_check_enabled=settings.guardrails_confidence_enabled,
            confidence_threshold=settings.guardrails_confidence_threshold,
            raise_on_block=settings.guardrails_raise_on_block,
        )
        self._guardrails = Guardrails(gc)

    def chat(self, user_id: str, question: str) -> ChatResponse:
        """Run a full question-answering turn (synchronous)."""
        if self.config.agentic_enabled and self._langgraph_agent is not None:
            return self._run_langgraph_turn(user_id, question)
        if self.config.agentic_enabled and self._react_loop is not None:
            return self._run_agentic_turn(user_id, question)
        return self._run_turn(user_id, question, sync=True)

    async def achat(self, user_id: str, question: str) -> ChatResponse:
        """Run a full question-answering turn (asynchronous)."""
        if self.config.agentic_enabled and self._langgraph_agent is not None:
            return await self._run_langgraph_turn_async(user_id, question)
        if self.config.agentic_enabled and self._react_loop is not None:
            return await self._run_agentic_turn_async(user_id, question)
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
        # 输入护栏检查
        blocked = self._check_input_guardrails(user_id, question)
        if blocked is not None:
            return blocked

        # -1. 优先命中语义缓存
        cached = self._try_semantic_cache(user_id, question)
        if cached is not None:
            return cached

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

        # 2. 混合检索（失败不中断主流程，降级为空上下文）
        kb_results: list[RetrievalResult] = []
        try:
            kb_results = self.config.knowledge_base.hybrid_search(search_query, top_k=top_k)
        except Exception as exc:
            logger.warning("Knowledge base retrieval failed: %s", exc)
        contexts = [r.text for r in kb_results]

        # 检索置信度检查
        self._check_confidence_guardrails(contexts, kb_results)

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

        # 输出护栏审核
        safe_answer = self._check_output_guardrails(answer)
        if safe_answer is not None:
            answer = safe_answer

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
        # 输入护栏检查
        blocked = self._check_input_guardrails(user_id, question)
        if blocked is not None:
            return blocked

        # -1. 优先命中语义缓存
        cached = self._try_semantic_cache(user_id, question)
        if cached is not None:
            return cached

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

        # 2. 混合检索（失败不中断主流程，降级为空上下文）
        kb_results: list[RetrievalResult] = []
        try:
            kb_results = self.config.knowledge_base.hybrid_search(search_query, top_k=top_k)
        except Exception as exc:
            logger.warning("Knowledge base retrieval failed: %s", exc)
        contexts = [r.text for r in kb_results]

        # 检索置信度检查
        self._check_confidence_guardrails(contexts, kb_results)

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

        # 输出护栏审核
        safe_answer = self._check_output_guardrails(answer)
        if safe_answer is not None:
            answer = safe_answer

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
        # 输入护栏检查
        blocked = self._check_input_guardrails(user_id, question)
        if blocked is not None:
            yield blocked.answer
            return

        # -1. 优先命中语义缓存
        cached = self._lookup_semantic_cache(user_id, question)
        if cached is not None:
            answer = cached["answer"]
            self.config.short_term_memory.add("user", question)
            self.config.short_term_memory.add("assistant", answer)
            self._archive_to_medium_term()
            yield answer
            return

        # Agentic 模式：执行一轮 ReAct 循环后模拟流式输出
        if self.config.agentic_enabled and self._react_loop is not None:
            result = await self._react_loop.arun(
                user_id, question, self._get_history_messages()
            )
            answer = result.answer
            contexts = [r.text for r in result.contexts]

            # 输出护栏审核
            safe_answer = self._check_output_guardrails(answer)
            if safe_answer is not None:
                answer = safe_answer

            yield answer

            # 本轮后处理：记忆、事实提取、评估、缓存
            self.config.short_term_memory.add("user", question)
            self.config.short_term_memory.add("assistant", answer)
            self._archive_to_medium_term()

            if self.config.long_term_memory is not None:
                facts = self.extractor.extract(question, answer)
                for fact in facts:
                    self.config.long_term_memory.remember(user_id, fact)

            self._evaluate(question, answer, contexts)

            self._store_semantic_cache(
                user_id=user_id,
                question=question,
                answer=answer,
                contexts=contexts,
                long_term_facts=result.long_term_facts,
                evaluation=None,
            )
            return

        #  0. Query transformation before retrieval
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

        kb_results: list[RetrievalResult] = []
        try:
            kb_results = self.config.knowledge_base.hybrid_search(search_query, top_k=top_k)
        except Exception as exc:
            logger.warning("Knowledge base retrieval failed: %s", exc)
        contexts = [r.text for r in kb_results]

        # 检索置信度检查
        self._check_confidence_guardrails(contexts, kb_results)

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

        # 输出护栏审核（流式全部接收后检查）
        safe_answer = self._check_output_guardrails(full_answer)
        if safe_answer is not None:
            full_answer = safe_answer

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

    # ------------------------------------------------------------------
    # Guardrails helpers (P2-3)
    # ------------------------------------------------------------------

    def _check_input_guardrails(
        self, user_id: str, question: str
    ) -> ChatResponse | None:
        """执行输入护栏检查。被硬拦截时返回安全兜底 ChatResponse。

        返回 None 表示通过，允许继续处理。
        """
        result = self._guardrails.check_input(question)
        if not result.blocked:
            return None

        logger.warning(
            "输入护栏拦截 user=%s blocked_by=%s", user_id, result.blocked_by
        )
        safe_answer = "抱歉，您的请求包含不安全内容，无法处理。"
        self.config.short_term_memory.add("user", question)
        self.config.short_term_memory.add("assistant", safe_answer)
        self._archive_to_medium_term()
        return ChatResponse(
            answer=safe_answer,
            search_query=question,
        )

    def _check_output_guardrails(self, answer: str) -> str | None:
        """对 LLM 输出执行毒性审核。被硬拦截时返回安全兜底文本。

        返回 None 表示通过，返回字符串表示替换后的安全回答。
        """
        result = self._guardrails.check_output(answer)
        if not result.blocked:
            return None

        logger.warning("输出护栏拦截 output_toxicity")
        return "抱歉，生成的回答包含不当内容，已被替换。"

    def _check_confidence_guardrails(
        self, contexts: list[str], retrieval_results: list[RetrievalResult]
    ) -> None:
        """检查检索置信度并在过低时记录警告。"""
        scores = [r.score for r in retrieval_results if hasattr(r, 'score')]
        result = self._guardrails.check_retrieval(contexts, scores if scores else None)
        if result.action.value == "warn":
            logger.warning("检索置信度警告: %s", result.message)

    # ------------------------------------------------------------------
    # Agentic RAG turn logic (P2-1)
    # ------------------------------------------------------------------

    def _run_agentic_turn(self, user_id: str, question: str) -> ChatResponse:
        """同步执行一轮 ReAct / 自我修正。"""
        assert self._react_loop is not None

        # 输入护栏检查
        blocked = self._check_input_guardrails(user_id, question)
        if blocked is not None:
            return blocked

        # 优先命中语义缓存，避免重复执行 ReAct 循环
        cached = self._try_semantic_cache(user_id, question)
        if cached is not None:
            return cached

        history = self._get_history_messages()
        result = self._react_loop.run(user_id, question, history)

        return self._finalize_agentic_turn(user_id, question, result)

    async def _run_agentic_turn_async(
        self, user_id: str, question: str
    ) -> ChatResponse:
        """异步执行一轮 ReAct / 自我修正。"""
        assert self._react_loop is not None

        # 输入护栏检查
        blocked = self._check_input_guardrails(user_id, question)
        if blocked is not None:
            return blocked

        # 优先命中语义缓存，避免重复执行 ReAct 循环
        cached = self._try_semantic_cache(user_id, question)
        if cached is not None:
            return cached

        history = self._get_history_messages()
        result = await self._react_loop.arun(user_id, question, history)

        return self._finalize_agentic_turn(user_id, question, result)

    def _run_langgraph_turn(self, user_id: str, question: str) -> ChatResponse:
        """LangGraph 同步执行一轮。"""
        assert self._langgraph_agent is not None

        blocked = self._check_input_guardrails(user_id, question)
        if blocked is not None:
            return blocked

        result = self._langgraph_agent.chat(
            user_id, question, self._get_history_messages()
        )
        return self._finalize_langgraph_turn(user_id, question, result)

    async def _run_langgraph_turn_async(
        self, user_id: str, question: str
    ) -> ChatResponse:
        """LangGraph 异步执行一轮。"""
        assert self._langgraph_agent is not None

        blocked = self._check_input_guardrails(user_id, question)
        if blocked is not None:
            return blocked

        result = await self._langgraph_agent.achat(
            user_id, question, self._get_history_messages()
        )
        return self._finalize_langgraph_turn(user_id, question, result)

    def _finalize_langgraph_turn(
        self, user_id: str, question: str, result: dict
    ) -> ChatResponse:
        """Post-process a LangGraph turn: memory, evaluation, cache."""
        answer: str = result.get("answer", "")
        contexts: list[str] = result.get("contexts", [])
        long_term_facts: list[str] = result.get("long_term_facts", [])
        search_query: str = result.get("search_query", question)

        safe_answer = self._check_output_guardrails(answer)
        if safe_answer is not None:
            answer = safe_answer

        self.config.short_term_memory.add("user", question)
        self.config.short_term_memory.add("assistant", answer)
        self._archive_to_medium_term()

        if self.config.long_term_memory is not None:
            facts = self.extractor.extract(question, answer)
            for fact in facts:
                self.config.long_term_memory.remember(user_id, fact)

        evaluation = self._evaluate(question, answer, contexts)

        return ChatResponse(
            answer=answer,
            contexts=[],
            long_term_facts=long_term_facts,
            search_query=search_query,
            evaluation=evaluation,
        )

    def _finalize_agentic_turn(
        self, user_id: str, question: str, result: "ReactResult"
    ) -> ChatResponse:
        """Post-process an agentic turn: memory, evaluation, cache."""
        answer = result.answer
        contexts = [r.text for r in result.contexts]

        # 输出护栏审核
        safe_answer = self._check_output_guardrails(answer)
        if safe_answer is not None:
            answer = safe_answer

        # Update short-term memory and archive
        self.config.short_term_memory.add("user", question)
        self.config.short_term_memory.add("assistant", answer)
        self._archive_to_medium_term()

        # Extract and store long-term facts
        if self.config.long_term_memory is not None:
            facts = self.extractor.extract(question, answer)
            for fact in facts:
                self.config.long_term_memory.remember(user_id, fact)

        # Evaluate
        evaluation = self._evaluate(question, answer, contexts)

        # Store in semantic cache
        self._store_semantic_cache(
            user_id=user_id,
            question=question,
            answer=answer,
            contexts=contexts,
            long_term_facts=result.long_term_facts,
            evaluation=evaluation,
        )

        return ChatResponse(
            answer=answer,
            contexts=result.contexts,
            long_term_facts=result.long_term_facts,
            search_query=result.search_query,
            evaluation=evaluation,
        )

    def _try_semantic_cache(
        self, user_id: str, question: str
    ) -> ChatResponse | None:
        """尝试命中语义缓存；命中则更新短期记忆并直接返回。"""
        if not self.config.semantic_cache:
            return None

        cached = self.config.semantic_cache.lookup(question, user_id)
        if cached is None:
            return None

        self.config.short_term_memory.add("user", question)
        self.config.short_term_memory.add("assistant", cached["answer"])
        self._archive_to_medium_term()
        return ChatResponse(
            answer=cached["answer"],
            contexts=[],
            long_term_facts=cached.get("long_term_facts", []),
            search_query=question,
            cache_hit=True,
            cached_query=cached.get("cached_query"),
            evaluation=cached.get("evaluation"),
        )

    def _get_history_messages(self) -> list[dict[str, str]]:
        """将短期记忆转换为 OpenAI 风格的消息列表。"""
        messages: list[dict[str, str]] = []
        for msg in self.config.short_term_memory.get_messages():
            if msg.role in ("user", "assistant"):
                messages.append({"role": msg.role, "content": msg.content})
        return messages

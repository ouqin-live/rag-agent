"""Agent orchestration: memory + knowledge base + generation + evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rag_agent.agentic import BaseTool, QueryRouter, SelfCorrector
from rag_agent.cache import SemanticCache
from rag_agent.config import get_settings
from rag_agent.evaluation import Evaluator
from rag_agent.evaluation.base import EvaluationResult
from rag_agent.guardrails import Guardrails, GuardrailsConfig
from rag_agent.knowledge import KnowledgeBase, RetrievalResult
from rag_agent.llm import BaseLLMClient
from rag_agent.memory import (
    LLMMemoryExtractor,
    LongTermMemory,
    MediumTermMemory,
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
    llm_client: BaseLLMClient
    medium_term_memory: MediumTermMemory | None = None
    long_term_memory: LongTermMemory | None = None
    evaluator: Evaluator | None = None
    query_transformer: QueryTransformer | None = None
    semantic_cache: SemanticCache | None = None
    system_prompt: str = field(default_factory=_default_system_prompt)
    fallback_enabled: bool = True
    # Agentic RAG (P2-1) — LangGraph-based workflow
    router: QueryRouter | None = None
    tools: dict[str, BaseTool] | None = None
    self_corrector: SelfCorrector | None = None
    # Guardrails (P2-3)
    guardrails: Guardrails | None = None


class Agent:
    """End-to-end RAG Agent with memory and auto-evaluation."""

    def __init__(self, config: AgentConfig):
        self.config = config
        # 默认初始化 query transformer（如未显式传入）
        if self.config.query_transformer is None:
            self.config.query_transformer = _build_query_transformer(
                self.config.llm_client
            )
        # 有 LLM 时优先用 LLM 提取事实，失败自动降级到规则提取
        self.extractor = LLMMemoryExtractor(self.config.llm_client)
        # 初始化护栏（必须在 _init_agentic 之前）
        self._init_guardrails()
        # 初始化 LangGraph 工作流
        self._init_agentic()

    def _init_agentic(self) -> None:
        """Initialize the LangGraph agentic workflow."""
        self._langgraph_agent: Any = None
        if self.config.knowledge_base is None:
            return

        settings = get_settings()

        # 默认的 router 和 self_corrector
        from rag_agent.agentic import RuleBasedRouter

        router = self.config.router or RuleBasedRouter()
        self_corrector = self.config.self_corrector or SelfCorrector(
            llm_client=self.config.llm_client,
            threshold=settings.agentic_faithfulness_threshold,
            max_iterations=settings.agentic_max_iterations,
        )

        from rag_agent.graph import LangGraphAgent, build_agentic_graph

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
            semantic_cache=self.config.semantic_cache,
            guardrails=self._guardrails,
            evaluator=self.config.evaluator,
            memory_extractor=self.extractor,
        )
        self._langgraph_agent = LangGraphAgent(graph)

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_id: str, question: str) -> ChatResponse:
        """Run a full question-answering turn."""
        return self._run_langgraph_turn(user_id, question)

    async def achat(self, user_id: str, question: str) -> ChatResponse:
        """Run a full question-answering turn (asynchronous)."""
        return await self._run_langgraph_turn_async(user_id, question)

    async def achat_stream(
        self,
        user_id: str,
        question: str,
    ):
        """Stream the answer token by token.

        Input guardrails, semantic cache, output guardrails, LTM and
        evaluation are all handled inside the LangGraph workflow.
        """
        result = await self._langgraph_agent.achat(
            user_id, question, self._get_history_messages()
        )
        answer = result.get("answer", "")

        yield answer

        self.config.short_term_memory.add("user", question)
        self.config.short_term_memory.add("assistant", answer)
        self._archive_to_medium_term()

        # Store to semantic cache for future reuse
        cache_hit = result.get("cache_hit", False)
        if not cache_hit and self.config.semantic_cache is not None:
            try:
                self.config.semantic_cache.store(
                    query=question,
                    user_id=user_id,
                    answer=answer,
                    contexts=result.get("contexts", []),
                    long_term_facts=result.get("long_term_facts", []),
                    evaluation=result.get("evaluation"),
                )
            except Exception as exc:
                logger.warning("缓存存储失败(stream): %s", exc)

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
    # Internal: LangGraph turn
    # ------------------------------------------------------------------

    def _run_langgraph_turn(self, user_id: str, question: str) -> ChatResponse:
        assert self._langgraph_agent is not None

        result = self._langgraph_agent.chat(
            user_id, question, self._get_history_messages()
        )
        return self._finalize_turn(user_id, question, result)

    async def _run_langgraph_turn_async(
        self, user_id: str, question: str
    ) -> ChatResponse:
        assert self._langgraph_agent is not None

        result = await self._langgraph_agent.achat(
            user_id, question, self._get_history_messages()
        )
        return self._finalize_turn(user_id, question, result)

    def _finalize_turn(
        self, user_id: str, question: str, result: dict[str, Any]
    ) -> ChatResponse:
        """Post-process a turn: STM update, cache storage, build response."""
        answer: str = result.get("answer", "")
        long_term_facts: list[str] = result.get("long_term_facts", [])
        search_query: str = result.get("search_query", question)
        cache_hit: bool = result.get("cache_hit", False)
        cached_query: str | None = result.get("cached_answer") if cache_hit else None
        evaluation: EvaluationResult | None = result.get("evaluation")

        # Short-term memory update
        self.config.short_term_memory.add("user", question)
        self.config.short_term_memory.add("assistant", answer)
        self._archive_to_medium_term()

        # Store to semantic cache for future reuse
        if not cache_hit and self.config.semantic_cache is not None:
            contexts: list[str] = result.get("contexts", [])
            try:
                self.config.semantic_cache.store(
                    query=question,
                    user_id=user_id,
                    answer=answer,
                    contexts=contexts,
                    long_term_facts=long_term_facts,
                    evaluation=evaluation,
                )
            except Exception as exc:
                logger.warning("缓存存储失败: %s", exc)

        return ChatResponse(
            answer=answer,
            contexts=[],
            long_term_facts=long_term_facts,
            search_query=search_query,
            cache_hit=cache_hit,
            cached_query=cached_query,
            evaluation=evaluation,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

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

    def _archive_to_medium_term(self) -> None:
        """当短期记忆超过限制时，将旧轮次归档到中期记忆。"""
        if self.config.medium_term_memory is None:
            return
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
            result: list = []
            for msg in messages[:overflow]:
                result.append(msg)
            return result
        return []



    def _check_input_guardrails(
        self, user_id: str, question: str
    ) -> ChatResponse | None:
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
        return ChatResponse(answer=safe_answer, search_query=question)

    def _check_output_guardrails(self, answer: str) -> str | None:
        result = self._guardrails.check_output(answer)
        if not result.blocked:
            return None
        logger.warning("输出护栏拦截 output_toxicity")
        return "抱歉，生成的回答包含不当内容，已被替换。"

    def _check_confidence_guardrails(
        self, contexts: list[str], retrieval_results: list[RetrievalResult]
    ) -> None:
        scores = [r.score for r in retrieval_results if hasattr(r, 'score')]
        result = self._guardrails.check_retrieval(contexts, scores if scores else None)
        if result.action.value == "warn":
            logger.warning("检索置信度警告: %s", result.message)

    # ------------------------------------------------------------------
    # Semantic cache helpers
    # ------------------------------------------------------------------

    def _try_semantic_cache(
        self, user_id: str, question: str
    ) -> ChatResponse | None:
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

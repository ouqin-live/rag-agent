"""ReAct 风格循环：检索 → 生成 → 反思 →（修正/再次检索）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rag_agent.agentic.base import AgenticContext, ToolResult
from rag_agent.agentic.router import QueryRouter, RouteDecision, RuleBasedRouter
from rag_agent.agentic.self_correction import SelfCorrector
from rag_agent.agentic.tools import CalculatorTool, DatetimeTool
from rag_agent.knowledge import KnowledgeBase, RetrievalResult
from rag_agent.retrieval import IdentityTransformer, QueryTransformer

if TYPE_CHECKING:
    from rag_agent.evaluation import Evaluator
    from rag_agent.llm import BaseLLMClient
    from rag_agent.memory.long_term import LongTermMemory
    from rag_agent.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)


@dataclass
class ReactResult:
    """ReAct 循环产出的最终结果。"""

    answer: str
    search_query: str = ""
    contexts: list[RetrievalResult] = field(default_factory=list)
    long_term_facts: list[str] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    iterations: int = 1
    reasoning: str = ""


class ReactLoop:
    """Agentic RAG 的 ReAct 循环。

    每轮对话执行以下步骤：

    1. 将问题路由到合适的数据源/工具（知识库、长期记忆、计算器等）。
    2. 使用查询改写器对原始问题进行改写（如指代消解、问题标准化）。
    3. 进入 ReAct 循环：
       3.1 从选定的数据源和工具中检索证据；
       3.2 用大模型生成答案（失败时返回兜底回答）；
       3.3 反思答案质量（如忠实度）；
       3.4 若质量不足，重写查询并再次检索，否则退出循环。
    4. 达到最大迭代次数后，将最终答案和中间证据打包返回。
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        knowledge_base: KnowledgeBase,
        router: QueryRouter | None = None,
        long_term_memory: LongTermMemory | None = None,
        short_term_memory: ShortTermMemory | None = None,
        self_corrector: SelfCorrector | None = None,
        query_transformer: QueryTransformer | None = None,
        evaluator: Evaluator | None = None,
        medium_term_memory: Any | None = None,
        tools: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 2,
        top_k: int = 5,
        temperature: float = 0.3,
    ):
        self.llm_client = llm_client
        self.knowledge_base = knowledge_base
        self.router = router or RuleBasedRouter()
        self.long_term_memory = long_term_memory
        self.short_term_memory = short_term_memory
        self.self_corrector = self_corrector or SelfCorrector(
            llm_client=llm_client, evaluator=evaluator
        )
        self.query_transformer = query_transformer or IdentityTransformer()
        self.medium_term_memory = medium_term_memory
        self.tools = self._build_default_tools()
        if tools:
            self.tools.update(tools)
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_iterations = max_iterations
        self.top_k = top_k
        self.temperature = temperature

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "你是一个严谨的 RAG 助手。请仅根据提供的参考资料、用户相关信息和工具结果回答问题，"
            "不要编造参考资料之外的信息。如果参考资料不足，请明确说明。"
        )

    @staticmethod
    def _build_default_tools() -> dict[str, Any]:
        return {
            "calculator": CalculatorTool(),
            "datetime": DatetimeTool(),
        }

    def run(
        self,
        user_id: str,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> ReactResult:
        """同步执行 ReAct 循环。

        循环流程：检索 → 生成 → 反思 →（修正并再次检索）。
        当答案通过自我修正检查时提前退出，否则最多执行 ``self.max_iterations`` 次。
        """
        # 1. 初始化本轮上下文，并决定使用哪些数据源/工具。
        ctx = AgenticContext(question=question, max_iterations=self.max_iterations)
        route = self.router.route(question, history)
        ctx.reasoning = f"Route: {', '.join(route.selected_sources)}. {route.reasoning}"
        logger.info("Agentic route for %r: %s", question, route.selected_sources)

        # 2. 使用查询改写器对原始问题进行改写（如指代消解、问题标准化）。
        #    传给改写器的是原始 Message 对象列表，保证 RewritingTransformer 能正常解析历史。
        #    改写失败时回退到原始问题。
        transformer_history = (
            self.short_term_memory.get_messages() if self.short_term_memory else None
        )
        try:
            transformed = self.query_transformer.transform(question, transformer_history)
            search_query = transformed[0] if transformed else question
        except Exception as exc:
            logger.warning("Query transformation failed in agentic run: %s", exc)
            search_query = question

        # 3. 开始 ReAct 循环。检索查询可能会在修正阶段被重写。
        for iteration in range(self.max_iterations):
            ctx.iteration = iteration

            # 3.1 从选定的数据源和工具中检索证据。
            #     可能命中知识库、长期记忆、计算器等。
            self._retrieve(user_id, search_query, route, ctx)

            # 3.2 基于收集到的证据生成答案。
            ctx.answer = self._generate_answer(question, history, ctx)

            # 3.3 反思：评估答案是否被充分支撑。
            #     自我修正器会返回一个分数，以及需要时重写后的查询。
            correction = self.self_corrector.check(
                question, ctx.answer, ctx.contexts, iteration
            )
            if not correction.needs_correction:
                # 答案已经足够好，记录推理过程并退出循环。
                ctx.reasoning += f" | Iteration {iteration + 1}: {correction.reasoning}"
                break

            # 3.4 触发修正：记录原因，使用重写后的查询，并清空证据以便下一轮重新收集。
            ctx.reasoning += f" | Iteration {iteration + 1}: {correction.reasoning}"
            search_query = correction.rewritten_query or question
            ctx.reset_evidence()
            logger.info("Self-correction iteration %d: %s", iteration + 1, search_query)

        # 4. 将最终答案和中间证据打包成结果对象返回。
        return ReactResult(
            answer=ctx.answer,
            search_query=search_query,
            contexts=self._kb_results_for_contexts(ctx.contexts),
            long_term_facts=ctx.long_term_facts,
            tool_results=[ToolResult(**r) for r in ctx.tool_results],
            iterations=ctx.iteration + 1,
            reasoning=ctx.reasoning,
        )

    async def arun(
        self,
        user_id: str,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> ReactResult:
        """异步执行 ReAct 循环。

        这是 ``run`` 的异步版本，遵循相同的 检索 → 生成 → 反思 → 修正 循环。
        """
        # 1. 初始化本轮上下文，并决定使用哪些数据源/工具。
        ctx = AgenticContext(question=question, max_iterations=self.max_iterations)
        route = self.router.route(question, history)
        ctx.reasoning = f"Route: {', '.join(route.selected_sources)}. {route.reasoning}"
        logger.info("Agentic route for %r: %s", question, route.selected_sources)

        # 2. 异步使用查询改写器对原始问题进行改写。
        #    传给改写器的是原始 Message 对象列表，保证 RewritingTransformer 能正常解析历史。
        transformer_history = (
            self.short_term_memory.get_messages() if self.short_term_memory else None
        )
        try:
            transformed = await self.query_transformer.atransform(question, transformer_history)
            search_query = transformed[0] if transformed else question
        except Exception as exc:
            logger.warning("Query transformation failed in agentic arun: %s", exc)
            search_query = question

        # 3. 开始 ReAct 循环。检索查询可能会在修正阶段被重写。
        for iteration in range(self.max_iterations):
            ctx.iteration = iteration

            # 3.1 从选定的数据源和工具中检索证据（同步 I/O）。
            self._retrieve(user_id, search_query, route, ctx)

            # 3.2 基于收集到的证据异步生成答案。
            ctx.answer = await self._agenerate_answer(question, history, ctx)

            # 3.3 反思：评估答案是否被充分支撑。
            correction = self.self_corrector.check(
                question, ctx.answer, ctx.contexts, iteration
            )
            if not correction.needs_correction:
                # 答案已经足够好，记录推理过程并退出循环。
                ctx.reasoning += f" | Iteration {iteration + 1}: {correction.reasoning}"
                break

            # 3.4 触发修正：使用重写后的查询并清空证据。
            ctx.reasoning += f" | Iteration {iteration + 1}: {correction.reasoning}"
            search_query = correction.rewritten_query or question
            ctx.reset_evidence()
            logger.info("Self-correction iteration %d: %s", iteration + 1, search_query)

        # 4. 将最终答案和中间证据打包成结果对象返回。
        return ReactResult(
            answer=ctx.answer,
            search_query=search_query,
            contexts=self._kb_results_for_contexts(ctx.contexts),
            long_term_facts=ctx.long_term_facts,
            tool_results=[ToolResult(**r) for r in ctx.tool_results],
            iterations=ctx.iteration + 1,
            reasoning=ctx.reasoning,
        )

    def _retrieve(
        self,
        user_id: str,
        query: str,
        route: RouteDecision,
        ctx: AgenticContext,
    ) -> None:
        """从选定的数据源和工具中收集证据。"""
        if route.use_long_term_memory and self.long_term_memory is not None:
            try:
                facts = self.long_term_memory.recall(user_id, query, top_k=3)
                ctx.long_term_facts.extend([f.content for f in facts])
            except Exception as exc:
                logger.warning("Long-term memory recall failed: %s", exc)

        if route.use_knowledge_base:
            try:
                results = self.knowledge_base.hybrid_search(query, top_k=self.top_k)
                ctx.contexts.extend([r.text for r in results])
            except Exception as exc:
                logger.warning("Knowledge base retrieval failed: %s", exc)

        for source in ("calculator", "datetime", "web_search"):
            if getattr(route, f"use_{source}", False) and source in self.tools:
                try:
                    result = self.tools[source].invoke(query)
                    ctx.tool_results.append(ToolResult(source=source, content=result).to_dict())
                except Exception as exc:
                    logger.warning("Tool %s invocation failed: %s", source, exc)

    def _build_messages(
        self,
        question: str,
        history: list[dict[str, str]] | None,
        ctx: AgenticContext,
    ) -> list[dict[str, str]]:
        """根据系统提示、证据和历史记录构建大模型输入消息。"""
        system_parts = [self.system_prompt]

        if ctx.long_term_facts:
            fact_block = "\n".join(f"- {f}" for f in ctx.long_term_facts)
            system_parts.append(f"\n[用户相关信息]\n{fact_block}")

        if ctx.contexts:
            context_block = "\n".join(f"- {c}" for c in ctx.contexts)
            system_parts.append(f"\n[参考资料]\n{context_block}")

        # 注入中期记忆（本次会话摘要）
        if self.medium_term_memory:
            summary = self.medium_term_memory.get_summary()
            if summary:
                system_parts.append(f"\n[本次会话摘要]\n{summary}")

        if ctx.tool_results:
            tool_block = "\n".join(
                f"- [{r['source']}] {r['content']}" for r in ctx.tool_results
            )
            system_parts.append(f"\n[工具结果]\n{tool_block}")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": "\n".join(system_parts)},
        ]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": question})
        return messages

    def _generate_answer(
        self,
        question: str,
        history: list[dict[str, str]] | None,
        ctx: AgenticContext,
    ) -> str:
        """生成答案；大模型失败时返回基于证据的兜底回答。"""
        messages = self._build_messages(question, history, ctx)
        try:
            return self.llm_client.generate(messages, temperature=self.temperature)
        except Exception as exc:
            logger.warning("Agentic LLM generation failed: %s", exc)
            return self._fallback_answer(question, ctx)

    async def _agenerate_answer(
        self,
        question: str,
        history: list[dict[str, str]] | None,
        ctx: AgenticContext,
    ) -> str:
        """异步生成答案；大模型失败时返回基于证据的兜底回答。"""
        messages = self._build_messages(question, history, ctx)
        try:
            return await self.llm_client.agenerate(messages, temperature=self.temperature)
        except Exception as exc:
            logger.warning("Agentic async LLM generation failed: %s", exc)
            return self._fallback_answer(question, ctx)

    def _fallback_answer(self, question: str, ctx: AgenticContext) -> str:
        """当大模型不可用时，基于已有证据给出兜底回答。"""
        if ctx.contexts:
            summary = "；".join(ctx.contexts[:2])
            return f"基于参考资料：{summary}。"
        if ctx.tool_results:
            return "；".join(r["content"] for r in ctx.tool_results[:2])
        return "当前无法生成回答，请稍后再试。"

    def _kb_results_for_contexts(
        self, contexts: list[str]
    ) -> list[RetrievalResult]:
        """将纯文本上下文列表转换为轻量的 RetrievalResult 对象列表。"""
        from rag_agent.knowledge.base import Chunk

        return [
            RetrievalResult(
                chunk=Chunk(id=f"ctx-{i}", text=text, doc_id=""),
                score=1.0,
            )
            for i, text in enumerate(contexts)
        ]

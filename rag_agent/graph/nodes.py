"""Node functions for the agentic LangGraph workflow.

Each function reads from ``GraphState`` and returns a partial dict of fields
to merge into the state. Dependencies (knowledge base, LLM, memory, etc.)
are injected via closure so the nodes stay pure.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langgraph.graph import END

from rag_agent.agentic.base import ToolResult
from rag_agent.agentic.router import RouteDecision, RuleBasedRouter
from rag_agent.graph.state import GraphState

logger = logging.getLogger(__name__)

# -- Node factory ------------------------------------------------------------


def make_nodes(
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
) -> dict[str, Callable[[GraphState], dict[str, Any]]]:
    """Create a dictionary of named node functions.

    Each node closes over the provided dependencies and can be added to a
    LangGraph ``StateGraph``.
    """
    _router = router or RuleBasedRouter()
    _tools = tools or {}
    _top_k = top_k
    _temperature = temperature
    _system_prompt = system_prompt

    def input_guardrail_node(state: GraphState) -> dict[str, Any]:
        """Check input safety: prompt injection + PII detection."""
        if guardrails is None:
            return {}
        question = state.get("question", "")
        result = guardrails.check_input(question)
        if result.blocked:
            logger.warning(
                "输入护栏拦截 blocked_by=%s", result.blocked_by
            )
            return {
                "guardrail_blocked": True,
                "guardrail_message": result.blocked_by,
                "answer": "抱歉，您的请求包含不安全内容，无法处理。",
            }
        return {"guardrail_blocked": False}

    def cache_lookup_node(state: GraphState) -> dict[str, Any]:
        """Check semantic cache for a similar previous query."""
        if semantic_cache is None:
            return {}
        question = state.get("question", "")
        user_id = state.get("user_id", "")
        cached = semantic_cache.lookup(question, user_id)
        if cached is None:
            return {"cache_hit": False}
        logger.info(
            "语义缓存命中 user=%s query=%r", user_id, question[:50]
        )
        return {
            "cache_hit": True,
            "cached_answer": cached.get("answer", ""),
            "answer": cached.get("answer", ""),
            "contexts": cached.get("contexts", []),
            "long_term_facts": cached.get("long_term_facts", []),
            "evaluation": cached.get("evaluation"),
            "search_query": cached.get("cached_query", question),
        }

    def route_node(state: GraphState) -> dict[str, Any]:
        """Decide which data sources / tools are needed."""
        question = state.get("question", "")
        history = state.get("chat_history")
        decision: RouteDecision = _router.route(question, history)
        return {
            "use_knowledge_base": decision.use_knowledge_base,
            "use_long_term_memory": decision.use_long_term_memory,
            "use_calculator": decision.use_calculator,
            "use_datetime": decision.use_datetime,
            "use_web_search": decision.use_web_search,
            "route_reasoning": decision.reasoning,
        }

    def transform_query_node(state: GraphState) -> dict[str, Any]:
        """Rewrite the question for better retrieval."""
        question = state.get("question", "")
        if query_transformer is not None:
            try:
                history = state.get("chat_history")
                transformed = query_transformer.transform(question, history)
                search_query = transformed[0] if transformed else question
            except Exception as exc:
                logger.warning("Query transform failed: %s", exc)
                search_query = question
        else:
            search_query = question
        return {"search_query": search_query}

    def retrieve_node(state: GraphState) -> dict[str, Any]:
        """Collect evidence from knowledge base, memory, and tools."""
        query = state.get("search_query") or state.get("question", "")
        user_id = state.get("user_id", "")
        contexts: list[str] = []
        facts: list[str] = []
        tool_results: list[dict[str, Any]] = []

        # Long-term memory
        if state.get("use_long_term_memory") and long_term_memory is not None:
            try:
                recalled = long_term_memory.recall(user_id, query, top_k=3)
                facts = [f.content for f in recalled]
            except Exception as exc:
                logger.warning("LTM recall failed: %s", exc)

        # Knowledge base
        if state.get("use_knowledge_base", True):
            try:
                results = knowledge_base.hybrid_search(query, top_k=_top_k)
                contexts = [r.text for r in results]
            except Exception as exc:
                logger.warning("KB retrieval failed: %s", exc)

        # Tools
        for source in ("calculator", "datetime", "web_search"):
            if state.get(f"use_{source}") and source in _tools:
                try:
                    result = _tools[source].invoke(query)
                    tool_results.append(
                        ToolResult(source=source, content=result).to_dict()
                    )
                except Exception as exc:
                    logger.warning("Tool %s failed: %s", source, exc)

        return {
            "contexts": contexts,
            "long_term_facts": facts,
            "tool_results": tool_results,
        }

    def generate_node(state: GraphState) -> dict[str, Any]:
        """Generate an answer using evidence and tools results."""
        question = state.get("question", "")
        answer = state.get("answer", "")
        # Skip generation if we already have a cached answer.
        if state.get("cache_hit") and answer:
            return {}

        messages = _build_messages(
            system_prompt=_system_prompt,
            question=question,
            history=state.get("chat_history"),
            contexts=state.get("contexts", []),
            long_term_facts=state.get("long_term_facts", []),
            tool_results=state.get("tool_results", []),
        )

        try:
            answer = llm_client.generate(messages, temperature=_temperature)
        except Exception as exc:
            logger.warning("LLM generation failed: %s", exc)
            answer = _fallback_answer(state)

        return {"answer": answer}

    def self_correction_node(state: GraphState) -> dict[str, Any]:
        """Evaluate answer quality and decide if correction is needed."""
        if self_corrector is None:
            return {"needs_correction": False, "correction_message": "no corrector"}

        iteration = state.get("iteration", 0)
        correction = self_corrector.check(
            question=state.get("question", ""),
            answer=state.get("answer", ""),
            contexts=state.get("contexts", []),
            iteration=iteration,
        )
        return {
            "correction_score": correction.score,
            "needs_correction": correction.needs_correction,
            "correction_message": correction.reasoning,
            "search_query": correction.rewritten_query or state.get("search_query", ""),
            "iteration": iteration + 1,
        }

    def output_guardrail_node(state: GraphState) -> dict[str, Any]:
        """Check output for toxic/harmful content."""
        if guardrails is None:
            return {}
        answer = state.get("answer", "")
        if not answer:
            return {}
        result = guardrails.check_output(answer)
        if result.blocked:
            logger.warning("输出护栏拦截 output_toxicity")
            return {"answer": "抱歉，生成的回答包含不当内容，已被替换。"}
        return {}

    def remember_node(state: GraphState) -> dict[str, Any]:
        """Extract and store long-term memory facts from this turn."""
        if long_term_memory is None:
            return {}
        question = state.get("question", "")
        answer = state.get("answer", "")
        user_id = state.get("user_id", "")
        extractor = memory_extractor
        if extractor is None:
            from rag_agent.memory.extractor import RuleBasedMemoryExtractor
            extractor = RuleBasedMemoryExtractor()
        try:
            facts = extractor.extract(question, answer)
            for fact in facts:
                long_term_memory.remember(user_id, fact)
            if facts:
                logger.debug("长期记忆存储 %d 条事实", len(facts))
        except Exception as exc:
            logger.warning("长期记忆提取失败: %s", exc)
        return {}

    def evaluate_node(state: GraphState) -> dict[str, Any]:
        """Score answer quality and persist the evaluation result."""
        if evaluator is None:
            return {}
        # Skip evaluation if we already have one from cache.
        if state.get("evaluation"):
            return {}
        question = state.get("question", "")
        answer = state.get("answer", "")
        contexts = state.get("contexts", [])
        try:
            result = evaluator.evaluate(question, answer, contexts)
            return {"evaluation": result}
        except Exception as exc:
            logger.warning("评估失败: %s", exc)
            return {}

    return {
        "input_guardrail": input_guardrail_node,
        "cache_lookup": cache_lookup_node,
        "route": route_node,
        "transform_query": transform_query_node,
        "retrieve": retrieve_node,
        "generate": generate_node,
        "output_guardrail": output_guardrail_node,
        "self_correction": self_correction_node,
        "remember": remember_node,
        "evaluate": evaluate_node,
    }


# -- Routing helpers ---------------------------------------------------------

def decide_after_route(state: GraphState) -> str:
    """Conditional edge: after routing, go to retrieve (skip if no sources)."""
    # Always attempt retrieve; the retrieve node handles empty results gracefully.
    return "retrieve"


def decide_after_correction(state: GraphState) -> str:
    """Conditional edge: after self-correction, loop back or proceed to remember."""
    max_iterations = 2
    iteration = state.get("iteration", 0)
    needs_correction = state.get("needs_correction", False)

    if needs_correction and iteration < max_iterations:
        return "retrieve"
    return "remember"


def decide_after_input_guardrail(state: GraphState) -> str:
    """Conditional edge: after input guardrail, end if blocked."""
    if state.get("guardrail_blocked"):
        return END
    return "cache_lookup"


def decide_after_cache(state: GraphState) -> str:
    """Conditional edge: after cache lookup, end if hit."""
    if state.get("cache_hit"):
        return END
    return "route"


# -- Internal helpers --------------------------------------------------------

def _build_messages(
    system_prompt: str,
    question: str,
    history: list[dict[str, str]] | None,
    contexts: list[str],
    long_term_facts: list[str],
    tool_results: list[dict[str, Any]],
) -> list[dict[str, str]]:
    parts = [system_prompt]

    if long_term_facts:
        parts.append("\n[用户相关信息]\n" + "\n".join(f"- {f}" for f in long_term_facts))
    if contexts:
        parts.append("\n[参考资料]\n" + "\n".join(f"- {c}" for c in contexts))
    if tool_results:
        parts.append("\n[工具结果]\n" + "\n".join(
            f"- [{r['source']}] {r['content']}" for r in tool_results
        ))

    messages: list[dict[str, str]] = [{"role": "system", "content": "\n".join(parts)}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})
    return messages


def _fallback_answer(state: GraphState) -> str:
    contexts = state.get("contexts", [])
    tool_results = state.get("tool_results", [])
    if contexts:
        return "基于参考资料：" + "；".join(contexts[:2])
    if tool_results:
        return "；".join(r["content"] for r in tool_results[:2])
    return "当前无法生成回答，请稍后再试。"

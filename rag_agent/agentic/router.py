"""Query routing: decide which data sources / tools a question needs."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_agent.llm import BaseLLMClient

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    """Decision produced by a query router."""

    use_knowledge_base: bool = True
    use_long_term_memory: bool = False
    use_calculator: bool = False
    use_datetime: bool = False
    use_web_search: bool = False
    reasoning: str = ""

    @property
    def selected_sources(self) -> list[str]:
        """Return human-readable list of selected sources."""
        sources = []
        if self.use_knowledge_base:
            sources.append("knowledge_base")
        if self.use_long_term_memory:
            sources.append("long_term_memory")
        if self.use_calculator:
            sources.append("calculator")
        if self.use_datetime:
            sources.append("datetime")
        if self.use_web_search:
            sources.append("web_search")
        return sources


class QueryRouter(ABC):
    """Decide which sources/tools should be used for a question."""

    @abstractmethod
    def route(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> RouteDecision:
        ...


class RuleBasedRouter(QueryRouter):
    """Route queries using lightweight heuristics.

    No LLM is required, so it is fast and deterministic.
    """

    _MATH_OPERATORS = ("+", "-", "*", "/", "^", "=", "等于", "多少", "计算")
    _DATETIME_KEYWORDS = (
        "现在时间",
        "当前时间",
        "今天日期",
        "今天几号",
        "星期几",
        "几点",
        "时间",
        "日期",
    )
    _PERSONAL_KEYWORDS = (
        "我喜欢",
        "我讨厌",
        "我的",
        "我记得",
        "我上次",
        "个人信息",
        "我叫",
        "我是",
    )

    def route(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> RouteDecision:
        q = question.strip()
        lower = q.lower()

        # 1. Calculator: explicit arithmetic or quantity questions
        if self._looks_like_math(q, lower):
            return RouteDecision(
                use_calculator=True,
                use_knowledge_base=False,
                reasoning="Question asks for a numeric calculation.",
            )

        # 2. Datetime: current time / date
        if any(k in q for k in self._DATETIME_KEYWORDS):
            return RouteDecision(
                use_datetime=True,
                use_knowledge_base=False,
                reasoning="Question asks for current time or date.",
            )

        # 3. Personal facts: prefer long-term memory, optionally KB
        if any(k in q for k in self._PERSONAL_KEYWORDS):
            return RouteDecision(
                use_long_term_memory=True,
                use_knowledge_base=True,
                reasoning="Question refers to personal preferences or history.",
            )

        # 4. Default: knowledge base
        return RouteDecision(
            use_knowledge_base=True,
            reasoning="Default route to knowledge base.",
        )

    def _looks_like_math(self, q: str, lower: str) -> bool:
        has_operator = any(op in q for op in ("+", "-", "*", "/", "^"))
        has_numbers = bool(re.search(r"\d", q))
        has_math_words = any(k in lower for k in ("等于", "多少", "计算", "结果是"))
        # Require both numbers and an operator/math word to avoid false positives
        return has_numbers and (has_operator or has_math_words)


class LLMQueryRouter(QueryRouter):
    """Route queries by asking an LLM to produce a JSON routing decision."""

    DEFAULT_PROMPT = """You are a query router for a RAG Agent.
Given the user question below, decide which data sources or tools are needed.

Available sources:
- knowledge_base: general domain knowledge
- long_term_memory: personal facts about the user
- calculator: arithmetic / math questions
- datetime: current time or date
- web_search: real-time information (not implemented yet, keep False)

Return ONLY a JSON object in this exact format:
{
  "use_knowledge_base": true,
  "use_long_term_memory": false,
  "use_calculator": false,
  "use_datetime": false,
  "use_web_search": false,
  "reasoning": "short reason"
}

Question: {question}"""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt: str | None = None,
    ):
        self.llm_client = llm_client
        self.prompt = prompt or self.DEFAULT_PROMPT

    def route(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> RouteDecision:
        try:
            response = self.llm_client.generate(
                messages=[
                    {"role": "user", "content": self.prompt.format(question=question)}
                ],
                temperature=0.0,
                max_tokens=200,
            )
            data = json.loads(self._extract_json(response))
            return RouteDecision(
                use_knowledge_base=bool(data.get("use_knowledge_base", True)),
                use_long_term_memory=bool(data.get("use_long_term_memory", False)),
                use_calculator=bool(data.get("use_calculator", False)),
                use_datetime=bool(data.get("use_datetime", False)),
                use_web_search=bool(data.get("use_web_search", False)),
                reasoning=str(data.get("reasoning", "")),
            )
        except Exception as exc:
            logger.warning("LLM query routing failed (%s). Falling back to rule router.", exc)
            return RuleBasedRouter().route(question, history)

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        return text

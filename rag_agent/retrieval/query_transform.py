"""Query transformation strategies for improving retrieval recall."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from rag_agent.llm import BaseLLMClient

if TYPE_CHECKING:
    from rag_agent.memory.short_term import Message

logger = logging.getLogger(__name__)


class QueryTransformer(ABC):
    """Transform a user query into one or more search-friendly queries."""

    @abstractmethod
    async def atransform(
        self,
        query: str,
        history: list[Message] | None = None,
    ) -> list[str]:
        """Async transformation. Returns a list of queries to search with."""
        ...

    def transform(
        self,
        query: str,
        history: list[Message] | None = None,
    ) -> list[str]:
        """Synchronous wrapper. Subclasses should override this for sync use."""
        try:
            return asyncio.run(self.atransform(query, history))
        except Exception as exc:
            logger.warning("Query transformation failed: %s", exc)
            return [query]


class IdentityTransformer(QueryTransformer):
    """No-op transformer: returns the original query as-is."""

    async def atransform(
        self,
        query: str,
        history: list[Message] | None = None,
    ) -> list[str]:
        return [query]

    def transform(
        self,
        query: str,
        history: list[Message] | None = None,
    ) -> list[str]:
        return [query]


class RewritingTransformer(QueryTransformer):
    """Rewrite the query to resolve pronouns and colloquialisms.

    Uses the conversation history (if available) to resolve references like
    "it", "that", "this" into concrete entities before retrieval.
    """

    DEFAULT_PROMPT = """请把以下用户问题改写成一个适合向量检索的标准问题。

要求：
1. 消除指代词（如“它”、“那个”、“这个”），用上文提到的具体概念替换。
2. 补充必要的上下文，但只基于对话历史，不要添加文档中没有的信息。
3. 保持原意，不要扩展问题范围。
4. 只返回改写后的问题，不要解释、不要输出多余内容。

{history}
当前问题：{query}

改写后的问题："""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt: str | None = None,
        max_history_turns: int = 3,
    ):
        self.llm_client = llm_client
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.max_history_turns = max_history_turns

    async def atransform(
        self,
        query: str,
        history: list[Message] | None = None,
    ) -> list[str]:
        history_text = self._format_history(history or [])
        prompt = self.prompt.format(history=history_text, query=query)
        try:
            rewritten = await self.llm_client.agenerate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            rewritten = rewritten.strip().strip('"').strip("'")
            if rewritten and rewritten != query:
                logger.info("Query rewritten: %r -> %r", query, rewritten)
                return [rewritten]
        except Exception as exc:
            logger.warning("Query rewriting failed: %s", exc)
        return [query]

    def transform(
        self,
        query: str,
        history: list[Message] | None = None,
    ) -> list[str]:
        """Synchronous wrapper using thread pool for the async LLM call."""
        try:
            return asyncio.run(self.atransform(query, history))
        except Exception as exc:
            logger.warning("Query transformation failed: %s", exc)
            return [query]

    def _format_history(self, history: list[Message]) -> str:
        if not history:
            return ""
        # Keep only the most recent complete turns to control token usage.
        turns: list[list[Message]] = []
        current_turn: list[Message] = []
        for msg in history:
            if msg.role == "user":
                if current_turn:
                    turns.append(current_turn)
                current_turn = [msg]
            else:
                current_turn.append(msg)
        if current_turn:
            turns.append(current_turn)

        kept_turns = turns[-self.max_history_turns :]
        lines = ["对话历史："]
        for turn in kept_turns:
            for msg in turn:
                role_label = "User" if msg.role == "user" else "Assistant"
                lines.append(f"{role_label}: {msg.content}")
        return "\n".join(lines)

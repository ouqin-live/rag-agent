"""Extract memorable facts from conversation turns."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Iterable

from rag_agent.llm import BaseLLMClient


class MemoryExtractor(ABC):
    """Extract facts about the user from a conversation turn."""

    @abstractmethod
    def extract(self, user_message: str, assistant_message: str) -> list[str]:
        """Return a list of concise fact strings."""
        ...


class RuleBasedMemoryExtractor(MemoryExtractor):
    """Rule-based extractor for Chinese and English preference/fact statements."""

    # Patterns that indicate a user preference, identity, or constraint.
    PATTERNS = [
        # Chinese
        r"我(?:喜欢|爱|偏好|习惯|常用|主要用|想用|要用)(.+?)[。！？;；,，]",
        r"我(?:是|从事|做|担任)(.+?)[。！？;；,，]",
        r"(?:请)(?:用|以|按照)(.+?)(?:回答|回复|生成|解释)",
        r"(?:不要)(?:用|以|按照)(.+?)(?:回答|回复|生成|解释)",
        r"(?:回答|回复|生成|解释)时(?:请|要|记得)(.+?)[。！？;；]",
        r"我(?:不懂|不熟悉|不了解|没学过)(.+?)[。！？;；]",
        r"我(?:熟悉|擅长|精通|了解)(.+?)[。！？;；]",
        # English
        r"I (?:like|love|prefer|usually use|want to use|mainly use|work with) (.+?)[.!?;,]",
        r"I am (?:a|an) (.+?)[.!?;,]",
        r"Please (?:answer|reply|respond|explain) (?:in|using|with) (.+?)[.!?;]",
        r"Do not (?:use|answer|reply) (?:in|using|with) (.+?)[.!?;]",
        r"I (?:don't know|am not familiar with|am not good at) (.+?)[.!?;]",
        r"I (?:know|am familiar with|am good at) (.+?)[.!?;]",
    ]

    def __init__(self, min_length: int = 3, max_length: int = 120):
        self.min_length = min_length
        self.max_length = max_length
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.PATTERNS]

    def extract(self, user_message: str, assistant_message: str) -> list[str]:
        facts: list[str] = []
        text = f"{user_message} {assistant_message}"
        for pattern in self._compiled:
            for match in pattern.finditer(text):
                fact = match.group(0).strip()
                if self._is_valid(fact):
                    facts.append(fact)
        return self._deduplicate(facts)

    def _is_valid(self, fact: str) -> bool:
        if len(fact) < self.min_length or len(fact) > self.max_length:
            return False
        # Skip pure questions and commands
        if fact.endswith(("?", "？")):
            return False
        # Skip sentences that look like the assistant agreeing
        if fact.lower().startswith(("好的", "明白", "收到", "sure", "ok", "okay")):
            return False
        return True

    @staticmethod
    def _deduplicate(facts: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for fact in facts:
            key = fact.lower().replace(" ", "")
            if key not in seen:
                seen.add(key)
                result.append(fact)
        return result


class LLMMemoryExtractor(MemoryExtractor):
    """Use LLM to extract user facts, falling back to rule-based on failure."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self._fallback = RuleBasedMemoryExtractor()

    def extract(self, user_message: str, assistant_message: str) -> list[str]:
        prompt = f"""从以下对话中提取用户的关键事实和偏好（如语言偏好、职业、技术栈等）。
每条事实用一句简短的话表达。只返回 JSON 数组，不要返回其他内容。
若无事实返回空数组：[]

User: {user_message}
Assistant: {assistant_message}

JSON: """
        try:
            response = self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            facts = json.loads(self._extract_json(response))
            if isinstance(facts, list) and all(isinstance(f, str) for f in facts):
                return [f.strip() for f in facts if f.strip()]
        except Exception:
            pass
        return self._fallback.extract(user_message, assistant_message)

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        return text

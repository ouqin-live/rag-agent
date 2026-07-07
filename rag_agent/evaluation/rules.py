"""Rule-based checks for generated answers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


class RuleResult:
    """Result of a single rule check."""

    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message


class RuleChecker(ABC):
    """Check an answer against a set of rules."""

    @abstractmethod
    def check(
        self, question: str, answer: str, contexts: list[str]
    ) -> list[RuleResult]:
        ...


class DefaultRuleChecker(RuleChecker):
    """Default set of safety and quality rules."""

    def __init__(self, min_length: int = 5, max_length: int = 2000):
        self.min_length = min_length
        self.max_length = max_length
        self.sensitive_words: list[str] = []

    def check(
        self, question: str, answer: str, contexts: list[str]
    ) -> list[RuleResult]:
        results: list[RuleResult] = []
        results.append(self._check_not_empty(answer))
        results.append(self._check_length(answer))
        results.append(self._check_refusal(answer))
        results.append(self._check_sensitive(answer))
        results.append(self._check_hallucination(answer, contexts))
        return results

    def _check_not_empty(self, answer: str) -> RuleResult:
        passed = bool(answer.strip())
        return RuleResult(
            name="not_empty",
            passed=passed,
            message="" if passed else "回答为空",
        )

    def _check_length(self, answer: str) -> RuleResult:
        length = len(answer)
        passed = self.min_length <= length <= self.max_length
        return RuleResult(
            name="length_ok",
            passed=passed,
            message="" if passed else f"回答长度 {length} 不在 [{self.min_length}, {self.max_length}] 范围内",
        )

    def _check_refusal(self, answer: str) -> RuleResult:
        patterns = [
            r"我不知道",
            r"无法回答",
            r"没有相关信息",
            r"I don't know",
            r"I cannot answer",
        ]
        combined = re.compile("|".join(patterns), re.IGNORECASE)
        passed = not combined.search(answer)
        return RuleResult(
            name="no_refusal",
            passed=passed,
            message="" if passed else "回答包含拒绝表述",
        )

    def _check_sensitive(self, answer: str) -> RuleResult:
        if not self.sensitive_words:
            return RuleResult(name="no_sensitive", passed=True)
        found = [w for w in self.sensitive_words if w in answer]
        passed = not found
        return RuleResult(
            name="no_sensitive",
            passed=passed,
            message="" if passed else f"回答包含敏感词: {found}",
        )

    def _check_hallucination(self, answer: str, contexts: list[str]) -> RuleResult:
        """Simple heuristic: flag if answer contains digits/entities not in contexts."""
        if not contexts:
            return RuleResult(name="no_obvious_hallucination", passed=True)
        answer_numbers = set(re.findall(r"\d+", answer))
        context_numbers: set[str] = set()
        for ctx in contexts:
            context_numbers |= set(re.findall(r"\d+", ctx))
        extra = answer_numbers - context_numbers
        passed = not extra
        return RuleResult(
            name="no_obvious_hallucination",
            passed=passed,
            message="" if passed else f"回答包含上下文未出现的数字: {extra}",
        )

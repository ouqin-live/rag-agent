"""Base abstractions for evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class EvaluationResult:
    """Result of evaluating a single Q&A pair."""

    question: str
    answer: str
    contexts: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    passed_rules: list[str] = field(default_factory=list)
    failed_rules: list[str] = field(default_factory=list)
    overall_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_failure(self, threshold: float = 0.6) -> bool:
        return self.overall_score < threshold


class BaseMetric(ABC):
    """A metric that scores a generated answer."""

    name: str = "base"

    @abstractmethod
    def score(self, question: str, answer: str, contexts: list[str]) -> float:
        """Return a score between 0.0 and 1.0."""
        ...

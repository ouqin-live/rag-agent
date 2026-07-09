"""Base abstractions for the agentic RAG module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rag_agent.evaluation.base import EvaluationResult


@dataclass
class AgenticContext:
    """Mutable state for a single agentic turn.

    Carries the question, retrieved evidence, generated answer, and loop
    control flags across ReAct / self-correction iterations.
    """

    question: str
    search_query: str = ""
    contexts: list[str] = field(default_factory=list)
    long_term_facts: list[str] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    iteration: int = 0
    max_iterations: int = 2
    evaluation: EvaluationResult | None = None
    needs_correction: bool = False
    reasoning: str = ""

    def reset_evidence(self) -> None:
        """Clear evidence collected in the previous iteration."""
        self.contexts = []
        self.long_term_facts = []
        self.tool_results = []


@dataclass
class ToolResult:
    """Result produced by a tool invocation."""

    source: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "content": self.content}


class BaseTool(ABC):
    """Abstract tool that can be invoked by the agentic loop."""

    name: str = "base_tool"

    @abstractmethod
    def invoke(self, query: str) -> str:
        """Execute the tool and return a text result."""
        ...

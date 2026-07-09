"""Self-correction loop: detect low-quality answers and rewrite queries."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_agent.evaluation.base import BaseMetric, EvaluationResult
from rag_agent.evaluation.metrics import FaithfulnessMetric

if TYPE_CHECKING:
    from rag_agent.evaluation import Evaluator
    from rag_agent.llm import BaseLLMClient

logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """Outcome of a self-correction check."""

    needs_correction: bool
    score: float
    rewritten_query: str = ""
    reasoning: str = ""


class SelfCorrector:
    """Decide whether an answer needs correction and produce a better query.

    The corrector uses a fast faithfulness metric by default. When the score is
    below ``threshold``, it asks the LLM to rewrite the search query so that a
    supplementary retrieval can be performed.
    """

    DEFAULT_REWRITE_PROMPT = """The answer below may be poorly supported by the retrieved contexts.
Please rewrite the original question into a focused search query that retrieves evidence missing from the current answer.

Original question: {question}
Current answer: {answer}
Retrieved contexts:
{contexts}

Return ONLY the rewritten search query, no explanation."""

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        metric: BaseMetric | None = None,
        evaluator: Evaluator | None = None,
        threshold: float = 0.5,
        max_iterations: int = 2,
        rewrite_prompt: str | None = None,
    ):
        self.llm_client = llm_client
        # 优先使用传入的 metric；其次尝试从 evaluator 中提取 faithfulness 指标；
        # 最后使用默认的 FaithfulnessMetric。
        self.metric = metric or self._extract_faithfulness_metric(evaluator) or FaithfulnessMetric(llm=llm_client)
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.rewrite_prompt = rewrite_prompt or self.DEFAULT_REWRITE_PROMPT

    @staticmethod
    def _extract_faithfulness_metric(evaluator: Evaluator | None) -> BaseMetric | None:
        """从 evaluator 中提取名为 faithfulness 的指标（如果存在）。"""
        if evaluator is None:
            return None
        for metric in evaluator.metrics:
            if metric.name == "faithfulness":
                return metric
        return None

    def check(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        iteration: int,
    ) -> CorrectionResult:
        """Return whether the answer needs correction and a rewritten query."""
        if iteration >= self.max_iterations:
            return CorrectionResult(
                needs_correction=False,
                score=0.0,
                reasoning="Reached maximum self-correction iterations.",
            )

        score = self.metric.score(question, answer, contexts)
        if score >= self.threshold:
            return CorrectionResult(
                needs_correction=False,
                score=score,
                reasoning=f"Faithfulness score {score:.2f} meets threshold.",
            )

        rewritten = self._rewrite_query(question, answer, contexts)
        return CorrectionResult(
            needs_correction=True,
            score=score,
            rewritten_query=rewritten,
            reasoning=f"Faithfulness score {score:.2f} below threshold {self.threshold:.2f}; rewriting query.",
        )

    def _rewrite_query(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> str:
        if self.llm_client is None:
            return question

        context_text = "\n\n".join(f"- {c}" for c in contexts)
        prompt = self.rewrite_prompt.format(
            question=question,
            answer=answer,
            contexts=context_text or "（无上下文）",
        )
        try:
            rewritten = self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            rewritten = rewritten.strip().strip('"').strip("'")
            if rewritten and rewritten != question:
                logger.info("Rewriting query for correction: %r -> %r", question, rewritten)
                return rewritten
        except Exception as exc:
            logger.warning("Query rewrite failed: %s", exc)
        return question

    def create_evaluation(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        score: float,
    ) -> EvaluationResult:
        """Create a lightweight evaluation result from the self-correction score."""
        return EvaluationResult(
            question=question,
            answer=answer,
            contexts=list(contexts),
            scores={"faithfulness": score},
            overall_score=score,
        )

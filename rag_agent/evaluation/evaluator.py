"""Evaluator that combines metrics, rules, and persistence."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from rag_agent.evaluation.base import BaseMetric, EvaluationResult
from rag_agent.evaluation.metrics import (
    AnswerRelevanceMetric,
    ContextPrecisionMetric,
    FaithfulnessMetric,
)
from rag_agent.evaluation.rules import DefaultRuleChecker, RuleChecker
from rag_agent.llm import BaseLLMClient, MockLLMClient

logger = logging.getLogger(__name__)


class Evaluator:
    """Run a configured set of metrics and rules and persist the results."""

    def __init__(
        self,
        metrics: list[BaseMetric] | None = None,
        rules: RuleChecker | None = None,
        db_path: str | Path | None = None,
        failure_threshold: float | None = None,
    ):
        from rag_agent.config import get_settings

        settings = get_settings()
        failure_threshold = (
            failure_threshold if failure_threshold is not None else settings.eval_failure_threshold
        )
        self.metrics = metrics or [
            FaithfulnessMetric(),
            AnswerRelevanceMetric(),
            ContextPrecisionMetric(),
        ]
        self.rules = rules or DefaultRuleChecker(
            min_length=settings.eval_answer_min_length,
            max_length=settings.eval_answer_max_length,
        )
        self.db_path = Path(db_path) if db_path else Path("data/eval/evaluations.db")
        self.failure_threshold = failure_threshold

        # 规则严重程度权重：值越大扣分越多
        self._rule_severity: dict[str, float] = {
            "not_empty": 0.5,
            "length_ok": 0.05,
            "no_refusal": 0.3,
            "no_sensitive": 0.5,
            "no_obvious_hallucination": 0.3,
        }
        self._init_db()

    @classmethod
    def with_llm(
        cls,
        llm: BaseLLMClient,
        db_path: str | Path | None = None,
        failure_threshold: float = 0.6,
    ) -> "Evaluator":
        """Convenience constructor that wires LLM into all metrics."""
        metrics = [
            FaithfulnessMetric(llm=llm),
            AnswerRelevanceMetric(llm=llm),
            ContextPrecisionMetric(llm=llm),
        ]
        return cls(
            metrics=metrics,
            db_path=db_path,
            failure_threshold=failure_threshold,
        )

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    contexts TEXT,
                    scores TEXT,
                    overall_score REAL,
                    passed_rules TEXT,
                    failed_rules TEXT,
                    is_failure BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_evaluations_failure
                    ON evaluations(is_failure, created_at);
                """
            )

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> EvaluationResult:
        """Evaluate a single Q&A pair and persist the result."""
        scores: dict[str, float] = {}
        for metric in self.metrics:
            try:
                scores[metric.name] = metric.score(question, answer, contexts)
            except Exception as exc:
                logger.warning("Metric %s failed: %s", metric.name, exc)
                scores[metric.name] = 0.0

        rule_results = self.rules.check(question, answer, contexts)
        passed = [r.name for r in rule_results if r.passed]
        failed = [
            f"{r.name}: {r.message}" for r in rule_results if not r.passed and r.message
        ]

        overall = self._compute_overall(scores, rule_results)
        result = EvaluationResult(
            question=question,
            answer=answer,
            contexts=list(contexts),
            scores=scores,
            passed_rules=passed,
            failed_rules=failed,
            overall_score=overall,
        )
        self._persist(result)
        return result

    def _compute_overall(
        self,
        scores: dict[str, float],
        rule_results: list,  # list[RuleResult]
    ) -> float:
        """计算综合分：指标平均分 - 按严重程度加权的规则扣分。"""
        if not scores:
            return 0.0
        avg_score = sum(scores.values()) / len(scores)
        # 按严重程度累加扣分
        penalty = sum(
            self._rule_severity.get(r.name, 0.0)
            for r in rule_results
            if not r.passed
        )
        return max(0.0, avg_score - penalty)

    def _persist(self, result: EvaluationResult) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO evaluations
                (question, answer, contexts, scores, overall_score,
                 passed_rules, failed_rules, is_failure)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.question,
                    result.answer,
                    json.dumps(result.contexts, ensure_ascii=False),
                    json.dumps(result.scores, ensure_ascii=False),
                    result.overall_score,
                    json.dumps(result.passed_rules, ensure_ascii=False),
                    json.dumps(result.failed_rules, ensure_ascii=False),
                    int(result.overall_score < self.failure_threshold),
                ),
            )

    def get_recent_evaluations(
        self, limit: int = 20
    ) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM evaluations ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

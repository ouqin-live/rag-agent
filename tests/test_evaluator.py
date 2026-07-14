"""Tests for the evaluator and rule checker."""

from __future__ import annotations

from pathlib import Path

from rag_agent.evaluation.evaluator import Evaluator
from rag_agent.evaluation.rules import DefaultRuleChecker


def test_default_rule_checker_flags_empty_answer() -> None:
    checker = DefaultRuleChecker(min_length=5, max_length=2000)
    results = checker.check("What is RAG?", "", ["RAG is retrieval augmented generation."])

    result_map = {r.name: r for r in results}
    assert result_map["not_empty"].passed is False
    assert result_map["length_ok"].passed is False


def test_default_rule_checker_passes_valid_answer() -> None:
    checker = DefaultRuleChecker(min_length=5, max_length=2000)
    answer = "RAG stands for retrieval augmented generation."
    results = checker.check("What is RAG?", answer, [answer])

    assert all(r.passed for r in results)


def test_default_rule_checker_flags_refusal() -> None:
    checker = DefaultRuleChecker()
    results = checker.check("What is X?", "我不知道", ["X is Y."])

    result_map = {r.name: r for r in results}
    assert result_map["no_refusal"].passed is False


def test_default_rule_checker_flags_hallucinated_numbers() -> None:
    checker = DefaultRuleChecker()
    answer = "The system has 42 servers."
    contexts = ["The system has 10 servers."]
    results = checker.check("How many servers?", answer, contexts)

    result_map = {r.name: r for r in results}
    assert result_map["no_obvious_hallucination"].passed is False


def test_evaluator_records_results(temp_dir: Path) -> None:
    db_path = temp_dir / "eval.db"
    evaluator = Evaluator(
        metrics=[],
        rules=DefaultRuleChecker(),
        db_path=db_path,
        failure_threshold=0.6,
    )

    result = evaluator.evaluate(
        question="What is RAG?",
        answer="RAG is retrieval augmented generation.",
        contexts=["RAG is retrieval augmented generation."],
    )

    assert result.overall_score >= 0.0
    assert result.is_failure is False

    # The evaluation should be persisted and retrievable.
    records = evaluator.get_recent_evaluations(limit=10)
    assert len(records) == 1
    assert records[0]["question"] == "What is RAG?"

"""RAGAS-style metrics with LLM-based scoring and offline fallbacks."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Iterable

from rag_agent.evaluation.base import BaseMetric
from rag_agent.llm import BaseLLMClient

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for fallback overlap metrics."""
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    return [t for t in text.split() if t]


def _overlap_score(a: str, b: str) -> float:
    """Jaccard-style overlap between two texts."""
    tokens_a = set(_tokenize(a))
    tokens_b = set(_tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))


def _extract_number_like(text: str) -> set[str]:
    """Extract numbers and alphanumeric entities for faithfulness fallback."""
    numbers = set(re.findall(r"\d+", text))
    words = set(re.findall(r"\b[A-Za-z]{4,}\b", text))
    return numbers | words


class FaithfulnessMetric(BaseMetric):
    """Measures whether the answer can be inferred from the retrieved contexts."""

    name = "faithfulness"

    def __init__(self, llm: BaseLLMClient | None = None):
        self.llm = llm

    def score(self, question: str, answer: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0

        if self.llm is None:
            return self._fallback_score(answer, contexts)

        context_text = "\n\n".join(f"[{i}] {ctx}" for i, ctx in enumerate(contexts))
        prompt = f"""You are evaluating the faithfulness of an answer.
Given the following contexts and answer, determine whether each statement in the answer is supported by the contexts.

Contexts:
{context_text}

Answer:
{answer}

Return ONLY a JSON object in this exact format:
{{"supported": <int>, "unsupported": <int>}}
Do not include any other text."""

        try:
            response = self.llm.generate(
                messages=[
                    {"role": "system", "content": "You are a strict faithfulness evaluator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            result = json.loads(self._extract_json(response))
            supported = int(result.get("supported", 0))
            unsupported = int(result.get("unsupported", 0))
            total = supported + unsupported
            return supported / total if total > 0 else 0.0
        except Exception as exc:
            logger.warning("Faithfulness LLM evaluation failed (%s). Using fallback.", exc)
            return self._fallback_score(answer, contexts)

    def _fallback_score(self, answer: str, contexts: list[str]) -> float:
        """Fallback: check that answer numbers/entities appear in contexts."""
        answer_entities = _extract_number_like(answer)
        if not answer_entities:
            return 1.0
        context_entities = set()
        for ctx in contexts:
            context_entities |= _extract_number_like(ctx)
        supported = answer_entities & context_entities
        return len(supported) / len(answer_entities)

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        return text


class AnswerRelevanceMetric(BaseMetric):
    """Measures how relevant the answer is to the question."""

    name = "answer_relevance"

    def __init__(self, llm: BaseLLMClient | None = None):
        self.llm = llm

    def score(self, question: str, answer: str, contexts: list[str]) -> float:
        if self.llm is None:
            return _overlap_score(question, answer)

        prompt = f"""You are evaluating answer relevance.
Rate how well the following answer addresses the question on a scale of 0 to 10, where 10 means fully relevant and 0 means completely irrelevant.

Question: {question}
Answer: {answer}

Return ONLY a JSON object in this exact format:
{{"score": <int 0-10>}}
Do not include any other text."""

        try:
            response = self.llm.generate(
                messages=[
                    {"role": "system", "content": "You are a relevance evaluator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            result = json.loads(self._extract_json(response))
            raw = int(result.get("score", 0))
            return max(0.0, min(1.0, raw / 10.0))
        except Exception as exc:
            logger.warning("Answer relevance LLM evaluation failed (%s). Using fallback.", exc)
            return _overlap_score(question, answer)

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        return text


class ContextPrecisionMetric(BaseMetric):
    """Measures the proportion of retrieved contexts that are relevant to the question."""

    name = "context_precision"

    def __init__(self, llm: BaseLLMClient | None = None):
        self.llm = llm

    def score(self, question: str, answer: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0

        if self.llm is None:
            return self._fallback_score(question, contexts)

        context_text = "\n\n".join(f"[{i}] {ctx}" for i, ctx in enumerate(contexts))
        prompt = f"""You are evaluating context precision.
For each context below, decide whether it is relevant to answering the question.
Return ONLY a JSON object mapping index to 1 (relevant) or 0 (not relevant).

Question: {question}

Contexts:
{context_text}

Example output: {{"0": 1, "1": 0}}
Do not include any other text."""

        try:
            response = self.llm.generate(
                messages=[
                    {"role": "system", "content": "You are a context precision evaluator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            result = json.loads(self._extract_json(response))
            relevant = sum(1 for k, v in result.items() if int(v) == 1)
            return relevant / len(contexts)
        except Exception as exc:
            logger.warning("Context precision LLM evaluation failed (%s). Using fallback.", exc)
            return self._fallback_score(question, contexts)

    def _fallback_score(self, question: str, contexts: list[str]) -> float:
        scores = [_overlap_score(question, ctx) for ctx in contexts]
        # Count contexts with reasonable overlap as relevant
        relevant = sum(1 for s in scores if s > 0.1)
        return relevant / len(contexts) if contexts else 0.0

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        return text

"""Resilience utilities: retry, fallback, and health state for external calls.

This module implements the P1-8 fault-tolerance requirements:

* Exponential-backoff retry for transient LLM/network failures
* Automatic model fallback when the primary LLM fails
* A simple health-state machine that degrades unhealthy models
* A final deterministic fallback when no model is available
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from openai import APIError, APITimeoutError, RateLimitError

from rag_agent.llm import BaseLLMClient, MockLLMClient

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

DEFAULT_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    APITimeoutError,
    RateLimitError,
    APIError,
    TimeoutError,
    ConnectionError,
)


@dataclass
class RetryConfig:
    """Configuration for exponential-backoff retry."""

    max_retries: int = 3
    backoff_factor: float = 2.0
    max_backoff: float = 60.0
    retryable_exceptions: tuple[type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS
    retryable_checker: Callable[[Exception], bool] | None = None

    def should_retry(self, exc: Exception) -> bool:
        """Return True if the exception warrants a retry."""
        if not isinstance(exc, self.retryable_exceptions):
            return False
        if self.retryable_checker is not None:
            return self.retryable_checker(exc)
        return True


def _calculate_backoff(attempt: int, config: RetryConfig) -> float:
    """Compute sleep duration for the given attempt (0-indexed).

    Uses exponential backoff capped at ``config.max_backoff``.
    """
    return min(config.backoff_factor**attempt, config.max_backoff)


def with_retry(config: RetryConfig | None = None) -> Callable[[F], F]:
    """Decorator that retries a callable with exponential backoff.

    Works for both synchronous and asynchronous functions. The wrapped function
    is re-invoked up to ``config.max_retries`` additional times when a
    retryable exception is raised.
    """
    cfg = config or RetryConfig()

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exc: Exception | None = None
                for attempt in range(cfg.max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:
                        last_exc = exc
                        if attempt == cfg.max_retries or not cfg.should_retry(exc):
                            raise
                        sleep_seconds = _calculate_backoff(attempt, cfg)
                        logger.warning(
                            "%s failed (attempt %d/%d): %s. Retrying in %.2fs...",
                            func.__name__,
                            attempt + 1,
                            cfg.max_retries + 1,
                            exc,
                            sleep_seconds,
                        )
                        await asyncio.sleep(sleep_seconds)
                raise last_exc  # pragma: no cover

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(cfg.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt == cfg.max_retries or not cfg.should_retry(exc):
                        raise
                    sleep_seconds = _calculate_backoff(attempt, cfg)
                    logger.warning(
                        "%s failed (attempt %d/%d): %s. Retrying in %.2fs...",
                        func.__name__,
                        attempt + 1,
                        cfg.max_retries + 1,
                        exc,
                        sleep_seconds,
                    )
                    time.sleep(sleep_seconds)
            raise last_exc  # pragma: no cover

        return sync_wrapper  # type: ignore[return-value]

    return decorator


@dataclass
class ModelHealthState:
    """Tracks per-model failure counts for graceful degradation.

    A model is considered unhealthy after ``failure_threshold`` consecutive
    failures. A successful call resets the failure counter.
    """

    failure_threshold: int = 3
    _failures: dict[str, int] = field(default_factory=dict)
    _last_failure: dict[str, float] = field(default_factory=dict)

    def mark_failure(self, model: str) -> None:
        """Record a failure for ``model`` and bump its failure counter."""
        self._failures[model] = self._failures.get(model, 0) + 1
        self._last_failure[model] = time.time()
        logger.warning(
            "Model %s failure count: %d/%d",
            model,
            self._failures[model],
            self.failure_threshold,
        )

    def mark_success(self, model: str) -> None:
        """Clear failure state for ``model`` after a successful call."""
        if model in self._failures:
            logger.info("Model %s recovered, clearing failure count", model)
            del self._failures[model]
            self._last_failure.pop(model, None)

    def is_healthy(self, model: str) -> bool:
        """Return True if the model has not exceeded the failure threshold."""
        return self._failures.get(model, 0) < self.failure_threshold

    def healthy_models(self, models: list[str]) -> list[str]:
        """Filter ``models`` to only those currently considered healthy."""
        return [m for m in models if self.is_healthy(m)]


class ResilientLLMClient(BaseLLMClient):
    """LLM client wrapper with retry, model fallback, and degradation state machine.

    The wrapper transparently retries transient failures on the primary model,
    falls back through a list of backup models, and finally returns a
    deterministic mock response if every model is unavailable.

    Streaming methods do not retry the same model (generators cannot be
    rewound), but they do fall back to the next healthy model on failure.
    """

    def __init__(
        self,
        client: BaseLLMClient,
        fallback_models: list[str] | None = None,
        retry_config: RetryConfig | None = None,
        health_state: ModelHealthState | None = None,
        final_fallback: BaseLLMClient | None = None,
    ):
        self._client = client
        self.fallback_models = list(fallback_models or [])
        self.retry_config = retry_config or RetryConfig()
        self.health_state = health_state or ModelHealthState()
        self._final_fallback = final_fallback or MockLLMClient()

    @property
    def default_model(self) -> str | None:
        """Proxy the underlying client's default model."""
        return getattr(self._client, "default_model", None)

    def _all_models(self, model: str | None) -> list[str]:
        """Return the primary model followed by unique fallback models."""
        primary = model or self.default_model or ""
        seen = {primary} if primary else set()
        models: list[str] = [primary] if primary else []
        for m in self.fallback_models:
            if m and m not in seen:
                models.append(m)
                seen.add(m)
        return models

    def _generate_once(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """Invoke the underlying sync ``generate`` with retry."""

        @with_retry(self.retry_config)
        def _call() -> str:
            return self._client.generate(
                messages,
                model=model,
                temperature=temperature,
                **kwargs,
            )

        return _call()

    async def _agenerate_once(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """Invoke the underlying async ``agenerate`` with retry."""

        @with_retry(self.retry_config)
        async def _call() -> str:
            return await self._client.agenerate(
                messages,
                model=model,
                temperature=temperature,
                **kwargs,
            )

        return await _call()

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        """Generate with retry and model fallback."""
        for m in self._all_models(model):
            if not self.health_state.is_healthy(m):
                logger.warning("Skipping unhealthy model %s", m)
                continue
            try:
                result = self._generate_once(m, messages, temperature, **kwargs)
                self.health_state.mark_success(m)
                return result
            except Exception as exc:
                self.health_state.mark_failure(m)
                logger.warning("Model %s generation failed: %s", m, exc)

        logger.warning("All LLM models failed; using final fallback.")
        return self._final_fallback.generate(messages, model, temperature, **kwargs)

    async def agenerate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        """Async generate with retry and model fallback."""
        for m in self._all_models(model):
            if not self.health_state.is_healthy(m):
                logger.warning("Skipping unhealthy model %s", m)
                continue
            try:
                result = await self._agenerate_once(m, messages, temperature, **kwargs)
                self.health_state.mark_success(m)
                return result
            except Exception as exc:
                self.health_state.mark_failure(m)
                logger.warning("Model %s async generation failed: %s", m, exc)

        logger.warning("All LLM models failed; using final fallback.")
        return await self._final_fallback.agenerate(
            messages, model, temperature, **kwargs
        )

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ):
        """Stream with model fallback (no per-model retry for streams)."""
        for m in self._all_models(model):
            if not self.health_state.is_healthy(m):
                logger.warning("Skipping unhealthy model %s", m)
                continue
            try:
                stream = self._client.generate_stream(
                    messages,
                    model=m,
                    temperature=temperature,
                    **kwargs,
                )
                self.health_state.mark_success(m)
                return stream
            except Exception as exc:
                self.health_state.mark_failure(m)
                logger.warning("Model %s streaming failed: %s", m, exc)

        logger.warning("All LLM models failed; using final fallback stream.")
        return self._final_fallback.generate_stream(
            messages, model, temperature, **kwargs
        )

    async def agenerate_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ):
        """Async stream with model fallback (no per-model retry for streams)."""
        for m in self._all_models(model):
            if not self.health_state.is_healthy(m):
                logger.warning("Skipping unhealthy model %s", m)
                continue
            try:
                stream = await self._client.agenerate_stream(
                    messages,
                    model=m,
                    temperature=temperature,
                    **kwargs,
                )
                self.health_state.mark_success(m)
                return stream
            except Exception as exc:
                self.health_state.mark_failure(m)
                logger.warning("Model %s async streaming failed: %s", m, exc)

        logger.warning("All LLM models failed; using final fallback stream.")
        return await self._final_fallback.agenerate_stream(
            messages, model, temperature, **kwargs
        )

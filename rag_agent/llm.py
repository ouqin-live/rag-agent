"""LLM client wrapper with OpenAI-compatible API support and fallback generation."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

# 自动加载项目根目录的 .env 文件
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

logger = logging.getLogger(__name__)


def _load_settings_values() -> dict[str, Any]:
    """Load defaults from application settings when available."""
    try:
        from rag_agent.config import get_settings

        settings = get_settings()
        return {
            "base_url": settings.llm_base_url,
            "api_key": settings.llm_api_key,
            "default_model": settings.llm_model,
            "timeout": settings.llm_timeout,
        }
    except Exception:
        return {}


class BaseLLMClient(ABC):
    """Abstract LLM client."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        ...

    @abstractmethod
    async def agenerate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        """Async generation. Subclasses may default to thread-pool execution."""
        ...

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ):
        """Yield generated text chunks as a stream.

        The default implementation falls back to ``generate`` and yields the
        full response as a single chunk. Subclasses can override for native
        streaming.
        """
        yield self.generate(messages, model, temperature, **kwargs)

    async def agenerate_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ):
        """Async streaming generator."""
        for chunk in self.generate_stream(messages, model, temperature, **kwargs):
            yield chunk


class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI-compatible client with IdeaLab-specific error handling."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ):
        settings = _load_settings_values()
        self.base_url = base_url or settings.get("base_url") or os.getenv("OPENAI_BASE_URL", "")
        self.api_key = (
            api_key
            or settings.get("api_key")
            or os.getenv("AI_STUDIO_TOKEN")
            or os.getenv("OPENAI_API_KEY", "")
        )
        self.default_model = default_model or settings.get("default_model") or os.getenv("OPENAI_MODEL", "")
        self.default_headers = default_headers or {}
        self.timeout = timeout or settings.get("timeout") or 60.0

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
            timeout=self.timeout,
        )
        self._async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
            timeout=self.timeout,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            **kwargs,
        )
        # IdeaLab may wrap errors in a custom success field
        if hasattr(response, "success") and response.success is False:
            err_msg = getattr(response, "errMsg", "未知错误")
            raise RuntimeError(f"IdeaLab API 错误: {err_msg}")
        return response.choices[0].message.content or ""

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ):
        """Stream tokens from the OpenAI-compatible endpoint."""
        stream = self._client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def agenerate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        response = await self._async_client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            **kwargs,
        )
        if hasattr(response, "success") and response.success is False:
            err_msg = getattr(response, "errMsg", "未知错误")
            raise RuntimeError(f"IdeaLab API 错误: {err_msg}")
        return response.choices[0].message.content or ""

    async def agenerate_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ):
        """Async stream tokens."""
        stream = await self._async_client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class MockLLMClient(BaseLLMClient):
    """Deterministic fallback client used when no LLM service is available.

    Responses are matched against the last user message to avoid false matches
    from injected context/history in the full prompt.
    """

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}

    def _match_response(self, messages: list[dict[str, str]]) -> str:
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break

        # Return a canned response if the last user message matches a known pattern
        for key, value in self.responses.items():
            if key in last_user:
                return value
        return "模拟回答：基于提供的参考资料无法给出准确判断。"

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        return self._match_response(messages)

    async def agenerate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        return self._match_response(messages)

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ):
        yield self._match_response(messages)

    async def agenerate_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ):
        yield self._match_response(messages)


def get_llm_client() -> BaseLLMClient:
    """Factory that returns a resilient LLM client.

    The returned client wraps an OpenAI-compatible client with exponential
    backoff retry, automatic fallback to backup models, and a final mock
    fallback when no model is available.
    """
    try:
        client = OpenAICompatibleClient()
    except Exception as exc:
        logger.warning("Failed to create OpenAI-compatible client: %s. Using mock.", exc)
        return MockLLMClient()

    from rag_agent.resilience import ModelHealthState, ResilientLLMClient, RetryConfig

    try:
        settings = _load_settings_values()
        # _load_settings_values returns partial settings; use config directly
        from rag_agent.config import get_settings

        settings = get_settings()
        fallback_models = [
            m.strip()
            for m in (settings.llm_fallback_models or "").split(",")
            if m.strip()
        ]
        retry_config = RetryConfig(
            max_retries=settings.llm_max_retries,
            backoff_factor=settings.llm_retry_backoff,
        )
        health_state = ModelHealthState(
            failure_threshold=settings.llm_health_failure_threshold,
        )
    except Exception as exc:
        logger.warning("Failed to load resilience settings: %s. Using defaults.", exc)
        fallback_models = []
        retry_config = RetryConfig()
        health_state = ModelHealthState()

    return ResilientLLMClient(
        client=client,
        fallback_models=fallback_models,
        retry_config=retry_config,
        health_state=health_state,
        final_fallback=MockLLMClient(),
    )

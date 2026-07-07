"""LLM client wrapper with OpenAI-compatible API support and fallback generation."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# 自动加载项目根目录的 .env 文件
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

logger = logging.getLogger(__name__)


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


class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI-compatible client with IdeaLab-specific error handling."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
        self.api_key = api_key or os.getenv("AI_STUDIO_TOKEN") or os.getenv("OPENAI_API_KEY", "")
        self.default_model = default_model or os.getenv("OPENAI_MODEL", "")
        self.default_headers = default_headers or {}

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
            timeout=timeout,
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


class MockLLMClient(BaseLLMClient):
    """Deterministic fallback client used when no LLM service is available.

    Responses are matched against the last user message to avoid false matches
    from injected context/history in the full prompt.
    """

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
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


def get_llm_client() -> BaseLLMClient:
    """Factory that returns an OpenAI-compatible client when credentials are available."""
    try:
        return OpenAICompatibleClient()
    except Exception as exc:
        logger.warning("Failed to create OpenAI-compatible client: %s. Using mock.", exc)
        return MockLLMClient()

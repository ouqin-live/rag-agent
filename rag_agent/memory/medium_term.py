"""Medium-term memory: conversation session summary."""

from __future__ import annotations

from rag_agent.llm import BaseLLMClient
from rag_agent.memory.short_term import Message


class MediumTermMemory:
    """Keeps a running summary of the current conversation session.

    When short-term memory exceeds its capacity, old turns are fed here
    to be compressed into a single paragraph of key information.
    """

    def __init__(self, llm_client: BaseLLMClient, max_summary_len: int = 500):
        self.llm_client = llm_client
        self.max_summary_len = max_summary_len
        self.summary = ""

    def update(self, messages: list[Message]) -> None:
        """Merge a block of old conversation turns into the session summary."""
        if not messages:
            return
        history = "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in messages
        )
        prompt = f"""将以下新对话合并到当前会话摘要中。保留关键信息和用户偏好，控制长度。

当前摘要：{self.summary or '（尚无摘要）'}
新对话：{history}

返回更新后的摘要（纯文本，不超过{self.max_summary_len}字）："""
        try:
            self.summary = self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=self.max_summary_len,
            )
        except Exception:
            # 降级：直接拼接
            if self.summary:
                self.summary += " " + history[:200]
            else:
                self.summary = history[:self.max_summary_len]

    def get_summary(self) -> str:
        """Return the current session summary."""
        return self.summary

    def clear(self) -> None:
        """Reset the session summary."""
        self.summary = ""

"""Short-term conversational memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


@dataclass
class Message:
    """A single turn in a conversation."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ShortTermMemory:
    """Keeps the most recent N turns of a conversation in memory.

    A "turn" is defined as a user message plus the following assistant message.
    When the limit is exceeded, the oldest complete turn is dropped.
    """

    def __init__(self, max_turns: int = 6):
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.max_turns = max_turns
        self._messages: list[Message] = []

    def add(self, role: Literal["user", "assistant", "system"], content: str) -> None:
        """Append a message to the conversation history."""
        self._messages.append(Message(role=role, content=content))
        self._enforce_limit()

    def get_messages(self) -> list[Message]:
        """Return the current conversation history."""
        return list(self._messages)

    def get_history_text(self) -> str:
        """Return history formatted as a single string for prompt injection."""
        lines = []
        for msg in self._messages:
            name = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{name}: {msg.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset the conversation history."""
        self._messages.clear()

    def _enforce_limit(self) -> None:
        """Drop oldest complete turns while keeping at most max_turns turns."""
        turns = []
        current_turn: list[Message] = []
        for msg in self._messages:
            if msg.role == "user":
                if current_turn:
                    turns.append(current_turn)
                current_turn = [msg]
            else:
                current_turn.append(msg)
        if current_turn:
            turns.append(current_turn)

        if len(turns) > self.max_turns:
            # Keep the most recent max_turns complete turns
            kept_turns = turns[-self.max_turns :]
            self._messages = [msg for turn in kept_turns for msg in turn]

    def __len__(self) -> int:
        return len(self._messages)

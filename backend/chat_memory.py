"""Conversation memory for context-aware follow-up questions."""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str


class ChatMemory:
    """Maintains a sliding window of recent conversation history.

    Stores the last N exchanges (user + assistant pairs) and formats
    them for inclusion in the RAG prompt. This enables the LLM to
    understand references like "it", "that", or "explain more".

    Usage:
        memory = ChatMemory(max_exchanges=5)
        memory.add("user", "What is RAG?")
        memory.add("assistant", "RAG stands for...")
        memory.add("user", "What are its benefits?")
        print(memory.format_for_prompt())
    """

    def __init__(self, max_exchanges: int = 5) -> None:
        if max_exchanges < 1:
            raise ValueError("max_exchanges must be at least 1")
        self._max_exchanges = max_exchanges
        self._messages: list[Message] = []

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Return the number of stored messages."""
        return len(self._messages)

    def add(self, role: str, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            role: "user" or "assistant".
            content: The message text.
        """
        self._messages.append(Message(role=role, content=content))
        self._trim()

    def get_history(self) -> list[Message]:
        """Return the current conversation history."""
        return list(self._messages)

    def format_for_prompt(self) -> str:
        """Format recent conversation into a string for the LLM prompt.

        Returns a block like:
            Conversation history:
            User: What is RAG?
            Assistant: RAG stands for Retrieval-Augmented Generation.
            User: What are its benefits?

        Returns an empty string if there is no history.
        """
        if not self._messages:
            return ""

        lines = ["Conversation history:"]
        for msg in self._messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset the conversation history."""
        self._messages.clear()

    # ── Private helpers ─────────────────────────────────────────────────

    def _trim(self) -> None:
        """Remove oldest messages, keeping at most max_exchanges exchanges.

        An exchange is a user + assistant pair (2 messages). We keep
        the most recent exchanges and drop the rest.
        """
        max_messages = self._max_exchanges * 2
        while len(self._messages) > max_messages:
            self._messages.pop(0)

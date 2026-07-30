"""Session-scoped conversation history for multi-turn chat.

Keeps the last few Q&A turns per session so Agent 2 can resolve
follow-up questions ("what about his education?") against earlier
context. In-memory by design, matching the vector store's lifecycle:
one server session = one conversation workspace.
"""

from collections import defaultdict, deque

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Enough context for natural follow-ups without bloating prompts.
MAX_TURNS = 6


class ChatHistory:
    """Stores recent (question, answer) turns per session id."""

    def __init__(self, max_turns: int = MAX_TURNS) -> None:
        self._turns: dict[str, deque[tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=max_turns)
        )

    def add(self, session_id: str, question: str, answer: str) -> None:
        """Record one completed turn for the session."""
        self._turns[session_id].append((question, answer))

    def format(self, session_id: str) -> str:
        """Render the session's history as a prompt-ready transcript.

        Returns:
            "User: ...\\nAssistant: ..." lines, or an empty string when
            the session has no history yet (first question).
        """
        turns = self._turns.get(session_id)
        if not turns:
            return ""
        return "\n".join(
            f"User: {question}\nAssistant: {answer}" for question, answer in turns
        )

    def clear(self, session_id: str) -> None:
        """Forget a session's history (used by the UI's New Chat)."""
        self._turns.pop(session_id, None)
        logger.info("Chat history cleared for session '%s'", session_id)


# Single shared instance, like the vector store manager.
chat_history = ChatHistory()

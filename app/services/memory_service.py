"""Persistent memory service -- conversation history that survives
restarts, plus intent/action records and preferences.

Drop-in upgrade of the in-memory ChatHistory: the ``add`` / ``format``
/ ``clear`` interface is identical, so the workflow stages switched to
this service with a one-line import change.

Resilience contract: if the database is unavailable (not initialised,
connection lost), every method degrades to the in-memory fallback and
logs a warning -- memory quality degrades, requests never fail.
"""

from typing import Any

from sqlalchemy import delete, func, select

from app.services.history import ChatHistory
from app.utils.logger import get_logger

logger = get_logger(__name__)

MAX_TURNS = 6  # matches the previous in-memory behavior


class MemoryService:
    """Database-backed conversation memory with in-memory fallback."""

    def __init__(self) -> None:
        self._fallback = ChatHistory(max_turns=MAX_TURNS)
        self._db_ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Connect and create tables; fall back silently on failure."""
        try:
            from app.db.database import init_engine

            init_engine()
            self._db_ready = True
        except Exception as exc:
            self._db_ready = False
            logger.warning(
                "Persistent memory unavailable (%s); using in-memory history",
                exc,
            )

    @property
    def persistent(self) -> bool:
        """Whether history is currently database-backed."""
        return self._db_ready

    # ------------------------------------------------------------------
    # ChatHistory-compatible interface (used by the workflow stages)
    # ------------------------------------------------------------------
    def add(self, session_id: str, question: str, answer: str) -> None:
        """Record one completed turn for the session."""
        self._fallback.add(session_id, question, answer)  # fast local cache
        if not self._db_ready:
            return
        try:
            from app.db.database import get_session
            from app.db.models import ChatMessageRecord, ChatSessionRecord

            with get_session() as db:
                if db.get(ChatSessionRecord, session_id) is None:
                    db.add(ChatSessionRecord(id=session_id))
                db.add(ChatMessageRecord(session_id=session_id, role="user", content=question))
                db.add(ChatMessageRecord(session_id=session_id, role="assistant", content=answer))
                db.commit()
        except Exception as exc:
            logger.warning("Memory write failed (%s); in-memory copy kept", exc)

    def format(self, session_id: str) -> str:
        """Render recent history as a prompt-ready transcript."""
        if self._db_ready:
            try:
                from app.db.database import get_session
                from app.db.models import ChatMessageRecord

                with get_session() as db:
                    rows = db.execute(
                        select(ChatMessageRecord)
                        .where(ChatMessageRecord.session_id == session_id)
                        .order_by(ChatMessageRecord.id.desc())
                        .limit(MAX_TURNS * 2)
                    ).scalars().all()
                rows.reverse()
                return "\n".join(
                    ("User: " if row.role == "user" else "Assistant: ") + row.content
                    for row in rows
                )
            except Exception as exc:
                logger.warning("Memory read failed (%s); using in-memory", exc)
        return self._fallback.format(session_id)

    def clear(self, session_id: str) -> None:
        """Forget a session's history."""
        self._fallback.clear(session_id)
        if not self._db_ready:
            return
        try:
            from app.db.database import get_session
            from app.db.models import ChatMessageRecord, ChatSessionRecord

            with get_session() as db:
                db.execute(delete(ChatMessageRecord).where(ChatMessageRecord.session_id == session_id))
                db.execute(delete(ChatSessionRecord).where(ChatSessionRecord.id == session_id))
                db.commit()
        except Exception as exc:
            logger.warning("Memory clear failed (%s)", exc)

    # ------------------------------------------------------------------
    # Intent / action records (Agents 5 and 6)
    # ------------------------------------------------------------------
    def record_intent(
        self, session_id: str, question: str, intent: str,
        confidence: float, method: str,
    ) -> None:
        """Persist one Agent 5 verdict."""
        if not self._db_ready:
            return
        try:
            from app.db.database import get_session
            from app.db.models import IntentRecord

            with get_session() as db:
                db.add(IntentRecord(
                    session_id=session_id, question=question[:2000],
                    intent=intent, confidence=confidence, method=method,
                ))
                db.commit()
        except Exception as exc:
            logger.warning("Intent record failed (%s)", exc)

    def record_action(
        self, session_id: str, action: str, status: str, file: str | None
    ) -> None:
        """Persist one Agent 6 execution."""
        if not self._db_ready:
            return
        try:
            from app.db.database import get_session
            from app.db.models import ActionRecord

            with get_session() as db:
                db.add(ActionRecord(
                    session_id=session_id, action=action, status=status, file=file,
                ))
                db.commit()
        except Exception as exc:
            logger.warning("Action record failed (%s)", exc)

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------
    def set_preference(self, session_id: str, key: str, value: str) -> None:
        """Store one per-session preference (upsert)."""
        if not self._db_ready:
            return
        try:
            from app.db.database import get_session
            from app.db.models import PreferenceRecord

            with get_session() as db:
                existing = db.get(PreferenceRecord, (session_id, key))
                if existing is None:
                    db.add(PreferenceRecord(session_id=session_id, key=key, value=value))
                else:
                    existing.value = value
                db.commit()
        except Exception as exc:
            logger.warning("Preference write failed (%s)", exc)

    def get_preference(self, session_id: str, key: str) -> str | None:
        """Read one per-session preference."""
        if not self._db_ready:
            return None
        try:
            from app.db.database import get_session
            from app.db.models import PreferenceRecord

            with get_session() as db:
                row = db.get(PreferenceRecord, (session_id, key))
                return row.value if row else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Introspection (admin dashboard)
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Aggregate memory statistics for monitoring."""
        base: dict[str, Any] = {"persistent": self._db_ready}
        if not self._db_ready:
            return base
        try:
            from app.db.database import get_session
            from app.db.models import (
                ActionRecord,
                ChatMessageRecord,
                ChatSessionRecord,
                IntentRecord,
            )

            with get_session() as db:
                count = lambda model: db.execute(  # noqa: E731
                    select(func.count()).select_from(model)
                ).scalar_one()
                base.update(
                    sessions=count(ChatSessionRecord),
                    messages=count(ChatMessageRecord),
                    intents=count(IntentRecord),
                    actions=count(ActionRecord),
                )
                intent_rows = db.execute(
                    select(IntentRecord.intent, func.count())
                    .group_by(IntentRecord.intent)
                    .order_by(func.count().desc())
                ).all()
                base["intent_breakdown"] = {name: n for name, n in intent_rows}
        except Exception as exc:
            logger.warning("Memory stats failed (%s)", exc)
        return base


# Single shared instance used by the workflow stages and routes.
memory = MemoryService()

"""Workflow context -- the single state object that flows through the
pipeline.

Stages communicate only through this context (never with each other),
which is what makes every stage independently replaceable: as long as a
stage reads and writes the agreed fields, its internals are free to
change.
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.documents import Document

from app.models.schemas import AskResponse


@dataclass
class StageTrace:
    """Execution record of one pipeline stage (for logs, API and UI)."""

    name: str
    status: str  # "completed" | "skipped" | "failed"
    duration_ms: float
    note: str = ""


@dataclass
class WorkflowContext:
    """Mutable state shared by every stage of one workflow run.

    The first block is the caller's input; everything below it is
    written by the stages as the request advances through the pipeline.
    """

    # --- Inputs (set by the caller) ---
    question: str
    session_id: str = "default"
    doc_id: str | None = None
    # Optional Agent 4 output (application, summary, user_intent, ...)
    # attached when the question is about what's on the user's screen.
    screen_context: dict[str, Any] | None = None
    # Streaming hook: when set, the Guide stage emits answer tokens
    # through it as they are generated (see chat_service.stream_events).
    token_callback: Callable[[str], None] | None = field(
        default=None, repr=False
    )
    # Set when the client disconnects mid-stream; the engine stops before
    # the next stage so we don't burn LLM quota on an abandoned request.
    cancel_event: threading.Event | None = field(default=None, repr=False)

    @property
    def cancelled(self) -> bool:
        """True when the caller has asked to abort this workflow."""
        return self.cancel_event is not None and self.cancel_event.is_set()

    # --- Written by Observe ---
    observations: dict[str, Any] = field(default_factory=dict)

    # --- Written by Understand ---
    standalone_question: str = ""
    intent: str = "question_answering"
    intent_confidence: float = 1.0
    recommended_workflow: str = "rag_answer"
    intent_method: str = "default"

    # --- Written by Analyze ---
    routed_doc_id: str | None = None
    chunks: list[Document] = field(default_factory=list)

    # --- Written by Guide ---
    response: AskResponse | None = None

    # --- Written by Automate ---
    automation_result: dict[str, Any] | None = None

    # --- Written by the engine ---
    trace: list[StageTrace] = field(default_factory=list)

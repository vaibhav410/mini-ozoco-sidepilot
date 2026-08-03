"""Question-answering orchestration -- delegated to the workflow engine.

Every question travels the SidePilot pipeline:

    Observe -> Understand -> Analyze -> Guide -> Automate

Two entry points, one pipeline:

- ``answer_question``: blocking; returns the complete AskResponse.
- ``stream_events``:   generator of live events (stage progress,
  answer tokens, final response) consumed by the SSE route.

This module stays the stable boundary between routes and the workflow.
"""

import queue
import threading
from typing import Any, Iterator

from app.models.schemas import AskResponse, IntentInfo, WorkflowStageInfo
from app.rag.prompt import NOT_FOUND_TOKEN
from app.utils.errors import AppError
from app.utils.logger import get_logger
from app.workflow import StageTrace, WorkflowContext, get_workflow_engine

logger = get_logger(__name__)

_SENTINEL = object()
# Max time stream_events() blocks per queue poll. Bounds how long a
# disconnect check in the route can be delayed by a single call -- it
# is NOT how fast an in-flight LLM call aborts (that isn't cancellable
# through LangChain), but it does stop the *next* stage from starting.
_POLL_SECONDS = 0.2


def answer_question(
    question: str,
    doc_id: str | None = None,
    session_id: str = "default",
    screen_context: dict[str, Any] | None = None,
) -> AskResponse:
    """Answer a user question by running the full workflow pipeline.

    Args:
        question: The user's natural-language question.
        doc_id: Optional explicit document to answer from.
        session_id: Conversation session; enables follow-up questions.
        screen_context: Optional Agent 4 screen analysis to ground the
            request in what the user is currently looking at.

    Returns:
        AskResponse with answer, routing, sources, validation verdict,
        the ``found`` flag, intent, and the executed workflow trace.

    Raises:
        NoDocumentsError: Nothing has been uploaded yet.
        DocumentNotFoundError: The given doc_id does not exist.
        AIServiceError: All LLM providers failed while generating.
    """
    context = WorkflowContext(
        question=question,
        doc_id=doc_id,
        session_id=session_id,
        screen_context=screen_context,
    )
    get_workflow_engine().run(context)
    return _finalize(context)


def stream_events(
    question: str,
    doc_id: str | None = None,
    session_id: str = "default",
    screen_context: dict[str, Any] | None = None,
    cancel_event: threading.Event | None = None,
) -> Iterator[dict[str, Any] | None]:
    """Run the pipeline while yielding live events for the SSE route.

    Event shapes (``{"event": ..., "data": ...}``):

    - ``stage``: one pipeline stage finished (name, status, duration_ms)
    - ``token``: next fragment of the generated answer
    - ``final``: the complete AskResponse (same JSON as POST /ask)
    - ``error``: a failure (detail, status)

    The pipeline runs in a worker thread; this generator drains a queue,
    so tokens reach the client the moment the model produces them. Each
    poll is bounded (``_POLL_SECONDS``), and a bare ``None`` is yielded
    on every timeout with nothing new -- this hands control back to the
    caller regularly so it can check for a client disconnect and call
    ``cancel_event.set()`` without waiting on a full pipeline stage.

    Args:
        cancel_event: Externally owned cancellation flag. If omitted,
            one is created (workflow still stops itself on cancellation,
            just with nothing external able to trigger it).
    """
    cancel_event = cancel_event or threading.Event()
    events: queue.Queue = queue.Queue()

    def emit(event: str, data: Any) -> None:
        events.put({"event": event, "data": data})

    gate = _TokenGate(lambda token: emit("token", {"text": token}))

    def on_stage(trace: StageTrace) -> None:
        emit(
            "stage",
            {
                "name": trace.name,
                "status": trace.status,
                "duration_ms": round(trace.duration_ms, 1),
                "note": trace.note,
            },
        )

    context = WorkflowContext(
        question=question,
        doc_id=doc_id,
        session_id=session_id,
        screen_context=screen_context,
        token_callback=gate.push,
        cancel_event=cancel_event,
    )

    # Propagate the request id into the worker thread: contextvars do
    # not cross thread boundaries on their own.
    from app.utils.request_context import request_id_var

    request_id = request_id_var.get()

    def work() -> None:
        request_id_var.set(request_id)
        try:
            get_workflow_engine().run(context, on_stage=on_stage)
            gate.flush()
            if context.cancelled:
                # Expected outcome, not a failure: the client is already
                # gone (nobody will receive this "final"), so there is
                # nothing useful to build or log as an error.
                logger.info("Streaming pipeline stopped: client disconnected")
            else:
                emit("final", _finalize(context).model_dump())
        except AppError as err:
            emit("error", {"detail": err.detail, "status": err.status_code})
        except Exception:
            logger.exception("Unexpected error in streaming pipeline")
            emit("error", {"detail": "Unexpected server error.", "status": 500})
        finally:
            events.put(_SENTINEL)

    threading.Thread(target=work, daemon=True).start()

    try:
        while True:
            try:
                item = events.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                # Nothing new yet -- hand control back to the caller so
                # it can check for a client disconnect. This is what
                # makes cancellation actually responsive: without it,
                # the generator stays blocked inside events.get() for
                # however long the current stage's LLM call takes.
                yield None
                continue
            if item is _SENTINEL:
                break
            yield item
    finally:
        # Reached on normal completion AND when the caller stops
        # iterating (route detected a disconnect and broke its loop).
        cancel_event.set()


def _finalize(context: WorkflowContext) -> AskResponse:
    """Attach the trace and intent verdict to the pipeline's response."""
    if context.response is None:  # defensive: Guide/Automate always set it
        raise RuntimeError("Workflow finished without producing a response")

    context.response.workflow = [
        WorkflowStageInfo(
            name=trace.name,
            status=trace.status,
            duration_ms=round(trace.duration_ms, 1),
        )
        for trace in context.trace
    ]
    context.response.intent = IntentInfo(
        intent=context.intent,
        confidence=round(context.intent_confidence, 2),
        recommended_workflow=context.recommended_workflow,
        method=context.intent_method,
    )
    return context.response


class _TokenGate:
    """Holds back the first tokens so the NOT_FOUND sentinel never
    reaches the client as visible text.

    The gate buffers until it has seen enough characters to rule out
    the sentinel prefix, then flushes and passes everything through.
    If the sentinel is detected, all tokens are suppressed -- the
    ``final`` event carries the proper "not found" answer instead.
    """

    def __init__(self, emit) -> None:
        self._emit = emit
        self._buffer = ""
        self._decided = False
        self._suppress = False

    def push(self, token: str) -> None:
        """Receive one token from the model."""
        if self._decided:
            if not self._suppress:
                self._emit(token)
            return
        self._buffer += token
        if len(self._buffer.lstrip()) >= len(NOT_FOUND_TOKEN):
            self._decide()

    def flush(self) -> None:
        """Called when generation ends; emits any undecided buffer."""
        if not self._decided:
            self._decide()

    def _decide(self) -> None:
        self._decided = True
        self._suppress = self._buffer.lstrip().startswith(NOT_FOUND_TOKEN)
        if not self._suppress and self._buffer:
            self._emit(self._buffer)

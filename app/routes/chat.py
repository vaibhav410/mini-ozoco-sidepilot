"""POST /ask and /ask/stream -- question answering (HTTP layer only)."""

import json
import threading

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.models.schemas import AskRequest, AskResponse
from app.services.chat_service import answer_question, stream_events
from app.utils.errors import AppError
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Chat"])

_STOP = object()


def _safe_next(iterator):
    """``next(iterator)`` that returns a sentinel instead of raising
    StopIteration -- keeps the polling loop below simple to read."""
    try:
        return next(iterator)
    except StopIteration:
        return _STOP


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about the uploaded documents",
    description=(
        "Runs Agent 2: routes the question to the relevant document, "
        "retrieves the top-k chunks from FAISS, and generates a grounded "
        "Gemini answer with source references. Off-topic questions get an "
        "explicit 'not found' response instead of a hallucination."
    ),
)
def ask_question(request: AskRequest) -> AskResponse:
    """Handle a question request."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        return answer_question(
            question,
            doc_id=request.doc_id,
            session_id=request.session_id,
            screen_context=request.screen_context,
        )
    except AppError as err:
        raise HTTPException(status_code=err.status_code, detail=err.detail)
    except Exception:
        logger.exception("Unexpected error while answering question")
        raise HTTPException(status_code=500, detail="Unexpected server error.")


@router.post(
    "/ask/stream",
    summary="Ask a question with a streamed (SSE) response",
    description=(
        "Same pipeline as POST /ask, delivered as Server-Sent Events: "
        "`stage` events report live pipeline progress (Observe -> "
        "Understand -> Analyze -> Guide -> Automate), `token` events "
        "stream the answer as it is generated, and the closing `final` "
        "event carries the complete AskResponse JSON. An `error` event "
        "replaces `final` on failure."
    ),
)
def ask_question_stream(http_request: Request, request: AskRequest) -> StreamingResponse:
    """Handle a streaming question request.

    The event generator polls the pipeline's queue with a short timeout
    (see ``chat_service._POLL_SECONDS``) instead of blocking forever, so
    this loop regularly gets control back to check
    ``http_request.is_disconnected()``. The moment a disconnect is
    detected, ``cancel_event`` is set -- the workflow engine checks it
    between stages and stops advancing, instead of running the full
    pipeline for a client that already left (verified: previously a
    dropped connection let the backend run to completion regardless).
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    cancel_event = threading.Event()
    events = stream_events(
        question,
        doc_id=request.doc_id,
        session_id=request.session_id,
        screen_context=request.screen_context,
        cancel_event=cancel_event,
    )

    async def sse():
        try:
            while True:
                if await http_request.is_disconnected():
                    logger.info("Client disconnected; cancelling stream")
                    break
                item = await run_in_threadpool(_safe_next, events)
                if item is _STOP:
                    break
                if item is None:  # poll timeout, nothing new yet
                    continue
                yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"
        finally:
            cancel_event.set()

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx)
        },
    )

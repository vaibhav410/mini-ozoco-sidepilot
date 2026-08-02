"""POST /ask and /ask/stream -- question answering (HTTP layer only)."""

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import AskRequest, AskResponse
from app.services.chat_service import answer_question, stream_events
from app.utils.errors import AppError
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Chat"])


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
def ask_question_stream(request: AskRequest) -> StreamingResponse:
    """Handle a streaming question request."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    def sse() -> "Generator[str, None, None]":  # noqa: F821 - doc only
        for event in stream_events(
            question,
            doc_id=request.doc_id,
            session_id=request.session_id,
            screen_context=request.screen_context,
        ):
            yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx)
        },
    )

"""POST /ask -- question answering endpoint (HTTP layer only)."""

from fastapi import APIRouter, HTTPException

from app.models.schemas import AskRequest, AskResponse
from app.services.chat_service import answer_question
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
            question, doc_id=request.doc_id, session_id=request.session_id
        )
    except AppError as err:
        raise HTTPException(status_code=err.status_code, detail=err.detail)
    except Exception:
        logger.exception("Unexpected error while answering question")
        raise HTTPException(status_code=500, detail="Unexpected server error.")

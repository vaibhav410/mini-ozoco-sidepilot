"""POST /intent/detect -- intent detection endpoint (HTTP layer only)."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.agents.intent_agent import get_intent_agent
from app.models.schemas import IntentDetectRequest, IntentInfo
from app.rag.vector_store import vector_store_manager
from app.utils.auth import require_user
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Intent"], dependencies=[Depends(require_user)])


@router.post(
    "/intent/detect",
    response_model=IntentInfo,
    summary="Detect the intent behind a user request",
    description=(
        "Runs Agent 5: classifies the request into one of the SidePilot "
        "intents (question_answering, summarization, screen_help, "
        "automation, email, export, search, navigation, classification) "
        "and recommends the workflow to fulfil it. High-precision "
        "heuristics answer instantly; ambiguous requests use the LLM. "
        "Works with or without uploaded documents."
    ),
)
async def detect_intent(request: IntentDetectRequest) -> IntentInfo:
    """Handle a standalone intent detection request."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = await run_in_threadpool(
            get_intent_agent().detect,
            question,
            vector_store_manager.registry,
            request.screen_context,
        )
    except Exception:
        logger.exception("Unexpected error while detecting intent")
        raise HTTPException(status_code=500, detail="Unexpected server error.")
    return IntentInfo(
        intent=result.intent,
        confidence=round(result.confidence, 2),
        recommended_workflow=result.recommended_workflow,
        method=result.method,
    )

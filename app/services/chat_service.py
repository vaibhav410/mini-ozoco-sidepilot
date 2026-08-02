"""Question-answering orchestration -- delegated to the workflow engine.

Every question now travels the SidePilot pipeline:

    Observe -> Understand -> Analyze -> Guide -> Automate

The stage implementations (app/workflow/stages.py) run the same
three-agent logic as before -- condense, route, retrieve, answer,
validate -- so behavior and the /ask response contract are unchanged;
the pipeline just adds per-stage tracing on top.

This module stays as the stable entry point for the routes.
"""

from typing import Any

from app.models.schemas import AskResponse, IntentInfo, WorkflowStageInfo
from app.utils.logger import get_logger
from app.workflow import WorkflowContext, get_workflow_engine

logger = get_logger(__name__)


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
        the ``found`` flag, and the executed workflow trace.

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

    if context.response is None:  # defensive: Guide always sets it
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

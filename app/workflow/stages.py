"""The five pipeline stages: Observe, Understand, Analyze, Guide, Automate.

Each stage is a small class with one ``run(context)`` method that reads
and mutates the shared :class:`WorkflowContext`, then returns a short
note for the trace. Stages never call each other -- the engine owns the
ordering -- so any stage can be replaced or extended without touching
the rest.

Responsibilities:

- Observe     gather facts: documents, chat history, screen context
- Understand  resolve follow-ups, detect the user's intent
- Analyze     route the question and retrieve supporting chunks
- Guide       generate and validate the grounded answer
- Automate    execute follow-up actions (handler slot for Agent 6)

A note starting with ``"skipped"`` marks the stage as skipped in the
trace -- see the engine's status convention.
"""

from typing import Callable

from app.agents.response_agent import get_response_agent
from app.agents.validation_agent import get_validation_agent
from app.models.schemas import AskResponse, Source, ValidationInfo
from app.rag.prompt import NOT_FOUND_TOKEN
from app.rag.retriever import retrieve_chunks
from app.rag.vector_store import vector_store_manager
from app.services.history import chat_history
from app.utils.errors import DocumentNotFoundError, NoDocumentsError
from app.utils.logger import get_logger
from app.workflow.context import WorkflowContext

logger = get_logger(__name__)

NOT_FOUND_MESSAGE = (
    "The answer to this question was not found in the uploaded documents."
)
UNSUPPORTED_MESSAGE = (
    "The generated answer could not be verified against the documents, "
    "so it was withheld to avoid giving you unreliable information."
)
SNIPPET_LENGTH = 240

# Intents that trigger the Automate stage (fulfilled by Agent 6 in a
# later module; every other intent skips automation entirely).
AUTOMATION_INTENTS = {"automation", "email", "export"}

# Signature for a pluggable intent detector (Module 2 injects Agent 5
# here; until then a trivial default keeps behavior identical).
IntentDetector = Callable[[WorkflowContext], tuple[str, float]]


class ObserveStage:
    """Gather everything already known before any LLM call is made."""

    name = "observe"

    def run(self, context: WorkflowContext) -> str:
        registry = vector_store_manager.registry
        if not registry:
            raise NoDocumentsError(
                "No documents uploaded yet. Upload a PDF or TXT file first."
            )
        if context.doc_id is not None and context.doc_id not in registry:
            raise DocumentNotFoundError(
                f"No document found with id '{context.doc_id}'."
            )

        history_text = chat_history.format(context.session_id)
        context.observations = {
            "documents": len(registry),
            "history_text": history_text,
            "has_history": bool(history_text),
            "has_screen_context": context.screen_context is not None,
        }
        return (
            f"{len(registry)} document(s), history={bool(history_text)}, "
            f"screen={context.screen_context is not None}"
        )


class UnderstandStage:
    """Resolve follow-up references and detect the user's intent."""

    name = "understand"

    def __init__(self, intent_detector: IntentDetector | None = None) -> None:
        # Dependency injection point: Module 2 plugs Agent 5 in here.
        self._detect_intent = intent_detector or _default_intent

    def run(self, context: WorkflowContext) -> str:
        history = context.observations.get("history_text", "")
        agent2 = get_response_agent()
        context.standalone_question = (
            agent2.condense(context.question, history)
            if history
            else context.question
        )
        context.intent, context.intent_confidence = self._detect_intent(context)
        return f"intent={context.intent} ({context.intent_confidence:.2f})"


class AnalyzeStage:
    """Route the question to the right document and retrieve context."""

    name = "analyze"

    def run(self, context: WorkflowContext) -> str:
        registry = vector_store_manager.registry
        context.routed_doc_id = context.doc_id or get_response_agent().route(
            context.standalone_question, registry
        )
        context.chunks = retrieve_chunks(
            context.standalone_question, doc_id=context.routed_doc_id
        )
        if not context.chunks:
            context.response = _not_found(context, NOT_FOUND_MESSAGE)
            return "no relevant context found"
        return (
            f"routed={context.routed_doc_id or 'all documents'}, "
            f"chunks={len(context.chunks)}"
        )


class GuideStage:
    """Generate the grounded answer and validate it (Agents 2 + 3)."""

    name = "guide"

    def run(self, context: WorkflowContext) -> str:
        if context.response is not None:
            return "skipped (already resolved by an earlier stage)"

        raw_answer = get_response_agent().answer(
            context.standalone_question, context.chunks
        )
        if NOT_FOUND_TOKEN in raw_answer:
            context.response = _not_found(context, NOT_FOUND_MESSAGE)
            return "answer not present in the documents"

        verdict = get_validation_agent().validate(
            context.standalone_question, raw_answer, context.chunks
        )
        if not verdict["supported"]:
            logger.warning(
                "Agent 3 rejected the draft answer: %s", verdict["reason"]
            )
            response = _not_found(context, UNSUPPORTED_MESSAGE)
            response.validation = ValidationInfo(
                checked=True, supported=False, confidence=verdict["confidence"]
            )
            context.response = response
            return "draft rejected by validation"

        sources = [
            Source(
                filename=chunk.metadata.get("filename", "unknown"),
                page=(
                    chunk.metadata["page"] + 1
                    if isinstance(chunk.metadata.get("page"), int)
                    else None
                ),
                snippet=_snippet(chunk.page_content),
            )
            for chunk in context.chunks
        ]
        chat_history.add(context.session_id, context.question, raw_answer)
        context.response = AskResponse(
            answer=raw_answer,
            routed_document=_routed_filename(context),
            sources=sources,
            found=True,
            validation=ValidationInfo(
                checked=True, supported=True, confidence=verdict["confidence"]
            ),
        )
        return f"grounded answer, confidence={verdict['confidence']}"


class AutomateStage:
    """Execute follow-up actions for automation intents.

    Handler slot for the automation agent (Agent 6, next modules).
    Until it lands, non-automation intents skip cleanly and automation
    intents record that no handler is installed yet.
    """

    name = "automate"

    def run(self, context: WorkflowContext) -> str:
        if context.intent not in AUTOMATION_INTENTS:
            return f"skipped (intent '{context.intent}' needs no automation)"
        return "skipped (automation agent not installed yet)"


def build_default_stages() -> list:
    """The standard SidePilot pipeline, in poster order."""
    return [
        ObserveStage(),
        UnderstandStage(),
        AnalyzeStage(),
        GuideStage(),
        AutomateStage(),
    ]


def _default_intent(context: WorkflowContext) -> tuple[str, float]:
    """Trivial detector: everything is Q&A until Agent 5 lands."""
    return "question_answering", 1.0


def _routed_filename(context: WorkflowContext) -> str | None:
    """Filename of the routed document, or None for 'all documents'."""
    if context.routed_doc_id is None:
        return None
    meta = vector_store_manager.registry.get(context.routed_doc_id)
    return meta["filename"] if meta else None


def _not_found(context: WorkflowContext, message: str) -> AskResponse:
    """Build the explicit 'not found' response and record the turn."""
    chat_history.add(context.session_id, context.question, message)
    return AskResponse(
        answer=message,
        routed_document=_routed_filename(context),
        sources=[],
        found=False,
        validation=ValidationInfo(checked=False),
    )


def _snippet(text: str) -> str:
    """Trim chunk text to a short display snippet."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= SNIPPET_LENGTH:
        return cleaned
    return cleaned[:SNIPPET_LENGTH].rsplit(" ", 1)[0] + " ..."

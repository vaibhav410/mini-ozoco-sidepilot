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

from langchain_core.documents import Document

from app.agents.intent_agent import IntentResult, get_intent_agent
from app.agents.response_agent import get_response_agent
from app.agents.validation_agent import get_validation_agent
from app.models.schemas import (
    AskResponse,
    AutomationInfo,
    Source,
    ValidationInfo,
)
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

# Signature for a pluggable intent detector. The default is Agent 5;
# tests can inject a stub through UnderstandStage's constructor.
IntentDetector = Callable[[WorkflowContext], IntentResult]


def _agent5_intent(context: WorkflowContext) -> IntentResult:
    """Default detector: Agent 5 over the live registry + screen context."""
    return get_intent_agent().detect(
        context.standalone_question,
        vector_store_manager.registry,
        context.screen_context,
    )


class ObserveStage:
    """Gather everything already known before any LLM call is made."""

    name = "observe"

    def run(self, context: WorkflowContext) -> str:
        registry = vector_store_manager.registry
        # Screen-only mode: with screen context attached, the assistant
        # can help even before any document is uploaded.
        if not registry and context.screen_context is None:
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
        # Dependency injection point: Agent 5 by default, any callable
        # with the IntentDetector signature for tests/extensions.
        self._detect_intent = intent_detector or _agent5_intent

    def run(self, context: WorkflowContext) -> str:
        history = context.observations.get("history_text", "")
        agent2 = get_response_agent()
        context.standalone_question = (
            agent2.condense(context.question, history)
            if history
            else context.question
        )
        result = self._detect_intent(context)
        context.intent = result.intent
        context.intent_confidence = result.confidence
        context.recommended_workflow = result.recommended_workflow
        context.intent_method = result.method
        return (
            f"intent={result.intent} ({result.confidence:.2f}, "
            f"{result.method}) -> {result.recommended_workflow}"
        )


class AnalyzeStage:
    """Route the question to the right document and retrieve context."""

    name = "analyze"

    def run(self, context: WorkflowContext) -> str:
        registry = vector_store_manager.registry
        if registry:
            context.routed_doc_id = context.doc_id or get_response_agent().route(
                context.standalone_question, registry
            )
            context.chunks = retrieve_chunks(
                context.standalone_question, doc_id=context.routed_doc_id
            )
            note = (
                f"routed={context.routed_doc_id or 'all documents'}, "
                f"chunks={len(context.chunks)}"
            )
        else:
            note = "no documents (screen-only mode)"

        # Screen context becomes one extra grounding chunk, so answers
        # can reference what the user is currently looking at and the
        # validator can check claims against it like any other source.
        if context.screen_context is not None:
            context.chunks = [*context.chunks, _screen_chunk(context.screen_context)]
            note += ", +screen context"

        if not context.chunks:
            context.response = _not_found(context, NOT_FOUND_MESSAGE)
            return "no relevant context found"
        return note


class GuideStage:
    """Generate the grounded answer and validate it (Agents 2 + 3)."""

    name = "guide"

    def run(self, context: WorkflowContext) -> str:
        if context.response is not None:
            return "skipped (already resolved by an earlier stage)"
        if context.intent in AUTOMATION_INTENTS:
            # Automation requests are fulfilled by the Automate stage
            # (Agent 6), not by a RAG answer.
            return "skipped (automation intent -- deferred to Automate)"

        agent2 = get_response_agent()
        if context.token_callback is not None:
            raw_answer = agent2.answer_stream(
                context.standalone_question, context.chunks, context.token_callback
            )
        else:
            raw_answer = agent2.answer(context.standalone_question, context.chunks)
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
    """Execute follow-up actions for automation intents via Agent 6."""

    name = "automate"

    def run(self, context: WorkflowContext) -> str:
        if context.intent not in AUTOMATION_INTENTS:
            return f"skipped (intent '{context.intent}' needs no automation)"

        # Imported lazily so the Q&A path never pays for automation deps.
        from dataclasses import asdict

        from app.agents.automation_agent import get_automation_agent

        outcome = get_automation_agent().execute(context)
        context.automation_result = asdict(outcome)

        chat_history.add(context.session_id, context.question, outcome.detail)
        context.response = AskResponse(
            answer=outcome.detail,
            routed_document=_routed_filename(context),
            sources=[],
            found=outcome.status == "completed",
            validation=ValidationInfo(checked=False),
            automation=AutomationInfo(
                action=outcome.action,
                status=outcome.status,
                file=outcome.file,
                download_url=outcome.download_url,
                extra=outcome.extra,
            ),
        )
        return f"{outcome.action}: {outcome.status}"


def build_default_stages() -> list:
    """The standard SidePilot pipeline, in poster order."""
    return [
        ObserveStage(),
        UnderstandStage(),
        AnalyzeStage(),
        GuideStage(),
        AutomateStage(),
    ]


def _screen_chunk(screen_context: dict) -> Document:
    """Turn Agent 4's screen analysis into a retrievable grounding chunk."""
    content = "\n".join(
        part
        for part in (
            f"Application on screen: {screen_context.get('application', 'Unknown')}",
            f"What is happening: {screen_context.get('summary', '')}",
            f"On-screen text: {screen_context.get('detected_text', '')}",
        )
        if part.split(": ", 1)[-1].strip()
    )
    return Document(page_content=content, metadata={"filename": "Current screen"})


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

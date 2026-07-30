"""Question-answering orchestration -- the three-agent pipeline.

Flow per question:
  guards -> history condense (Agent 2) -> routing (Agent 2)
        -> retrieval -> draft answer (Agent 2)
        -> grounding validation (Agent 3) -> response + history update

Routes call this; this calls agents and the retriever. No HTTP code.
"""

from app.agents.response_agent import get_response_agent
from app.agents.validation_agent import get_validation_agent
from app.models.schemas import AskResponse, Source, ValidationInfo
from app.rag.prompt import NOT_FOUND_TOKEN
from app.rag.retriever import retrieve_chunks
from app.rag.vector_store import vector_store_manager
from app.services.history import chat_history
from app.utils.errors import DocumentNotFoundError, NoDocumentsError
from app.utils.logger import get_logger

logger = get_logger(__name__)

NOT_FOUND_MESSAGE = (
    "The answer to this question was not found in the uploaded documents."
)
UNSUPPORTED_MESSAGE = (
    "The generated answer could not be verified against the documents, "
    "so it was withheld to avoid giving you unreliable information."
)
SNIPPET_LENGTH = 240


def answer_question(
    question: str, doc_id: str | None = None, session_id: str = "default"
) -> AskResponse:
    """Answer a user question with conversational RAG and validation.

    Args:
        question: The user's natural-language question.
        doc_id: Optional explicit document to answer from.
        session_id: Conversation session; enables follow-up questions.

    Returns:
        AskResponse with answer, routing, sources, validation verdict,
        and the ``found`` flag (False = grounded "not found").

    Raises:
        NoDocumentsError: Nothing has been uploaded yet.
        DocumentNotFoundError: The given doc_id does not exist.
        AIServiceError: All LLM providers failed while generating.
    """
    registry = vector_store_manager.registry

    # Guards: an empty index or unknown doc_id are user errors, not AI ones.
    if not registry:
        raise NoDocumentsError(
            "No documents uploaded yet. Upload a PDF or TXT file first."
        )
    if doc_id is not None and doc_id not in registry:
        raise DocumentNotFoundError(f"No document found with id '{doc_id}'.")

    agent2 = get_response_agent()

    # AGENT 2 step 0 -- resolve follow-ups against chat history.
    # First questions have no history and skip this LLM call.
    history = chat_history.format(session_id)
    standalone = agent2.condense(question, history) if history else question

    # AGENT 2 step 1 -- routing (skipped when the caller pinned a doc_id).
    routed_id = doc_id or agent2.route(standalone, registry)
    routed_name = registry[routed_id]["filename"] if routed_id else None

    # AGENT 2 step 2 -- retrieval.
    chunks = retrieve_chunks(standalone, doc_id=routed_id)
    if not chunks:
        return _not_found(question, session_id, routed_name, NOT_FOUND_MESSAGE)

    # AGENT 2 step 3 -- grounded draft answer.
    raw_answer = agent2.answer(standalone, chunks)
    if NOT_FOUND_TOKEN in raw_answer:
        return _not_found(question, session_id, routed_name, NOT_FOUND_MESSAGE)

    # AGENT 3 -- validate the draft against the retrieved context.
    verdict = get_validation_agent().validate(standalone, raw_answer, chunks)
    if not verdict["supported"]:
        logger.warning("Agent 3 rejected the draft answer: %s", verdict["reason"])
        response = _not_found(question, session_id, routed_name, UNSUPPORTED_MESSAGE)
        response.validation = ValidationInfo(
            checked=True, supported=False, confidence=verdict["confidence"]
        )
        return response

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
        for chunk in chunks
    ]

    chat_history.add(session_id, question, raw_answer)
    return AskResponse(
        answer=raw_answer,
        routed_document=routed_name,
        sources=sources,
        found=True,
        validation=ValidationInfo(
            checked=True, supported=True, confidence=verdict["confidence"]
        ),
    )


def _not_found(
    question: str, session_id: str, routed_name: str | None, message: str
) -> AskResponse:
    """Build the explicit 'not found' response and record the turn."""
    chat_history.add(session_id, question, message)
    return AskResponse(
        answer=message,
        routed_document=routed_name,
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

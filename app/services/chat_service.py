"""Question-answering orchestration.

The full answer pipeline lives here:
guard -> Agent 2 routing -> retrieval -> Agent 2 generation -> sources.
Routes call this; this calls the agent and the retriever. No HTTP code.
"""

from app.agents.response_agent import get_response_agent
from app.models.schemas import AskResponse, Source
from app.rag.prompt import NOT_FOUND_TOKEN
from app.rag.retriever import retrieve_chunks
from app.rag.vector_store import vector_store_manager
from app.utils.errors import DocumentNotFoundError, NoDocumentsError
from app.utils.logger import get_logger

logger = get_logger(__name__)

NOT_FOUND_MESSAGE = (
    "The answer to this question was not found in the uploaded documents."
)
SNIPPET_LENGTH = 240


def answer_question(question: str, doc_id: str | None = None) -> AskResponse:
    """Answer a user question with RAG, orchestrated by Agent 2.

    Args:
        question: The user's natural-language question.
        doc_id: Optional explicit document to answer from.

    Returns:
        AskResponse with the answer, routing info, sources, and the
        ``found`` flag (False = grounded "not found", never a guess).

    Raises:
        NoDocumentsError: Nothing has been uploaded yet.
        DocumentNotFoundError: The given doc_id does not exist.
        AIServiceError: Gemini failed while generating.
    """
    registry = vector_store_manager.registry

    # Guards: an empty index or unknown doc_id are user errors, not AI ones.
    if not registry:
        raise NoDocumentsError(
            "No documents uploaded yet. Upload a PDF or TXT file first."
        )
    if doc_id is not None and doc_id not in registry:
        raise DocumentNotFoundError(f"No document found with id '{doc_id}'.")

    agent = get_response_agent()

    # AGENT 2 step 1 -- routing (skipped when the caller pinned a doc_id).
    routed_id = doc_id or agent.route(question, registry)
    routed_name = registry[routed_id]["filename"] if routed_id else None

    # AGENT 2 step 2 -- retrieval.
    chunks = retrieve_chunks(question, doc_id=routed_id)
    if not chunks:
        return AskResponse(
            answer=NOT_FOUND_MESSAGE,
            routed_document=routed_name,
            sources=[],
            found=False,
        )

    # AGENT 2 step 3 -- grounded generation.
    raw_answer = agent.answer(question, chunks)

    # The sentinel means: context did not contain the answer. We return a
    # clean message instead of letting the model improvise one.
    if NOT_FOUND_TOKEN in raw_answer:
        return AskResponse(
            answer=NOT_FOUND_MESSAGE,
            routed_document=routed_name,
            sources=[],
            found=False,
        )

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

    return AskResponse(
        answer=raw_answer,
        routed_document=routed_name,
        sources=sources,
        found=True,
    )


def _snippet(text: str) -> str:
    """Trim chunk text to a short display snippet."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= SNIPPET_LENGTH:
        return cleaned
    return cleaned[:SNIPPET_LENGTH].rsplit(" ", 1)[0] + " ..."

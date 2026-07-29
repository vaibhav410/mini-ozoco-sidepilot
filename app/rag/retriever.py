"""Retriever: fetch the top-k chunks relevant to a question.

Thin, intentionally simple layer over the vector store so retrieval
behavior (k, filtering, logging) lives in exactly one place.
"""

from langchain_core.documents import Document

from app.config import settings
from app.rag.vector_store import vector_store_manager
from app.utils.logger import get_logger

logger = get_logger(__name__)


def retrieve_chunks(
    question: str, doc_id: str | None = None, k: int | None = None
) -> list[Document]:
    """Return the chunks most relevant to the question.

    Args:
        question: The user's natural-language question.
        doc_id: Optional document filter (set by Agent 2's routing).
        k: How many chunks to retrieve; defaults to ``settings.top_k``.

    Returns:
        Up to ``k`` chunks ordered by semantic similarity.
    """
    top_k = k or settings.top_k
    chunks = vector_store_manager.search(question, k=top_k, doc_id=doc_id)
    logger.info(
        "Retrieved %d chunk(s) for question (doc filter: %s)",
        len(chunks),
        doc_id or "all documents",
    )
    return chunks

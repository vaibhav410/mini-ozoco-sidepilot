"""Agent 2 -- Response Generation / Routing Agent.

Runs once per user question. First routes the question to the relevant
document (using the registry Agent 1 built), then generates a grounded
answer from retrieved context. It never invents information: if the
context lacks the answer, it emits the NOT_FOUND sentinel.
"""

from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from app.agents.llm import build_llm, model_used
from app.rag.prompt import ANSWER_PROMPT, NOT_FOUND_TOKEN, ROUTING_PROMPT
from app.utils.errors import ai_service_error_from
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ResponseAgent:
    """Routes questions and generates grounded answers using Gemini."""

    def __init__(self) -> None:
        # Falls back to Groq automatically when Gemini is unavailable.
        llm = build_llm(temperature=0.2)
        self._routing_chain = ROUTING_PROMPT | llm
        self._answer_chain = ANSWER_PROMPT | llm

    def route(self, question: str, registry: dict[str, dict[str, Any]]) -> str | None:
        """Pick the document most relevant to the question.

        Args:
            question: The user's question.
            registry: ``doc_id -> metadata`` written by Agent 1.

        Returns:
            The chosen ``doc_id``, or ``None`` meaning "no specific
            document" (search across all of them). With a single uploaded
            document, routing is trivial and skips the LLM call entirely.
        """
        if len(registry) == 1:
            only_doc = next(iter(registry))
            logger.info("AGENT 2 | routing: single document -> %s", only_doc)
            return only_doc

        # Registry preserves insertion order, so numbering the documents
        # lets the routing prompt resolve "this document" to the newest.
        documents_block = "\n".join(
            f"{position}. doc_id: {doc_id} | file: {meta['filename']} | "
            f"category: {meta['category']} | summary: {meta['summary']}"
            for position, (doc_id, meta) in enumerate(registry.items(), start=1)
        )
        try:
            response = self._routing_chain.invoke(
                {"documents": documents_block, "question": question}
            )
        except Exception as exc:
            # Routing is an optimization -- fall back to searching all
            # documents rather than failing the whole request.
            logger.warning("Agent 2 routing call failed (%s); searching all", exc)
            return None

        choice = str(response.content).strip().strip("`'\"")
        if choice in registry:
            logger.info("AGENT 2 | routed question to doc_id=%s", choice)
            return choice
        logger.info("AGENT 2 | routing result '%s' -> searching all documents", choice)
        return None

    def answer(self, question: str, chunks: list[Document]) -> str:
        """Generate a grounded answer from the retrieved chunks.

        Args:
            question: The user's question.
            chunks: Top-k chunks returned by the retriever.

        Returns:
            The raw answer text; may be the ``NOT_FOUND_TOKEN`` sentinel.

        Raises:
            AIServiceError: If the Gemini call fails.
        """
        context = "\n\n---\n\n".join(
            f"[Source: {chunk.metadata.get('filename', 'unknown')}"
            f"{self._page_ref(chunk)}]\n{chunk.page_content}"
            for chunk in chunks
        )
        try:
            response = self._answer_chain.invoke(
                {"context": context, "question": question}
            )
        except Exception as exc:
            logger.error("Agent 2 Gemini call failed: %s", exc)
            raise ai_service_error_from(exc) from exc

        answer_text = str(response.content).strip()
        grounded = NOT_FOUND_TOKEN not in answer_text
        logger.info(
            "AGENT 2 | answer generated (grounded=%s, %d context chunks) [model: %s]",
            grounded,
            len(chunks),
            model_used(response),
        )
        return answer_text

    @staticmethod
    def _page_ref(chunk: Document) -> str:
        """Format a page reference like ', page 3' when available."""
        page = chunk.metadata.get("page")
        return f", page {page + 1}" if isinstance(page, int) else ""


@lru_cache(maxsize=1)
def get_response_agent() -> ResponseAgent:
    """Return the shared Agent 2 instance (created lazily)."""
    return ResponseAgent()

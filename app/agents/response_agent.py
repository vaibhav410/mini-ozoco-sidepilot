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
from app.rag.prompt import (
    ANSWER_PROMPT,
    CONDENSE_PROMPT,
    NOT_FOUND_TOKEN,
    ROUTING_PROMPT,
)
from app.utils.errors import ai_service_error_from
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ResponseAgent:
    """Routes questions and generates grounded answers using Gemini."""

    def __init__(self) -> None:
        # Falls back to Groq automatically when Gemini is unavailable.
        llm = build_llm(temperature=0.2)
        self._condense_chain = CONDENSE_PROMPT | llm
        self._routing_chain = ROUTING_PROMPT | llm
        self._answer_chain = ANSWER_PROMPT | llm

    def condense(self, question: str, history: str) -> str:
        """Rewrite a follow-up question as a standalone question.

        Uses the recent chat history to resolve references ("he", "that
        document"). Called only when history exists -- first questions
        skip this LLM call entirely.

        Args:
            question: The user's raw question.
            history: Prompt-ready transcript of recent turns.

        Returns:
            The standalone question, or the original question if the
            rewrite fails (condensation is an optimization, not a gate).
        """
        try:
            response = self._condense_chain.invoke(
                {"history": history, "question": question}
            )
        except Exception as exc:
            logger.warning("Agent 2 condense call failed (%s); using original", exc)
            return question
        standalone = str(response.content).strip().strip('"')
        if not standalone:
            return question
        if standalone.lower() != question.lower():
            logger.info("AGENT 2 | follow-up condensed to: %s", standalone)
        return standalone

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
        try:
            response = self._answer_chain.invoke(
                {"context": self._format_context(chunks), "question": question}
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

    def answer_stream(self, question, chunks: list[Document], on_token) -> str:
        """Generate a grounded answer, emitting tokens as they arrive.

        Args:
            question: The user's question.
            chunks: Top-k chunks returned by the retriever.
            on_token: Callable fired with each generated text fragment.

        Returns:
            The complete answer text (same contract as :meth:`answer`).

        Raises:
            AIServiceError: If the streaming call fails.
        """
        parts: list[str] = []
        try:
            for piece in self._answer_chain.stream(
                {"context": self._format_context(chunks), "question": question}
            ):
                text = str(piece.content)
                if text:
                    parts.append(text)
                    on_token(text)
        except Exception as exc:
            logger.error("Agent 2 streaming call failed: %s", exc)
            raise ai_service_error_from(exc) from exc

        answer_text = "".join(parts).strip()
        logger.info(
            "AGENT 2 | streamed answer (%d chars, %d context chunks)",
            len(answer_text),
            len(chunks),
        )
        return answer_text

    @classmethod
    def _format_context(cls, chunks: list[Document]) -> str:
        """Render retrieved chunks as the prompt's context block."""
        return "\n\n---\n\n".join(
            f"[Source: {chunk.metadata.get('filename', 'unknown')}"
            f"{cls._page_ref(chunk)}]\n{chunk.page_content}"
            for chunk in chunks
        )

    @staticmethod
    def _page_ref(chunk: Document) -> str:
        """Format a page reference like ', page 3' when available."""
        page = chunk.metadata.get("page")
        return f", page {page + 1}" if isinstance(page, int) else ""


@lru_cache(maxsize=1)
def get_response_agent() -> ResponseAgent:
    """Return the shared Agent 2 instance (created lazily)."""
    return ResponseAgent()

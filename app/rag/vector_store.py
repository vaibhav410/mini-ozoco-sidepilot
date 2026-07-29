"""FAISS vector store manager + in-memory document registry.

This module is the shared state both agents communicate through:

- Agent 1 writes: chunks stamped with metadata (doc_id, filename,
  category) into FAISS, and the document's profile into the registry.
- Agent 2 reads: the registry for routing, and FAISS for retrieval.
"""

from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.rag.embeddings import get_embeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Owns the FAISS index and the registry of uploaded documents."""

    def __init__(self) -> None:
        self._index: FAISS | None = None
        # doc_id -> {filename, category, summary, topics, chunks}
        self.registry: dict[str, dict[str, Any]] = {}

    @property
    def document_count(self) -> int:
        """Number of documents currently indexed."""
        return len(self.registry)

    def add_document(
        self,
        doc_id: str,
        filename: str,
        category: str,
        summary: str,
        topics: list[str],
        chunks: list[Document],
    ) -> int:
        """Index a processed document's chunks and record it in the registry.

        Args:
            doc_id: Unique id assigned at upload time.
            filename: Original filename (used in source references).
            category: Category assigned by Agent 1.
            summary: Short summary produced by Agent 1.
            topics: Key topics produced by Agent 1.
            chunks: Chunked documents from the splitter.

        Returns:
            The number of chunks indexed.
        """
        for chunk_no, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "category": category,
                    "chunk_no": chunk_no,
                }
            )

        if self._index is None:
            # First document: create the FAISS index from these chunks.
            self._index = FAISS.from_documents(chunks, get_embeddings())
        else:
            self._index.add_documents(chunks)

        self.registry[doc_id] = {
            "filename": filename,
            "category": category,
            "summary": summary,
            "topics": topics,
            "chunks": len(chunks),
        }
        logger.info(
            "Indexed '%s' (doc_id=%s, category=%s) with %d chunks",
            filename,
            doc_id,
            category,
            len(chunks),
        )
        return len(chunks)

    def search(
        self, query: str, k: int, doc_id: str | None = None
    ) -> list[Document]:
        """Semantic similarity search over the index.

        Args:
            query: The user's question.
            k: Number of chunks to return.
            doc_id: Optional filter so only one document's chunks match.

        Returns:
            The top-k most similar chunks (empty list if the index is empty).
        """
        if self._index is None:
            return []
        metadata_filter = {"doc_id": doc_id} if doc_id else None
        return self._index.similarity_search(query, k=k, filter=metadata_filter)


# Single shared instance -- the communication channel between the agents.
vector_store_manager = VectorStoreManager()

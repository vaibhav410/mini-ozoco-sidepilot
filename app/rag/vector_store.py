"""FAISS vector store manager + in-memory document registry.

This module is the shared state both agents communicate through:

- Agent 1 writes: chunks stamped with metadata (doc_id, filename,
  category) into FAISS, and the document's profile into the registry.
- Agent 2 reads: the registry for routing, and FAISS for retrieval.
"""

import json
import threading
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import settings
from app.rag.embeddings import get_embeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_REGISTRY_FILE = "registry.json"


class VectorStoreManager:
    """Owns the FAISS index and the registry of uploaded documents.

    Thread-safe: a single lock guards every mutation of the index and
    registry, so concurrent uploads/deletes can't corrupt shared state.
    The index and registry are persisted to disk so documents survive
    a restart (matching the persistence of chat history).
    """

    def __init__(self) -> None:
        self._index: FAISS | None = None
        # doc_id -> {filename, category, summary, topics, chunks, chunk_ids}
        self.registry: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

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

        with self._lock:
            if self._index is None:
                # First document: create the FAISS index from these chunks.
                self._index = FAISS.from_documents(chunks, get_embeddings())
                chunk_ids = list(self._index.index_to_docstore_id.values())
            else:
                chunk_ids = self._index.add_documents(chunks)

            self.registry[doc_id] = {
                "filename": filename,
                "category": category,
                "summary": summary,
                "topics": topics,
                "chunks": len(chunks),
                # Stored so the document can be deleted from FAISS later.
                "chunk_ids": chunk_ids,
            }
            self._persist()
        logger.info(
            "Indexed '%s' (doc_id=%s, category=%s) with %d chunks",
            filename,
            doc_id,
            category,
            len(chunks),
        )
        return len(chunks)

    def remove_document(self, doc_id: str) -> bool:
        """Delete one document's chunks from FAISS and the registry.

        Args:
            doc_id: Id assigned at upload time.

        Returns:
            True if the document existed and was removed.
        """
        with self._lock:
            meta = self.registry.pop(doc_id, None)
            if meta is None:
                return False
            if self._index is not None and meta.get("chunk_ids"):
                try:
                    self._index.delete(meta["chunk_ids"])
                except Exception as exc:  # index inconsistency must not 500
                    logger.warning("FAISS delete failed for %s: %s", doc_id, exc)
            if not self.registry:
                self._index = None  # empty index -> fresh start
            self._persist()
        logger.info(
            "Removed '%s' (doc_id=%s) from the index", meta["filename"], doc_id
        )
        return True

    # ------------------------------------------------------------------
    # Persistence: index + registry survive restarts (BUG-01)
    # ------------------------------------------------------------------
    def _persist(self) -> None:
        """Save index + registry to disk. Caller holds the lock."""
        if not settings.index_dir:
            return
        try:
            registry_path = settings.index_dir / _REGISTRY_FILE
            if self._index is None:
                # Empty store: clear any stale files.
                registry_path.unlink(missing_ok=True)
                (settings.index_dir / "index.faiss").unlink(missing_ok=True)
                (settings.index_dir / "index.pkl").unlink(missing_ok=True)
                return
            self._index.save_local(str(settings.index_dir))
            registry_path.write_text(json.dumps(self.registry), encoding="utf-8")
        except Exception as exc:  # persistence must never break a request
            logger.warning("Vector store persist failed: %s", exc)

    def load(self) -> None:
        """Restore index + registry from disk at startup, if present."""
        if not settings.index_dir:
            return
        registry_path = settings.index_dir / _REGISTRY_FILE
        faiss_path = settings.index_dir / "index.faiss"
        if not (registry_path.exists() and faiss_path.exists()):
            return
        try:
            with self._lock:
                self._index = FAISS.load_local(
                    str(settings.index_dir),
                    get_embeddings(),
                    allow_dangerous_deserialization=True,
                )
                self.registry = json.loads(
                    registry_path.read_text(encoding="utf-8")
                )
            logger.info(
                "Restored %d document(s) from %s",
                len(self.registry),
                settings.index_dir,
            )
        except Exception as exc:
            logger.warning("Vector store restore failed (%s); starting empty", exc)
            self._index = None
            self.registry = {}

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

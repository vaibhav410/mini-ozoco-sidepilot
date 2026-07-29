"""Document upload orchestration.

The full ingestion pipeline lives here:
validate -> save -> load -> Agent 1 (classify/summarize) -> chunk -> index.
Routes call this; this calls agents and RAG modules. No HTTP code here.
"""

from pathlib import Path
from uuid import uuid4

from app.agents.document_agent import get_document_agent
from app.config import settings
from app.models.schemas import UploadResponse
from app.rag.loader import SUPPORTED_EXTENSIONS, documents_to_text, load_document
from app.rag.splitter import split_documents
from app.rag.vector_store import vector_store_manager
from app.utils.errors import (
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFileError,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def process_upload(filename: str, data: bytes) -> UploadResponse:
    """Run the complete ingestion pipeline for one uploaded file.

    Synchronous on purpose: the route offloads this whole pipeline to
    FastAPI's threadpool so the blocking work (Gemini call, embeddings,
    FAISS indexing) never stalls the event loop. Taking (filename, bytes)
    instead of an UploadFile also keeps this layer free of FastAPI types.

    Args:
        filename: Original filename of the upload.
        data: Raw file bytes.

    Returns:
        UploadResponse with the document id and Agent 1's analysis.

    Raises:
        UnsupportedFileError: Bad extension or unparseable file.
        FileTooLargeError: File exceeds the configured limit.
        EmptyDocumentError: File has no readable text.
        AIServiceError: Gemini failed during classification.
    """
    original_name = Path(filename or "upload").name
    suffix = Path(original_name).suffix.lower()

    # 1. Validate extension before doing any work.
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(
            f"Unsupported file type '{suffix or 'unknown'}'. "
            "Supported types: PDF, TXT."
        )

    # 2. Validate size.
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise FileTooLargeError(
            f"File is too large. Maximum size is {settings.max_file_size_mb} MB."
        )
    if not data:
        raise EmptyDocumentError("The uploaded file is empty.")

    # 3. Save with a unique prefix so same-named files never collide.
    doc_id = uuid4().hex[:8]
    saved_path = settings.upload_dir / f"{doc_id}_{original_name}"
    saved_path.write_bytes(data)
    logger.info("Saved upload '%s' as %s", original_name, saved_path.name)

    # 4. Extract text (raises for corrupt/empty documents).
    documents = load_document(saved_path)
    full_text = documents_to_text(documents)

    # 5. AGENT 1: classify + summarize + topics.
    analysis = get_document_agent().analyze(full_text, original_name)

    # 6. Chunk and index with Agent 1's metadata attached (this is the
    #    hand-off point between Agent 1 and Agent 2).
    chunks = split_documents(documents)
    chunks_indexed = vector_store_manager.add_document(
        doc_id=doc_id,
        filename=original_name,
        category=analysis["category"],
        summary=analysis["summary"],
        topics=analysis["topics"],
        chunks=chunks,
    )

    return UploadResponse(
        doc_id=doc_id,
        filename=original_name,
        category=analysis["category"],
        summary=analysis["summary"],
        topics=analysis["topics"],
        chunks_indexed=chunks_indexed,
        status="indexed",
    )

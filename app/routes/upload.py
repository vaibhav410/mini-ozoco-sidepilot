"""POST /upload -- document upload endpoint (HTTP layer only)."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.models.schemas import (
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentsResponse,
    UploadResponse,
)
from app.rag.vector_store import vector_store_manager
from app.services.document_service import process_upload, remove_document
from app.utils.errors import AppError
from app.utils.logger import get_logger
from app.utils.security import require_admin

logger = get_logger(__name__)

router = APIRouter(tags=["Documents"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a PDF/TXT document",
    description=(
        "Uploads one document, runs Agent 1 (classification + summary), "
        "chunks and indexes it into the FAISS vector store."
    ),
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Handle a document upload request.

    The file is read asynchronously; the blocking pipeline (Agent 1,
    embeddings, FAISS) runs in the threadpool so the server stays
    responsive while a document is being processed.
    """
    data = await file.read()
    try:
        return await run_in_threadpool(process_upload, file.filename or "upload", data)
    except AppError as err:
        # Expected domain failures -> clean HTTP error with right status.
        raise HTTPException(status_code=err.status_code, detail=err.detail)
    except Exception:
        logger.exception("Unexpected error while processing upload")
        raise HTTPException(status_code=500, detail="Unexpected server error.")


@router.get(
    "/documents",
    response_model=DocumentsResponse,
    summary="List all indexed documents",
    description="Returns every document in the vector store with the "
    "category and summary produced by Agent 1.",
)
def list_documents() -> DocumentsResponse:
    """Expose the document registry built by Agent 1."""
    return DocumentsResponse(
        documents=[
            DocumentInfo(
                doc_id=doc_id,
                filename=meta["filename"],
                category=meta["category"],
                summary=meta["summary"],
                chunks=meta["chunks"],
            )
            for doc_id, meta in vector_store_manager.registry.items()
        ]
    )


@router.delete(
    "/documents/{doc_id}",
    response_model=DeleteDocumentResponse,
    summary="Remove a document from the index",
    description="Deletes the document's chunks from FAISS, its registry "
    "entry, and the saved file -- so its content can no longer influence "
    "answers or drafts. Requires ADMIN_TOKEN when one is configured.",
    dependencies=[Depends(require_admin)],
)
def delete_document(doc_id: str) -> DeleteDocumentResponse:
    """Handle a document removal request."""
    if not remove_document(doc_id):
        raise HTTPException(
            status_code=404, detail=f"No document found with id '{doc_id}'."
        )
    return DeleteDocumentResponse(doc_id=doc_id)

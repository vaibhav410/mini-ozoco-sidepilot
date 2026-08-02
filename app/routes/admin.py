"""GET /admin -- monitoring dashboard and its stats API (HTTP layer only)."""

import sys
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.config import settings
from app.integrations.filesystem import list_exports
from app.rag.vector_store import vector_store_manager
from app.services.memory_service import memory
from app.utils.logger import get_logger
from app.utils.metrics import metrics

logger = get_logger(__name__)

router = APIRouter(tags=["Admin"])

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"


@router.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    """Serve the admin dashboard page."""
    return FileResponse(
        STATIC_DIR / "admin.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.get(
    "/admin/stats",
    summary="System, pipeline and memory statistics",
    description=(
        "Everything the monitoring dashboard shows: process health and "
        "memory usage, model configuration, indexed documents, "
        "persistent-memory counts, per-stage pipeline timings, and "
        "per-endpoint HTTP timings."
    ),
)
def admin_stats() -> dict:
    """Aggregate live statistics from every subsystem."""
    return {
        "system": {
            "python": sys.version.split()[0],
            "process_memory_mb": _process_memory_mb(),
            "model": settings.gemini_model,
            "vision_model": settings.gemini_vision_model,
            "fallback_model": settings.groq_model if settings.groq_api_key else None,
            "embeddings_backend": settings.embeddings_backend,
        },
        "documents": [
            {
                "doc_id": doc_id,
                "filename": meta["filename"],
                "category": meta["category"],
                "chunks": meta["chunks"],
            }
            for doc_id, meta in vector_store_manager.registry.items()
        ],
        "memory": memory.stats(),
        "exports": len(list_exports()),
        "metrics": metrics.snapshot(),
    }


def _process_memory_mb() -> float | None:
    """Resident memory of this process; None if psutil is missing."""
    try:
        import os

        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None

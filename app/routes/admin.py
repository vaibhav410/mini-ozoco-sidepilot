"""GET /admin -- monitoring dashboard and its stats API (HTTP layer only)."""

import sys
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.integrations.filesystem import list_exports
from app.rag.vector_store import vector_store_manager
from app.services.memory_service import memory
from app.utils.auth import get_session_role, require_admin_any
from app.utils.logger import get_logger
from app.utils.metrics import metrics

logger = get_logger(__name__)

router = APIRouter(tags=["Admin"])

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/admin", include_in_schema=False)
def admin_page(request: Request, token: str = ""):
    """Serve the admin dashboard shell -- ADMIN_EMAIL only.

    Gated server-side by the Clerk session cookie: a signed-out visitor
    is bounced to the landing page's sign-in flow, and a signed-in
    non-admin is forwarded to "/app" rather than ever seeing the
    dashboard shell -- this dashboard is never exposed to regular
    users, matching the same 401-vs-403 distinction require_admin_any
    enforces on the /admin/stats API underneath. The historical
    ?token=<ADMIN_TOKEN> query path is unaffected, for non-browser
    access without a Clerk session at all.
    """
    if settings.admin_token and token and token == settings.admin_token:
        pass
    else:
        role = get_session_role(request)
        if role == "anonymous":
            return RedirectResponse(url="/?signin=1&redirect_url=/admin")
        if role != "admin":
            return RedirectResponse(url="/app")
    response = templates.TemplateResponse(
        request,
        "admin.html",
        {"clerk_publishable_key": settings.clerk_publishable_key},
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@router.get(
    "/admin/stats",
    summary="System, pipeline and memory statistics",
    description=(
        "Everything the monitoring dashboard shows: process health and "
        "memory usage, model configuration, indexed documents, "
        "persistent-memory counts, per-stage pipeline timings, and "
        "per-endpoint HTTP timings. Requires ADMIN_TOKEN or a signed-in "
        "admin session."
    ),
    dependencies=[Depends(require_admin_any)],
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

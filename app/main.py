"""Mini OZOCO SidePilot AI System -- FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.config import settings
from app.models.schemas import HealthResponse
from app.rag.vector_store import vector_store_manager
from app.routes.chat import router as chat_router
from app.routes.exports import router as exports_router
from app.routes.intent import router as intent_router
from app.routes.screen import router as screen_router
from app.routes.upload import router as upload_router
from app.utils.logger import get_logger

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook: validate configuration before serving."""
    settings.validate()  # fail fast if GOOGLE_API_KEY is missing

    # Persistent memory: PostgreSQL/SQLite when reachable, in-memory
    # fallback otherwise -- the app serves either way.
    from app.services.memory_service import memory

    memory.initialize()
    logger.info(
        "Startup OK | model=%s | embeddings=%s | top_k=%d",
        settings.gemini_model,
        settings.embedding_model,
        settings.top_k,
    )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Mini OZOCO SidePilot AI System",
    description=(
        "Document Q&A with conversational RAG and a three-agent workflow: "
        "**Agent 1** classifies and summarizes uploaded documents; "
        "**Agent 2** resolves follow-ups via chat history, routes questions, "
        "retrieves context from FAISS, and generates grounded answers with "
        "source references; **Agent 3** validates every answer against the "
        "sources before it is returned; **Agent 4** understands screenshots "
        "of the user's screen via Gemini Vision with an OCR fallback."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(screen_router)
app.include_router(intent_router)
app.include_router(exports_router)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service health check",
)
def health() -> HealthResponse:
    """Report service status and how many documents are indexed."""
    return HealthResponse(
        status="ok", documents_indexed=vector_store_manager.document_count
    )


@app.get("/", include_in_schema=False)
def serve_ui() -> FileResponse:
    """Serve the single-page web UI.

    No-cache headers ensure browsers always load the latest UI after a
    redeploy instead of showing a stale cached version.
    """
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )

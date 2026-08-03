"""Vercel entrypoint -- exposes the same FastAPI app as a serverless function.

Unlike Render/Railway/Fly, Vercel's filesystem is read-only outside /tmp
and every invocation can land on a different, short-lived instance: the
in-memory document registry, the FAISS index on disk, and a local SQLite
file are NOT reliably shared across requests here. UPLOAD_DIR/INDEX_DIR/
EXPORTS_DIR must point under /tmp and DATABASE_URL should point at a real
hosted Postgres for anything beyond a single-request demo.
"""

from app.main import app

__all__ = ["app"]

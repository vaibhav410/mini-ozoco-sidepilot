"""SQLAlchemy engine and session factory.

Created lazily by :func:`init_engine` so importing the module never
opens a connection, and the app can degrade to in-memory history when
no database is reachable.
"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def init_engine() -> Engine:
    """Create (once) and return the engine for the configured URL.

    Returns:
        The SQLAlchemy engine.

    Raises:
        Exception: If the database is unreachable or the URL invalid --
            callers treat this as "persistence unavailable".
    """
    global _engine, _session_factory
    if _engine is not None and _session_factory is not None:
        return _engine

    url = settings.database_url
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
        # Ensure the parent directory of the SQLite file exists.
        db_path = url.replace("sqlite:///", "", 1)
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)

    from app.db import models  # noqa: F401 - register tables on Base

    Base.metadata.create_all(engine)

    # Globals are assigned only after everything succeeded, so a failed
    # attempt never leaves a half-initialised engine behind.
    _session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    _engine = engine

    backend = "postgresql" if url.startswith("postgres") else url.split(":", 1)[0]
    logger.info("Database ready (backend=%s)", backend)
    return _engine


def get_session() -> Session:
    """Open a new ORM session (engine must be initialised first)."""
    if _session_factory is None:
        raise RuntimeError("Database engine not initialised")
    return _session_factory()

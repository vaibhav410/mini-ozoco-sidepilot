"""Filesystem integration: safe management of the exports directory.

Every generated artifact (email drafts, exported summaries, action
plans) lives in one exports directory. All paths pass through
``safe_export_path`` so a crafted filename can never escape it.
"""

import re
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
_MAX_SLUG = 40


def exports_dir() -> Path:
    """Return the exports directory, creating it on first use."""
    directory = settings.exports_dir
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_export_path(filename: str) -> Path:
    """Resolve a filename inside the exports directory, safely.

    Args:
        filename: The requested file name (client- or agent-supplied).

    Returns:
        An absolute path inside the exports directory.

    Raises:
        ValueError: If the name is empty after sanitisation.
    """
    name = Path(filename).name  # strips any directory components
    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid export filename: {filename!r}")
    return exports_dir() / name


def timestamped_name(title: str, extension: str) -> str:
    """Build a unique, filesystem-friendly export filename.

    Args:
        title: Human title to slugify (e.g. "Summary of resume.pdf").
        extension: File extension without the dot (e.g. "md", "pdf").

    Returns:
        A name like ``summary-of-resume-20260803-104512.md``.
    """
    slug = _SLUG_PATTERN.sub("-", title.lower()).strip("-")[:_MAX_SLUG] or "export"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{slug}-{stamp}.{extension}"


def save_text(filename: str, content: str) -> Path:
    """Write a text file into the exports directory.

    Args:
        filename: Target file name (sanitised).
        content: UTF-8 text content.

    Returns:
        The path of the written file.
    """
    path = safe_export_path(filename)
    path.write_text(content, encoding="utf-8")
    logger.info("Saved export %s (%d bytes)", path.name, path.stat().st_size)
    return path


def list_exports() -> list[dict]:
    """List generated files, newest first (for the API and admin UI)."""
    files = [
        {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(
                timespec="seconds"
            ),
        }
        for path in exports_dir().iterdir()
        if path.is_file()
    ]
    return sorted(files, key=lambda f: f["modified"], reverse=True)

"""Application-wide logging setup.

Usage in any module:

    from app.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Document classified as %s", category)
"""

import logging
import sys

# One consistent format everywhere: time | level | module | message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root() -> None:
    """Configure the root logger exactly once for the whole process."""
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stdout,
    )
    # Quiet down noisy third-party libraries so our own logs stay readable.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger named after the calling module.

    Args:
        name: Usually ``__name__`` of the module requesting the logger.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    _configure_root()
    return logging.getLogger(name)

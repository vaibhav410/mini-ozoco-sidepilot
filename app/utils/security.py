"""Lightweight, dependency-free security primitives.

- A per-IP sliding-window rate limiter (protects LLM quota from bursts).
- An early Content-Length guard (rejects oversized bodies before they
  are buffered into memory -- the upload memory-DoS fix).

Admin authorization lives in app.utils.auth (require_admin_any), which
combines the ADMIN_TOKEN check with Clerk-based login.

All are process-local and need no Redis, so they run on the free tier.
"""

import threading
import time
from collections import defaultdict, deque

from app.config import settings

# Largest body we will ever accept, across any endpoint (documents,
# images, audio). Requests above this are rejected before being read.
_ABS_MAX_BODY_BYTES = max(
    settings.max_file_size_mb,
    settings.max_image_size_mb,
    16,  # audio + form overhead floor
) * 1024 * 1024 + (1024 * 1024)  # +1 MB multipart overhead headroom


class RateLimiter:
    """Per-client sliding-window request limiter."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self._limit = limit
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str, now: float | None = None) -> bool:
        """Return True if this client may proceed, recording the hit."""
        if self._limit <= 0:  # 0/negative disables limiting
            return True
        now = now if now is not None else time.monotonic()
        with self._lock:
            hits = self._hits[client]
            cutoff = now - self._window
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            return True


rate_limiter = RateLimiter(
    settings.rate_limit_requests, settings.rate_limit_window_seconds
)


def max_body_bytes() -> int:
    """Absolute maximum request body size accepted by any endpoint."""
    return _ABS_MAX_BODY_BYTES

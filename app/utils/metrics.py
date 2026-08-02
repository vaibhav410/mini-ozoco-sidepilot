"""In-process metrics registry: counters and rolling timings.

Deliberately dependency-free (no Prometheus client) so it runs on the
Render free tier; the admin dashboard reads ``snapshot()``. Timings
keep a rolling window per key -- enough for live averages/percentiles
without unbounded memory.
"""

import threading
import time
from collections import defaultdict, deque
from typing import Any

_WINDOW = 200  # samples kept per timing key


class MetricsRegistry:
    """Thread-safe counters + rolling timing statistics."""

    def __init__(self, window: int = _WINDOW) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._timings: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._totals: dict[str, int] = defaultdict(int)
        self._started = time.time()

    def increment(self, key: str, amount: int = 1) -> None:
        """Add to a named counter."""
        with self._lock:
            self._counters[key] += amount

    def record_timing(self, key: str, duration_ms: float) -> None:
        """Record one duration sample for a named operation."""
        with self._lock:
            self._timings[key].append(duration_ms)
            self._totals[key] += 1

    def snapshot(self) -> dict[str, Any]:
        """Aggregate view for the admin dashboard / monitoring."""
        with self._lock:
            timings: dict[str, Any] = {}
            for key, samples in self._timings.items():
                if not samples:
                    continue
                ordered = sorted(samples)
                p95_index = max(0, int(len(ordered) * 0.95) - 1)
                timings[key] = {
                    "total": self._totals[key],
                    "avg_ms": round(sum(samples) / len(samples), 1),
                    "p95_ms": round(ordered[p95_index], 1),
                    "last_ms": round(samples[-1], 1),
                }
            return {
                "uptime_seconds": round(time.time() - self._started, 1),
                "counters": dict(self._counters),
                "timings": timings,
            }


# Single shared registry for the whole process.
metrics = MetricsRegistry()

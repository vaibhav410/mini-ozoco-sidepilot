"""Per-request context shared across the stack.

A contextvar carries the request id from the HTTP middleware into every
log line (via the logging filter) without threading it through function
signatures. Worker threads must propagate it explicitly -- see
``chat_service.stream_events``.
"""

import contextvars

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)

"""Clerk authentication: verifies session tokens on protected routes.

Two dependencies cover every protected route in the app:

- ``require_user``: any signed-in Clerk account. Used on the functional
  API (``/ask``, ``/upload``, ``/screen/analyze``, ...) so a public
  deployment isn't wide open to anonymous quota abuse.
- ``require_admin_any``: the one admin account. Accepts EITHER the
  existing ``ADMIN_TOKEN`` header/query (unchanged, for curl/API use)
  OR a signed-in Clerk user whose email matches ``ADMIN_EMAIL``.

Session verification uses Clerk's official ``authenticate_request``
helper against the ``__session`` cookie or an ``Authorization: Bearer``
header (FastAPI's ``Request.headers`` already satisfies the
``Requestish`` protocol it expects, so it's passed straight through).

Fail-closed by design: if Clerk/ADMIN_EMAIL isn't configured, protected
routes reject with 503 rather than silently admitting every request --
the previous single-user demo's "open when unset" default is not safe
once real login is in the picture.
"""

import time
from typing import Any

from clerk_backend_api import AuthenticateRequestOptions, Clerk, authenticate_request
from fastapi import Header, HTTPException, Query, Request

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Cache of user_id -> (email, expiry) so the admin check doesn't hit
# Clerk's Users API on every single request. Session tokens carry email
# directly only if a custom claim is configured (Dashboard -> Sessions
# -> Customize session token -> add {"email": "{{user.primary_email_address}}"});
# without it, this cache is what keeps the fallback API call cheap.
_EMAIL_CACHE_TTL_SECONDS = 600
_email_cache: dict[str, tuple[str | None, float]] = {}


def _clerk_client() -> Clerk:
    return Clerk(bearer_auth=settings.clerk_secret_key)


def _auth_state(request: Request):
    """Verify the request's session token; never raises."""
    return authenticate_request(
        request,
        AuthenticateRequestOptions(
            secret_key=settings.clerk_secret_key,
            authorized_parties=list(settings.clerk_authorized_parties) or None,
        ),
    )


def require_user(request: Request) -> dict[str, Any]:
    """FastAPI dependency: reject unless a valid Clerk session is present.

    Returns:
        The verified session token's JWT payload (at least ``sub``,
        the Clerk user id).

    Raises:
        HTTPException: 503 if Clerk isn't configured; 401 if signed out.
    """
    if not settings.clerk_secret_key:
        raise HTTPException(status_code=503, detail="Authentication is not configured.")
    state = _auth_state(request)
    if not state.is_signed_in:
        raise HTTPException(status_code=401, detail="Sign in required.")
    return state.payload or {}


def _resolve_email(user_id: str) -> str | None:
    """Look up a Clerk user's primary email, cached for cheap re-checks."""
    if not user_id:
        return None
    cached = _email_cache.get(user_id)
    if cached and cached[1] > time.monotonic():
        return cached[0]

    email: str | None = None
    try:
        user = _clerk_client().users.get(user_id=user_id)
        for addr in user.email_addresses or []:
            if addr.id == user.primary_email_address_id:
                email = addr.email_address
                break
        if email is None and user.email_addresses:
            email = user.email_addresses[0].email_address
    except Exception as exc:
        logger.warning("Could not resolve email for Clerk user %s: %s", user_id, exc)

    _email_cache[user_id] = (email, time.monotonic() + _EMAIL_CACHE_TTL_SECONDS)
    return email


def _is_admin_session(request: Request) -> bool:
    """Whether the request carries a signed-in session for ADMIN_EMAIL."""
    if not (settings.clerk_secret_key and settings.admin_email):
        return False
    state = _auth_state(request)
    if not state.is_signed_in:
        return False
    payload = state.payload or {}
    email = payload.get("email") or _resolve_email(payload.get("sub", ""))
    return bool(email) and email.lower() == settings.admin_email.lower()


def require_admin_any(
    request: Request,
    authorization: str = Header(default=""),
    x_admin_token: str = Header(default=""),
    token: str = Query(default=""),
) -> None:
    """FastAPI dependency: ADMIN_TOKEN (any header/query form) OR a
    signed-in Clerk session for ADMIN_EMAIL.

    Fails closed: if neither mechanism is configured, every request is
    rejected rather than admitted -- this is a change from the old
    token-only checker, which allowed everyone through when
    ``ADMIN_TOKEN`` was left unset.

    Raises:
        HTTPException: 401 if neither check passes.
    """
    if settings.admin_token:
        supplied = (
            authorization.removeprefix("Bearer ").strip()
            or x_admin_token.strip()
            or token.strip()
        )
        if supplied == settings.admin_token:
            return
    if _is_admin_session(request):
        return
    raise HTTPException(status_code=401, detail="Admin authorization required.")

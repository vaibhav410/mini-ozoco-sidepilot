"""Gmail integration: email draft creation.

Honest about credentials: without Gmail OAuth configured, the draft is
generated as a standards-compliant ``.eml`` file (opens in Outlook,
Thunderbird, Apple Mail and imports into Gmail) plus a ``mailto:`` link
that opens the user's mail client pre-filled. When ``GMAIL_TOKEN_JSON``
is configured a real Gmail API draft can be created through the same
function surface -- callers never change.
"""

from email.message import EmailMessage
from urllib.parse import quote

from app.config import settings
from app.integrations.filesystem import safe_export_path, timestamped_name
from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_draft(to: str, subject: str, body: str) -> dict:
    """Create an email draft through the best available backend.

    Args:
        to: Recipient address ("" allowed -- user fills it in later).
        subject: Email subject line.
        body: Plain-text email body.

    Returns:
        Dict with ``backend`` ("gmail_api" or "eml_file"), ``file``
        (generated .eml filename, if any), and ``mailto`` (pre-filled
        mailto link).
    """
    # mailto links longer than ~2000 chars are silently dropped by
    # Windows/browsers, so the body is capped -- the .eml file and the
    # UI's copy button carry the full text.
    mailto = (
        f"mailto:{quote(to)}?subject={quote(subject[:150])}"
        f"&body={quote(body[:1200])}"
    )
    # Browser Gmail compose: works even on machines with no default
    # mail app configured (where mailto silently does nothing).
    gmail_web = (
        "https://mail.google.com/mail/?view=cm&fs=1"
        f"&to={quote(to)}&su={quote(subject[:150])}&body={quote(body[:1200])}"
    )

    if settings.gmail_token_json:
        result = _create_gmail_api_draft(to, subject, body)
        if result is not None:
            return {
                "backend": "gmail_api",
                "file": None,
                "mailto": mailto,
                "gmail_web": gmail_web,
                **result,
            }
        logger.warning("Gmail API draft failed; falling back to .eml file")

    message = EmailMessage()
    message["To"] = to or "recipient@example.com"
    message["Subject"] = subject
    message.set_content(body)

    path = safe_export_path(timestamped_name(f"draft-{subject or 'email'}", "eml"))
    path.write_bytes(bytes(message))
    logger.info("Email draft saved as %s", path.name)
    return {
        "backend": "eml_file",
        "file": path.name,
        "mailto": mailto,
        "gmail_web": gmail_web,
    }


def _create_gmail_api_draft(to: str, subject: str, body: str) -> dict | None:
    """Create a real Gmail draft via the API; None on any failure.

    Imported lazily: google-api-python-client is an optional dependency
    that only users who configure OAuth need to install.
    """
    try:
        import base64
        import json

        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        credentials = Credentials.from_authorized_user_info(
            json.loads(settings.gmail_token_json),
            scopes=["https://www.googleapis.com/auth/gmail.compose"],
        )
        service = build("gmail", "v1", credentials=credentials)

        message = EmailMessage()
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)
        encoded = base64.urlsafe_b64encode(bytes(message)).decode("ascii")

        draft = (
            service.users()
            .drafts()
            .create(userId="me", body={"message": {"raw": encoded}})
            .execute()
        )
        logger.info("Gmail API draft created (id=%s)", draft.get("id"))
        return {"draft_id": draft.get("id")}
    except Exception as exc:  # missing package, bad token, API error
        logger.warning("Gmail API unavailable: %s", exc)
        return None

"""Application error hierarchy.

Services raise these domain errors; routes translate them into HTTP
responses using ``status_code`` and ``detail``. This keeps FastAPI
imports out of the business-logic layer.
"""


class AppError(Exception):
    """Base class for expected, user-facing application failures."""

    status_code: int = 500

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class UnsupportedFileError(AppError):
    """Uploaded file type is not supported or the file is unreadable."""

    status_code = 400


class EmptyDocumentError(AppError):
    """The uploaded file contains no extractable text."""

    status_code = 400


class FileTooLargeError(AppError):
    """The uploaded file exceeds the configured size limit."""

    status_code = 413


class NoDocumentsError(AppError):
    """A question was asked before any document was uploaded."""

    status_code = 400


class DocumentNotFoundError(AppError):
    """The requested document id does not exist in the registry."""

    status_code = 404


class AIServiceError(AppError):
    """The Gemini API call failed (rate limit, network, timeout)."""

    status_code = 502


def ai_service_error_from(exc: Exception) -> AIServiceError:
    """Map a provider exception to a user-friendly AIServiceError.

    Distinguishes free-tier rate limiting (the most common failure during
    demos) from generic outages, so the user knows whether waiting helps.
    """
    text = str(exc)
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return AIServiceError(
            "Gemini rate limit reached (free tier). "
            "Please wait about a minute and try again."
        )
    return AIServiceError("AI service is temporarily unavailable. Please try again.")

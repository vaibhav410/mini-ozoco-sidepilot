"""Speech-to-text service backed by Groq Whisper.

The browser records audio with MediaRecorder (supported everywhere,
unlike the Web Speech API) and posts it here; Groq's hosted Whisper
transcribes it. Reuses the existing GROQ_API_KEY -- no new secrets.
"""

import httpx

from app.config import settings
from app.utils.errors import AIServiceError, EmptyDocumentError
from app.utils.logger import get_logger

logger = get_logger(__name__)

GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MAX_AUDIO_BYTES = 15 * 1024 * 1024
_TIMEOUT_SECONDS = 60


def transcribe_audio(filename: str, data: bytes, content_type: str) -> str:
    """Transcribe one recorded audio clip to text.

    Args:
        filename: Client filename (extension hints the container type).
        data: Raw audio bytes (webm/ogg/wav/mp3 ...).
        content_type: MIME type reported by the browser.

    Returns:
        The transcribed text (may be empty for silence).

    Raises:
        EmptyDocumentError: No audio data was sent or it is too large.
        AIServiceError: Groq is not configured or the API call failed.
    """
    if not data:
        raise EmptyDocumentError("No audio received.")
    if len(data) > MAX_AUDIO_BYTES:
        raise EmptyDocumentError("Audio clip is too large (max 15 MB).")
    if not settings.groq_api_key:
        raise AIServiceError(
            "Speech-to-text is not configured: set GROQ_API_KEY to enable "
            "voice input."
        )

    try:
        response = httpx.post(
            GROQ_TRANSCRIBE_URL,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            data={"model": settings.speech_model, "response_format": "json"},
            files={
                "file": (
                    filename or "voice.webm",
                    data,
                    content_type or "application/octet-stream",
                )
            },
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Whisper transcription failed (%d): %s",
            exc.response.status_code,
            exc.response.text[:200],
        )
        raise AIServiceError(
            "Voice transcription failed. Please try again."
        ) from exc
    except httpx.HTTPError as exc:
        logger.error("Whisper request error: %s", exc)
        raise AIServiceError(
            "Voice transcription is temporarily unavailable."
        ) from exc

    text = str(response.json().get("text", "")).strip()
    logger.info(
        "Transcribed %d KB of audio -> %d chars [model: %s]",
        len(data) // 1024,
        len(text),
        settings.speech_model,
    )
    return text

"""POST /speech/transcribe -- voice input endpoint (HTTP layer only)."""

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.models.schemas import TranscribeResponse
from app.services.speech_service import transcribe_audio
from app.utils.errors import AppError
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Speech"])


@router.post(
    "/speech/transcribe",
    response_model=TranscribeResponse,
    summary="Transcribe recorded voice input to text",
    description=(
        "Accepts an audio clip recorded in the browser (webm/ogg/wav/mp3) "
        "and returns the transcribed text via Groq Whisper. Works in every "
        "browser, unlike the Web Speech API."
    ),
)
async def transcribe(file: UploadFile = File(...)) -> TranscribeResponse:
    """Handle a voice transcription request."""
    data = await file.read()
    try:
        text = await run_in_threadpool(
            transcribe_audio,
            file.filename or "voice.webm",
            data,
            file.content_type or "",
        )
    except AppError as err:
        raise HTTPException(status_code=err.status_code, detail=err.detail)
    except Exception:
        logger.exception("Unexpected error while transcribing audio")
        raise HTTPException(status_code=500, detail="Unexpected server error.")
    return TranscribeResponse(text=text)

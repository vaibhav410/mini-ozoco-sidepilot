"""POST /screen/analyze -- screen understanding endpoint (HTTP layer only)."""

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.models.schemas import ScreenAnalyzeResponse
from app.services.screen_service import analyze_screenshot
from app.utils.errors import AppError
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Screen Understanding"])


@router.post(
    "/screen/analyze",
    response_model=ScreenAnalyzeResponse,
    summary="Analyze a screenshot of the user's screen",
    description=(
        "Runs Agent 4: a PNG/JPEG screenshot is analyzed with Gemini "
        "Vision to detect the visible application, the on-screen text, "
        "a summary of what is happening, the user's likely intent, and "
        "suggested next actions. If the vision call fails, the image "
        "falls back to local OCR (PyTesseract) interpreted by the text "
        "LLM, so the endpoint degrades gracefully instead of erroring."
    ),
)
async def analyze_screen(file: UploadFile = File(...)) -> ScreenAnalyzeResponse:
    """Handle a screenshot analysis request.

    The file is read asynchronously; the blocking pipeline (Pillow,
    Gemini Vision, OCR) runs in the threadpool so the server stays
    responsive while a screenshot is being analyzed.
    """
    data = await file.read()
    try:
        return await run_in_threadpool(
            analyze_screenshot, file.filename or "screenshot", data
        )
    except AppError as err:
        raise HTTPException(status_code=err.status_code, detail=err.detail)
    except Exception:
        logger.exception("Unexpected error while analyzing screenshot")
        raise HTTPException(status_code=500, detail="Unexpected server error.")

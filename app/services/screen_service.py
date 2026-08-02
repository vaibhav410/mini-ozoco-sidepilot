"""Screen understanding orchestration.

The full pipeline for one screenshot lives here:

validate -> decode + downscale (Pillow) -> Agent 4 Gemini Vision
        -> [on vision failure] OCR (PyTesseract) -> text-LLM interpretation
        -> [on LLM failure] raw OCR text only
        -> structured ScreenAnalyzeResponse

Routes call this; this calls the agent and the OCR service. No HTTP
code here, mirroring document_service / chat_service.
"""

import base64
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from app.agents.vision_agent import get_screen_agent
from app.config import settings
from app.models.schemas import ScreenAnalyzeResponse
from app.services.ocr_service import extract_text
from app.utils.errors import (
    AIServiceError,
    FileTooLargeError,
    InvalidImageError,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Pillow format name -> MIME type sent to Gemini.
SUPPORTED_IMAGE_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg"}

# Screenshots larger than this are downscaled before the vision call:
# plenty of detail for understanding, much smaller request payload.
MAX_DIMENSION = 1600

OCR_ONLY_SUMMARY = (
    "Raw on-screen text was extracted with OCR, but no AI provider was "
    "available to interpret it."
)


def analyze_screenshot(filename: str, data: bytes) -> ScreenAnalyzeResponse:
    """Run the complete screen-understanding pipeline for one screenshot.

    Synchronous on purpose (like ``process_upload``): the route offloads
    it to FastAPI's threadpool so the blocking work (Pillow, Gemini, OCR)
    never stalls the event loop.

    Args:
        filename: Original filename of the upload (logging context only).
        data: Raw image bytes.

    Returns:
        ScreenAnalyzeResponse with the structured screen understanding
        and the ``analysis_method`` that produced it.

    Raises:
        InvalidImageError: Not a readable PNG/JPEG image.
        FileTooLargeError: Image exceeds the configured size limit.
        AIServiceError: Vision failed AND no OCR text could be extracted.
    """
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if not data:
        raise InvalidImageError("The uploaded image is empty.")
    if len(data) > max_bytes:
        raise FileTooLargeError(
            f"Image is too large. Maximum size is {settings.max_image_size_mb} MB."
        )

    image, mime_type = _decode_image(data)
    payload, payload_mime = _prepare_payload(image, data, mime_type)
    image_b64 = base64.b64encode(payload).decode("ascii")
    logger.info(
        "Screenshot '%s' accepted (%s, %dx%d, %d KB payload)",
        filename,
        payload_mime,
        image.width,
        image.height,
        len(payload) // 1024,
    )

    agent = get_screen_agent()

    # Primary path: one multimodal Gemini call.
    try:
        analysis = agent.analyze_image(image_b64, payload_mime)
        return ScreenAnalyzeResponse(**analysis, analysis_method="gemini_vision")
    except Exception as exc:
        logger.warning("Vision analysis failed (%s); trying OCR fallback", exc)

    # Fallback path: OCR on the full-resolution image, then the text LLM.
    ocr_text = extract_text(image)
    if not ocr_text:
        raise AIServiceError(
            "Screen analysis is temporarily unavailable: the vision model "
            "failed and OCR could not extract any text. Please try again."
        )
    try:
        analysis = agent.analyze_ocr_text(ocr_text)
        return ScreenAnalyzeResponse(**analysis, analysis_method="ocr_llm")
    except Exception as exc:
        # Last resort: return the raw OCR text so the user still gets
        # *something* useful even with every AI provider down.
        logger.warning("OCR interpretation failed (%s); returning raw text", exc)
        return ScreenAnalyzeResponse(
            application="Unknown",
            activity="Could not be determined.",
            detected_text=ocr_text[:4000],
            summary=OCR_ONLY_SUMMARY,
            user_intent="unknown",
            suggested_actions=[],
            analysis_method="ocr_only",
        )


def _decode_image(data: bytes) -> tuple[Image.Image, str]:
    """Decode and validate the upload as a PNG or JPEG image.

    Returns:
        The opened Pillow image and its MIME type.

    Raises:
        InvalidImageError: Corrupt data or an unsupported format.
    """
    try:
        # verify() detects truncated/corrupt files but consumes the
        # object, so the image is reopened for actual use afterwards.
        probe = Image.open(BytesIO(data))
        probe.verify()
        image = Image.open(BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(
            "The uploaded file could not be read as an image. "
            "Supported formats: PNG, JPEG."
        ) from exc

    image_format = (image.format or "").upper()
    if image_format not in SUPPORTED_IMAGE_FORMATS:
        raise InvalidImageError(
            f"Unsupported image format '{image_format or 'unknown'}'. "
            "Supported formats: PNG, JPEG."
        )
    return image, SUPPORTED_IMAGE_FORMATS[image_format]


def _prepare_payload(
    image: Image.Image, original: bytes, mime_type: str
) -> tuple[bytes, str]:
    """Downscale oversized screenshots for the vision call.

    Small images pass through untouched (no re-encode, no quality loss).
    Large ones are resized on a copy -- the original stays full-resolution
    for the OCR fallback -- and re-encoded as PNG, which keeps UI text
    crisp.

    Returns:
        The payload bytes to send and their MIME type.
    """
    if max(image.size) <= MAX_DIMENSION:
        return original, mime_type

    scaled = image.copy()
    scaled.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
    buffer = BytesIO()
    scaled.save(buffer, format="PNG")
    logger.info(
        "Screenshot downscaled %dx%d -> %dx%d for vision call",
        image.width,
        image.height,
        scaled.width,
        scaled.height,
    )
    return buffer.getvalue(), "image/png"

"""OCR fallback for screen understanding (PyTesseract).

Used only when the Gemini Vision call fails (rate limit, outage): the
screenshot's raw text is extracted locally and handed to the text LLM
instead. Missing dependencies are treated as "OCR unavailable", never
as a crash -- the vision path must keep working without Tesseract
installed (e.g. on the Render free tier).
"""

from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import settings
from app.utils.logger import get_logger

if TYPE_CHECKING:  # Pillow types only needed for annotations
    from PIL.Image import Image

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_pytesseract():
    """Import and configure pytesseract once; ``None`` if unavailable."""
    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract is not installed; OCR fallback disabled")
        return None
    if settings.tesseract_cmd:
        # Windows installs Tesseract outside PATH by default; the .env
        # can point straight at the binary.
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    return pytesseract


def ocr_available() -> bool:
    """Report whether the OCR fallback can run in this environment."""
    return _load_pytesseract() is not None


def extract_text(image: "Image") -> str:
    """Run OCR over a screenshot and return the extracted text.

    Args:
        image: The decoded Pillow image (full resolution -- OCR accuracy
            drops on downscaled screenshots).

    Returns:
        The extracted text, or an empty string on any failure (missing
        Tesseract binary, unreadable image). OCR is a fallback, so it
        degrades silently rather than raising.
    """
    pytesseract = _load_pytesseract()
    if pytesseract is None:
        return ""
    try:
        text = pytesseract.image_to_string(image)
    except Exception as exc:  # binary missing / crashed on this image
        logger.warning("OCR extraction failed: %s", exc)
        return ""
    text = text.strip()
    logger.info("OCR extracted %d characters from screenshot", len(text))
    return text

"""Agent 4 -- Screen Understanding / Vision Agent.

Analyzes a screenshot of the user's screen and produces a structured
understanding: the visible application, what the user is doing, the key
on-screen text, the likely intent, and suggested next actions.

Two analysis paths, chosen by the service layer:

- ``analyze_image``: the primary path -- one multimodal Gemini call with
  the screenshot inline. It never uses the Groq fallback because the
  Groq model is text-only; the real fallback for vision is OCR.
- ``analyze_ocr_text``: the fallback path -- interprets PyTesseract's
  raw text with the shared text LLM (which itself falls back from
  Gemini to Groq like every other agent).

Both paths return the same dict shape, so downstream consumers (the
route today, the workflow engine in a later module) never care which
path produced the result.
"""

from functools import lru_cache

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.llm import build_llm, model_used
from app.config import settings
from app.rag.prompt import OCR_SCREEN_PROMPT, SCREEN_UNDERSTANDING_INSTRUCTIONS
from app.utils.errors import ai_service_error_from
from app.utils.json_utils import extract_json_object
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Bounds keep responses compact and prompts cheap.
MAX_ACTIONS = 5
MAX_TEXT_CHARS = 4000
MAX_OCR_PROMPT_CHARS = 6000


class ScreenAgent:
    """Understands screenshots via Gemini Vision, with an OCR text path."""

    def __init__(self) -> None:
        # Vision must run on Gemini directly (no .with_fallbacks): the
        # configured Groq model cannot accept images, so a failed vision
        # call falls back to OCR in the service layer instead.
        self._vision_llm = ChatGoogleGenerativeAI(
            model=settings.gemini_vision_model,
            google_api_key=settings.google_api_key,
            temperature=0.2,
            max_retries=1,  # fail fast so the OCR fallback engages quickly
        )
        # Text-only interpretation of OCR output reuses the shared
        # factory and inherits the Gemini -> Groq fallback.
        self._ocr_chain = OCR_SCREEN_PROMPT | build_llm(temperature=0.2)

    def analyze_image(self, image_b64: str, mime_type: str) -> dict:
        """Understand a screenshot with a single multimodal Gemini call.

        Args:
            image_b64: Base64-encoded image payload (already validated
                and downscaled by the service layer).
            mime_type: ``image/png`` or ``image/jpeg``.

        Returns:
            Dict with keys ``application``, ``activity``,
            ``detected_text``, ``summary``, ``user_intent``,
            ``suggested_actions``.

        Raises:
            AIServiceError: If the Gemini Vision call fails.
        """
        message = HumanMessage(
            content=[
                {"type": "text", "text": SCREEN_UNDERSTANDING_INSTRUCTIONS},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                },
            ]
        )
        try:
            response = self._vision_llm.invoke([message])
        except Exception as exc:
            logger.error("Agent 4 vision call failed: %s", exc)
            raise ai_service_error_from(exc) from exc

        result = self._parse(str(response.content))
        logger.info(
            "AGENT 4 | screen analyzed: app='%s' intent='%s' [model: %s]",
            result["application"],
            result["user_intent"],
            model_used(response),
        )
        return result

    def analyze_ocr_text(self, ocr_text: str) -> dict:
        """Understand a screenshot from its OCR-extracted text.

        Args:
            ocr_text: Raw text produced by the OCR service.

        Returns:
            The same dict shape as :meth:`analyze_image`.

        Raises:
            AIServiceError: If all text LLM providers fail.
        """
        excerpt = ocr_text[:MAX_OCR_PROMPT_CHARS]
        try:
            response = self._ocr_chain.invoke({"ocr_text": excerpt})
        except Exception as exc:
            logger.error("Agent 4 OCR-interpretation call failed: %s", exc)
            raise ai_service_error_from(exc) from exc

        result = self._parse(str(response.content), fallback_text=excerpt)
        logger.info(
            "AGENT 4 | OCR text interpreted: app='%s' intent='%s' [model: %s]",
            result["application"],
            result["user_intent"],
            model_used(response),
        )
        return result

    @staticmethod
    def _parse(raw: str, fallback_text: str = "") -> dict:
        """Normalize the model's JSON into the agreed result shape.

        A malformed response must not fail the request: missing fields
        degrade to neutral defaults, and if no JSON was returned at all
        the raw prose becomes the summary (graceful degradation).
        """
        data = extract_json_object(raw) or {}
        prose = " ".join(raw.split())

        def field(key: str, default: str = "") -> str:
            return str(data.get(key, "") or "").strip() or default

        actions_raw = data.get("suggested_actions", [])
        if not isinstance(actions_raw, list):
            actions_raw = [actions_raw]
        actions = [str(a).strip() for a in actions_raw if str(a).strip()]

        return {
            "application": field("application", "Unknown"),
            "activity": field("activity", "Could not be determined."),
            "detected_text": field("detected_text", fallback_text)[:MAX_TEXT_CHARS],
            "summary": field("summary", prose[:400] or "No summary available."),
            "user_intent": field("user_intent", "unknown"),
            "suggested_actions": actions[:MAX_ACTIONS],
        }


@lru_cache(maxsize=1)
def get_screen_agent() -> ScreenAgent:
    """Return the shared Agent 4 instance (created lazily)."""
    return ScreenAgent()

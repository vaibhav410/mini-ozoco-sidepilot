"""Agent 1 -- Document Understanding / Classification Agent.

Runs once per uploaded document. Classifies it into a category,
writes a short summary, and extracts key topics. Its structured output
becomes the metadata Agent 2 later uses for routing.
"""

import json
import re
from functools import lru_cache

from app.agents.llm import build_llm, model_used
from app.rag.prompt import CLASSIFICATION_PROMPT
from app.utils.errors import ai_service_error_from
from app.utils.logger import get_logger

logger = get_logger(__name__)

# The document is classified from its beginning -- more than enough
# signal for category/summary while keeping the Gemini call small.
MAX_ANALYSIS_CHARS = 4000

VALID_CATEGORIES = {
    "Resume",
    "Invoice",
    "Research Paper",
    "Report",
    "General Document",
}


class DocumentAgent:
    """Classifies and summarizes uploaded documents using Gemini."""

    def __init__(self) -> None:
        # Low temperature -> stable, factual output. Falls back to Groq
        # automatically when Gemini is unavailable (see app.agents.llm).
        self._chain = CLASSIFICATION_PROMPT | build_llm(temperature=0.1)

    def analyze(self, text: str, filename: str) -> dict:
        """Classify a document and produce summary + topics.

        Args:
            text: Full extracted document text.
            filename: Original filename (context hint only).

        Returns:
            Dict with keys ``category``, ``summary``, ``topics``.

        Raises:
            AIServiceError: If the Gemini call itself fails.
        """
        excerpt = text[:MAX_ANALYSIS_CHARS]
        try:
            response = self._chain.invoke(
                {"filename": filename, "text": excerpt}
            )
        except Exception as exc:
            logger.error("Agent 1 Gemini call failed: %s", exc)
            raise ai_service_error_from(exc) from exc

        result = self._parse(str(response.content), filename)
        logger.info(
            "AGENT 1 | '%s' classified as '%s' (topics: %s) [model: %s]",
            filename,
            result["category"],
            ", ".join(result["topics"]) or "-",
            model_used(response),
        )
        return result

    @staticmethod
    def _parse(raw: str, filename: str) -> dict:
        """Parse Gemini's JSON output with a safe fallback.

        A malformed LLM response must not fail the upload -- we degrade
        to 'General Document' and keep going (graceful degradation).
        """
        fallback = {
            "category": "General Document",
            "summary": f"Uploaded document '{filename}'.",
            "topics": [],
        }
        # Strip optional ```json fences and grab the outermost {...} block.
        cleaned = re.sub(r"```(?:json)?", "", raw).strip("` \n")
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            logger.warning("Agent 1 returned non-JSON output; using fallback")
            return fallback
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Agent 1 returned invalid JSON; using fallback")
            return fallback

        category = str(data.get("category", "")).strip()
        if category not in VALID_CATEGORIES:
            category = "General Document"
        summary = str(data.get("summary", "")).strip() or fallback["summary"]
        topics_raw = data.get("topics", [])
        topics = [str(t).strip() for t in topics_raw if str(t).strip()][:6]

        return {"category": category, "summary": summary, "topics": topics}


@lru_cache(maxsize=1)
def get_document_agent() -> DocumentAgent:
    """Return the shared Agent 1 instance (created lazily, after config
    validation has run at startup)."""
    return DocumentAgent()

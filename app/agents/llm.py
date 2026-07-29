"""Chat model factory shared by both agents.

Builds the primary Gemini model and, when a Groq key is configured,
wraps it with an automatic fallback: if the Gemini call raises (rate
limit, outage), the exact same prompt is retried on Groq. Both agents
stay provider-agnostic -- they just call `build_llm(temperature)`.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_llm(temperature: float):
    """Return the chat model for an agent, with optional Groq fallback.

    Args:
        temperature: Sampling temperature for both primary and fallback.

    Returns:
        A LangChain runnable chat model: Gemini alone, or Gemini with a
        Groq fallback when GROQ_API_KEY is configured.
    """
    primary = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
        # Fail fast: one retry only, so the fallback engages in seconds
        # instead of waiting out the SDK's long exponential backoff.
        max_retries=1,
    )
    if not settings.groq_api_key:
        return primary

    # Imported lazily so the app runs fine without the optional package.
    from langchain_groq import ChatGroq

    fallback = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=temperature,
    )
    logger.info(
        "Groq fallback enabled: %s -> %s", settings.gemini_model, settings.groq_model
    )
    return primary.with_fallbacks([fallback])


def model_used(response) -> str:
    """Best-effort name of the model that actually produced a response."""
    meta = getattr(response, "response_metadata", {}) or {}
    return meta.get("model_name") or settings.gemini_model

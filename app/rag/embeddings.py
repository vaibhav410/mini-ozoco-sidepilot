"""Shared embedding model with two interchangeable backends.

Chosen by EMBEDDINGS_BACKEND in .env:

- "local"  (default): HuggingFace sentence-transformer running on this
  machine via PyTorch. Free, fast, fully offline indexing -- the
  assignment's primary stack.
- "gemini": Google's embedding API (gemini-embedding-001). No PyTorch in
  the process, so small-RAM hosts (e.g. Render's 512 MB free tier) can
  run the app. Uses the same GOOGLE_API_KEY; embeddings have their own
  free-tier quota pool separate from the chat models.

Either way the model is created exactly once and cached for the process.
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return the process-wide embedding model (created on first call).

    Returns:
        A LangChain embeddings instance for the configured backend.
    """
    if settings.embeddings_backend == "gemini":
        # Imported lazily: this path must work without PyTorch or
        # sentence-transformers installed at all.
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        logger.info("Loading REMOTE embeddings: Gemini embedding API")
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.google_api_key,
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Loading LOCAL embedding model: %s", settings.embedding_model)
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)

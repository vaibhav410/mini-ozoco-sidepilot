"""Central configuration for the Mini OZOCO SidePilot AI System.

Every setting the application needs is read here, once, from environment
variables (loaded from the `.env` file). No other module should call
`os.getenv` directly -- they import `settings` instead.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains `app/` (and `.env`, `uploads/`).
BASE_DIR = Path(__file__).resolve().parents[1]

# Read the .env file at the project root and load its values into the
# process environment. Real environment variables take precedence.
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment variables."""

    # --- AI providers ---
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    # Optional fallback provider used automatically when Gemini fails.
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    # "local" runs the sentence-transformer on this machine (default).
    # "gemini" calls the Gemini embedding API instead -- used on
    # small-RAM hosts (e.g. Render free tier) where PyTorch cannot fit.
    embeddings_backend: str = os.getenv("EMBEDDINGS_BACKEND", "local")

    # --- RAG tuning ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    # --- Uploads ---
    upload_dir: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
    )
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))

    def validate(self) -> None:
        """Fail fast at startup if a required setting is missing.

        Raises:
            RuntimeError: If the Gemini API key was not provided.
        """
        if not self.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env "
                "and add your Gemini API key."
            )
        self.upload_dir.mkdir(parents=True, exist_ok=True)


# Single shared instance imported by the rest of the application.
settings = Settings()

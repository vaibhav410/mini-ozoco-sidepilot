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

    # --- Screen understanding (Agent 4) ---
    # gemini-2.5-flash is multimodal, so vision reuses the chat model by
    # default; override GEMINI_VISION_MODEL to use a dedicated one.
    gemini_vision_model: str = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
    # Secondary vision model tried when the primary fails. Free-tier
    # quotas are PER MODEL, so flash-lite usually still has quota left
    # when flash is exhausted -- and Groq can't do vision at all.
    gemini_vision_fallback_model: str = os.getenv(
        "GEMINI_VISION_FALLBACK_MODEL", "gemini-2.5-flash-lite"
    )
    max_image_size_mb: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "8"))
    # Optional explicit path to the Tesseract binary for the OCR fallback
    # (Windows installs it outside PATH); empty means "use PATH".
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "")

    # --- RAG tuning ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    # --- Uploads ---
    upload_dir: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
    )
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))

    # --- Speech (voice input) ---
    # Groq-hosted Whisper model for /speech/transcribe.
    speech_model: str = os.getenv("SPEECH_MODEL", "whisper-large-v3-turbo")

    # --- Persistent memory ---
    # PostgreSQL in production (postgresql+psycopg2://user:pass@host/db);
    # defaults to a local SQLite file so dev + free hosting need no setup.
    database_url: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'sidepilot.db'}"
    )

    # --- Automation & integrations (Agent 6) ---
    exports_dir: Path = field(
        default_factory=lambda: BASE_DIR / os.getenv("EXPORTS_DIR", "exports")
    )
    # Optional Gmail OAuth token JSON for real API drafts; empty means
    # drafts are generated as downloadable .eml files instead.
    gmail_token_json: str = os.getenv("GMAIL_TOKEN_JSON", "")

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
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        if self.database_url.startswith("sqlite"):
            (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)


# Single shared instance imported by the rest of the application.
settings = Settings()

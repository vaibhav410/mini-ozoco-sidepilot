"""Document loading: turn an uploaded PDF/TXT file into raw text.

Uses LangChain loaders: PyPDFLoader for PDFs (page-aware, keeps page
numbers for source references) and TextLoader for plain text.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from app.utils.errors import EmptyDocumentError, UnsupportedFileError
from app.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_document(path: Path) -> list[Document]:
    """Load a PDF or TXT file into LangChain documents.

    Args:
        path: Path of the saved uploaded file.

    Returns:
        A list of ``Document`` objects (one per PDF page, or one for TXT),
        each carrying ``page_content`` and metadata.

    Raises:
        UnsupportedFileError: If the extension is unsupported or the file
            cannot be parsed.
        EmptyDocumentError: If no readable text was extracted.
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            documents = PyPDFLoader(str(path)).load()
        except Exception as exc:  # corrupt / password-protected / not a PDF
            logger.error("PDF parsing failed for %s: %s", path.name, exc)
            raise UnsupportedFileError(
                "The file could not be read as a PDF. It may be corrupt "
                "or password-protected."
            ) from exc
    elif suffix == ".txt":
        try:
            documents = TextLoader(str(path), encoding="utf-8").load()
        except (UnicodeDecodeError, RuntimeError):
            # Not UTF-8 -- retry with a permissive single-byte encoding.
            documents = TextLoader(str(path), encoding="latin-1").load()
    else:
        raise UnsupportedFileError(
            f"Unsupported file type '{suffix}'. Supported types: PDF, TXT."
        )

    full_text = "\n".join(doc.page_content for doc in documents).strip()
    if not full_text:
        raise EmptyDocumentError(
            "No readable text found in the document. Scanned/image-based "
            "PDFs are not supported."
        )

    # Reject binary/garbage masquerading as text: real documents are
    # overwhelmingly printable. A low printable ratio means the file was
    # decoded from bytes that are not actually text (indexing it would
    # only pollute retrieval and waste the classification call).
    if suffix == ".txt" and _printable_ratio(full_text) < 0.85:
        raise UnsupportedFileError(
            "This file does not appear to be readable text. Please upload "
            "a valid PDF or TXT document."
        )

    logger.info(
        "Loaded %s: %d page(s), %d characters", path.name, len(documents), len(full_text)
    )
    return documents


def documents_to_text(documents: list[Document]) -> str:
    """Join loaded documents into one plain-text string (for Agent 1)."""
    return "\n".join(doc.page_content for doc in documents).strip()


def _printable_ratio(text: str, sample: int = 4000) -> float:
    """Fraction of characters that are printable/whitespace (0.0-1.0)."""
    excerpt = text[:sample]
    if not excerpt:
        return 0.0
    printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in excerpt)
    return printable / len(excerpt)

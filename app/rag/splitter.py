"""Text chunking: split loaded documents into overlapping chunks.

RecursiveCharacterTextSplitter tries to split on paragraph boundaries
first, then sentences, then words -- so chunks keep semantic meaning.
The overlap prevents an answer from being cut in half at a boundary.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for embedding.

    Args:
        documents: Loaded documents (from :mod:`app.rag.loader`).

    Returns:
        A list of chunk ``Document`` objects, each inheriting the source
        document's metadata (e.g. page number).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        "Split %d document page(s) into %d chunk(s) (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks

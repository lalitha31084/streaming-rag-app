"""
Document ingestion helpers.

Provides utility functions for text extraction and chunking that are
shared between the API server and the background worker.
"""

import io
import logging

logger = logging.getLogger(__name__)


def extract_text(filename: str, content: bytes) -> str:
    """
    Extract text content from uploaded file bytes.
    Supports PDF documents via pypdf and plain text/markdown files.
    """
    if filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(text.strip())
            extracted = "\n\n".join(pages_text)
            if extracted.strip():
                logger.info("Successfully extracted %d characters from PDF '%s'", len(extracted), filename)
                return extracted
        except Exception as e:
            logger.warning("pypdf extraction failed for '%s': %s. Falling back to UTF-8 decode.", filename, e)

    # Default fallback: UTF-8 with lenient error handling
    return content.decode("utf-8", errors="ignore")


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Split text into overlapping windows of chunk_size characters.
    Overlap ensures no information is lost at chunk boundaries.
    """
    if not text or not text.strip():
        return []

    chunks: list[str] = []
    step = max(chunk_size - chunk_overlap, 1)
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks

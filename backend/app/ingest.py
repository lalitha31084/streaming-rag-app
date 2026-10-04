"""
Document ingestion helpers.

Provides utility functions for text extraction, section-aware semantic chunking,
and metadata generation shared between the API server and the background worker.
"""

import io
import logging
import re

logger = logging.getLogger(__name__)


def clean_extracted_text(text: str) -> str:
    """
    Clean typical PDF extraction anomalies and OCR ligatures:
    - Split words like 'F ull' -> 'Full', 'T ools' -> 'Tools'
    - Normalize unicode dashes, bullet points, and replacement characters
    """
    if not text:
        return ""
    # Fix split words from PDF fonts
    text = re.sub(r"\bF\s+ull\b", "Full", text)
    text = re.sub(r"\bT\s+ools\b", "Tools", text)
    # Normalize unicode hyphens and replacement characters
    text = text.replace("\ufffd", "–")
    text = re.sub(r"[\u2013\u2014]", "–", text)
    # Normalize bullets
    text = re.sub(r"[•·▪●]", "•", text)
    return text.strip()


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
                return clean_extracted_text(extracted)
        except Exception as e:
            logger.warning("pypdf extraction failed for '%s': %s. Falling back to UTF-8 decode.", filename, e)

    # Default fallback: UTF-8 with lenient error handling
    raw = content.decode("utf-8", errors="ignore")
    return clean_extracted_text(raw)


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


def chunk_document(
    text: str,
    filename: str = "document",
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Section-aware semantic chunking with rich metadata.
    
    1. Detects structured resume sections:
       - Contact & Objective
       - Education
       - Experience / Internship
       - Projects (splits each project into its own isolated chunk)
       - Technical Skills
       - Achievements / Competitive Programming
       - Certifications
    2. Attaches rich metadata to every chunk:
       - source: filename
       - section: section name
       - title: title / project name
       - page: estimated page number (0-indexed)
       - chunk_index: integer index
    3. Falls back to sliding-window chunks if no section headers exist.
    
    Returns:
        list of dicts: [{"text": str, "metadata": dict}, ...]
    """
    cleaned = clean_extracted_text(text)
    if not cleaned:
        return []

    # Section headers regex (case-insensitive for resume sections)
    sec_pattern = r"(?m)^(Objective|Education|Experience|Projects|Technical Skills|Achievements|Certifications)\b"
    matches = list(re.finditer(sec_pattern, cleaned, re.IGNORECASE))

    # If no standard resume sections are detected, fall back to sliding window with metadata
    if not matches:
        base_chunks = chunk_text(cleaned, chunk_size, chunk_overlap)
        return [
            {
                "text": c,
                "metadata": {
                    "source": filename,
                    "section": "General",
                    "title": filename,
                    "page": i // 3,
                    "chunk_index": i,
                },
            }
            for i, c in enumerate(base_chunks)
        ]

    results: list[dict] = []
    chunk_idx = 0

    # Preamble / Contact info before the first section
    first_start = matches[0].start()
    if first_start > 0:
        header_text = cleaned[:first_start].strip()
        if header_text:
            results.append({
                "text": header_text,
                "metadata": {
                    "source": filename,
                    "section": "Personal Info",
                    "title": "Contact & Personal Information",
                    "page": 0,
                    "chunk_index": chunk_idx,
                },
            })
            chunk_idx += 1

    for i in range(len(matches)):
        sec_name = matches[i].group(1).title()
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        sec_raw = cleaned[start:end].strip()

        # Isolate individual projects within the Projects section
        if sec_name.lower() == "projects":
            # Strip the leading "Projects" label for cleaner project headers
            sec_body = re.sub(r"^projects\s*\n+", "", sec_raw, flags=re.IGNORECASE).strip()
            # Split projects by header lines (e.g. Project Name|Tech Stack View Project)
            proj_parts = re.split(r"(?m)^([A-Za-z0-9\s\-]+(?:\|[^\n]+)?)\n(?=\s*[•\-])", sec_body)
            if len(proj_parts) > 1:
                for p_idx in range(1, len(proj_parts), 2):
                    title_line = proj_parts[p_idx].strip()
                    proj_content = proj_parts[p_idx + 1].strip()
                    clean_title = title_line.split("|")[0].replace("View Project", "").strip()
                    clean_title = re.sub(r"^projects\s*\n*", "", clean_title, flags=re.IGNORECASE).strip()

                    full_proj_text = f"Project: {title_line}\n{proj_content}"
                    results.append({
                        "text": full_proj_text,
                        "metadata": {
                            "source": filename,
                            "section": "Projects",
                            "title": clean_title,
                            "project": clean_title,
                            "page": 0 if p_idx < 3 else 1,
                            "chunk_index": chunk_idx,
                        },
                    })
                    chunk_idx += 1
                continue

        # Standard sections (Education, Experience, Technical Skills, Achievements, Certifications)
        page_num = 0 if sec_name in ["Objective", "Education", "Experience"] else 1
        results.append({
            "text": sec_raw,
            "metadata": {
                "source": filename,
                "section": sec_name,
                "title": sec_name,
                "page": page_num,
                "chunk_index": chunk_idx,
            },
        })
        chunk_idx += 1

    return results

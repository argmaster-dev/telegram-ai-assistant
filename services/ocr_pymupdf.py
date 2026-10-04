"""PyMuPDF wrapper for digital PDF text extraction (no OCR).

Used by ``ocr_router`` when ``ocr_detect.is_digital_pdf`` says the file has
embedded text. For scan PDFs we fall back to ``ocr_easy.extract_from_pdf_pages``.
"""
from __future__ import annotations

import fitz


def extract_text_from_digital_pdf(pdf_bytes: bytes) -> list[str]:
    """Return per-page text extracted via PyMuPDF ``page.get_text()``.

    The returned list length equals the PDF's page count; pages with no
    extractable text contribute an empty string.
    """
    pages: list[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pages.append(page.get_text())
    return pages

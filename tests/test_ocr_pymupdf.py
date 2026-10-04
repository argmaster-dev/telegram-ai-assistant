"""Tests for services.ocr_pymupdf.extract_text_from_digital_pdf."""
from __future__ import annotations

from services import ocr_pymupdf


def test_extract_from_digital_pdf_returns_pages(digital_pdf_bytes, digital_pdf_keyword):
    pages = ocr_pymupdf.extract_text_from_digital_pdf(digital_pdf_bytes)
    assert isinstance(pages, list)
    assert len(pages) >= 1, "expected at least one page from synthetic digital PDF"
    assert digital_pdf_keyword in pages[0], (
        f"expected keyword {digital_pdf_keyword!r} on page 1, got: {pages[0][:200]!r}"
    )

"""Heuristic to distinguish digital PDFs (embedded text) from scan PDFs.

If the first ``max_pages`` pages cumulatively expose at least
``threshold_chars`` non-whitespace characters via PyMuPDF's
``page.get_text()``, the file is treated as a digital PDF. Otherwise it's a
scan and ``ocr_router`` will route it through EasyOCR.

The threshold (default 100 chars across 3 pages) catches the typical case of
a digital PDF whose first page is a title page with just a few words while
also rejecting scans where the only "text" is a small embedded form field.
"""
from __future__ import annotations

import fitz


def is_digital_pdf(
    pdf_bytes: bytes,
    *,
    threshold_chars: int = 100,
    max_pages: int = 3,
) -> bool:
    total = 0
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_idx in range(min(len(doc), max_pages)):
            text = doc.load_page(page_idx).get_text()
            total += len(text.strip())
            if total >= threshold_chars:
                return True
    return False

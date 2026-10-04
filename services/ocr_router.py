"""OCR dispatcher: routes by MIME + privacy mode to the right engine.

PRIVATE mode (local CPU):
  * ``image/*``           → ``ocr_easy.extract_from_image``
  * ``application/pdf`` + digital → ``ocr_pymupdf.extract_text_from_digital_pdf``
  * ``application/pdf`` + scan    → ``ocr_easy.extract_from_pdf_pages``

PUBLIC mode (Gemini Files API; requires explicit user consent
flow):
  * ``image/*``           → ``ocr_gemini.extract_image_text``
  * ``application/pdf``   → ``ocr_gemini.extract_pdf_text``
  * Missing ``gemini_client``       → ``ConfigError``
  * ``PRIVACY_STRICT=true``         → ``PrivacyStrictBlocked``
  * Gemini network/API failure      → ``GeminiUnavailable``
    (NEVER silent-fallback to PRIVATE — the user explicitly chose PUBLIC for
    quality and must not silently get a degraded result.)
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

from services import ocr_detect, ocr_easy, ocr_gemini, ocr_pymupdf
from services.gemini_client import GeminiClient
from services.ocr_errors import (
    ConfigError,
    GeminiUnavailable,
    OcrError,
    PrivacyStrictBlocked,
)


logger = logging.getLogger(__name__)


Mode = Literal["private", "public"]


_PAGE_HEADER = "\n\n--- page {n} ---\n\n"


def _join_pages(pages: list[str]) -> str:
    """Concatenate per-page strings with a clear page-boundary marker.

    Empty pages are skipped to keep output compact.
    """
    parts: list[str] = []
    for i, page in enumerate(pages, start=1):
        if not page or not page.strip():
            continue
        parts.append(_PAGE_HEADER.format(n=i))
        parts.append(page)
    return "".join(parts).lstrip()


async def extract_text(
    data: bytes,
    mime: str,
    mode: Mode,
    *,
    gemini_client: Optional[GeminiClient] = None,
) -> str:
    """Extract text from ``data`` according to ``mime`` and privacy ``mode``.

    Returns the extracted plain text (or markdown for PUBLIC).

    Raises:
        ConfigError: invalid mode/mime, or missing ``gemini_client`` in PUBLIC mode
        PrivacyStrictBlocked: PUBLIC + ``PRIVACY_STRICT=true`` (via gemini_client)
        GeminiUnavailable: PUBLIC + Gemini API problems
    """
    if mode not in ("private", "public"):
        raise ConfigError(f"invalid mode: {mode!r}")

    is_image = mime.startswith("image/")
    is_pdf = mime == "application/pdf"
    if not (is_image or is_pdf):
        raise ConfigError(f"unsupported mime: {mime!r}")

    if mode == "private":
        if is_image:
            return await ocr_easy.extract_from_image(data)
        # PDF
        if ocr_detect.is_digital_pdf(data):
            logger.debug("ocr_router PRIVATE: digital pdf → PyMuPDF")
            pages = ocr_pymupdf.extract_text_from_digital_pdf(data)
        else:
            logger.debug("ocr_router PRIVATE: scan pdf → EasyOCR per page")
            pages = await ocr_easy.extract_from_pdf_pages(data)
        return _join_pages(pages)

    # mode == "public"
    if gemini_client is None:
        raise ConfigError(
            "PUBLIC mode requires gemini_client (set GEMINI_API_KEY in .env)"
        )

    if is_image:
        return await ocr_gemini.extract_image_text(gemini_client, data, mime)
    return await ocr_gemini.extract_pdf_text(gemini_client, data)


__all__ = [
    "extract_text",
    "Mode",
    "OcrError",
    "ConfigError",
    "PrivacyStrictBlocked",
    "GeminiUnavailable",
]

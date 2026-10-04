"""High-level OCR adapter on top of ``GeminiClient`` Files API.

Used in PUBLIC mode by ``ocr_router.extract_text``. Each call follows the
upload → extract → delete (best-effort) sequence. Network failures bubble up
as ``GeminiUnavailable`` so the caller can surface the right user message
instead of silently degrading to local OCR.
"""
from __future__ import annotations

import logging

from services.gemini_client import GeminiClient


logger = logging.getLogger(__name__)


async def extract_pdf_text(client: GeminiClient, pdf_bytes: bytes) -> str:
    """Upload PDF → extract markdown → cleanup. Returns markdown text."""
    file_uri = await client.upload_file(pdf_bytes, "application/pdf")
    try:
        text = await client.extract_text_from_file(file_uri, "application/pdf")
    finally:
        try:
            await client.delete_file(file_uri)
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_file after extract_pdf_text failed: %s", exc)
    return text


async def extract_image_text(
    client: GeminiClient,
    img_bytes: bytes,
    mime_type: str,
) -> str:
    """Same upload/extract/delete flow for an image MIME type."""
    file_uri = await client.upload_file(img_bytes, mime_type)
    try:
        text = await client.extract_text_from_file(file_uri, mime_type)
    finally:
        try:
            await client.delete_file(file_uri)
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_file after extract_image_text failed: %s", exc)
    return text

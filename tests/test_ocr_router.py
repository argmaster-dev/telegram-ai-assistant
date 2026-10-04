"""Tests for services.ocr_router.extract_text — dispatcher mode×mime matrix."""
from __future__ import annotations

import httpx
import pytest

from services import ocr_router
from services.gemini_client import GeminiClient
from services.ocr_errors import ConfigError, PrivacyStrictBlocked


# ---- PRIVATE branch -------------------------------------------------------

@pytest.mark.asyncio
async def test_private_image_uses_easyocr(passport_jpg_bytes):
    text = await ocr_router.extract_text(
        passport_jpg_bytes, "image/jpeg", "private"
    )
    assert isinstance(text, str)
    assert text.strip(), "expected non-empty OCR output for synthetic passport"


@pytest.mark.asyncio
async def test_private_digital_pdf_uses_pymupdf(
    digital_pdf_bytes, digital_pdf_keyword
):
    text = await ocr_router.extract_text(
        digital_pdf_bytes, "application/pdf", "private"
    )
    assert digital_pdf_keyword in text


@pytest.mark.asyncio
async def test_private_scan_pdf_uses_easyocr(scan_pdf_bytes):
    """Scan-only PDF should be routed to EasyOCR per page (returns str)."""
    text = await ocr_router.extract_text(
        scan_pdf_bytes, "application/pdf", "private"
    )
    assert isinstance(text, str)
    # Synthetic scan pixels don't have to OCR perfectly — we only require it
    # ran without raising and returned a string.


# ---- PUBLIC branch --------------------------------------------------------

def _gemini_mock_handler(payload_text: str = "# from gemini"):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/upload/v1beta/files":
            return httpx.Response(
                200,
                json={"file": {"uri": "https://x/files/p1", "name": "files/p1"}},
            )
        if ":generateContent" in request.url.path:
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": payload_text}]}}]},
            )
        return httpx.Response(200)
    return handler


@pytest.mark.asyncio
async def test_public_pdf_uses_gemini(digital_pdf_bytes):
    client = GeminiClient(
        api_key="k",
        transport=httpx.MockTransport(_gemini_mock_handler("# from gemini")),
        rate_limit_sec=0.0,
    )
    text = await ocr_router.extract_text(
        digital_pdf_bytes, "application/pdf", "public", gemini_client=client
    )
    assert text == "# from gemini"


@pytest.mark.asyncio
async def test_public_image_uses_gemini():
    client = GeminiClient(
        api_key="k",
        transport=httpx.MockTransport(_gemini_mock_handler("image markdown")),
        rate_limit_sec=0.0,
    )
    text = await ocr_router.extract_text(
        b"jpg-bytes", "image/jpeg", "public", gemini_client=client
    )
    assert text == "image markdown"


# ---- Error paths ----------------------------------------------------------

@pytest.mark.asyncio
async def test_public_without_client_raises_config_error(digital_pdf_bytes):
    with pytest.raises(ConfigError, match="PUBLIC mode requires"):
        await ocr_router.extract_text(
            digital_pdf_bytes, "application/pdf", "public"
        )


@pytest.mark.asyncio
async def test_public_with_privacy_strict_raises(digital_pdf_bytes):
    client = GeminiClient(
        api_key="k",
        privacy_strict=True,
        transport=httpx.MockTransport(_gemini_mock_handler()),
        rate_limit_sec=0.0,
    )
    with pytest.raises(PrivacyStrictBlocked):
        await ocr_router.extract_text(
            digital_pdf_bytes,
            "application/pdf",
            "public",
            gemini_client=client,
        )


@pytest.mark.asyncio
async def test_invalid_mode_raises_config_error():
    with pytest.raises(ConfigError, match="invalid mode"):
        await ocr_router.extract_text(b"x", "image/png", "wrong")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_unsupported_mime_raises_config_error():
    with pytest.raises(ConfigError, match="unsupported mime"):
        await ocr_router.extract_text(b"x", "video/mp4", "private")

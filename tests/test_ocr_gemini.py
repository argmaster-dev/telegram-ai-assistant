"""Tests for services.ocr_gemini high-level adapters — Gemini mocked."""
from __future__ import annotations

import httpx
import pytest

from services import ocr_gemini
from services.gemini_client import GeminiClient
from services.ocr_errors import GeminiUnavailable


@pytest.mark.asyncio
async def test_extract_pdf_text_full_flow():
    """upload → extract → delete (best effort), all observed in order."""
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/upload/v1beta/files":
            return httpx.Response(
                200,
                json={"file": {"uri": "https://x/files/u1", "name": "files/u1"}},
            )
        if ":generateContent" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "candidates": [{"content": {"parts": [{"text": "result md"}]}}]
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200)
        return httpx.Response(404)

    client = GeminiClient(
        api_key="k", transport=httpx.MockTransport(handler), rate_limit_sec=0.0
    )
    text = await ocr_gemini.extract_pdf_text(client, b"pdf-bytes")
    assert text == "result md"

    methods = [m for (m, _p) in calls]
    paths = [p for (_m, p) in calls]
    assert any("/upload/" in p for p in paths)
    assert any(":generateContent" in p for p in paths)
    assert "DELETE" in methods, "expected best-effort delete after extract"


@pytest.mark.asyncio
async def test_extract_pdf_text_propagates_gemini_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = GeminiClient(
        api_key="k", transport=httpx.MockTransport(handler), rate_limit_sec=0.0
    )
    with pytest.raises(GeminiUnavailable):
        await ocr_gemini.extract_pdf_text(client, b"pdf")


@pytest.mark.asyncio
async def test_extract_image_text_full_flow():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/upload/v1beta/files":
            return httpx.Response(
                200,
                json={"file": {"uri": "https://x/files/i1", "name": "files/i1"}},
            )
        if ":generateContent" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "candidates": [{"content": {"parts": [{"text": "image text"}]}}]
                },
            )
        return httpx.Response(200)

    client = GeminiClient(
        api_key="k", transport=httpx.MockTransport(handler), rate_limit_sec=0.0
    )
    text = await ocr_gemini.extract_image_text(client, b"jpg", "image/jpeg")
    assert text == "image text"

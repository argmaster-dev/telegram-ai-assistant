"""Tests for services.gemini_client.GeminiClient — mocked via httpx.MockTransport.

No real Gemini API calls.
"""
from __future__ import annotations

import httpx
import pytest

from services.gemini_client import GeminiClient
from services.ocr_errors import GeminiUnavailable, PrivacyStrictBlocked


def _client(handler) -> GeminiClient:
    return GeminiClient(
        api_key="test-key",
        privacy_strict=False,
        transport=httpx.MockTransport(handler),
        rate_limit_sec=0.0,
    )


@pytest.mark.asyncio
async def test_upload_file_returns_uri():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/upload/v1beta/files"
        assert request.headers.get("X-Goog-Upload-Protocol") == "raw"
        return httpx.Response(
            200,
            json={
                "file": {
                    "uri": "https://gemini/files/abc-123",
                    "name": "files/abc-123",
                }
            },
        )

    client = _client(handler)
    uri = await client.upload_file(b"hello", "application/pdf")
    assert uri == "https://gemini/files/abc-123"


@pytest.mark.asyncio
async def test_extract_text_from_file_returns_markdown():
    def handler(request: httpx.Request) -> httpx.Response:
        assert ":generateContent" in request.url.path
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "# heading\nbody"}]}}
                ]
            },
        )

    client = _client(handler)
    text = await client.extract_text_from_file(
        "https://gemini/files/abc-123", "application/pdf"
    )
    assert text == "# heading\nbody"


@pytest.mark.asyncio
async def test_delete_file_does_not_raise_on_200():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert "/files/abc-123" in request.url.path
        return httpx.Response(200)

    client = _client(handler)
    await client.delete_file("files/abc-123")  # must not raise


@pytest.mark.asyncio
async def test_delete_file_swallows_404():
    """Files API auto-expires — delete must be best-effort."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    client = _client(handler)
    await client.delete_file("files/missing")  # must not raise


@pytest.mark.asyncio
async def test_privacy_strict_blocks_upload():
    client = GeminiClient(
        api_key="test",
        privacy_strict=True,
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"file": {"uri": "x"}})
        ),
        rate_limit_sec=0.0,
    )
    with pytest.raises(PrivacyStrictBlocked):
        await client.upload_file(b"x", "application/pdf")


@pytest.mark.asyncio
async def test_privacy_strict_blocks_extract():
    client = GeminiClient(
        api_key="test", privacy_strict=True, rate_limit_sec=0.0
    )
    with pytest.raises(PrivacyStrictBlocked):
        await client.extract_text_from_file("uri", "application/pdf")


@pytest.mark.asyncio
async def test_5xx_raises_gemini_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream broken")

    client = _client(handler)
    with pytest.raises(GeminiUnavailable, match="HTTP 503"):
        await client.upload_file(b"x", "application/pdf")


@pytest.mark.asyncio
async def test_extract_text_unexpected_response_shape_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    client = _client(handler)
    with pytest.raises(GeminiUnavailable, match="unexpected response shape"):
        await client.extract_text_from_file("uri", "application/pdf")

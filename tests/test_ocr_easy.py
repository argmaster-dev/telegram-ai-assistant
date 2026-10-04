"""Tests for services.ocr_easy — uses the real EasyOCR Reader.

First test in a session triggers Reader init (~5–10s). All later calls reuse
the singleton, ~1–2s per image.
"""
from __future__ import annotations

import pytest

from services import ocr_easy


@pytest.mark.asyncio
async def test_extract_from_image_returns_nonempty_string(passport_jpg_bytes):
    text = await ocr_easy.extract_from_image(passport_jpg_bytes)
    assert isinstance(text, str)
    assert text.strip(), "EasyOCR returned empty string for synthetic passport sample"


@pytest.mark.asyncio
async def test_get_reader_is_singleton():
    first = await ocr_easy._get_reader()
    second = await ocr_easy._get_reader()
    assert first is second, "EasyOCR Reader must be a session-wide singleton"


@pytest.mark.asyncio
async def test_reader_is_cpu_only():
    """Reader created with gpu=False — no CUDA device should be attached."""
    reader = await ocr_easy._get_reader()
    device = getattr(reader, "device", None)
    if device is None:
        # Older EasyOCR versions stored the choice as a private flag instead
        device = str(getattr(reader, "_device", "cpu"))
    assert "cpu" in str(device).lower(), f"reader device should be CPU, got {device!r}"

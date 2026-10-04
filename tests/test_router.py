"""Tests for services.router.classify — Ollama is mocked, no real HTTP."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from services import router
from services.ollama_client import OllamaInvalidJSON, OllamaUnavailable


@pytest.mark.asyncio
async def test_classify_reminder_intent():
    mock_gen = AsyncMock(return_value={"intent": "reminder", "confidence": 0.95})
    with patch("services.router.generate_json", mock_gen):
        result = await router.classify(
            "напомни завтра в 14 позвонить", base_url="http://tg_ollama:11434"
        )
    assert result["intent"] == "reminder"
    assert result["confidence"] == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_classify_unknown_intent():
    mock_gen = AsyncMock(return_value={"intent": "unknown", "confidence": 0.2})
    with patch("services.router.generate_json", mock_gen):
        result = await router.classify("привет", base_url="http://x")
    assert result["intent"] == "unknown"
    assert result["confidence"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_classify_invalid_json_falls_back_to_unknown():
    mock_gen = AsyncMock(side_effect=OllamaInvalidJSON("not json"))
    with patch("services.router.generate_json", mock_gen):
        result = await router.classify("что-то", base_url="http://x")
    assert result["intent"] == "unknown"
    assert result["confidence"] == 0.0


@pytest.mark.asyncio
async def test_classify_unreachable_falls_back_to_unknown():
    mock_gen = AsyncMock(side_effect=OllamaUnavailable("down"))
    with patch("services.router.generate_json", mock_gen):
        result = await router.classify("что-то", base_url="http://x")
    assert result["intent"] == "unknown"
    assert result["confidence"] == 0.0


@pytest.mark.asyncio
async def test_classify_unexpected_intent_label_normalized_to_unknown():
    mock_gen = AsyncMock(return_value={"intent": "random_label", "confidence": 1.0})
    with patch("services.router.generate_json", mock_gen):
        result = await router.classify("что-то", base_url="http://x")
    assert result["intent"] == "unknown"

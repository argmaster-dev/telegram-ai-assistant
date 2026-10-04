"""router classifies new task / subscription intents.

These tests complement ``tests/test_router.py`` (which covers the binary
reminder/unknown world). The Ollama call is mocked — we only
verify that ``services.router.classify`` correctly normalises the new
``"task"`` and ``"subscription"`` intent labels and rejects unknown labels.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from services import router


@pytest.mark.asyncio
async def test_router_returns_task_intent():
    mock_gen = AsyncMock(return_value={"intent": "task", "confidence": 0.92})
    with patch("services.router.generate_json", mock_gen):
        result = await router.classify(
            "добавь задачу купить молоко", base_url="http://x"
        )
    assert result["intent"] == "task"
    assert result["confidence"] == pytest.approx(0.92)


@pytest.mark.asyncio
async def test_router_returns_subscription_intent():
    mock_gen = AsyncMock(return_value={"intent": "subscription", "confidence": 0.88})
    with patch("services.router.generate_json", mock_gen):
        result = await router.classify(
            "Spotify 169₽ Сбер каждое 15 число", base_url="http://x"
        )
    assert result["intent"] == "subscription"


@pytest.mark.asyncio
async def test_router_still_classifies_reminder():
    """Regression: extending Intent set must not break the reminder path."""
    mock_gen = AsyncMock(return_value={"intent": "reminder", "confidence": 0.95})
    with patch("services.router.generate_json", mock_gen):
        result = await router.classify(
            "напомни завтра в 14 позвонить", base_url="http://x"
        )
    assert result["intent"] == "reminder"


@pytest.mark.asyncio
async def test_router_rejects_unknown_intent_label():
    """Foreign label like 'todo' or 'shopping' coerces to unknown."""
    mock_gen = AsyncMock(return_value={"intent": "todo", "confidence": 0.9})
    with patch("services.router.generate_json", mock_gen):
        result = await router.classify("что-то", base_url="http://x")
    assert result["intent"] == "unknown"

"""Tests for services.pipeline.classify_and_parse — unified single-call flow.

intent values ``reminder_single`` / ``reminder_recurring`` / ``unknown``.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest

from services import pipeline
from services.ollama_client import OllamaUnavailable
from services.reminders import ReminderParseError


TZ = ZoneInfo("Europe/Moscow")


def _future_iso(hours: int = 2) -> str:
    return (datetime.now(tz=TZ) + timedelta(hours=hours)).isoformat()


def _past_iso(hours: int = 1) -> str:
    return (datetime.now(tz=TZ) - timedelta(hours=hours)).isoformat()


# ---------------------------------------------------------------------------
# reminder_single
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_classify_and_parse_valid_single_reminder():
    mock_gen = AsyncMock(
        return_value={
            "intent": "reminder_single",
            "confidence": 0.95,
            "title": "позвонить в страховую",
            "when_at": _future_iso(2),
            "rrule": "",
            "exdates": [],
            "exclude_holidays": "none",
        }
    )
    with patch("services.pipeline.generate_json", mock_gen):
        result = await pipeline.classify_and_parse(
            "напомни через 2 часа позвонить в страховую",
            base_url="http://x",
            user_id=42,
        )
    assert result.intent == "reminder_single"
    assert result.confidence == pytest.approx(0.95)
    assert result.reminder is not None
    assert result.reminder.title == "позвонить в страховую"
    assert result.reminder.user_id == 42
    assert result.reminder.rrule is None
    assert result.reminder.when_at is not None
    assert result.reminder.when_at > datetime.now(tz=TZ)


@pytest.mark.asyncio
async def test_classify_and_parse_legacy_reminder_label_is_treated_as_single():
    """Older output format (intent='reminder') is accepted as alias."""
    mock_gen = AsyncMock(
        return_value={
            "intent": "reminder",  # legacy
            "confidence": 0.9,
            "title": "test",
            "when_at": _future_iso(1),
        }
    )
    with patch("services.pipeline.generate_json", mock_gen):
        result = await pipeline.classify_and_parse("…", base_url="http://x")
    assert result.intent == "reminder_single"
    assert result.reminder is not None


# ---------------------------------------------------------------------------
# reminder_recurring
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_classify_and_parse_valid_recurring_reminder():
    mock_gen = AsyncMock(
        return_value={
            "intent": "reminder_recurring",
            "confidence": 0.97,
            "title": "сдать отчёт",
            "when_at": "",
            "rrule": "FREQ=WEEKLY;BYDAY=FR;BYHOUR=18;BYMINUTE=0",
            "exdates": [],
            "exclude_holidays": "none",
        }
    )
    with patch("services.pipeline.generate_json", mock_gen):
        result = await pipeline.classify_and_parse(
            "каждую пятницу в 18 сдать отчёт",
            base_url="http://x",
            user_id=7,
        )
    assert result.intent == "reminder_recurring"
    assert result.reminder is not None
    assert result.reminder.title == "сдать отчёт"
    assert result.reminder.user_id == 7
    assert result.reminder.rrule == "FREQ=WEEKLY;BYDAY=FR;BYHOUR=18;BYMINUTE=0"
    assert result.reminder.exclude_holidays == "none"
    # next-fire cache must be populated and in the future
    assert result.reminder.when_at is not None
    assert result.reminder.when_at > datetime.now(tz=TZ)


@pytest.mark.asyncio
async def test_classify_and_parse_recurring_with_ru_holidays():
    mock_gen = AsyncMock(
        return_value={
            "intent": "reminder_recurring",
            "confidence": 0.95,
            "title": "еженедельный отчёт",
            "when_at": "",
            "rrule": "FREQ=WEEKLY;BYDAY=FR;BYHOUR=18;BYMINUTE=0",
            "exdates": [],
            "exclude_holidays": "ru",
        }
    )
    with patch("services.pipeline.generate_json", mock_gen):
        result = await pipeline.classify_and_parse(
            "каждую пятницу в 18, кроме праздников РФ",
            base_url="http://x",
        )
    assert result.intent == "reminder_recurring"
    assert result.reminder is not None
    assert result.reminder.exclude_holidays == "ru"


@pytest.mark.asyncio
async def test_classify_and_parse_recurring_empty_rrule_raises():
    mock_gen = AsyncMock(
        return_value={
            "intent": "reminder_recurring",
            "confidence": 0.8,
            "title": "x",
            "when_at": "",
            "rrule": "",
            "exdates": [],
            "exclude_holidays": "none",
        }
    )
    with patch("services.pipeline.generate_json", mock_gen):
        with pytest.raises(ReminderParseError, match="rrule empty"):
            await pipeline.classify_and_parse("…", base_url="http://x")


@pytest.mark.asyncio
async def test_classify_and_parse_recurring_invalid_rrule_raises():
    mock_gen = AsyncMock(
        return_value={
            "intent": "reminder_recurring",
            "confidence": 0.8,
            "title": "x",
            "when_at": "",
            "rrule": "FREQ=HOURLY;BYHOUR=9",  # HOURLY not in whitelist
            "exdates": [],
            "exclude_holidays": "none",
        }
    )
    with patch("services.pipeline.generate_json", mock_gen):
        with pytest.raises(ReminderParseError, match="FREQ"):
            await pipeline.classify_and_parse("…", base_url="http://x")


# ---------------------------------------------------------------------------
# unknown / failures
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_classify_and_parse_unknown_intent():
    mock_gen = AsyncMock(
        return_value={
            "intent": "unknown",
            "confidence": 0.1,
            "title": "",
            "when_at": "",
            "rrule": "",
            "exdates": [],
            "exclude_holidays": "none",
        }
    )
    with patch("services.pipeline.generate_json", mock_gen):
        result = await pipeline.classify_and_parse("привет", base_url="http://x")
    assert result.intent == "unknown"
    assert result.reminder is None


@pytest.mark.asyncio
async def test_classify_and_parse_ollama_down_falls_back_to_unknown():
    mock_gen = AsyncMock(side_effect=OllamaUnavailable("down"))
    with patch("services.pipeline.generate_json", mock_gen):
        result = await pipeline.classify_and_parse("что-то", base_url="http://x")
    assert result.intent == "unknown"
    assert result.confidence == 0.0
    assert result.reminder is None


@pytest.mark.asyncio
async def test_classify_and_parse_single_empty_title_raises():
    mock_gen = AsyncMock(
        return_value={
            "intent": "reminder_single",
            "confidence": 0.9,
            "title": "",
            "when_at": _future_iso(1),
        }
    )
    with patch("services.pipeline.generate_json", mock_gen):
        with pytest.raises(ReminderParseError, match="title empty"):
            await pipeline.classify_and_parse("…", base_url="http://x")


@pytest.mark.asyncio
async def test_classify_and_parse_single_invalid_iso_raises():
    mock_gen = AsyncMock(
        return_value={
            "intent": "reminder_single",
            "confidence": 0.9,
            "title": "x",
            "when_at": "not a date",
        }
    )
    with patch("services.pipeline.generate_json", mock_gen):
        with pytest.raises(ReminderParseError, match="invalid ISO"):
            await pipeline.classify_and_parse("…", base_url="http://x")


@pytest.mark.asyncio
async def test_classify_and_parse_single_past_raises():
    mock_gen = AsyncMock(
        return_value={
            "intent": "reminder_single",
            "confidence": 0.9,
            "title": "x",
            "when_at": _past_iso(2),
        }
    )
    with patch("services.pipeline.generate_json", mock_gen):
        with pytest.raises(ReminderParseError, match="future"):
            await pipeline.classify_and_parse("…", base_url="http://x")

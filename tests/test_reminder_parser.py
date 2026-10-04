"""Tests for services.reminders.parse — Ollama is mocked, no real HTTP."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest

from services import reminders as reminder_parser
from services.reminders import ReminderParseError


TZ = ZoneInfo("Europe/Moscow")


def _future_iso(hours: int = 2) -> str:
    return (datetime.now(tz=TZ) + timedelta(hours=hours)).isoformat()


def _past_iso(hours: int = 1) -> str:
    return (datetime.now(tz=TZ) - timedelta(hours=hours)).isoformat()


@pytest.mark.asyncio
async def test_parse_valid_future_datetime():
    mock_gen = AsyncMock(
        return_value={"title": "позвонить в страховую", "when_at": _future_iso(2)}
    )
    with patch("services.reminders.generate_json", mock_gen):
        reminder = await reminder_parser.parse(
            "напомни через 2 часа позвонить в страховую",
            base_url="http://x",
            user_id=42,
        )
    assert reminder.title == "позвонить в страховую"
    assert reminder.user_id == 42
    assert reminder.when_at is not None
    assert reminder.when_at > datetime.now(tz=TZ)
    assert reminder.source == "text"
    assert reminder.status == "scheduled"


@pytest.mark.asyncio
async def test_parse_empty_title_raises():
    mock_gen = AsyncMock(return_value={"title": "", "when_at": _future_iso()})
    with patch("services.reminders.generate_json", mock_gen):
        with pytest.raises(ReminderParseError, match="empty title"):
            await reminder_parser.parse("…", base_url="http://x")


@pytest.mark.asyncio
async def test_parse_empty_when_at_raises():
    mock_gen = AsyncMock(return_value={"title": "что-то", "when_at": ""})
    with patch("services.reminders.generate_json", mock_gen):
        with pytest.raises(ReminderParseError, match="empty when_at"):
            await reminder_parser.parse("…", base_url="http://x")


@pytest.mark.asyncio
async def test_parse_invalid_iso_raises():
    mock_gen = AsyncMock(return_value={"title": "x", "when_at": "not a date"})
    with patch("services.reminders.generate_json", mock_gen):
        with pytest.raises(ReminderParseError, match="invalid ISO"):
            await reminder_parser.parse("…", base_url="http://x")


@pytest.mark.asyncio
async def test_parse_past_datetime_raises():
    mock_gen = AsyncMock(return_value={"title": "x", "when_at": _past_iso(3)})
    with patch("services.reminders.generate_json", mock_gen):
        with pytest.raises(ReminderParseError, match="future"):
            await reminder_parser.parse("…", base_url="http://x")

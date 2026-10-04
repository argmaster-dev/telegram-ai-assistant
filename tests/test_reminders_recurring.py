"""Tests for services.reminders.parse_recurring — Ollama mocked."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest

from services import reminders as reminder_parser
from services.reminders import ReminderParseError


TZ = ZoneInfo("Europe/Moscow")


@pytest.mark.asyncio
async def test_parse_recurring_weekly_friday():
    mock_gen = AsyncMock(
        return_value={
            "title": "сдать отчёт",
            "rrule": "FREQ=WEEKLY;BYDAY=FR;BYHOUR=18;BYMINUTE=0",
            "exdates": [],
            "exclude_holidays": "none",
        }
    )
    with patch("services.reminders.generate_json", mock_gen):
        reminder = await reminder_parser.parse_recurring(
            "каждую пятницу в 18:00 сдать отчёт",
            base_url="http://x",
            user_id=1,
        )
    assert reminder.title == "сдать отчёт"
    assert reminder.rrule == "FREQ=WEEKLY;BYDAY=FR;BYHOUR=18;BYMINUTE=0"
    assert reminder.exclude_holidays == "none"
    assert reminder.when_at is not None
    assert reminder.when_at > datetime.now(tz=TZ)


@pytest.mark.asyncio
async def test_parse_recurring_monthly_first_day():
    mock_gen = AsyncMock(
        return_value={
            "title": "оплатить ипотеку",
            "rrule": "FREQ=MONTHLY;BYMONTHDAY=1;BYHOUR=9;BYMINUTE=0",
            "exdates": [],
            "exclude_holidays": "none",
        }
    )
    with patch("services.reminders.generate_json", mock_gen):
        reminder = await reminder_parser.parse_recurring(
            "каждое 1 число в 9:00 оплатить ипотеку",
            base_url="http://x",
        )
    assert reminder.rrule.startswith("FREQ=MONTHLY")
    assert "BYMONTHDAY=1" in reminder.rrule


@pytest.mark.asyncio
async def test_parse_recurring_biweekly_tuesday():
    mock_gen = AsyncMock(
        return_value={
            "title": "к стоматологу",
            "rrule": "FREQ=WEEKLY;INTERVAL=2;BYDAY=TU;BYHOUR=12;BYMINUTE=0",
            "exdates": [],
            "exclude_holidays": "none",
        }
    )
    with patch("services.reminders.generate_json", mock_gen):
        reminder = await reminder_parser.parse_recurring(
            "каждые 2 недели во вторник в 12", base_url="http://x"
        )
    assert "INTERVAL=2" in reminder.rrule
    assert "BYDAY=TU" in reminder.rrule


@pytest.mark.asyncio
async def test_parse_recurring_with_ru_holidays():
    mock_gen = AsyncMock(
        return_value={
            "title": "еженедельный отчёт",
            "rrule": "FREQ=WEEKLY;BYDAY=FR;BYHOUR=18;BYMINUTE=0",
            "exdates": [],
            "exclude_holidays": "ru",
        }
    )
    with patch("services.reminders.generate_json", mock_gen):
        reminder = await reminder_parser.parse_recurring(
            "каждую пятницу в 18, кроме праздников РФ, еженедельный отчёт",
            base_url="http://x",
        )
    assert reminder.exclude_holidays == "ru"


@pytest.mark.asyncio
async def test_parse_recurring_weekdays():
    mock_gen = AsyncMock(
        return_value={
            "title": "утренняя планёрка",
            "rrule": "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9;BYMINUTE=0",
            "exdates": [],
            "exclude_holidays": "none",
        }
    )
    with patch("services.reminders.generate_json", mock_gen):
        reminder = await reminder_parser.parse_recurring(
            "по будням в 9 утренняя планёрка", base_url="http://x"
        )
    assert "BYDAY=MO,TU,WE,TH,FR" in reminder.rrule


@pytest.mark.asyncio
async def test_parse_recurring_invalid_freq_raises():
    """Model returns FREQ=HOURLY which is outside our whitelist."""
    mock_gen = AsyncMock(
        return_value={
            "title": "x",
            "rrule": "FREQ=HOURLY;BYHOUR=9",
            "exdates": [],
            "exclude_holidays": "none",
        }
    )
    with patch("services.reminders.generate_json", mock_gen):
        with pytest.raises(ReminderParseError, match="FREQ"):
            await reminder_parser.parse_recurring("…", base_url="http://x")


@pytest.mark.asyncio
async def test_parse_recurring_empty_rrule_raises():
    mock_gen = AsyncMock(
        return_value={
            "title": "x",
            "rrule": "",
            "exdates": [],
            "exclude_holidays": "none",
        }
    )
    with patch("services.reminders.generate_json", mock_gen):
        with pytest.raises(ReminderParseError, match="empty rrule"):
            await reminder_parser.parse_recurring("…", base_url="http://x")

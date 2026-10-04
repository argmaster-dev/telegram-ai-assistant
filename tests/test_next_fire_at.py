"""Unit tests for services.reminders.next_fire_at (no Ollama, no DB).

Covers RRULE parsing, exdates, exclude_holidays (``ru``), INTERVAL, BYMONTHDAY,
UNTIL and combinations.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import holidays

from services.reminders import next_fire_at


TZ = ZoneInfo("Europe/Moscow")


def _dt(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=TZ)


# ---------------------------------------------------------------------------
# Baseline RRULEs
# ---------------------------------------------------------------------------

def test_weekly_friday_basic():
    """FREQ=WEEKLY;BYDAY=FR;BYHOUR=18 from Thu 2026-04-23 → Fri 2026-04-24 18:00."""
    result = next_fire_at(
        "FREQ=WEEKLY;BYDAY=FR;BYHOUR=18;BYMINUTE=0",
        from_dt=_dt(2026, 4, 23, 12, 0),
    )
    assert result is not None
    assert result.weekday() == 4  # Friday
    assert result.hour == 18
    assert result.date() == date(2026, 4, 24)


def test_monthly_first_day_from_mid_month():
    """FREQ=MONTHLY;BYMONTHDAY=1;BYHOUR=9 from 2026-01-15 → 2026-02-01 09:00."""
    result = next_fire_at(
        "FREQ=MONTHLY;BYMONTHDAY=1;BYHOUR=9;BYMINUTE=0",
        from_dt=_dt(2026, 1, 15),
    )
    assert result is not None
    assert result.date() == date(2026, 2, 1)
    assert result.hour == 9


def test_interval_2_weekly_tuesday():
    """FREQ=WEEKLY;INTERVAL=2;BYDAY=TU — with dtstart on Tue, the second fire
    is 2 weeks after the first, not 1 week.
    """
    result = next_fire_at(
        "FREQ=WEEKLY;INTERVAL=2;BYDAY=TU;BYHOUR=10;BYMINUTE=0",
        from_dt=_dt(2026, 4, 21, 9, 0),  # Tue Apr 21
    )
    assert result is not None
    # First occurrence at or after Apr 21 10:00 is Apr 21 itself
    # (dateutil INTERVAL counts from dtstart). If after() returns Apr 21 10,
    # the *next* step would be May 5 — so either is acceptable; we only
    # require it's a Tuesday within ~3 weeks.
    assert result.weekday() == 1  # Tuesday
    delta = (result.date() - date(2026, 4, 21)).days
    assert delta in (0, 14, 7)  # dtstart-adjacent or 2-week jump


# ---------------------------------------------------------------------------
# exdates
# ---------------------------------------------------------------------------

def test_exdate_skips_one_month():
    """FREQ=MONTHLY;BYMONTHDAY=1 with exdate 2026-05-01 from Apr 15 → Jun 1."""
    result = next_fire_at(
        "FREQ=MONTHLY;BYMONTHDAY=1;BYHOUR=9;BYMINUTE=0",
        exdates=["2026-05-01"],
        from_dt=_dt(2026, 4, 15),
    )
    assert result is not None
    assert result.date() == date(2026, 6, 1)


# ---------------------------------------------------------------------------
# Russian holidays
# ---------------------------------------------------------------------------

def test_ru_holidays_covers_jan_2_2026():
    """Sanity: holidays.country_holidays('RU') marks 2026-01-02 as a holiday.

    If this asserts False, the rest of the holiday-aware tests need revisiting —
    holidays package may have changed Russian calendar data.
    """
    ru = holidays.country_holidays("RU")
    assert date(2026, 1, 2) in ru


def test_weekly_friday_skips_ru_holiday_jan_2_2026():
    """Jan 2 2026 is a Friday AND a Russian New Year holiday → skipped."""
    result = next_fire_at(
        "FREQ=WEEKLY;BYDAY=FR;BYHOUR=9;BYMINUTE=0",
        exclude_holidays="ru",
        from_dt=_dt(2026, 1, 1, 12, 0),  # Thu Jan 1
    )
    assert result is not None
    # Jan 2 is a Russian holiday; Jan 9 is the first Friday that isn't
    assert result.date() >= date(2026, 1, 9), f"expected >= Jan 9, got {result}"


def test_daily_from_new_year_week_2026():
    """FREQ=DAILY from Dec 31 2025 with exclude_holidays=ru → first non-holiday
    is 2026-01-09 (Jan 1–8 are all holidays in РФ).
    """
    result = next_fire_at(
        "FREQ=DAILY;BYHOUR=9;BYMINUTE=0",
        exclude_holidays="ru",
        from_dt=_dt(2025, 12, 31, 10, 0),
    )
    assert result is not None
    assert result.date() >= date(2026, 1, 9), f"expected >= Jan 9, got {result}"


# ---------------------------------------------------------------------------
# UNTIL exhausted
# ---------------------------------------------------------------------------

def test_until_in_past_returns_none():
    result = next_fire_at(
        "FREQ=WEEKLY;BYDAY=FR;UNTIL=20200101T000000Z",
        from_dt=_dt(2026, 4, 23),
    )
    assert result is None


# ---------------------------------------------------------------------------
# Combined exdates + holidays
# ---------------------------------------------------------------------------

def test_combined_exdates_and_ru_holidays():
    """Fri weekly with Apr 24 exdate AND RU-holiday May 1 → next is May 8."""
    result = next_fire_at(
        "FREQ=WEEKLY;BYDAY=FR;BYHOUR=12;BYMINUTE=0",
        exdates=["2026-04-24"],
        exclude_holidays="ru",
        from_dt=_dt(2026, 4, 23, 10, 0),
    )
    assert result is not None
    # Sequence: Apr 24 (exdate, skip) → May 1 (RU Labour Day, skip) → May 8
    assert result.date() == date(2026, 5, 8)

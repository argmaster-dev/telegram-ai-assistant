"""subscription stats aggregation + year forecast."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from services import storage, subscriptions
from services.intents import Subscription


_TZ = ZoneInfo("Europe/Moscow")


@pytest.mark.asyncio
async def test_compute_stats_empty_user(tmp_path):
    db_path = str(tmp_path / "e.db")
    await storage.init_db(db_path)
    stats = await subscriptions.compute_stats(db_path, user_id=42)
    assert stats == {"months": [], "year_forecast": {}}


@pytest.mark.asyncio
async def test_compute_stats_groups_by_month_and_currency(tmp_path):
    """3 monthly subs of mixed currencies → next 12 months populated correctly.

    Each subscription contributes 12 occurrences in a 12-month horizon. Total
    forecast = monthly_amount * 12 grouped by currency.
    """
    db_path = str(tmp_path / "g.db")
    await storage.init_db(db_path)
    user_id = 7

    today = datetime.now(tz=_TZ)
    # Pin next charges in the near future so they fall in the 12-month horizon.
    netflix = Subscription(
        user_id=user_id, service="Netflix", amount=499, currency="RUB",
        recurrence="monthly", day_of_month=20, notify_before_days=2,
        next_charge_at=datetime(today.year, today.month, 20, 10, tzinfo=_TZ)
        if today.day < 20
        else datetime(
            today.year if today.month < 12 else today.year + 1,
            today.month + 1 if today.month < 12 else 1,
            20, 10, tzinfo=_TZ,
        ),
    )
    spotify = Subscription(
        user_id=user_id, service="Spotify", amount=169, currency="RUB",
        recurrence="monthly", day_of_month=15, notify_before_days=2,
        next_charge_at=datetime(today.year, today.month, 15, 10, tzinfo=_TZ)
        if today.day < 15
        else datetime(
            today.year if today.month < 12 else today.year + 1,
            today.month + 1 if today.month < 12 else 1,
            15, 10, tzinfo=_TZ,
        ),
    )
    chatgpt = Subscription(
        user_id=user_id, service="ChatGPT", amount=20, currency="USD",
        recurrence="monthly", day_of_month=1, notify_before_days=2,
        next_charge_at=datetime(
            today.year if today.month < 12 else today.year + 1,
            today.month + 1 if today.month < 12 else 1,
            1, 10, tzinfo=_TZ,
        ),
    )
    for s in (netflix, spotify, chatgpt):
        await storage.save_subscription(db_path, s)

    stats = await subscriptions.compute_stats(db_path, user_id=user_id)
    months = stats["months"]
    forecast = stats["year_forecast"]

    # Forecast: monthly × 12 occurrences per year.
    assert forecast["RUB"] == (499 + 169) * 12
    assert forecast["USD"] == 20 * 12

    # Months list non-empty and sorted ascending.
    assert len(months) >= 1
    keys = [m["month"] for m in months]
    assert keys == sorted(keys)

    # Each month bucket has total_by_currency dict.
    for m in months:
        assert isinstance(m["total_by_currency"], dict)
        assert m["count"] >= 1


@pytest.mark.asyncio
async def test_compute_stats_yearly_only_one_per_year_in_forecast(tmp_path):
    """A yearly subscription contributes 1 occurrence to the forecast."""
    db_path = str(tmp_path / "y.db")
    await storage.init_db(db_path)
    user_id = 1

    today = datetime.now(tz=_TZ)
    # Yearly subscription with a charge date in current month + 6 months
    # to ensure it lands in the 12-month horizon.
    next_at = datetime(
        today.year + 1 if today.month >= 6 else today.year,
        today.month - 6 if today.month > 6 else today.month + 6,
        15, 10, tzinfo=_TZ,
    )
    yearly = Subscription(
        user_id=user_id, service="Hetzner", amount=12, currency="EUR",
        recurrence="yearly", day_of_month=15, notify_before_days=0,
        next_charge_at=next_at,
    )
    await storage.save_subscription(db_path, yearly)

    stats = await subscriptions.compute_stats(db_path, user_id=user_id)
    assert stats["year_forecast"]["EUR"] == 12  # 12 EUR × 1/year = 12

"""subscription parser, storage CRUD, compute_next_charge_at edges.

Ollama is mocked. ``compute_next_charge_at`` is exercised on the famous
"31 февраля" edge: in 2026 February has 28 days, in a leap year
(2028) — 29 days. Storage round-trip uses ``tmp_path`` SQLite.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest

from services import storage, subscriptions
from services.intents import Subscription
from services.ollama_client import OllamaUnavailable


_TZ = ZoneInfo("Europe/Moscow")


# ---------------------------------------------------------------------------
# parse_subscription — parser unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_subscription_netflix_rub_with_card():
    """User: 'Netflix 499 руб Тинькофф каждое 20 число'."""
    mock_gen = AsyncMock(return_value={
        "service": "Netflix",
        "amount": 499,
        "currency": "RUB",
        "card": "Тинькофф",
        "recurrence": "monthly",
        "day_of_month": 20,
        "notify_before_days": 2,
    })
    with patch("services.subscriptions.generate_json", mock_gen):
        sub = await subscriptions.parse_subscription(
            "Netflix 499 руб Тинькофф каждое 20 число",
            base_url="http://x",
            user_id=42,
            now=datetime(2026, 5, 1, 12, 0, tzinfo=_TZ),
        )
    assert sub.service == "Netflix"
    assert sub.amount == 499
    assert sub.currency == "RUB"
    assert sub.card == "Тинькофф"
    assert sub.recurrence == "monthly"
    assert sub.day_of_month == 20
    assert sub.notify_before_days == 2
    assert sub.user_id == 42
    # next_charge_at should be 2026-05-20 (this month, 20th hasn't passed yet)
    assert sub.next_charge_at is not None
    assert sub.next_charge_at.month == 5 and sub.next_charge_at.day == 20


@pytest.mark.asyncio
async def test_parse_subscription_chatgpt_usd_no_card():
    mock_gen = AsyncMock(return_value={
        "service": "ChatGPT Plus",
        "amount": 20,
        "currency": "USD",
        "card": "",
        "recurrence": "monthly",
        "day_of_month": 1,
        "notify_before_days": 2,
    })
    with patch("services.subscriptions.generate_json", mock_gen):
        sub = await subscriptions.parse_subscription(
            "ChatGPT Plus 20 долларов ежемесячно",
            base_url="http://x",
            now=datetime(2026, 5, 5, 12, 0, tzinfo=_TZ),
        )
    assert sub.service == "ChatGPT Plus"
    assert sub.currency == "USD"
    assert sub.card is None  # empty string → None


@pytest.mark.asyncio
async def test_parse_subscription_yearly_eur():
    mock_gen = AsyncMock(return_value={
        "service": "Hetzner",
        "amount": 12,
        "currency": "EUR",
        "card": "",
        "recurrence": "yearly",
        "day_of_month": None,
        "notify_before_days": 0,
    })
    with patch("services.subscriptions.generate_json", mock_gen):
        sub = await subscriptions.parse_subscription(
            "хостинг Hetzner 12 евро в год, без напоминания",
            base_url="http://x",
            now=datetime(2026, 5, 1, tzinfo=_TZ),
        )
    assert sub.recurrence == "yearly"
    assert sub.notify_before_days == 0


@pytest.mark.asyncio
async def test_parse_subscription_invalid_currency_falls_back_to_rub():
    mock_gen = AsyncMock(return_value={
        "service": "Test",
        "amount": 100,
        "currency": "BANANA",
        "recurrence": "monthly",
        "day_of_month": 5,
    })
    with patch("services.subscriptions.generate_json", mock_gen):
        sub = await subscriptions.parse_subscription(
            "Test 100 BANANA каждое 5 число", base_url="http://x"
        )
    assert sub.currency == "RUB"


@pytest.mark.asyncio
async def test_parse_subscription_amount_zero_raises():
    mock_gen = AsyncMock(return_value={
        "service": "X",
        "amount": 0,
        "currency": "RUB",
        "recurrence": "monthly",
        "day_of_month": 1,
    })
    with patch("services.subscriptions.generate_json", mock_gen):
        with pytest.raises(subscriptions.SubscriptionParseError):
            await subscriptions.parse_subscription("...", base_url="http://x")


@pytest.mark.asyncio
async def test_parse_subscription_empty_service_raises():
    mock_gen = AsyncMock(return_value={
        "service": "",
        "amount": 100,
        "currency": "RUB",
        "recurrence": "monthly",
        "day_of_month": 1,
    })
    with patch("services.subscriptions.generate_json", mock_gen):
        with pytest.raises(subscriptions.SubscriptionParseError):
            await subscriptions.parse_subscription("...", base_url="http://x")


@pytest.mark.asyncio
async def test_parse_subscription_ollama_down_raises():
    mock_gen = AsyncMock(side_effect=OllamaUnavailable("down"))
    with patch("services.subscriptions.generate_json", mock_gen):
        with pytest.raises(subscriptions.SubscriptionParseError):
            await subscriptions.parse_subscription("...", base_url="http://x")


# ---------------------------------------------------------------------------
# compute_next_charge_at — date math
# ---------------------------------------------------------------------------


def _sub(**kw) -> Subscription:
    base = dict(
        user_id=1, service="X", amount=100, currency="RUB",
        recurrence="monthly", day_of_month=15, notify_before_days=2,
    )
    base.update(kw)
    return Subscription(**base)


def test_compute_next_charge_thirty_first_february_non_leap():
    """day_of_month=31 + from_dt=2026-02-01 → 2026-02-28 (non-leap year)."""
    sub = _sub(day_of_month=31)
    nxt = subscriptions.compute_next_charge_at(
        sub, from_dt=datetime(2026, 2, 1, 0, 0, tzinfo=_TZ)
    )
    assert nxt is not None
    assert nxt.year == 2026 and nxt.month == 2 and nxt.day == 28


def test_compute_next_charge_thirty_first_february_leap_year():
    """day_of_month=31 + from_dt=2028-02-01 → 2028-02-29 (leap)."""
    sub = _sub(day_of_month=31)
    nxt = subscriptions.compute_next_charge_at(
        sub, from_dt=datetime(2028, 2, 1, 0, 0, tzinfo=_TZ)
    )
    assert nxt is not None
    assert nxt.year == 2028 and nxt.month == 2 and nxt.day == 29


def test_compute_next_charge_rolls_to_next_month_when_day_passed():
    """day_of_month=15 + from_dt=2026-05-20 → 2026-06-15."""
    sub = _sub(day_of_month=15)
    nxt = subscriptions.compute_next_charge_at(
        sub, from_dt=datetime(2026, 5, 20, 12, 0, tzinfo=_TZ)
    )
    assert nxt is not None
    assert nxt.year == 2026 and nxt.month == 6 and nxt.day == 15


def test_compute_next_charge_picks_this_month_when_day_in_future():
    """day_of_month=20 + from_dt=2026-05-10 → 2026-05-20."""
    sub = _sub(day_of_month=20)
    nxt = subscriptions.compute_next_charge_at(
        sub, from_dt=datetime(2026, 5, 10, 12, 0, tzinfo=_TZ)
    )
    assert nxt is not None
    assert nxt.year == 2026 and nxt.month == 5 and nxt.day == 20


def test_compute_next_charge_weekly_seven_days():
    sub = _sub(recurrence="weekly", day_of_month=None)
    base = datetime(2026, 5, 10, 12, 0, tzinfo=_TZ)
    nxt = subscriptions.compute_next_charge_at(sub, from_dt=base)
    assert nxt is not None
    delta = nxt - base
    assert delta.days == 7


def test_compute_next_charge_monthly_no_day_returns_none():
    sub = _sub(recurrence="monthly", day_of_month=None)
    nxt = subscriptions.compute_next_charge_at(
        sub, from_dt=datetime(2026, 5, 10, tzinfo=_TZ)
    )
    assert nxt is None


# ---------------------------------------------------------------------------
# Storage CRUD — round trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_get_list_subscription_round_trip(tmp_path):
    db_path = str(tmp_path / "s.db")
    await storage.init_db(db_path)

    s = Subscription(
        user_id=42, service="Netflix", amount=499, currency="RUB",
        card="Тинькофф", recurrence="monthly", day_of_month=20,
        next_charge_at=datetime(2026, 5, 20, 10, 0, tzinfo=_TZ),
        notify_before_days=2,
    )
    sub_id = await storage.save_subscription(db_path, s)
    assert sub_id > 0

    fetched = await storage.get_subscription(db_path, sub_id)
    assert fetched is not None
    assert fetched.service == "Netflix"
    assert fetched.amount == 499
    assert fetched.card == "Тинькофф"
    assert fetched.day_of_month == 20
    assert fetched.next_charge_at is not None
    assert fetched.next_charge_at.day == 20

    rows = await storage.list_subscriptions(db_path, 42)
    assert len(rows) == 1 and rows[0].id == sub_id

    # Different user → empty.
    other = await storage.list_subscriptions(db_path, 99)
    assert other == []


@pytest.mark.asyncio
async def test_update_subscription_field(tmp_path):
    db_path = str(tmp_path / "u.db")
    await storage.init_db(db_path)
    s = Subscription(
        user_id=1, service="X", amount=100, currency="RUB",
        recurrence="monthly", day_of_month=10, notify_before_days=2,
    )
    sub_id = await storage.save_subscription(db_path, s)

    updated = await storage.update_subscription_fields(
        db_path, sub_id, {"amount": 199.99}
    )
    assert updated is not None and updated.amount == pytest.approx(199.99)

    with pytest.raises(ValueError):
        await storage.update_subscription_fields(
            db_path, sub_id, {"user_id": 999}
        )


@pytest.mark.asyncio
async def test_delete_subscription_idempotent(tmp_path):
    db_path = str(tmp_path / "d.db")
    await storage.init_db(db_path)
    s = Subscription(
        user_id=1, service="Spotify", amount=169, currency="RUB",
        recurrence="monthly", day_of_month=15, notify_before_days=2,
    )
    sub_id = await storage.save_subscription(db_path, s)

    deleted = await storage.delete_subscription(db_path, sub_id)
    assert deleted is not None and deleted.id == sub_id

    again = await storage.delete_subscription(db_path, sub_id)
    assert again is None

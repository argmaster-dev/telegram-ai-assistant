"""Subscription parser + recurring date math + monthly stats.

``parse_subscription`` is a single Ollama call returning structured JSON
that maps onto ``Subscription``. ``compute_next_charge_at`` handles the
"31 февраля" edge: months without day-N coerce to the last day of the
month via ``calendar.monthrange``. ``compute_stats`` aggregates per-month
totals broken down by currency and produces a 12-month forecast.

"""
from __future__ import annotations

import calendar
import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from services import storage
from services.intents import Subscription
from services.ollama_client import OllamaError, generate_json
from utils.prompts import SUBSCRIPTION_PARSER_PROMPT


logger = logging.getLogger(__name__)


TZ = ZoneInfo("Europe/Moscow")

_VALID_CURRENCIES: frozenset[str] = frozenset({"RUB", "USD", "EUR", "GBP"})
_VALID_RECURRENCE: frozenset[str] = frozenset({"monthly", "yearly", "weekly"})


class SubscriptionParseError(Exception):
    """Raised when Ollama output can't be turned into a valid Subscription."""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


async def parse_subscription(
    text: str,
    *,
    base_url: str,
    user_id: int = 0,
    now: Optional[datetime] = None,
) -> Subscription:
    """Parse a free-text subscription description into a ``Subscription``.

    Caller is expected to (1) persist the returned object via
    ``storage.save_subscription`` and (2) compute initial ``next_charge_at``
    via ``compute_next_charge_at`` before scheduling.

    Errors:
      * Ollama unreachable / non-JSON → ``SubscriptionParseError``.
      * Empty service name OR amount <= 0 → ``SubscriptionParseError``.
    """
    if now is None:
        now = datetime.now(tz=TZ)

    try:
        raw = await generate_json(
            SUBSCRIPTION_PARSER_PROMPT.format(text=text), base_url=base_url
        )
    except OllamaError as exc:
        logger.warning("parse_subscription ollama failed: %s", exc)
        raise SubscriptionParseError(f"ollama unavailable: {exc}") from exc

    service = str(raw.get("service", "")).strip()
    if not service:
        raise SubscriptionParseError(
            f"parse_subscription returned empty service: {raw!r}"
        )

    try:
        amount = float(raw.get("amount", 0))
    except (TypeError, ValueError) as exc:
        raise SubscriptionParseError(
            f"non-numeric amount {raw.get('amount')!r}: {exc}"
        ) from exc
    if amount <= 0:
        raise SubscriptionParseError(
            f"amount must be > 0, got {amount} ({raw!r})"
        )

    currency_raw = str(raw.get("currency", "RUB")).strip().upper()
    currency = currency_raw if currency_raw in _VALID_CURRENCIES else "RUB"

    card_raw = raw.get("card", "")
    card: Optional[str] = str(card_raw).strip() or None

    recurrence_raw = str(raw.get("recurrence", "monthly")).strip().lower()
    recurrence = (
        recurrence_raw if recurrence_raw in _VALID_RECURRENCE else "monthly"
    )

    day_of_month_raw = raw.get("day_of_month")
    day_of_month: Optional[int] = None
    if day_of_month_raw is not None and day_of_month_raw != "":
        try:
            day_of_month = int(day_of_month_raw)
            if not 1 <= day_of_month <= 31:
                day_of_month = None
        except (TypeError, ValueError):
            day_of_month = None

    # For monthly without explicit day, default to 1st of next month.
    if recurrence == "monthly" and day_of_month is None:
        day_of_month = 1

    try:
        notify_before_days = int(raw.get("notify_before_days", 2))
        if notify_before_days < 0:
            notify_before_days = 0
    except (TypeError, ValueError):
        notify_before_days = 2

    sub = Subscription(
        user_id=user_id,
        service=service,
        amount=amount,
        currency=currency,
        card=card,
        recurrence=recurrence,
        day_of_month=day_of_month,
        notify_before_days=notify_before_days,
    )
    sub.next_charge_at = compute_next_charge_at(sub, from_dt=now)
    return sub


# ---------------------------------------------------------------------------
# Date math
# ---------------------------------------------------------------------------


def compute_next_charge_at(
    sub: Subscription, *, from_dt: Optional[datetime] = None
) -> Optional[datetime]:
    """Return the next charge moment strictly after ``from_dt``.

    Handles edge cases:
      * ``recurrence='monthly'``, ``day_of_month=31`` in February → 28th
        (or 29th in a leap year), via ``calendar.monthrange``.
      * ``recurrence='yearly'`` — uses ``day_of_month`` of the next anniversary
        of ``from_dt`` (we don't track ``birth_date`` separately for v1.0).
      * ``recurrence='weekly'`` — pins to "7 days from from_dt" without a
        weekday concept (good enough for v1.0; weekly subs are rare).
      * Returns ``None`` if no valid date can be produced.

    All datetimes are returned in Europe/Moscow.
    """
    if from_dt is None:
        from_dt = datetime.now(tz=TZ)
    if from_dt.tzinfo is None:
        from_dt = from_dt.replace(tzinfo=TZ)

    if sub.recurrence == "monthly":
        if sub.day_of_month is None:
            return None
        candidate_year = from_dt.year
        candidate_month = from_dt.month
        # Try this month first; if requested day already passed (or is today
        # and time has elapsed), roll to next month.
        for _ in range(13):  # safety cap: max ~13 month rolls
            last_day = calendar.monthrange(candidate_year, candidate_month)[1]
            day = min(sub.day_of_month, last_day)
            candidate = datetime(
                candidate_year, candidate_month, day,
                hour=10, minute=0, second=0, tzinfo=TZ,
            )
            if candidate > from_dt:
                return candidate
            # Roll to next month.
            candidate_month += 1
            if candidate_month > 12:
                candidate_month = 1
                candidate_year += 1
        return None

    if sub.recurrence == "yearly":
        if sub.day_of_month is None:
            # Default: next anniversary at same day-of-year (rare).
            year = from_dt.year + 1
            return datetime(year, from_dt.month, from_dt.day, hour=10, tzinfo=TZ)
        # Use January-{day} on the next year.
        year = from_dt.year if from_dt.month < 1 else from_dt.year + 1
        return datetime(year, 1, sub.day_of_month, hour=10, tzinfo=TZ)

    if sub.recurrence == "weekly":
        return from_dt + timedelta(days=7)

    return None


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


async def compute_stats(
    db_path: str, user_id: int, *, months: int = 12
) -> dict:
    """Return aggregated per-month totals and a 12-month forecast.

    Schema:
      {
        "months": [
          {
            "month": "2026-05",
            "total_by_currency": {"RUB": 668, "USD": 20},
            "count": 2,
          },
          ...
        ],
        "year_forecast": {"RUB": 8016, "USD": 240},
      }

    The forecast scales each subscription by its annual occurrence count
    (monthly × 12, yearly × 1, weekly × 52). Months with no subscriptions
    are omitted from ``months``.
    """
    subs = await storage.list_subscriptions(db_path, user_id)
    if not subs:
        return {"months": [], "year_forecast": {}}

    now = datetime.now(tz=TZ)
    horizon = now + timedelta(days=months * 31)

    # Project each subscription onto upcoming charge moments within the horizon.
    per_month: dict[str, dict] = {}
    for sub in subs:
        cursor = sub.next_charge_at or compute_next_charge_at(sub, from_dt=now)
        guard = 0  # safety: stop after enough rolls (12 months × 4 = 48 ticks)
        while cursor is not None and cursor <= horizon and guard < 60:
            key = cursor.strftime("%Y-%m")
            bucket = per_month.setdefault(
                key, {"month": key, "total_by_currency": {}, "count": 0}
            )
            bucket["total_by_currency"][sub.currency] = (
                bucket["total_by_currency"].get(sub.currency, 0.0) + sub.amount
            )
            bucket["count"] += 1
            # Advance to next occurrence.
            cursor = compute_next_charge_at(
                sub, from_dt=cursor + timedelta(days=1)
            )
            guard += 1

    months_list = sorted(per_month.values(), key=lambda b: b["month"])
    # Tidy floats to integers when whole — keeps the formatter simple.
    for b in months_list:
        b["total_by_currency"] = {
            ccy: (int(amt) if abs(amt - round(amt)) < 0.005 else round(amt, 2))
            for ccy, amt in b["total_by_currency"].items()
        }

    forecast: dict[str, float] = {}
    for sub in subs:
        per_year = {"monthly": 12, "yearly": 1, "weekly": 52}.get(sub.recurrence, 12)
        forecast[sub.currency] = forecast.get(sub.currency, 0.0) + sub.amount * per_year
    forecast = {
        ccy: (int(amt) if abs(amt - round(amt)) < 0.005 else round(amt, 2))
        for ccy, amt in forecast.items()
    }

    return {"months": months_list, "year_forecast": forecast}

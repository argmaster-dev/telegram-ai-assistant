"""Reminder parsing + RRULE-aware next-fire computation.

``parse(text)`` — single-shot parser.
``parse_recurring(text)``, ``_validate_rrule``, ``next_fire_at``.

The high-throughput free-text handler uses ``services.pipeline.classify_and_parse``
(one Ollama call for single+recurring+unknown). The standalone ``parse`` and
``parse_recurring`` remain for (a) unit tests and (b) FSM ``/remind_edit`` when
the user enters a new RRULE pattern.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import holidays
from dateutil.parser import ParserError, isoparse
from dateutil.rrule import rrulestr

from services.intents import Reminder
from services.ollama_client import OllamaError, generate_json
from utils.prompts import RECURRING_PARSER_PROMPT, REMINDER_PARSER_PROMPT


logger = logging.getLogger(__name__)


TZ = ZoneInfo("Europe/Moscow")

_ALLOWED_FREQ = {"DAILY", "WEEKLY", "MONTHLY", "YEARLY"}
_ALLOWED_EXCLUDE = {"none", "ru", "ru+custom"}


class ReminderParseError(Exception):
    """Raised when Ollama's output can't be turned into a valid Reminder."""


# ---------------------------------------------------------------------------
# single-shot parser
# ---------------------------------------------------------------------------

async def parse(
    text: str,
    *,
    base_url: str,
    user_id: int = 0,
    now: Optional[datetime] = None,
) -> Reminder:
    """Parse ``text`` into a single-shot ``Reminder``. Raises ``ReminderParseError``."""
    if now is None:
        now = datetime.now(tz=TZ)

    prompt = REMINDER_PARSER_PROMPT.format(
        now=now.strftime("%Y-%m-%d %H:%M:%S %z"),
        text=text,
    )

    try:
        raw = await generate_json(prompt, base_url=base_url)
    except OllamaError as exc:
        raise ReminderParseError(f"ollama failed: {exc}") from exc

    title = str(raw.get("title", "")).strip()
    when_at_raw = str(raw.get("when_at", "")).strip()

    if not title:
        raise ReminderParseError(f"empty title from ollama: {raw!r}")
    if not when_at_raw:
        raise ReminderParseError(f"empty when_at from ollama: {raw!r}")

    try:
        when_at = isoparse(when_at_raw)
    except (ParserError, ValueError) as exc:
        raise ReminderParseError(
            f"invalid ISO datetime {when_at_raw!r}: {exc}"
        ) from exc

    if when_at.tzinfo is None:
        when_at = when_at.replace(tzinfo=TZ)

    if when_at <= now:
        raise ReminderParseError(
            f"when_at must be strictly in the future: {when_at_raw} <= now {now.isoformat()}"
        )

    return Reminder(
        user_id=user_id,
        title=title,
        when_at=when_at,
        source="text",
        status="scheduled",
    )


# ---------------------------------------------------------------------------
# recurring parser + RRULE utilities
# ---------------------------------------------------------------------------

def _validate_rrule(rrule_str: str) -> None:
    """Validate an RRULE string (without DTSTART).

    Two checks:
      1. The string is parseable by ``dateutil.rrule.rrulestr`` when prefixed
         with ``RRULE:`` and given a ``dtstart``.
      2. ``FREQ`` value is in our whitelist (DAILY/WEEKLY/MONTHLY/YEARLY).
    """
    if not rrule_str or not rrule_str.strip():
        raise ReminderParseError("rrule is empty")

    parts: dict[str, str] = {}
    for chunk in rrule_str.split(";"):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            parts[k.strip().upper()] = v.strip()

    freq = parts.get("FREQ", "").upper()
    if freq not in _ALLOWED_FREQ:
        raise ReminderParseError(
            f"unsupported FREQ={freq!r}; expected one of {sorted(_ALLOWED_FREQ)}"
        )

    try:
        rrulestr(f"RRULE:{rrule_str}", dtstart=datetime.now(tz=TZ))
    except (ValueError, TypeError) as exc:
        raise ReminderParseError(f"invalid RRULE {rrule_str!r}: {exc}") from exc


def next_fire_at(
    rrule_str: str,
    exdates: Optional[Iterable[str]] = None,
    exclude_holidays: str = "none",
    from_dt: Optional[datetime] = None,
) -> Optional[datetime]:
    """Return the next fire datetime strictly after ``from_dt``, skipping
    ``exdates`` and (if ``exclude_holidays`` is ``"ru"`` or ``"ru+custom"``)
    Russian public holidays.

    Returns ``None`` if:
      * the rule is exhausted (e.g., UNTIL/COUNT passed), or
      * no non-excluded date found within 1-year lookahead (safety cap).
    """
    if from_dt is None:
        from_dt = datetime.now(tz=TZ)
    if from_dt.tzinfo is None:
        from_dt = from_dt.replace(tzinfo=TZ)

    try:
        rule = rrulestr(f"RRULE:{rrule_str}", dtstart=from_dt)
    except (ValueError, TypeError) as exc:
        raise ReminderParseError(f"invalid RRULE {rrule_str!r}: {exc}") from exc

    use_ru_holidays = exclude_holidays in ("ru", "ru+custom")
    ru_holidays = holidays.country_holidays("RU") if use_ru_holidays else None

    exdate_set: set = set()
    for raw in exdates or []:
        try:
            exdate_set.add(isoparse(str(raw)).date())
        except (ParserError, ValueError) as exc:
            raise ReminderParseError(f"invalid exdate {raw!r}: {exc}") from exc

    cursor = from_dt
    for _ in range(366):
        candidate = rule.after(cursor, inc=False)
        if candidate is None:
            return None
        if candidate.tzinfo is None:
            candidate = candidate.replace(tzinfo=TZ)

        if candidate.date() in exdate_set:
            cursor = candidate
            continue
        if ru_holidays is not None and candidate.date() in ru_holidays:
            cursor = candidate
            continue
        return candidate
    return None


async def parse_recurring(
    text: str,
    *,
    base_url: str,
    user_id: int = 0,
    now: Optional[datetime] = None,
) -> Reminder:
    """Parse ``text`` as a recurring reminder via a dedicated Ollama prompt.

    Used by FSM ``/remind_edit`` when the user enters a new pattern. The main
    free-text flow uses ``services.pipeline.classify_and_parse`` instead (single
    Ollama call handles all intents + parsing at once).
    """
    if now is None:
        now = datetime.now(tz=TZ)

    prompt = RECURRING_PARSER_PROMPT.format(
        now=now.strftime("%Y-%m-%d %H:%M:%S %z"),
        text=text,
    )

    try:
        raw = await generate_json(prompt, base_url=base_url)
    except OllamaError as exc:
        raise ReminderParseError(f"ollama failed: {exc}") from exc

    title = str(raw.get("title", "")).strip()
    rrule_str = str(raw.get("rrule", "")).strip()
    exdates_raw = raw.get("exdates", []) or []
    exclude_holidays = str(raw.get("exclude_holidays", "none")).strip() or "none"

    if not title:
        raise ReminderParseError(f"empty title from ollama: {raw!r}")
    if not rrule_str:
        raise ReminderParseError(f"empty rrule from ollama: {raw!r}")

    _validate_rrule(rrule_str)

    if exclude_holidays not in _ALLOWED_EXCLUDE:
        exclude_holidays = "none"

    normalized_exdates: list[str] = []
    for d in exdates_raw:
        try:
            normalized_exdates.append(isoparse(str(d)).date().isoformat())
        except (ParserError, ValueError) as exc:
            raise ReminderParseError(f"invalid exdate {d!r}: {exc}") from exc

    computed_next = next_fire_at(
        rrule_str, normalized_exdates, exclude_holidays, from_dt=now
    )
    if computed_next is None:
        raise ReminderParseError(
            f"rrule {rrule_str!r} has no valid next fire within 1 year"
        )

    return Reminder(
        user_id=user_id,
        title=title,
        when_at=computed_next,
        rrule=rrule_str,
        exdates=normalized_exdates,
        exclude_holidays=exclude_holidays,
        source="text",
        status="scheduled",
    )

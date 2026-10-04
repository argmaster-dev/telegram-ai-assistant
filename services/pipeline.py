"""Unified classify + parse for reminders (single-shot OR recurring) in ONE Ollama call.

This module established the pattern: one prompt does both intent classification
and reminder parsing, halving latency from ~5–8s to ~2–3s on ``qwen3:8b``.
extends the output to three intents: ``reminder_single``,
``reminder_recurring``, ``unknown``.

``services.router`` and standalone ``services.reminders.parse`` / ``parse_recurring``
remain for FSM ``/remind_edit`` (the user enters a new pattern in natural
language) and for dedicated unit tests.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from dateutil.parser import ParserError, isoparse

from services.intents import Reminder
from services.ollama_client import OllamaError, generate_json
from services.reminders import (
    TZ,
    ReminderParseError,
    _ALLOWED_EXCLUDE,
    _validate_rrule,
    next_fire_at,
)
from utils.prompts import UNIFIED_REMINDER_PROMPT


logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    intent: str           # "reminder_single" | "reminder_recurring" | "unknown"
    confidence: float
    reminder: Optional[Reminder]


def _normalise_intent(raw: str) -> str:
    """Map raw Ollama ``intent`` value to our canonical set.

    Accepts legacy ``"reminder"`` (older format) as alias for
    ``"reminder_single"`` so in-flight jobs stored by the previous prompt
    format keep working after upgrade.
    """
    if raw == "reminder_single" or raw == "reminder":
        return "reminder_single"
    if raw == "reminder_recurring":
        return "reminder_recurring"
    return "unknown"


async def classify_and_parse(
    text: str,
    *,
    base_url: str,
    user_id: int = 0,
    now: Optional[datetime] = None,
) -> PipelineResult:
    """Single Ollama call returning intent + optional parsed ``Reminder``.

    Errors:
      * Any Ollama failure (unreachable, non-JSON) → returns ``intent="unknown"``.
      * ``reminder_single`` but title/when_at missing/past → ``ReminderParseError``.
      * ``reminder_recurring`` but title/rrule missing/invalid → ``ReminderParseError``.
    """
    if now is None:
        now = datetime.now(tz=TZ)

    prompt = UNIFIED_REMINDER_PROMPT.format(
        now=now.strftime("%Y-%m-%d %H:%M:%S %z"),
        text=text,
    )

    try:
        raw = await generate_json(prompt, base_url=base_url)
    except OllamaError as exc:
        logger.warning("pipeline classify_and_parse (ollama): %s", exc)
        return PipelineResult(intent="unknown", confidence=0.0, reminder=None)

    intent = _normalise_intent(raw.get("intent", "unknown"))

    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if intent == "unknown":
        return PipelineResult(intent=intent, confidence=confidence, reminder=None)

    title = str(raw.get("title", "")).strip()
    if not title:
        raise ReminderParseError(f"intent={intent} but title empty: {raw!r}")

    # ---- single-shot branch ------------------------------------------------
    if intent == "reminder_single":
        when_at_raw = str(raw.get("when_at", "")).strip()
        if not when_at_raw:
            raise ReminderParseError(
                f"intent=reminder_single but when_at empty: {raw!r}"
            )

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

        reminder = Reminder(
            user_id=user_id,
            title=title,
            when_at=when_at,
            source="text",
            status="scheduled",
        )
        return PipelineResult(intent=intent, confidence=confidence, reminder=reminder)

    # ---- recurring branch --------------------------------------------------
    rrule_str = str(raw.get("rrule", "")).strip()
    if not rrule_str:
        raise ReminderParseError(
            f"intent=reminder_recurring but rrule empty: {raw!r}"
        )
    _validate_rrule(rrule_str)

    exclude_holidays = str(raw.get("exclude_holidays", "none")).strip() or "none"
    if exclude_holidays not in _ALLOWED_EXCLUDE:
        exclude_holidays = "none"

    exdates_raw = raw.get("exdates", []) or []
    normalised_exdates: list[str] = []
    for d in exdates_raw:
        try:
            normalised_exdates.append(isoparse(str(d)).date().isoformat())
        except (ParserError, ValueError) as exc:
            raise ReminderParseError(f"invalid exdate {d!r}: {exc}") from exc

    computed_next = next_fire_at(
        rrule_str, normalised_exdates, exclude_holidays, from_dt=now
    )
    if computed_next is None:
        raise ReminderParseError(
            f"rrule {rrule_str!r} has no valid next fire within 1 year"
        )

    reminder = Reminder(
        user_id=user_id,
        title=title,
        when_at=computed_next,
        rrule=rrule_str,
        exdates=normalised_exdates,
        exclude_holidays=exclude_holidays,
        source="text",
        status="scheduled",
    )
    return PipelineResult(intent=intent, confidence=confidence, reminder=reminder)

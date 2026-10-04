"""Task parser — natural-language → ``Task`` via Ollama.

Single Ollama call with ``TASK_PARSER_PROMPT`` extracts title, optional
deadline, priority and category. The router (``services.router.classify``)
already filtered intent="task" so this parser does not classify — it only
extracts structured fields. Defensive on every field: missing/invalid
deadline becomes ``None`` (open-ended task), unknown priority/category
fall back to defaults ``"med"`` / ``"general"``.

"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from dateutil.parser import ParserError, isoparse

from services.intents import Task
from services.ollama_client import OllamaError, generate_json
from utils.prompts import TASK_PARSER_PROMPT


logger = logging.getLogger(__name__)


TZ = ZoneInfo("Europe/Moscow")

_VALID_PRIORITIES: frozenset[str] = frozenset({"low", "med", "high"})


class TaskParseError(Exception):
    """Raised when Ollama returned task=True but title is empty."""


async def parse_task(
    text: str,
    *,
    base_url: str,
    user_id: int = 0,
    now: Optional[datetime] = None,
) -> Task:
    """Parse a free-text task description into a ``Task`` dataclass.

    Errors:
      * Ollama unreachable / non-JSON → ``TaskParseError`` (caller decides
        whether to retry, ask user, or degrade).
      * Empty title → ``TaskParseError`` (a task without a title is useless).
      * Past deadline → silently dropped (task becomes open-ended). We don't
        raise because the user might say "купить молоко на вчера" expecting
        the task to still be created without the date.
    """
    if now is None:
        now = datetime.now(tz=TZ)

    prompt = TASK_PARSER_PROMPT.format(
        now=now.strftime("%Y-%m-%d %H:%M:%S %z"),
        text=text,
    )

    try:
        raw = await generate_json(prompt, base_url=base_url)
    except OllamaError as exc:
        logger.warning("parse_task ollama failed: %s", exc)
        raise TaskParseError(f"ollama unavailable: {exc}") from exc

    title = str(raw.get("title", "")).strip()
    if not title:
        raise TaskParseError(f"parse_task returned empty title: {raw!r}")

    deadline = _parse_deadline(raw.get("deadline", ""), now=now)

    priority_raw = str(raw.get("priority", "med")).strip().lower()
    priority = priority_raw if priority_raw in _VALID_PRIORITIES else "med"

    category_raw = str(raw.get("category", "general")).strip().lower()
    # Categories are open-ended (any snake_case identifier) — only reject
    # empty or whitespace-only values. Fall back to 'general'.
    category = category_raw or "general"

    return Task(
        user_id=user_id,
        title=title,
        deadline=deadline,
        priority=priority,
        status="todo",
        category=category,
    )


def _parse_deadline(raw: object, *, now: datetime) -> Optional[datetime]:
    """Parse an ISO date or datetime; return None on empty/invalid/past.

    Date-only values (``"2026-05-01"``) are coerced to end-of-day in TZ
    (``23:59:59`` Europe/Moscow) so a task with deadline "to пятницы"
    doesn't auto-disappear at midnight.
    """
    s = str(raw).strip()
    if not s:
        return None
    try:
        parsed = isoparse(s)
    except (ParserError, ValueError):
        logger.debug("parse_task: invalid deadline %r — dropping", s)
        return None

    if parsed.tzinfo is None:
        # Date-only or naïve datetime — pin to Europe/Moscow end-of-day.
        parsed = parsed.replace(
            hour=23, minute=59, second=59, tzinfo=TZ
        ) if parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0 \
            else parsed.replace(tzinfo=TZ)

    if parsed <= now:
        logger.debug("parse_task: deadline %s is past — open-ended task", parsed)
        return None

    return parsed

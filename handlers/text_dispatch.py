"""Unified free-text dispatcher.

Single text-message handler that:
  1. Classifies intent via ``services.router.classify`` (reminder | task |
     subscription | unknown).
  2. For ``reminder`` — delegates to the unified reminder pipeline (single
     OR recurring) just like the previous ``reminders.handle_free_text`` did.
  3. For ``task`` — calls ``services.tasks.parse_task`` and hands off to
     ``handlers.tasks_handler.handle_parsed_task``.
  4. For ``subscription`` — calls ``services.subscriptions.parse_subscription``
     and hands off to ``handlers.subs_handler.handle_parsed_subscription``.
  5. For ``unknown`` (or low confidence) — sends the help text.

This handler MUST be the catch-all for ``F.text & ~F.text.startswith("/")``;
the per-domain handlers (``reminders.py``, ``tasks_handler.py``,
``subs_handler.py``) only register slash-commands and inline callbacks.
"""
from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.types import Message

from config import Settings
from services import pipeline, router as intent_router
from services.reminders import ReminderParseError
from services.scheduler import SchedulerService

# Per-domain helpers — imported lazily inside the dispatcher to avoid circular
# imports during module loading (handlers register routers at import time).


logger = logging.getLogger(__name__)

router = Router(name="text_dispatch")


# Confidence below this threshold turns any non-"unknown" intent into "unknown"
# so we don't accidentally schedule a reminder when the model was just guessing.
_MIN_CONFIDENCE = 0.5


_HELP_TEXT = (
    "Пока не понял, чего ты хочешь. Я умею:\n\n"
    "📅 <b>Напоминания</b>:\n"
    "• <i>«напомни завтра в 14 позвонить в страховую»</i>\n"
    "• <i>«каждую пятницу в 18 сдать отчёт»</i>\n\n"
    "🎯 <b>Задачи</b>:\n"
    "• <i>«добавь задачу купить молоко к пятнице, home»</i>\n"
    "• <i>«срочно подготовить отчёт до пятницы, work»</i>\n\n"
    "💳 <b>Подписки</b>:\n"
    "• <i>«Spotify 169₽ Сбер каждое 15 число»</i>\n"
    "• <i>«ChatGPT Plus 20 долларов ежемесячно»</i>\n\n"
    "📎 <b>Документы</b> — пришли фото или PDF.\n\n"
    "Команды: /reminders, /tasks, /subs, /docs, /start."
)


@router.message(StateFilter(None), F.text & ~F.text.startswith("/"))
async def dispatch(
    message: Message,
    scheduler: SchedulerService,
    settings: Settings,
    db_path: str,
) -> None:
    if not message.text:
        return

    user_id = message.from_user.id if message.from_user else 0

    # Step 1: 4-way intent classification (cheap single Ollama call).
    try:
        cls = await intent_router.classify(
            message.text, base_url=settings.OLLAMA_URL
        )
    except Exception as exc:
        logger.exception("text_dispatch: router.classify crashed: %s", exc)
        await message.answer(_HELP_TEXT)
        return

    intent = cls["intent"]
    confidence = cls["confidence"]
    if intent != "unknown" and confidence < _MIN_CONFIDENCE:
        logger.info(
            "text_dispatch: low confidence %.2f for intent=%s, treating as unknown",
            confidence, intent,
        )
        intent = "unknown"

    # Step 2: per-intent dispatch.
    if intent == "reminder":
        await _handle_reminder(message, scheduler, settings, user_id)
        return

    if intent == "task":
        await _handle_task(message, db_path, settings, user_id)
        return

    if intent == "subscription":
        await _handle_subscription(message, db_path, settings, scheduler, user_id)
        return

    # unknown
    await message.answer(_HELP_TEXT)


async def _handle_reminder(
    message: Message,
    scheduler: SchedulerService,
    settings: Settings,
    user_id: int,
) -> None:
    """Schedule single OR recurring reminder via the unified pipeline."""
    try:
        result = await pipeline.classify_and_parse(
            message.text or "",
            base_url=settings.OLLAMA_URL,
            user_id=user_id,
        )
    except ReminderParseError as exc:
        logger.info("reminder parse failed for chat=%s: %s", user_id, exc)
        await message.answer(
            "Не смог разобрать. Попробуй переформулировать, например: "
            "<i>«напомни завтра в 14 сдать отчёт»</i>."
        )
        return
    except Exception as exc:
        logger.exception("pipeline.classify_and_parse crashed: %s", exc)
        await message.answer("Что-то пошло не так. Попробуй ещё раз.")
        return

    if (
        result.intent not in ("reminder_single", "reminder_recurring")
        or result.reminder is None
    ):
        # Router said "reminder" but pipeline disagreed (rare race after model update).
        await message.answer(_HELP_TEXT)
        return

    try:
        await scheduler.schedule_reminder(result.reminder)
    except Exception as exc:
        logger.exception("scheduler.schedule_reminder failed: %s", exc)
        await message.answer(
            "Не смог поставить напоминание. Попробуй ещё раз чуть позже."
        )
        return

    # Re-use reminders.py formatters via local import (avoid circular at import time).
    from handlers.reminders import _build_reminder_kb, _format_when

    icon = "🔁" if result.reminder.rrule else "✅"
    confirm = (
        f"{icon} Напомню <b>{_format_when(result.reminder)}</b>:\n"
        f"«{result.reminder.title}»"
    )
    if result.reminder.rrule:
        tag = ""
        if result.reminder.exclude_holidays == "ru":
            tag = " (без праздников РФ)"
        elif result.reminder.exclude_holidays == "ru+custom":
            tag = " (без праздников РФ и доп.дат)"
        confirm += f"\n<code>{result.reminder.rrule}</code>{tag}"

    await message.answer(confirm, reply_markup=_build_reminder_kb(result.reminder))


async def _handle_task(
    message: Message, db_path: str, settings: Settings, user_id: int
) -> None:
    """Parse the message as a task and hand off to tasks_handler."""
    from handlers.tasks_handler import handle_parsed_task
    from services.tasks import TaskParseError, parse_task

    try:
        task = await parse_task(
            message.text or "",
            base_url=settings.OLLAMA_URL,
            user_id=user_id,
        )
    except TaskParseError as exc:
        logger.info("task parse failed for chat=%s: %s", user_id, exc)
        await message.answer(
            "Не смог разобрать задачу. Попробуй переформулировать, например: "
            "<i>«добавь задачу купить молоко к пятнице, home»</i>."
        )
        return

    await handle_parsed_task(message, task, db_path=db_path)


async def _handle_subscription(
    message: Message,
    db_path: str,
    settings: Settings,
    scheduler: SchedulerService,
    user_id: int,
) -> None:
    """Parse the message as a subscription. Subs handler arrives."""
    try:
        from handlers.subs_handler import handle_parsed_subscription
        from services.subscriptions import (
            SubscriptionParseError,
            parse_subscription,
        )
    except ImportError:
        # subscriptions module not available — degrade gracefully.
        logger.info("subscription path requested but not deployed yet")
        await message.answer(
            "Подписки появятся позже. Пока используй /tasks или /reminders."
        )
        return

    try:
        sub = await parse_subscription(
            message.text or "",
            base_url=settings.OLLAMA_URL,
            user_id=user_id,
        )
    except SubscriptionParseError as exc:
        logger.info("subscription parse failed for chat=%s: %s", user_id, exc)
        await message.answer(
            "Не смог разобрать подписку. Попробуй переформулировать, например: "
            "<i>«Spotify 169₽ Сбер каждое 15 число»</i>."
        )
        return

    await handle_parsed_subscription(
        message, sub, db_path=db_path, scheduler=scheduler
    )

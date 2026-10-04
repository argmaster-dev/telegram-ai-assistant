"""Reminder handlers: free-text pipeline, listing, editing, skip/cancel callbacks.

single-shot flow + ``/reminders`` + ``cancel:*`` callback.
* Dispatcher now handles both ``reminder_single`` and ``reminder_recurring``
    intents from the unified pipeline.
  * ``/remind_list`` (alias ``/reminders``) lists single + recurring with distinct
    icons and different button sets per reminder.
  * ``skip_next:<id>`` callback adds an exdate for the next occurrence of a
    recurring reminder and re-schedules.
  * ``/remind_edit <id>`` — two-step FSM to edit title / when_at / rrule.

``scheduler``, ``settings``, ``db_path`` are injected by aiogram via
``dp[key] = value`` (see ``bot.py``).
"""
from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from config import Settings
from services import reminder_store
from services import reminders as reminder_parser
from services.intents import Reminder
from services.reminders import ReminderParseError, next_fire_at
from services.scheduler import SchedulerService, _job_id


logger = logging.getLogger(__name__)

router = Router(name="reminders")

_TZ = ZoneInfo("Europe/Moscow")


class EditState(StatesGroup):
    """FSM states for ``/remind_edit``."""
    choose_field = State()
    enter_value = State()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_when(reminder: Reminder) -> str:
    return (
        reminder.when_at.strftime("%d.%m.%Y %H:%M")
        if reminder.when_at
        else "?"
    )


def _format_reminder_line(reminder: Reminder) -> str:
    # subscription-driven reminders get a 💳 badge so the user
    # can tell them apart from manual reminders in /reminders.
    if reminder.source == "subscription":
        icon = "💳"
    elif reminder.rrule:
        icon = "🔁"
    else:
        icon = "🔔"
    lines = [f"{icon} <b>#{reminder.id}</b> — {_format_when(reminder)}"]
    lines.append(f"«{reminder.title}»")
    if reminder.rrule:
        rule_tag = ""
        if reminder.exclude_holidays == "ru":
            rule_tag = " (без праздников РФ)"
        elif reminder.exclude_holidays == "ru+custom":
            rule_tag = " (без праздников РФ и доп.дат)"
        lines.append(f"<code>{reminder.rrule}</code>{rule_tag}")
    return "\n".join(lines)


def _build_reminder_kb(reminder: Reminder) -> InlineKeyboardMarkup:
    """Inline buttons: different sets for single vs recurring."""
    rows: list[list[InlineKeyboardButton]] = []
    if reminder.rrule:
        rows.append([
            InlineKeyboardButton(
                text="⏭ Пропустить ближайшее",
                callback_data=f"skip_next:{reminder.id}",
            )
        ])
        rows.append([
            InlineKeyboardButton(
                text="❌ Удалить правило",
                callback_data=f"cancel:{reminder.id}",
            )
        ])
    else:
        rows.append([
            InlineKeyboardButton(
                text="❌ Отменить",
                callback_data=f"cancel:{reminder.id}",
            )
        ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# ---------------------------------------------------------------------------
# /remind_list (alias /reminders)
# ---------------------------------------------------------------------------

@router.message(Command("remind_list", "reminders"))
async def list_reminders_cmd(
    message: Message,
    scheduler: SchedulerService,
) -> None:
    user_id = message.from_user.id if message.from_user else 0
    rems = await scheduler.list_scheduled(user_id)
    if not rems:
        await message.answer("📭 Нет активных напоминаний.")
        return

    await message.answer(f"📋 Активных напоминаний: {len(rems)}")
    for reminder in rems:
        await message.answer(
            _format_reminder_line(reminder),
            reply_markup=_build_reminder_kb(reminder),
        )


# ---------------------------------------------------------------------------
# cancel:<id> — full cancel (single OR recurring series)
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("cancel:"))
async def handle_cancel_callback(
    callback: CallbackQuery,
    scheduler: SchedulerService,
) -> None:
    data = callback.data or ""
    try:
        reminder_id = int(data.split(":", 1)[1])
    except (IndexError, ValueError):
        await callback.answer("Некорректный callback.")
        return

    removed = await scheduler.cancel(reminder_id)
    if not removed:
        await callback.answer("Напоминание уже не активно.")
        return

    await callback.answer("Отменено.")
    if callback.message and callback.message.text:
        try:
            await callback.message.edit_text(
                f"<s>{callback.message.text}</s>\n\n❌ отменено",
                reply_markup=None,
            )
        except Exception as exc:
            logger.debug("cancel: edit_text failed: %s", exc)


# ---------------------------------------------------------------------------
# skip_next:<id> — skip one occurrence of a recurring reminder
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("skip_next:"))
async def handle_skip_next_callback(
    callback: CallbackQuery,
    scheduler: SchedulerService,
    db_path: str,
) -> None:
    data = callback.data or ""
    try:
        reminder_id = int(data.split(":", 1)[1])
    except (IndexError, ValueError):
        await callback.answer("Некорректный callback.")
        return

    reminder = await reminder_store.get(db_path, reminder_id)
    if reminder is None or reminder.status != "scheduled" or not reminder.rrule:
        await callback.answer("Не могу пропустить — правило неактивно.")
        return
    if reminder.when_at is None:
        await callback.answer("Нет ближайшего срабатывания.")
        return

    skip_date = reminder.when_at.date().isoformat()
    if skip_date not in reminder.exdates:
        reminder.exdates.append(skip_date)

    try:
        next_dt = next_fire_at(
            reminder.rrule,
            reminder.exdates,
            reminder.exclude_holidays,
            from_dt=datetime.now(tz=_TZ),
        )
    except Exception as exc:
        logger.exception("skip_next: next_fire_at crashed: %s", exc)
        await callback.answer("Ошибка вычисления следующей даты.")
        return

    if next_dt is None:
        reminder.status = "done"
        await reminder_store.update(db_path, reminder)
        try:
            scheduler.scheduler.remove_job(_job_id(reminder_id))
        except Exception:
            pass
        await callback.answer("Пропущено. Больше правило не сработает.")
    else:
        reminder.when_at = next_dt
        await reminder_store.update(db_path, reminder)
        try:
            await scheduler.reschedule(reminder)
        except Exception as exc:
            logger.exception("skip_next: reschedule failed: %s", exc)
            await callback.answer("Обновил exdate, но переставить job не смог.")
            return
        await callback.answer(
            f"Пропущено. Следующее — {next_dt.strftime('%d.%m %H:%M')}"
        )

    if callback.message:
        try:
            kb = _build_reminder_kb(reminder) if reminder.status == "scheduled" else None
            await callback.message.edit_text(
                _format_reminder_line(reminder),
                reply_markup=kb,
            )
        except Exception as exc:
            logger.debug("skip_next: edit_text failed: %s", exc)


# ---------------------------------------------------------------------------
# /remind_edit <id> — FSM
# ---------------------------------------------------------------------------

@router.message(Command("remind_edit"))
async def remind_edit_cmd(
    message: Message,
    command: CommandObject,
    db_path: str,
    state: FSMContext,
) -> None:
    args = (command.args or "").strip()
    if not args:
        await message.answer("Использование: <code>/remind_edit &lt;id&gt;</code>")
        return
    try:
        reminder_id = int(args)
    except ValueError:
        await message.answer("ID должен быть числом.")
        return

    reminder = await reminder_store.get(db_path, reminder_id)
    if reminder is None:
        await message.answer(f"Напоминание #{reminder_id} не найдено.")
        return
    if reminder.status != "scheduled":
        await message.answer(f"Напоминание #{reminder_id} неактивно (status={reminder.status}).")
        return

    await state.set_state(EditState.choose_field)
    await state.update_data(reminder_id=reminder_id)

    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(text="📝 Текст", callback_data="edit_field:title")],
    ]
    if reminder.rrule:
        rows.append([
            InlineKeyboardButton(text="🔁 Правило (rrule)", callback_data="edit_field:rrule")
        ])
    else:
        rows.append([
            InlineKeyboardButton(text="📅 Время", callback_data="edit_field:when_at")
        ])
    rows.append([InlineKeyboardButton(text="✖️ Отмена", callback_data="edit_field:__cancel")])

    await message.answer(
        f"Редактируем:\n{_format_reminder_line(reminder)}\n\nЧто меняем?",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
    )


@router.callback_query(F.data.startswith("edit_field:"), EditState.choose_field)
async def handle_edit_field_choice(
    callback: CallbackQuery,
    state: FSMContext,
) -> None:
    data = callback.data or ""
    field = data.split(":", 1)[1] if ":" in data else ""

    if field == "__cancel":
        await state.clear()
        await callback.answer("Отменено.")
        if callback.message:
            try:
                await callback.message.edit_text("Редактирование отменено.", reply_markup=None)
            except Exception:
                pass
        return

    if field not in ("title", "rrule", "when_at"):
        await callback.answer("Неизвестное поле.")
        return

    await state.update_data(field=field)
    await state.set_state(EditState.enter_value)

    prompts = {
        "title": "Введи новый текст напоминания:",
        "rrule": "Введи новое правило (например, «каждую пятницу в 18 кроме праздников РФ»):",
        "when_at": "Введи новое время (например, «завтра в 15» или ISO «2026-05-01 14:00»):",
    }
    await callback.answer()
    if callback.message:
        try:
            await callback.message.edit_text(prompts[field], reply_markup=None)
        except Exception as exc:
            logger.debug("edit: edit_text failed: %s", exc)


@router.message(EditState.enter_value, F.text)
async def handle_edit_value(
    message: Message,
    state: FSMContext,
    scheduler: SchedulerService,
    settings: Settings,
    db_path: str,
) -> None:
    if not message.text:
        return

    data = await state.get_data()
    reminder_id = data.get("reminder_id")
    field = data.get("field")

    if reminder_id is None or field not in ("title", "rrule", "when_at"):
        await state.clear()
        await message.answer("Ошибка FSM — начни /remind_edit заново.")
        return

    reminder = await reminder_store.get(db_path, reminder_id)
    if reminder is None:
        await state.clear()
        await message.answer(f"Напоминание #{reminder_id} исчезло.")
        return

    new_value = message.text.strip()

    try:
        if field == "title":
            reminder.title = new_value
        elif field == "when_at":
            parsed = await reminder_parser.parse(
                new_value,
                base_url=settings.OLLAMA_URL,
                user_id=reminder.user_id,
            )
            reminder.when_at = parsed.when_at
            reminder.rrule = None  # switching to single-shot
            reminder.exdates = []
            reminder.exclude_holidays = "none"
        elif field == "rrule":
            parsed = await reminder_parser.parse_recurring(
                new_value,
                base_url=settings.OLLAMA_URL,
                user_id=reminder.user_id,
            )
            reminder.rrule = parsed.rrule
            reminder.exdates = parsed.exdates
            reminder.exclude_holidays = parsed.exclude_holidays
            reminder.when_at = parsed.when_at
    except ReminderParseError as exc:
        await message.answer(f"Не смог разобрать: {exc}\n\nПопробуй ещё раз.")
        return
    except Exception as exc:
        logger.exception("edit: unexpected: %s", exc)
        await message.answer("Неожиданная ошибка. Начни /remind_edit заново.")
        await state.clear()
        return

    try:
        await reminder_store.update(db_path, reminder)
        await scheduler.reschedule(reminder)
    except Exception as exc:
        logger.exception("edit: update/reschedule failed: %s", exc)
        await message.answer("Обновил запись, но перестановка job'а упала. /remind_edit ещё раз.")
        await state.clear()
        return

    await state.clear()
    await message.answer(
        f"✅ Обновлено.\n{_format_reminder_line(reminder)}",
        reply_markup=_build_reminder_kb(reminder),
    )


# ---------------------------------------------------------------------------
# Free-text entry was moved to ``handlers/text_dispatch.py``.
# This module now only owns slash commands + inline callbacks for reminders.
# Helper symbols ``_format_when`` and ``_build_reminder_kb`` above are imported
# by the dispatcher to render the confirmation message.
# ---------------------------------------------------------------------------

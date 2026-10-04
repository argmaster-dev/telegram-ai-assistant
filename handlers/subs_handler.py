"""Subscription command handlers.

Commands:
  /subs               — list user's subscriptions ordered by next charge.
  /subs_stats         — per-month aggregate + 12-month forecast.
  /sub_edit <id>      — FSM: choose field → enter value (re-schedules notification).
  /sub_delete <id>    — confirm → cancel APScheduler job → delete row.

Free-text subscription entry is dispatched through ``handlers.text_dispatch``
which calls ``handle_parsed_subscription`` here after a successful parse.

Inline buttons under a subscription block:
  ✏️ Edit | 🗑 Delete

Callbacks:
  subedit:<id>          — start /sub_edit FSM.
  subdel:<id>           — show delete-confirm prompt.
  subdelconfirm:<id>    — actually delete after confirm.
  subdelcancel:<id>     — abort delete.
  subeditfield:<f>      — pick field in FSM.
  subeditcancel         — cancel /sub_edit FSM.
"""
from __future__ import annotations

import logging
from typing import Optional

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

from services import storage, subscriptions
from services.intents import Subscription
from services.scheduler import SchedulerService
from utils.messages import (
    format_subscription_confirm_added,
    format_subscription_delete_confirm,
    format_subscription_detail,
    format_subs_stats,
    format_subscriptions_page,
)


logger = logging.getLogger(__name__)

router = Router(name="subs")


_VALID_RECURRENCE: frozenset[str] = frozenset({"monthly", "yearly", "weekly"})
_VALID_CURRENCIES: frozenset[str] = frozenset({"RUB", "USD", "EUR", "GBP"})


# ---------------------------------------------------------------------------
# FSM
# ---------------------------------------------------------------------------


class SubEditState(StatesGroup):
    choose_field = State()
    enter_value = State()


# ---------------------------------------------------------------------------
# Inline keyboards
# ---------------------------------------------------------------------------


def _build_sub_kb(sub: Subscription) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✏️ Edit", callback_data=f"subedit:{sub.id}"),
        InlineKeyboardButton(text="🗑 Delete", callback_data=f"subdel:{sub.id}"),
    ]])


def _build_edit_field_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🏷 Сервис", callback_data="subeditfield:service")],
        [InlineKeyboardButton(text="💰 Сумма", callback_data="subeditfield:amount")],
        [InlineKeyboardButton(text="💱 Валюта", callback_data="subeditfield:currency")],
        [InlineKeyboardButton(text="💳 Карта", callback_data="subeditfield:card")],
        [InlineKeyboardButton(text="🗓 День месяца", callback_data="subeditfield:day_of_month")],
        [InlineKeyboardButton(text="🔁 Периодичность", callback_data="subeditfield:recurrence")],
        [InlineKeyboardButton(text="🔔 Уведомление за", callback_data="subeditfield:notify_before_days")],
        [InlineKeyboardButton(text="✖️ Отмена", callback_data="subeditcancel")],
    ])


def _build_delete_confirm_kb(sub_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text="🗑 Удалить",
            callback_data=f"subdelconfirm:{sub_id}",
        ),
        InlineKeyboardButton(
            text="✖️ Отмена",
            callback_data=f"subdelcancel:{sub_id}",
        ),
    ]])


# ---------------------------------------------------------------------------
# Public helper used by handlers/text_dispatch.py after a successful parse
# ---------------------------------------------------------------------------


async def handle_parsed_subscription(
    message: Message,
    parsed: Subscription,
    *,
    db_path: str,
    scheduler: SchedulerService,
) -> None:
    """Persist subscription, schedule notification, reply with confirmation."""
    parsed.user_id = message.from_user.id if message.from_user else 0
    sub_id = await storage.save_subscription(db_path, parsed)
    parsed.id = sub_id

    # Schedule pre-charge notification (returns None if no future charge).
    job_id = None
    try:
        job_id = await scheduler.schedule_subscription(parsed)
    except Exception as exc:
        logger.exception(
            "schedule_subscription failed for sub #%d: %s", sub_id, exc
        )

    saved = await storage.get_subscription(db_path, sub_id) or parsed
    await message.answer(
        format_subscription_confirm_added(saved, reminder_active=bool(job_id)),
        reply_markup=_build_sub_kb(saved),
    )


# ---------------------------------------------------------------------------
# Listing commands
# ---------------------------------------------------------------------------


@router.message(Command("subs"))
async def cmd_subs(message: Message, db_path: str) -> None:
    user_id = message.from_user.id if message.from_user else 0
    subs = await storage.list_subscriptions(db_path, user_id)
    await message.answer(format_subscriptions_page(subs))
    for s in subs:
        await message.answer(
            format_subscription_detail(s),
            reply_markup=_build_sub_kb(s),
        )


@router.message(Command("subs_stats"))
async def cmd_subs_stats(message: Message, db_path: str) -> None:
    user_id = message.from_user.id if message.from_user else 0
    stats = await subscriptions.compute_stats(db_path, user_id)
    await message.answer(format_subs_stats(stats))


# ---------------------------------------------------------------------------
# /sub_delete + confirm + cancel-job
# ---------------------------------------------------------------------------


def _parse_id_arg(command: CommandObject) -> Optional[int]:
    args = (command.args or "").strip()
    if not args:
        return None
    try:
        return int(args)
    except ValueError:
        return None


@router.message(Command("sub_delete"))
async def cmd_sub_delete(
    message: Message, command: CommandObject, db_path: str
) -> None:
    sub_id = _parse_id_arg(command)
    if sub_id is None:
        await message.answer("Использование: <code>/sub_delete &lt;id&gt;</code>")
        return
    sub = await storage.get_subscription(db_path, sub_id)
    if sub is None:
        await message.answer(f"Подписка #{sub_id} не найдена.")
        return
    await message.answer(
        format_subscription_delete_confirm(sub),
        reply_markup=_build_delete_confirm_kb(sub_id),
    )


@router.callback_query(F.data.startswith("subdel:"))
async def cb_sub_delete_prompt(callback: CallbackQuery, db_path: str) -> None:
    sub_id = _id_from_callback(callback.data)
    if sub_id is None:
        await callback.answer("Некорректный callback.")
        return
    sub = await storage.get_subscription(db_path, sub_id)
    if sub is None:
        await callback.answer("Подписка исчезла.")
        return
    await callback.answer()
    if callback.message:
        try:
            await callback.message.edit_text(
                format_subscription_delete_confirm(sub),
                reply_markup=_build_delete_confirm_kb(sub_id),
            )
        except Exception as exc:
            logger.debug("subdel edit_text failed: %s", exc)


@router.callback_query(F.data.startswith("subdelconfirm:"))
async def cb_sub_delete_confirm(
    callback: CallbackQuery,
    db_path: str,
    scheduler: SchedulerService,
) -> None:
    """Transactional delete: cancel APScheduler job FIRST, then delete row.

    If the cancel raises (very unlikely with our idempotent wrapper), we abort
    and tell the user — leaves a row + a still-scheduled job rather than the
    opposite (orphan job firing into a deleted row).
    """
    sub_id = _id_from_callback(callback.data)
    if sub_id is None:
        await callback.answer("Некорректный callback.")
        return
    try:
        await scheduler.cancel_subscription(sub_id)
    except Exception as exc:
        logger.exception("cancel_subscription failed for #%d: %s", sub_id, exc)
        await callback.answer("Не удалось отменить уведомление, подписка не удалена.")
        return
    deleted = await storage.delete_subscription(db_path, sub_id)
    if deleted is None:
        await callback.answer("Уже удалена.")
    else:
        await callback.answer("Удалено.")
    if callback.message:
        try:
            await callback.message.edit_text(
                f"🗑 Подписка #{sub_id} удалена.",
                reply_markup=None,
            )
        except Exception as exc:
            logger.debug("subdelconfirm edit_text failed: %s", exc)


@router.callback_query(F.data.startswith("subdelcancel:"))
async def cb_sub_delete_cancel(
    callback: CallbackQuery, db_path: str
) -> None:
    sub_id = _id_from_callback(callback.data)
    if sub_id is None:
        await callback.answer()
        return
    sub = await storage.get_subscription(db_path, sub_id)
    await callback.answer("Отменено.")
    if callback.message and sub:
        try:
            await callback.message.edit_text(
                format_subscription_detail(sub),
                reply_markup=_build_sub_kb(sub),
            )
        except Exception as exc:
            logger.debug("subdelcancel edit_text failed: %s", exc)


# ---------------------------------------------------------------------------
# /sub_edit + FSM
# ---------------------------------------------------------------------------


@router.message(Command("sub_edit"))
async def cmd_sub_edit(
    message: Message,
    command: CommandObject,
    db_path: str,
    state: FSMContext,
) -> None:
    sub_id = _parse_id_arg(command)
    if sub_id is None:
        await message.answer("Использование: <code>/sub_edit &lt;id&gt;</code>")
        return
    sub = await storage.get_subscription(db_path, sub_id)
    if sub is None:
        await message.answer(f"Подписка #{sub_id} не найдена.")
        return
    await state.set_state(SubEditState.choose_field)
    await state.update_data(sub_id=sub_id)
    await message.answer(
        f"Редактируем:\n\n{format_subscription_detail(sub)}\n\nЧто меняем?",
        reply_markup=_build_edit_field_kb(),
    )


@router.callback_query(F.data.startswith("subedit:"))
async def cb_sub_edit_start(
    callback: CallbackQuery, db_path: str, state: FSMContext
) -> None:
    sub_id = _id_from_callback(callback.data)
    if sub_id is None:
        await callback.answer("Некорректный callback.")
        return
    sub = await storage.get_subscription(db_path, sub_id)
    if sub is None:
        await callback.answer("Подписка исчезла.")
        return
    await state.set_state(SubEditState.choose_field)
    await state.update_data(sub_id=sub_id)
    await callback.answer()
    if callback.message:
        try:
            await callback.message.edit_text(
                f"Редактируем:\n\n{format_subscription_detail(sub)}\n\nЧто меняем?",
                reply_markup=_build_edit_field_kb(),
            )
        except Exception as exc:
            logger.debug("subedit edit_text failed: %s", exc)


@router.callback_query(F.data.startswith("subeditfield:"), SubEditState.choose_field)
async def cb_sub_edit_field(callback: CallbackQuery, state: FSMContext) -> None:
    field = (callback.data or "").split(":", 1)[1] if ":" in (callback.data or "") else ""
    valid_fields = {
        "service", "amount", "currency", "card", "day_of_month",
        "recurrence", "notify_before_days",
    }
    if field not in valid_fields:
        await callback.answer("Неизвестное поле.")
        return
    await state.update_data(field=field)
    await state.set_state(SubEditState.enter_value)
    prompts = {
        "service": "Введи новое название сервиса:",
        "amount": "Введи новую сумму (число; для копеек — точка, например <code>19.99</code>).",
        "currency": "Введи валюту: <code>RUB</code>, <code>USD</code>, <code>EUR</code> или <code>GBP</code>.",
        "card": "Введи карту/банк или <code>none</code> чтобы убрать.",
        "day_of_month": "Введи день списания (1..31) или <code>none</code> чтобы убрать.",
        "recurrence": "Введи периодичность: <code>monthly</code>, <code>yearly</code> или <code>weekly</code>.",
        "notify_before_days": "Введи число дней до списания, за сколько уведомить (0 — без уведомления).",
    }
    await callback.answer()
    if callback.message:
        try:
            await callback.message.edit_text(prompts[field], reply_markup=None)
        except Exception as exc:
            logger.debug("subeditfield edit_text failed: %s", exc)


@router.callback_query(F.data == "subeditcancel")
async def cb_sub_edit_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.answer("Отменено.")
    if callback.message:
        try:
            await callback.message.edit_text(
                "Редактирование отменено.", reply_markup=None
            )
        except Exception as exc:
            logger.debug("subeditcancel edit_text failed: %s", exc)


@router.message(SubEditState.enter_value, F.text)
async def msg_sub_edit_value(
    message: Message,
    state: FSMContext,
    db_path: str,
    scheduler: SchedulerService,
) -> None:
    if not message.text:
        return
    data = await state.get_data()
    sub_id = data.get("sub_id")
    field = data.get("field")
    if sub_id is None or field is None:
        await state.clear()
        await message.answer("Сбой FSM — начни /sub_edit заново.")
        return

    new_value_raw = message.text.strip()
    update: dict = {}

    if field == "service":
        if not new_value_raw:
            await message.answer("Название не может быть пустым.")
            return
        update["service"] = new_value_raw
    elif field == "amount":
        try:
            v = float(new_value_raw.replace(",", "."))
        except ValueError:
            await message.answer("Сумма должна быть числом. Пример: <code>499</code> или <code>19.99</code>.")
            return
        if v <= 0:
            await message.answer("Сумма должна быть > 0.")
            return
        update["amount"] = v
    elif field == "currency":
        v = new_value_raw.upper()
        if v not in _VALID_CURRENCIES:
            await message.answer("Валюта: <code>RUB</code>, <code>USD</code>, <code>EUR</code> или <code>GBP</code>.")
            return
        update["currency"] = v
    elif field == "card":
        if new_value_raw.lower() in ("none", "нет", "убрать", "-"):
            update["card"] = None
        else:
            update["card"] = new_value_raw
    elif field == "day_of_month":
        if new_value_raw.lower() in ("none", "нет", "убрать", "-"):
            update["day_of_month"] = None
        else:
            try:
                d = int(new_value_raw)
            except ValueError:
                await message.answer("День должен быть числом 1..31.")
                return
            if not 1 <= d <= 31:
                await message.answer("День должен быть от 1 до 31.")
                return
            update["day_of_month"] = d
    elif field == "recurrence":
        v = new_value_raw.lower()
        if v not in _VALID_RECURRENCE:
            await message.answer("Только: <code>monthly</code>, <code>yearly</code>, <code>weekly</code>.")
            return
        update["recurrence"] = v
    elif field == "notify_before_days":
        try:
            n = int(new_value_raw)
        except ValueError:
            await message.answer("Число дней должно быть целым.")
            return
        if n < 0:
            await message.answer("Число дней не может быть отрицательным.")
            return
        update["notify_before_days"] = n

    sub = await storage.update_subscription_fields(db_path, sub_id, update)
    if sub is None:
        await state.clear()
        await message.answer(f"Подписка #{sub_id} исчезла во время правки.")
        return

    # Recompute next_charge_at after structural changes that affect schedule.
    if field in {"day_of_month", "recurrence", "notify_before_days"}:
        new_next = subscriptions.compute_next_charge_at(sub)
        if new_next is not None and new_next != sub.next_charge_at:
            sub = await storage.update_subscription_fields(
                db_path, sub_id, {"next_charge_at": new_next}
            ) or sub

    # Re-schedule notification after any change (cheap; idempotent).
    try:
        await scheduler.reschedule_subscription(sub)
    except Exception as exc:
        logger.exception("reschedule_subscription failed for #%d: %s", sub_id, exc)
        await state.clear()
        await message.answer(
            f"⚠️ Поле обновил, но переставить уведомление не смог: {exc}\n\n"
            f"{format_subscription_detail(sub)}"
        )
        return

    await state.clear()
    await message.answer(
        f"✅ Обновлено.\n\n{format_subscription_detail(sub)}",
        reply_markup=_build_sub_kb(sub),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _id_from_callback(data: Optional[str]) -> Optional[int]:
    if not data or ":" not in data:
        return None
    try:
        return int(data.split(":", 1)[1])
    except ValueError:
        return None

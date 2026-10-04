"""Task command handlers.

Commands:
  /tasks               — list active (status='todo') tasks.
  /tasks_all           — list all (active + done).
  /tasks_by_cat <cat>  — list active tasks in a category.
  /task_done <id>      — mark task as done.
  /task_undone <id>    — mark a done task back to todo (no separate /tasks_by_pri command).
  /task_edit <id>      — FSM: choose field (title/deadline/priority/category) → enter value.
  /task_delete <id>    — delete with confirm.

Free-text task entry happens through the unified text dispatcher (see
``handlers/text_dispatch.py``); this module provides only the *helper*
``handle_parsed_task`` that the dispatcher calls after a successful parse.

Inline buttons under a task block:
  ✅ Done | ✏️ Edit | 🗑 Delete

Callbacks:
  taskdone:<id>        — mark done (re-renders block as struck-through).
  taskundone:<id>      — mark back to todo.
  taskedit:<id>        — start /task_edit FSM.
  taskdel:<id>         — show delete-confirm prompt.
  taskdelconfirm:<id>  — actually delete after confirm.
  taskeditfield:<f>    — pick field in FSM (title/deadline/priority/category).
  taskeditcancel       — cancel /task_edit FSM.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
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
from dateutil.parser import ParserError, isoparse

from services import storage
from services.intents import Task
from utils.messages import (
    format_task_confirm_added,
    format_task_delete_confirm,
    format_task_detail,
    format_tasks_page,
)


logger = logging.getLogger(__name__)

router = Router(name="tasks")

_TZ = ZoneInfo("Europe/Moscow")
_VALID_PRIORITIES: frozenset[str] = frozenset({"low", "med", "high"})


# ---------------------------------------------------------------------------
# FSM
# ---------------------------------------------------------------------------


class TaskEditState(StatesGroup):
    """FSM for /task_edit and taskedit:<id> callback."""
    choose_field = State()
    enter_value = State()


# ---------------------------------------------------------------------------
# Inline keyboards
# ---------------------------------------------------------------------------


def _build_task_kb(task: Task) -> InlineKeyboardMarkup:
    """Inline buttons for a single task block."""
    if task.status == "done":
        return InlineKeyboardMarkup(
            inline_keyboard=[[
                InlineKeyboardButton(
                    text="↩️ Вернуть в todo",
                    callback_data=f"taskundone:{task.id}",
                ),
                InlineKeyboardButton(
                    text="🗑 Удалить",
                    callback_data=f"taskdel:{task.id}",
                ),
            ]]
        )
    return InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(
                text="✅ Done",
                callback_data=f"taskdone:{task.id}",
            ),
            InlineKeyboardButton(
                text="✏️ Edit",
                callback_data=f"taskedit:{task.id}",
            ),
            InlineKeyboardButton(
                text="🗑 Delete",
                callback_data=f"taskdel:{task.id}",
            ),
        ]]
    )


def _build_edit_field_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📝 Текст", callback_data="taskeditfield:title")],
        [InlineKeyboardButton(text="⏰ Дедлайн", callback_data="taskeditfield:deadline")],
        [InlineKeyboardButton(text="🏷 Приоритет", callback_data="taskeditfield:priority")],
        [InlineKeyboardButton(text="📁 Категория", callback_data="taskeditfield:category")],
        [InlineKeyboardButton(text="✖️ Отмена", callback_data="taskeditcancel")],
    ])


def _build_delete_confirm_kb(task_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text="🗑 Удалить",
            callback_data=f"taskdelconfirm:{task_id}",
        ),
        InlineKeyboardButton(
            text="✖️ Отмена",
            callback_data=f"taskdelcancel:{task_id}",
        ),
    ]])


# ---------------------------------------------------------------------------
# Public helper used by handlers/text_dispatch.py after a successful parse
# ---------------------------------------------------------------------------


async def handle_parsed_task(
    message: Message, parsed: Task, *, db_path: str
) -> None:
    """Persist a freshly-parsed Task and reply with confirmation + buttons."""
    parsed.user_id = message.from_user.id if message.from_user else 0
    task_id = await storage.save_task(db_path, parsed)
    parsed.id = task_id
    saved = await storage.get_task(db_path, task_id) or parsed
    await message.answer(
        format_task_confirm_added(saved),
        reply_markup=_build_task_kb(saved),
    )


# ---------------------------------------------------------------------------
# Listing commands
# ---------------------------------------------------------------------------


@router.message(Command("tasks"))
async def cmd_tasks(message: Message, db_path: str) -> None:
    user_id = message.from_user.id if message.from_user else 0
    tasks = await storage.list_tasks(db_path, user_id, status="todo")
    await message.answer(format_tasks_page(tasks, scope="todo"))
    for t in tasks:
        await message.answer(
            format_task_detail(t),
            reply_markup=_build_task_kb(t),
        )


@router.message(Command("tasks_all"))
async def cmd_tasks_all(message: Message, db_path: str) -> None:
    user_id = message.from_user.id if message.from_user else 0
    tasks = await storage.list_tasks(db_path, user_id, status=None)
    await message.answer(format_tasks_page(tasks, scope="all"))
    for t in tasks:
        await message.answer(
            format_task_detail(t),
            reply_markup=_build_task_kb(t),
        )


@router.message(Command("tasks_by_cat"))
async def cmd_tasks_by_cat(
    message: Message, command: CommandObject, db_path: str
) -> None:
    args = (command.args or "").strip().lower()
    if not args:
        await message.answer(
            "Использование: <code>/tasks_by_cat &lt;категория&gt;</code>\n"
            "Примеры категорий: <i>work, home, shopping, health, study, finance, general</i>"
        )
        return
    user_id = message.from_user.id if message.from_user else 0
    tasks = await storage.list_tasks_by_category(db_path, user_id, args)
    await message.answer(format_tasks_page(tasks, scope="by_cat"))
    for t in tasks:
        await message.answer(
            format_task_detail(t),
            reply_markup=_build_task_kb(t),
        )


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


def _parse_id_arg(command: CommandObject) -> Optional[int]:
    args = (command.args or "").strip()
    if not args:
        return None
    try:
        return int(args)
    except ValueError:
        return None


@router.message(Command("task_done"))
async def cmd_task_done(
    message: Message, command: CommandObject, db_path: str
) -> None:
    task_id = _parse_id_arg(command)
    if task_id is None:
        await message.answer("Использование: <code>/task_done &lt;id&gt;</code>")
        return
    task = await storage.mark_task_done(db_path, task_id)
    if task is None:
        await message.answer(f"Задача #{task_id} не найдена.")
        return
    await message.answer(
        f"✅ Задача #{task_id} закрыта.\n\n{format_task_detail(task)}",
        reply_markup=_build_task_kb(task),
    )


@router.message(Command("task_undone"))
async def cmd_task_undone(
    message: Message, command: CommandObject, db_path: str
) -> None:
    task_id = _parse_id_arg(command)
    if task_id is None:
        await message.answer("Использование: <code>/task_undone &lt;id&gt;</code>")
        return
    task = await storage.mark_task_todo(db_path, task_id)
    if task is None:
        await message.answer(f"Задача #{task_id} не найдена.")
        return
    await message.answer(
        f"↩️ Задача #{task_id} вернулась в todo.\n\n{format_task_detail(task)}",
        reply_markup=_build_task_kb(task),
    )


@router.callback_query(F.data.startswith("taskdone:"))
async def cb_task_done(callback: CallbackQuery, db_path: str) -> None:
    task_id = _id_from_callback(callback.data)
    if task_id is None:
        await callback.answer("Некорректный callback.")
        return
    task = await storage.mark_task_done(db_path, task_id)
    if task is None:
        await callback.answer("Задача исчезла.")
        return
    await callback.answer("Готово.")
    if callback.message:
        try:
            await callback.message.edit_text(
                format_task_detail(task),
                reply_markup=_build_task_kb(task),
            )
        except Exception as exc:
            logger.debug("taskdone edit_text failed: %s", exc)


@router.callback_query(F.data.startswith("taskundone:"))
async def cb_task_undone(callback: CallbackQuery, db_path: str) -> None:
    task_id = _id_from_callback(callback.data)
    if task_id is None:
        await callback.answer("Некорректный callback.")
        return
    task = await storage.mark_task_todo(db_path, task_id)
    if task is None:
        await callback.answer("Задача исчезла.")
        return
    await callback.answer("Возвращена в todo.")
    if callback.message:
        try:
            await callback.message.edit_text(
                format_task_detail(task),
                reply_markup=_build_task_kb(task),
            )
        except Exception as exc:
            logger.debug("taskundone edit_text failed: %s", exc)


# ---------------------------------------------------------------------------
# /task_delete + confirm callbacks
# ---------------------------------------------------------------------------


@router.message(Command("task_delete"))
async def cmd_task_delete(
    message: Message, command: CommandObject, db_path: str
) -> None:
    task_id = _parse_id_arg(command)
    if task_id is None:
        await message.answer("Использование: <code>/task_delete &lt;id&gt;</code>")
        return
    task = await storage.get_task(db_path, task_id)
    if task is None:
        await message.answer(f"Задача #{task_id} не найдена.")
        return
    await message.answer(
        format_task_delete_confirm(task),
        reply_markup=_build_delete_confirm_kb(task_id),
    )


@router.callback_query(F.data.startswith("taskdel:"))
async def cb_task_delete_prompt(
    callback: CallbackQuery, db_path: str
) -> None:
    task_id = _id_from_callback(callback.data)
    if task_id is None:
        await callback.answer("Некорректный callback.")
        return
    task = await storage.get_task(db_path, task_id)
    if task is None:
        await callback.answer("Задача исчезла.")
        return
    await callback.answer()
    if callback.message:
        try:
            await callback.message.edit_text(
                format_task_delete_confirm(task),
                reply_markup=_build_delete_confirm_kb(task_id),
            )
        except Exception as exc:
            logger.debug("taskdel edit_text failed: %s", exc)


@router.callback_query(F.data.startswith("taskdelconfirm:"))
async def cb_task_delete_confirm(
    callback: CallbackQuery, db_path: str
) -> None:
    task_id = _id_from_callback(callback.data)
    if task_id is None:
        await callback.answer("Некорректный callback.")
        return
    deleted = await storage.delete_task(db_path, task_id)
    if deleted is None:
        await callback.answer("Уже удалено.")
    else:
        await callback.answer("Удалено.")
    if callback.message:
        try:
            await callback.message.edit_text(
                f"🗑 Задача #{task_id} удалена.",
                reply_markup=None,
            )
        except Exception as exc:
            logger.debug("taskdelconfirm edit_text failed: %s", exc)


@router.callback_query(F.data.startswith("taskdelcancel:"))
async def cb_task_delete_cancel(
    callback: CallbackQuery, db_path: str
) -> None:
    task_id = _id_from_callback(callback.data)
    if task_id is None:
        await callback.answer()
        return
    task = await storage.get_task(db_path, task_id)
    await callback.answer("Отменено.")
    if callback.message and task:
        try:
            await callback.message.edit_text(
                format_task_detail(task),
                reply_markup=_build_task_kb(task),
            )
        except Exception as exc:
            logger.debug("taskdelcancel edit_text failed: %s", exc)


# ---------------------------------------------------------------------------
# /task_edit + FSM
# ---------------------------------------------------------------------------


@router.message(Command("task_edit"))
async def cmd_task_edit(
    message: Message,
    command: CommandObject,
    db_path: str,
    state: FSMContext,
) -> None:
    task_id = _parse_id_arg(command)
    if task_id is None:
        await message.answer("Использование: <code>/task_edit &lt;id&gt;</code>")
        return
    task = await storage.get_task(db_path, task_id)
    if task is None:
        await message.answer(f"Задача #{task_id} не найдена.")
        return

    await state.set_state(TaskEditState.choose_field)
    await state.update_data(task_id=task_id)
    await message.answer(
        f"Редактируем:\n\n{format_task_detail(task)}\n\nЧто меняем?",
        reply_markup=_build_edit_field_kb(),
    )


@router.callback_query(F.data.startswith("taskedit:"))
async def cb_task_edit_start(
    callback: CallbackQuery, db_path: str, state: FSMContext
) -> None:
    task_id = _id_from_callback(callback.data)
    if task_id is None:
        await callback.answer("Некорректный callback.")
        return
    task = await storage.get_task(db_path, task_id)
    if task is None:
        await callback.answer("Задача исчезла.")
        return
    await state.set_state(TaskEditState.choose_field)
    await state.update_data(task_id=task_id)
    await callback.answer()
    if callback.message:
        try:
            await callback.message.edit_text(
                f"Редактируем:\n\n{format_task_detail(task)}\n\nЧто меняем?",
                reply_markup=_build_edit_field_kb(),
            )
        except Exception as exc:
            logger.debug("taskedit edit_text failed: %s", exc)


@router.callback_query(F.data.startswith("taskeditfield:"), TaskEditState.choose_field)
async def cb_task_edit_field(
    callback: CallbackQuery, state: FSMContext
) -> None:
    field = (callback.data or "").split(":", 1)[1] if ":" in (callback.data or "") else ""
    if field not in ("title", "deadline", "priority", "category"):
        await callback.answer("Неизвестное поле.")
        return
    await state.update_data(field=field)
    await state.set_state(TaskEditState.enter_value)
    prompts = {
        "title": "Введи новый текст задачи:",
        "deadline": (
            "Введи новый дедлайн ISO-датой (например, <code>2026-05-10</code> "
            "или <code>2026-05-10 18:00</code>) либо <code>none</code> чтобы убрать."
        ),
        "priority": "Введи новый приоритет: <code>low</code>, <code>med</code> или <code>high</code>.",
        "category": "Введи новую категорию (одно слово, snake_case).",
    }
    await callback.answer()
    if callback.message:
        try:
            await callback.message.edit_text(prompts[field], reply_markup=None)
        except Exception as exc:
            logger.debug("taskeditfield edit_text failed: %s", exc)


@router.callback_query(F.data == "taskeditcancel")
async def cb_task_edit_cancel(
    callback: CallbackQuery, state: FSMContext
) -> None:
    await state.clear()
    await callback.answer("Отменено.")
    if callback.message:
        try:
            await callback.message.edit_text(
                "Редактирование отменено.", reply_markup=None
            )
        except Exception as exc:
            logger.debug("taskeditcancel edit_text failed: %s", exc)


@router.message(TaskEditState.enter_value, F.text)
async def msg_task_edit_value(
    message: Message, state: FSMContext, db_path: str
) -> None:
    if not message.text:
        return
    data = await state.get_data()
    task_id = data.get("task_id")
    field = data.get("field")
    if task_id is None or field not in ("title", "deadline", "priority", "category"):
        await state.clear()
        await message.answer("Сбой FSM — попробуй /task_edit заново.")
        return

    new_value_raw = message.text.strip()
    update: dict = {}
    if field == "title":
        if not new_value_raw:
            await message.answer("Текст задачи не может быть пустым.")
            return
        update["title"] = new_value_raw
    elif field == "deadline":
        if new_value_raw.lower() in ("none", "нет", "убрать", "-"):
            update["deadline"] = None
        else:
            try:
                parsed = isoparse(new_value_raw)
            except (ParserError, ValueError):
                await message.answer(
                    "Не понял дату. Пример: <code>2026-05-10</code> или "
                    "<code>2026-05-10 18:00</code>. Попробуй ещё раз."
                )
                return
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=_TZ)
            if parsed <= datetime.now(tz=_TZ):
                await message.answer("Дедлайн должен быть в будущем. Попробуй ещё раз.")
                return
            update["deadline"] = parsed
    elif field == "priority":
        v = new_value_raw.lower()
        if v not in _VALID_PRIORITIES:
            await message.answer(
                "Приоритет: только <code>low</code>, <code>med</code>, <code>high</code>."
            )
            return
        update["priority"] = v
    elif field == "category":
        v = new_value_raw.lower().replace(" ", "_")
        if not v:
            await message.answer("Категория не может быть пустой.")
            return
        update["category"] = v

    task = await storage.update_task_fields(db_path, task_id, update)
    await state.clear()
    if task is None:
        await message.answer(f"Задача #{task_id} исчезла во время правки.")
        return
    await message.answer(
        f"✅ Обновлено.\n\n{format_task_detail(task)}",
        reply_markup=_build_task_kb(task),
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

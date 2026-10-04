"""document CRUD commands and search handlers.

Commands:
  /docs                       list with pagination (10 per page)
  /docs_find <query>          merge FTS5 + Qdrant ranked results
  /docs_by_type <type>        filter by document type
  /docs_by_tag <tag>          filter by tag (exact match)
  /doc_edit <id>              menu → pick field → new value
  /doc_delete <id>            confirm → triple cleanup (Qdrant + file + SQLite)
  /doc_tag <id> <csv>         add tags inline
  /doc_untag <id> <tag>       remove tag inline

Callback patterns (disjoint from existing doctype/docmode/photoact/...):
  docspage:<page>
  docshow:<id>
  docedit_go:<id>
  docedit_field:<id>:<field>   field ∈ {type, tags, mode, pii}
  docedit_type:<id>:<newtype>
  docedit_type_ok:<id>:<newtype>
  docedit_pii:<id>:<yes|no>
  docedit_mode:<id>:<local_only|redacted_indexed>
  docdel_go:<id>
  docdel:<id>:<confirm|cancel>
"""
from __future__ import annotations

import logging
from pathlib import Path
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

from config import Settings
from services import jobs, qdrant_store, storage
from utils.messages import (
    doc_type_label,
    format_delete_confirm,
    format_doc_detail,
    format_docs_page,
    format_edit_menu,
    format_search_result,
    format_tag_result,
)


logger = logging.getLogger(__name__)

router = Router(name="documents_crud")


_PAGE_SIZE = 10
_DOC_TYPES: tuple[str, ...] = (
    "passport", "snils", "driving", "inn", "policy", "contract", "other",
)


class DocEditState(StatesGroup):
    enter_tags = State()


# ---------------------------------------------------------------------------
# /docs — paginated list
# ---------------------------------------------------------------------------


async def _fetch_page(
    db_path: str, user_id: int, page: int
) -> tuple[list[storage.Document], int, int]:
    """Return (docs_on_page, total_pages, total_count). Page is 0-indexed."""
    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT COUNT(*) FROM documents WHERE user_id = ?", (user_id,)
        )
        (total,) = await cur.fetchone()
    total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            f"SELECT {storage._DOC_COLUMNS} FROM documents "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (user_id, _PAGE_SIZE, page * _PAGE_SIZE),
        )
        rows = await cur.fetchall()
    docs = [
        d for d in (storage._row_to_document(r) for r in rows) if d is not None
    ]
    return docs, total_pages, total


def _docs_page_keyboard(
    docs, page: int, total_pages: int
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    # Per-document row of action buttons.
    for d in docs:
        rows.append(
            [
                InlineKeyboardButton(text=f"#{d.id}", callback_data=f"docshow:{d.id}"),
                InlineKeyboardButton(text="✏️", callback_data=f"docedit_go:{d.id}"),
                InlineKeyboardButton(text="🗑", callback_data=f"docdel_go:{d.id}"),
            ]
        )
    # Navigation row.
    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="← Prev", callback_data=f"docspage:{page - 1}"))
    nav.append(
        InlineKeyboardButton(
            text=f"{page + 1}/{total_pages}", callback_data="docspage:noop"
        )
    )
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton(text="Next →", callback_data=f"docspage:{page + 1}"))
    rows.append(nav)
    return InlineKeyboardMarkup(inline_keyboard=rows)


@router.message(Command("docs"))
async def on_docs(message: Message, db_path: str) -> None:
    docs, total_pages, total = await _fetch_page(db_path, message.chat.id, 0)
    await message.answer(
        format_docs_page(docs, 0, total_pages, total),
        reply_markup=_docs_page_keyboard(docs, 0, total_pages),
    )


@router.callback_query(F.data.startswith("docspage:"))
async def on_docs_page(callback: CallbackQuery, db_path: str) -> None:
    raw = callback.data.split(":", 1)[1]
    if raw == "noop":
        await callback.answer()
        return
    page = int(raw)
    docs, total_pages, total = await _fetch_page(db_path, callback.message.chat.id, page)
    await callback.message.edit_text(
        format_docs_page(docs, page, total_pages, total),
        reply_markup=_docs_page_keyboard(docs, page, total_pages),
    )
    await callback.answer()


@router.callback_query(F.data.startswith("docshow:"))
async def on_docshow(callback: CallbackQuery, db_path: str) -> None:
    doc_id = int(callback.data.split(":", 1)[1])
    doc = await storage.find_document_by_id(db_path, doc_id)
    if doc is None:
        await callback.answer("Документ не найден", show_alert=True)
        return
    joblist = await jobs.get_by_target(db_path, "document", doc_id)
    await callback.message.answer(format_doc_detail(doc, joblist))
    await callback.answer()


# ---------------------------------------------------------------------------
# /docs_by_type, /docs_by_tag
# ---------------------------------------------------------------------------


@router.message(Command("docs_by_type"))
async def on_docs_by_type(
    message: Message, command: CommandObject, db_path: str
) -> None:
    arg = (command.args or "").strip().lower()
    if arg not in _DOC_TYPES:
        await message.answer(
            "Укажи тип: " + " / ".join(_DOC_TYPES) + "\nПример: /docs_by_type passport"
        )
        return
    docs = await storage.find_document_by_type(db_path, message.chat.id, arg)
    if not docs:
        await message.answer(f"Документов типа {doc_type_label(arg)} нет.")
        return
    lines = [f"📎 Документы типа {doc_type_label(arg)} ({len(docs)}):"]
    for d in docs:
        lines.append(f"#{d.id} · {d.tags or '—'} · {d.created_at}")
    await message.answer("\n".join(lines))


@router.message(Command("docs_by_tag"))
async def on_docs_by_tag(
    message: Message, command: CommandObject, db_path: str
) -> None:
    arg = (command.args or "").strip()
    if not arg:
        await message.answer("Укажи тег: /docs_by_tag важное")
        return
    try:
        docs = await storage.find_by_tag(db_path, message.chat.id, arg)
    except ValueError as exc:
        await message.answer(f"Неверный тег: {exc}")
        return
    if not docs:
        await message.answer(f"Документов с тегом «{arg}» нет.")
        return
    lines = [f"📎 Документы с тегом «{arg}» ({len(docs)}):"]
    for d in docs:
        lines.append(f"#{d.id} · {doc_type_label(d.type)} · {d.created_at}")
    await message.answer("\n".join(lines))


# ---------------------------------------------------------------------------
# /docs_find — FTS + Qdrant merge ranking
# ---------------------------------------------------------------------------


async def _search_merged(
    db_path: str, qdrant_url: str, user_id: int, query: str, limit: int = 10
) -> list[dict]:
    """FTS5 + Qdrant merge. Returns hits sorted by combined score."""
    fts_rows = await storage.full_text_search(db_path, user_id, query, limit=20)
    try:
        qdrant_hits = await qdrant_store.search(
            qdrant_url, query, top_k=20, filter_={"user_id": user_id}
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qdrant search failed in merge: %s", exc)
        qdrant_hits = []

    # Collect per doc_id.
    merged: dict[int, dict] = {}
    for doc, relevance in fts_rows:
        merged[doc.id] = {
            "doc": doc,
            "fts_score": float(relevance),
            "qdrant_score": 0.0,
            "source": "fts",
        }
    for h in qdrant_hits:
        payload = h.get("payload") or {}
        doc_id = payload.get("doc_id")
        score = float(h.get("score", 0.0))
        if doc_id is None:
            continue
        if doc_id in merged:
            merged[doc_id]["qdrant_score"] = score
            merged[doc_id]["source"] = "both"
        else:
            doc = await storage.find_document_by_id(db_path, doc_id)
            if doc is None:
                continue
            merged[doc_id] = {
                "doc": doc,
                "fts_score": 0.0,
                "qdrant_score": score,
                "source": "qdrant",
            }

    # Combined score: simple sum of normalised contributions.
    hits = sorted(
        merged.values(),
        key=lambda m: m["fts_score"] + m["qdrant_score"],
        reverse=True,
    )[:limit]
    for m in hits:
        m["score"] = m["fts_score"] + m["qdrant_score"]
    return hits


@router.message(Command("docs_find"))
async def on_docs_find(
    message: Message, command: CommandObject, db_path: str, settings: Settings
) -> None:
    query = (command.args or "").strip()
    if not query:
        await message.answer("Укажи поисковый запрос: /docs_find паспорт")
        return
    hits = await _search_merged(
        db_path, settings.QDRANT_URL, message.chat.id, query, limit=10
    )
    await message.answer(format_search_result(hits, query))


# ---------------------------------------------------------------------------
# /doc_edit — FSM menu
# ---------------------------------------------------------------------------


def _edit_menu_keyboard(doc_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Тип", callback_data=f"docedit_field:{doc_id}:type")],
            [InlineKeyboardButton(text="Теги", callback_data=f"docedit_field:{doc_id}:tags")],
            [InlineKeyboardButton(text="Режим хранения", callback_data=f"docedit_field:{doc_id}:mode")],
            [InlineKeyboardButton(text="PII (есть/нет)", callback_data=f"docedit_field:{doc_id}:pii")],
            [InlineKeyboardButton(text="❌ Отмена", callback_data=f"docedit_field:{doc_id}:cancel")],
        ]
    )


def _type_pick_keyboard(doc_id: int) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for t in _DOC_TYPES:
        row.append(
            InlineKeyboardButton(
                text=doc_type_label(t), callback_data=f"docedit_type:{doc_id}:{t}"
            )
        )
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _type_confirm_keyboard(doc_id: int, new_type: str, cur_type: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"✅ Сменить {cur_type} → {new_type}",
                    callback_data=f"docedit_type_ok:{doc_id}:{new_type}",
                )
            ],
            [InlineKeyboardButton(text="❌ Отмена", callback_data=f"docedit_field:{doc_id}:cancel")],
        ]
    )


def _pii_keyboard(doc_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Да, есть ПД", callback_data=f"docedit_pii:{doc_id}:yes"),
                InlineKeyboardButton(text="Нет ПД", callback_data=f"docedit_pii:{doc_id}:no"),
            ]
        ]
    )


def _mode_keyboard(doc_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔒 local-only", callback_data=f"docedit_mode:{doc_id}:local_only"
                )
            ],
            [
                InlineKeyboardButton(
                    text="🎭 redacted_indexed",
                    callback_data=f"docedit_mode:{doc_id}:redacted_indexed",
                )
            ],
        ]
    )


@router.message(Command("doc_edit"))
async def on_doc_edit(
    message: Message, command: CommandObject, db_path: str
) -> None:
    try:
        doc_id = int((command.args or "").strip())
    except ValueError:
        await message.answer("Использование: /doc_edit <id>")
        return
    doc = await storage.find_document_by_id(db_path, doc_id)
    if doc is None or doc.user_id != message.chat.id:
        await message.answer(f"Документ #{doc_id} не найден.")
        return
    await message.answer(format_edit_menu(doc), reply_markup=_edit_menu_keyboard(doc_id))


@router.callback_query(F.data.startswith("docedit_go:"))
async def on_docedit_go(callback: CallbackQuery, db_path: str) -> None:
    doc_id = int(callback.data.split(":", 1)[1])
    doc = await storage.find_document_by_id(db_path, doc_id)
    if doc is None:
        await callback.answer("Документ не найден", show_alert=True)
        return
    await callback.message.answer(
        format_edit_menu(doc), reply_markup=_edit_menu_keyboard(doc_id)
    )
    await callback.answer()


@router.callback_query(F.data.startswith("docedit_field:"))
async def on_docedit_field(
    callback: CallbackQuery, state: FSMContext, db_path: str
) -> None:
    _, raw_id, field = callback.data.split(":", 2)
    doc_id = int(raw_id)
    doc = await storage.find_document_by_id(db_path, doc_id)
    if doc is None:
        await callback.answer("Документ не найден", show_alert=True)
        return
    if field == "cancel":
        await state.clear()
        await callback.message.edit_text("Отменено.")
        await callback.answer()
        return
    if field == "type":
        await callback.message.edit_text(
            f"Выбери новый тип (текущий: {doc_type_label(doc.type)}):",
            reply_markup=_type_pick_keyboard(doc_id),
        )
    elif field == "tags":
        await state.set_state(DocEditState.enter_tags)
        await state.update_data(edit_doc_id=doc_id)
        await callback.message.edit_text(
            f"Введи теги через запятую (текущие: {doc.tags or '—'}).\n"
            f"Пример: работа, важное"
        )
    elif field == "mode":
        await callback.message.edit_text(
            f"Режим хранения (текущий: {doc.storage_mode}):",
            reply_markup=_mode_keyboard(doc_id),
        )
    elif field == "pii":
        await callback.message.edit_text(
            f"Пометить наличие ПД (сейчас: {'есть' if doc.has_pii else 'нет'}):",
            reply_markup=_pii_keyboard(doc_id),
        )
    await callback.answer()


@router.callback_query(F.data.startswith("docedit_type:"))
async def on_docedit_type_pick(callback: CallbackQuery, db_path: str) -> None:
    _, raw_id, new_type = callback.data.split(":", 2)
    doc_id = int(raw_id)
    doc = await storage.find_document_by_id(db_path, doc_id)
    if doc is None:
        await callback.answer("Документ не найден", show_alert=True)
        return
    if new_type not in _DOC_TYPES:
        await callback.answer("Неверный тип", show_alert=True)
        return
    # Type change requires an extra confirm step (§4 DoD).
    await callback.message.edit_text(
        f"Точно сменить тип `{doc.type}` → `{new_type}`?\n"
        f"(Это не перезапускает PII-детектор.)",
        reply_markup=_type_confirm_keyboard(doc_id, new_type, doc.type),
    )
    await callback.answer()


@router.callback_query(F.data.startswith("docedit_type_ok:"))
async def on_docedit_type_ok(callback: CallbackQuery, db_path: str) -> None:
    _, raw_id, new_type = callback.data.split(":", 2)
    doc_id = int(raw_id)
    await storage.update_document(
        db_path, doc_id, {"type": new_type, "type_confirmed_by_user": True}
    )
    await callback.message.edit_text(
        f"✅ Тип документа #{doc_id} обновлён на {doc_type_label(new_type)}."
    )
    await callback.answer()


@router.callback_query(F.data.startswith("docedit_pii:"))
async def on_docedit_pii(callback: CallbackQuery, db_path: str) -> None:
    _, raw_id, yn = callback.data.split(":", 2)
    doc_id = int(raw_id)
    has_pii = yn == "yes"
    await storage.update_document(db_path, doc_id, {"has_pii": has_pii})
    await callback.message.edit_text(
        f"✅ Документ #{doc_id}: has_pii = {'есть' if has_pii else 'нет'}."
    )
    await callback.answer()


@router.callback_query(F.data.startswith("docedit_mode:"))
async def on_docedit_mode(
    callback: CallbackQuery, db_path: str, settings: Settings
) -> None:
    _, raw_id, new_mode = callback.data.split(":", 2)
    doc_id = int(raw_id)
    doc = await storage.find_document_by_id(db_path, doc_id)
    if doc is None:
        await callback.answer("Документ не найден", show_alert=True)
        return
    if new_mode not in ("local_only", "redacted_indexed"):
        await callback.answer("Неверный режим", show_alert=True)
        return
    if doc.storage_mode == new_mode:
        await callback.message.edit_text(f"Режим уже {new_mode}.")
        await callback.answer()
        return

    if new_mode == "redacted_indexed":
        # local_only → redacted_indexed: update + enqueue embed job.
        await storage.update_document(db_path, doc_id, {"storage_mode": new_mode})
        job_id = await jobs.enqueue(
            db_path,
            type_="embed",
            target_kind="document",
            target_id=doc_id,
            payload={"user_id": doc.user_id, "reindex": True},
        )
        eta = await jobs.estimate_eta(db_path, "embed_chunk", units=1)
        await callback.message.edit_text(
            f"✅ Документ #{doc_id}: режим → redacted_indexed. "
            f"Поставил в очередь на индексацию (job #{job_id}, ETA ~{eta} сек)."
        )
    else:
        # redacted_indexed → local_only: synchronous Qdrant delete.
        try:
            await qdrant_store.delete(settings.QDRANT_URL, doc_id=doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("qdrant delete on mode-change failed: %s", exc)
        await storage.update_document(db_path, doc_id, {"storage_mode": new_mode})
        await callback.message.edit_text(
            f"✅ Документ #{doc_id}: режим → local_only. "
            f"Убрал из поискового индекса, файл остался на диске."
        )
    await callback.answer()


@router.message(DocEditState.enter_tags, F.text)
async def on_docedit_enter_tags(
    message: Message, state: FSMContext, db_path: str
) -> None:
    data = await state.get_data()
    doc_id = int(data.get("edit_doc_id", 0))
    raw = (message.text or "").strip()
    tags_list = [t.strip() for t in raw.split(",") if t.strip()]
    try:
        # Full replace: remove existing, add new.
        doc = await storage.find_document_by_id(db_path, doc_id)
        if doc and doc.tags:
            existing = [t.strip() for t in doc.tags.split(",") if t.strip()]
            if existing:
                await storage.remove_tags(db_path, doc_id, existing)
        if tags_list:
            final = await storage.add_tags(db_path, doc_id, tags_list)
        else:
            final = []
    except ValueError as exc:
        await message.answer(f"❌ Невалидные теги: {exc}")
        await state.clear()
        return
    await state.clear()
    await message.answer(format_tag_result(doc_id, final))


# ---------------------------------------------------------------------------
# /doc_tag, /doc_untag (без FSM — command args)
# ---------------------------------------------------------------------------


@router.message(Command("doc_tag"))
async def on_doc_tag(
    message: Message, command: CommandObject, db_path: str
) -> None:
    parts = (command.args or "").split(None, 1)
    if len(parts) != 2:
        await message.answer("Использование: /doc_tag <id> tag1, tag2, ...")
        return
    try:
        doc_id = int(parts[0])
    except ValueError:
        await message.answer("Первый аргумент — число (id документа).")
        return
    tags = [t.strip() for t in parts[1].split(",") if t.strip()]
    try:
        final = await storage.add_tags(db_path, doc_id, tags)
    except ValueError as exc:
        await message.answer(f"❌ {exc}")
        return
    except LookupError:
        await message.answer(f"Документ #{doc_id} не найден.")
        return
    await message.answer(format_tag_result(doc_id, final))


@router.message(Command("doc_untag"))
async def on_doc_untag(
    message: Message, command: CommandObject, db_path: str
) -> None:
    parts = (command.args or "").split(None, 1)
    if len(parts) != 2:
        await message.answer("Использование: /doc_untag <id> <tag>")
        return
    try:
        doc_id = int(parts[0])
    except ValueError:
        await message.answer("Первый аргумент — число (id документа).")
        return
    try:
        final = await storage.remove_tags(db_path, doc_id, [parts[1].strip()])
    except ValueError as exc:
        await message.answer(f"❌ {exc}")
        return
    except LookupError:
        await message.answer(f"Документ #{doc_id} не найден.")
        return
    await message.answer(format_tag_result(doc_id, final, action="обновлены"))


# ---------------------------------------------------------------------------
# /doc_delete — confirm + triple cleanup (Qdrant → file → SQLite)
# ---------------------------------------------------------------------------


def _delete_confirm_keyboard(doc_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Да, удалить", callback_data=f"docdel:{doc_id}:confirm"),
                InlineKeyboardButton(text="❌ Отмена", callback_data=f"docdel:{doc_id}:cancel"),
            ]
        ]
    )


@router.message(Command("doc_delete"))
async def on_doc_delete(
    message: Message, command: CommandObject, db_path: str
) -> None:
    try:
        doc_id = int((command.args or "").strip())
    except ValueError:
        await message.answer("Использование: /doc_delete <id>")
        return
    doc = await storage.find_document_by_id(db_path, doc_id)
    if doc is None or doc.user_id != message.chat.id:
        await message.answer(f"Документ #{doc_id} не найден.")
        return
    await message.answer(
        format_delete_confirm(doc), reply_markup=_delete_confirm_keyboard(doc_id)
    )


@router.callback_query(F.data.startswith("docdel_go:"))
async def on_docdel_go(callback: CallbackQuery, db_path: str) -> None:
    doc_id = int(callback.data.split(":", 1)[1])
    doc = await storage.find_document_by_id(db_path, doc_id)
    if doc is None:
        await callback.answer("Документ не найден", show_alert=True)
        return
    await callback.message.answer(
        format_delete_confirm(doc), reply_markup=_delete_confirm_keyboard(doc_id)
    )
    await callback.answer()


@router.callback_query(F.data.startswith("docdel:"))
async def on_docdel(
    callback: CallbackQuery, db_path: str, settings: Settings
) -> None:
    _, raw_id, action = callback.data.split(":", 2)
    doc_id = int(raw_id)
    if action == "cancel":
        await callback.message.edit_text("Удаление отменено.")
        await callback.answer()
        return

    doc = await storage.find_document_by_id(db_path, doc_id)
    if doc is None:
        await callback.message.edit_text(f"Документ #{doc_id} не найден.")
        await callback.answer()
        return

    # Order: Qdrant → file → SQLite (§6 risk).
    if doc.storage_mode == "redacted_indexed":
        try:
            await qdrant_store.delete(settings.QDRANT_URL, doc_id=doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("qdrant delete failed for doc #%s: %s", doc_id, exc)

    try:
        if doc.original_path:
            Path(doc.original_path).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(
            "file unlink failed for doc #%s (%s): %s — zombi row possible",
            doc_id, doc.original_path, exc,
        )

    await storage.delete_document(db_path, doc_id)
    await callback.message.edit_text(f"✅ Документ #{doc_id} удалён.")
    await callback.answer()

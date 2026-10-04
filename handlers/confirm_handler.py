"""FSM + callback handlers for the document intent / confirmation flow.

PDF uploads go straight to the 2-step confirmation (PDF is assumed to be a
document):

    document_handler  →  ConfirmType.choosing
                        ↓
                     ConfirmPiiAndMode.deciding
                        ↓
                     encrypt_file + save_document + jobs.enqueue

Photos go through an extra intent-picker first (a photo is NOT necessarily
a document — patch, bug #3):

    photo_handler  →  PhotoIntent.choosing   (4 buttons)
                     • 💾 Сохранить как документ  → ConfirmType.choosing
                     • 🔍 Распознать текст        → one-shot OCR, no save
                     • 🌐 Найти в интернете       → placeholder
                     • ❌ Отмена                  → delete tmp

FSM data carried between all states:
  - tmp_path    : absolute str path to the plaintext tmp file
  - sha256      : SHA-256 hex of the file (for dedup)
  - mime        : MIME type ('image/jpeg' | 'application/pdf')
  - chosen_type : set by ConfirmType.choosing → used in ConfirmPiiAndMode
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from pathlib import Path
from typing import Optional

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from config import Settings
from services import jobs, storage
from services.encryption import encrypt_file
from services.gemini_client import GeminiClient
from services.intents import Document
from services.ocr_errors import OcrError
from services.ocr_router import extract_text
from utils.messages import (
    doc_type_label,
    format_accepted_reply,
    format_aborted_reply,
    format_duplicate_reply,
)


logger = logging.getLogger(__name__)


router = Router(name="documents_confirm")


# ---------------------------------------------------------------------------
# FSM state groups
# ---------------------------------------------------------------------------


class PhotoIntent(StatesGroup):
    """Initial pick-what-to-do-with-this-photo step (photos only, not PDFs)."""
    choosing = State()


class ConfirmType(StatesGroup):
    choosing = State()


class ConfirmPiiAndMode(StatesGroup):
    deciding = State()


# ---------------------------------------------------------------------------
# Inline keyboards
# ---------------------------------------------------------------------------


_DOC_TYPES: tuple[str, ...] = (
    "passport",
    "snils",
    "driving",
    "inn",
    "policy",
    "contract",
    "other",
)


def _type_keyboard() -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for t in _DOC_TYPES:
        row.append(
            InlineKeyboardButton(text=doc_type_label(t), callback_data=f"doctype:{t}")
        )
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton(text="🗑 Удалить", callback_data="doctype:abort")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _photo_intent_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="💾 Сохранить как документ",
                    callback_data="photoact:save",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🔍 Распознать текст",
                    callback_data="photoact:ocr",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🌐 Найти в интернете (🔜 скоро)",
                    callback_data="photoact:web",
                )
            ],
            [
                InlineKeyboardButton(
                    text="❌ Отмена",
                    callback_data="photoact:cancel",
                )
            ],
        ]
    )


def _mode_keyboard(allow_public: bool) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="🔒 Только локально (без индексации)",
                callback_data="docmode:local_only",
            )
        ],
        [
            InlineKeyboardButton(
                text="🎭 Обезличить и индексировать (локально)",
                callback_data="docmode:redacted_indexed",
            )
        ],
    ]
    if allow_public:
        rows.append(
            [
                InlineKeyboardButton(
                    text="☁️ Без ПД — через Gemini (качество выше)",
                    callback_data="docmode:public",
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton(text="🗑 Удалить", callback_data="docmode:abort")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


# ---------------------------------------------------------------------------
# Entry points used by photo_handler & document_handler
# ---------------------------------------------------------------------------


async def _stage_upload(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    db_path: str,
    data: bytes,
    mime: str,
    suffix: str,
) -> Optional[Path]:
    """Dedup + tmp-save + populate FSM data. Returns the tmp_path or None on dup.

    Common to both ``start_confirm_flow`` (PDF) and ``start_photo_intent_flow``
    (photo). On dedup hit sends the duplicate reply to the user itself; on
    fresh upload leaves the FSM state UNSET (caller sets PhotoIntent/ConfirmType).
    """
    sha = hashlib.sha256(data).hexdigest()
    user_id = message.chat.id

    existing = await storage.find_document_by_sha256(db_path, user_id, sha)
    if existing is not None:
        await message.answer(format_duplicate_reply(existing.id))
        return None

    tmp_dir = Path(settings.TMP_DIR)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4()}{suffix}"
    tmp_path.write_bytes(data)

    await state.update_data(
        tmp_path=str(tmp_path),
        sha256=sha,
        mime=mime,
    )
    return tmp_path


async def start_confirm_flow(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    db_path: str,
    data: bytes,
    mime: str,
    suffix: str,
) -> None:
    """PDF entry: stage tmp, enter ConfirmType.choosing with inline keyboard."""
    tmp_path = await _stage_upload(
        message=message, state=state, settings=settings, db_path=db_path,
        data=data, mime=mime, suffix=suffix,
    )
    if tmp_path is None:
        return
    await state.set_state(ConfirmType.choosing)
    await message.answer(
        "Какой это документ? Выберите тип из списка:",
        reply_markup=_type_keyboard(),
    )


async def start_photo_intent_flow(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    db_path: str,
    data: bytes,
    mime: str,
    suffix: str,
) -> None:
    """Photo entry: show the 4-button intent picker (save/ocr/web/cancel)."""
    tmp_path = await _stage_upload(
        message=message, state=state, settings=settings, db_path=db_path,
        data=data, mime=mime, suffix=suffix,
    )
    if tmp_path is None:
        return
    await state.set_state(PhotoIntent.choosing)
    await message.answer(
        "Что сделать с этим фото?",
        reply_markup=_photo_intent_keyboard(),
    )


# ---------------------------------------------------------------------------
# Inline OCR helper (used by PhotoIntent button AND by caption-shortcut in
# photo_handler / document_handler — bug #3 + OCR on demand §1.3)
# ---------------------------------------------------------------------------


_TG_MSG_MAX = 3900  # 4096 is the Telegram hard limit; keep a margin for UI text


async def _ocr_bytes_to_text(
    data: bytes, mime: str, settings: Settings
) -> tuple[str, str]:
    """Run OCR on raw bytes. Returns (text, engine_label).

    Picks PUBLIC (Gemini Vision) when available and not PRIVACY_STRICT,
    otherwise falls back to PRIVATE (EasyOCR / PyMuPDF via ocr_router).
    """
    use_public = bool(settings.GEMINI_API_KEY) and not settings.PRIVACY_STRICT
    ocr_mode = "public" if use_public else "private"
    gemini_client = None
    if ocr_mode == "public":
        gemini_client = GeminiClient(
            api_key=settings.GEMINI_API_KEY,
            privacy_strict=settings.PRIVACY_STRICT,
        )
    text = await extract_text(data, mime, ocr_mode, gemini_client=gemini_client)
    return text, "Gemini Vision" if use_public else "локальный OCR"


def format_ocr_reply(text: str) -> str:
    """Trim + format OCR result for a chat reply."""
    if not text.strip():
        return "OCR не нашёл текста на файле."
    body = text[:_TG_MSG_MAX]
    suffix = "" if len(text) <= _TG_MSG_MAX else f"\n\n… (обрезано, всего {len(text)} симв.)"
    return f"🔍 Распознанный текст:\n\n{body}{suffix}"


async def run_inline_ocr_for_upload(
    message: Message, data: bytes, mime: str, settings: Settings
) -> None:
    """Caption-shortcut path: run OCR and reply in chat without saving the file.

    Used by ``photo_handler``/``document_handler`` when the caption contains
    a trigger word («распознай», «ocr», «прочти», …). No FSM state is set,
    no tmp file is written, nothing is stored — just answer and return.
    """
    try:
        notice = await message.answer("🔍 Распознаю текст…")
    except Exception:  # noqa: BLE001
        notice = None

    try:
        text, engine = await _ocr_bytes_to_text(data, mime, settings)
    except OcrError as exc:
        logger.warning("caption-shortcut OCR failed: %s", exc)
        await message.answer(f"❌ OCR упал: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("caption-shortcut OCR crashed")
        await message.answer(f"❌ OCR упал неожиданно: {exc}")
        return

    reply = format_ocr_reply(text)
    # Try to edit the notice message if we were able to post one; otherwise
    # send as a fresh reply.
    if notice is not None:
        try:
            await notice.edit_text(reply, parse_mode=None)
            return
        except Exception:  # noqa: BLE001
            pass
    await message.answer(reply, parse_mode=None)


# ---------------------------------------------------------------------------
# Step 0: photo intent (photos only — PDFs skip this and go straight to ConfirmType)
# ---------------------------------------------------------------------------


@router.callback_query(PhotoIntent.choosing, F.data == "photoact:save")
async def on_photo_save_as_doc(callback: CallbackQuery, state: FSMContext) -> None:
    """User picked «Сохранить как документ» — jump into the standard ConfirmType flow."""
    await state.set_state(ConfirmType.choosing)
    await callback.message.edit_text(
        "Какой это документ? Выберите тип из списка:",
        reply_markup=_type_keyboard(),
    )
    await callback.answer()


@router.callback_query(PhotoIntent.choosing, F.data == "photoact:ocr")
async def on_photo_ocr(
    callback: CallbackQuery,
    state: FSMContext,
    settings: Settings,
) -> None:
    """User picked «Распознать текст» — one-shot OCR, reply with text, no save."""
    data = await state.get_data()
    tmp_path = Path(data.get("tmp_path", ""))
    mime = str(data.get("mime", "image/jpeg"))

    if not tmp_path.exists():
        await callback.message.edit_text(
            "⚠️ Временный файл пропал, отправь фото ещё раз."
        )
        await state.clear()
        await callback.answer()
        return

    await callback.message.edit_text("🔍 Распознаю текст…")
    await callback.answer()

    try:
        text, _engine = await _ocr_bytes_to_text(
            tmp_path.read_bytes(), mime, settings
        )
    except OcrError as exc:
        logger.warning("photo OCR failed: %s", exc)
        await callback.message.edit_text(f"❌ OCR упал: {exc}")
        await _cleanup_tmp(tmp_path)
        await state.clear()
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("photo OCR crashed")
        await callback.message.edit_text(f"❌ OCR упал неожиданно: {exc}")
        await _cleanup_tmp(tmp_path)
        await state.clear()
        return

    await _cleanup_tmp(tmp_path)
    await state.clear()
    await callback.message.edit_text(format_ocr_reply(text), parse_mode=None)


@router.callback_query(PhotoIntent.choosing, F.data == "photoact:web")
async def on_photo_web_search(
    callback: CallbackQuery, state: FSMContext
) -> None:
    """Placeholder (Gemini Vision + grounded search)."""
    data = await state.get_data()
    tmp_path = Path(data.get("tmp_path", ""))
    await _cleanup_tmp(tmp_path)
    await state.clear()
    await callback.message.edit_text(
        "🌐 Поиск по картинке через Gemini Vision — будет в следующей версии. "
        "Сейчас могу только распознать текст или сохранить как документ."
    )
    await callback.answer()


@router.callback_query(PhotoIntent.choosing, F.data == "photoact:cancel")
async def on_photo_cancel(
    callback: CallbackQuery, state: FSMContext
) -> None:
    data = await state.get_data()
    tmp_path = Path(data.get("tmp_path", ""))
    await _cleanup_tmp(tmp_path)
    await state.clear()
    await callback.message.edit_text(
        "Ок, не сохраняю. Временный файл удалён."
    )
    await callback.answer()


# ---------------------------------------------------------------------------
# Step 1: document type chosen
# ---------------------------------------------------------------------------


@router.callback_query(ConfirmType.choosing, F.data.startswith("doctype:"))
async def on_type_chosen(
    callback: CallbackQuery,
    state: FSMContext,
    settings: Settings,
) -> None:
    choice = callback.data.split(":", 1)[1]
    data = await state.get_data()
    tmp_path = Path(data.get("tmp_path", ""))

    if choice == "abort":
        await _cleanup_tmp(tmp_path)
        await state.clear()
        await callback.message.edit_text(format_aborted_reply())
        await callback.answer()
        return

    if choice not in _DOC_TYPES:
        await callback.answer("Неизвестный тип", show_alert=True)
        return

    allow_public = bool(settings.GEMINI_API_KEY) and not settings.PRIVACY_STRICT

    await state.update_data(chosen_type=choice)
    await state.set_state(ConfirmPiiAndMode.deciding)
    await callback.message.edit_text(
        f"Тип: <b>{doc_type_label(choice)}</b>.\n\n"
        f"Как хранить? PII-детектор проверит содержимое после OCR, "
        f"но решение о режиме нужно принять сейчас:",
        reply_markup=_mode_keyboard(allow_public=allow_public),
    )
    await callback.answer()


# ---------------------------------------------------------------------------
# Step 2: storage mode chosen → encrypt + save + enqueue
# ---------------------------------------------------------------------------


@router.callback_query(ConfirmPiiAndMode.deciding, F.data.startswith("docmode:"))
async def on_mode_chosen(
    callback: CallbackQuery,
    state: FSMContext,
    settings: Settings,
    db_path: str,
) -> None:
    choice = callback.data.split(":", 1)[1]
    data = await state.get_data()
    tmp_path = Path(data.get("tmp_path", ""))
    sha = str(data.get("sha256", ""))
    mime = str(data.get("mime", "application/pdf"))
    doc_type = str(data.get("chosen_type", "other"))

    if choice == "abort":
        await _cleanup_tmp(tmp_path)
        await state.clear()
        await callback.message.edit_text(format_aborted_reply())
        await callback.answer()
        return

    if choice not in ("local_only", "redacted_indexed", "public"):
        await callback.answer("Неизвестный режим", show_alert=True)
        return

    # Reject 'public' if not actually allowed (defensive: keyboard should have hidden it).
    if choice == "public" and (not settings.GEMINI_API_KEY or settings.PRIVACY_STRICT):
        await callback.answer(
            "PUBLIC-режим недоступен: проверьте GEMINI_API_KEY / PRIVACY_STRICT",
            show_alert=True,
        )
        return

    storage_mode = "local_only" if choice == "local_only" else "redacted_indexed"
    ocr_mode = "public" if choice == "public" else "private"

    user_id = callback.message.chat.id

    # 1. Insert Document with placeholder path to get doc_id.
    doc = Document(
        user_id=user_id,
        type=doc_type,
        type_confirmed_by_user=True,
        has_pii=False,            # not yet known — worker will update after OCR if needed
        storage_mode=storage_mode,
        original_path="",          # filled below
        original_sha256=sha,
    )
    doc_id = await storage.save_document(db_path, doc)

    # 2. Encrypt tmp → data/documents/<user_id>/<doc_id>.bin  (unlinks tmp)
    enc_path = Path(settings.DOCS_DIR) / str(user_id) / f"{doc_id}.bin"
    try:
        encrypt_file(tmp_path, enc_path, settings.DOCS_ENCRYPTION_KEY)
    except FileNotFoundError:
        # tmp already missing (race / reboot) — fail gracefully.
        await callback.message.edit_text(
            "⚠️ Временный файл пропал, попробуйте отправить документ ещё раз."
        )
        await state.clear()
        await callback.answer()
        return
    await storage.update_document_path(db_path, doc_id, str(enc_path))

    # 3. Enqueue OCR job.
    job_id = await jobs.enqueue(
        db_path,
        type_="ocr",
        target_kind="document",
        target_id=doc_id,
        payload={
            "mode": ocr_mode,
            "storage_mode": storage_mode,
            "mime": mime,
            "user_id": user_id,
            "doc_type": doc_type,
        },
    )
    eta = await jobs.estimate_eta(db_path, f"ocr_{ocr_mode}_page", units=1)
    logger.info(
        "doc %s enqueued as job %s (type=%s, mode=%s, storage=%s, eta=%ds)",
        doc_id, job_id, doc_type, ocr_mode, storage_mode, eta,
    )

    await state.clear()
    await callback.message.edit_text(
        format_accepted_reply(doc_id=doc_id, eta_seconds=eta, ocr_mode=ocr_mode)
    )
    await callback.answer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _cleanup_tmp(tmp_path: Optional[Path]) -> None:
    try:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
    except OSError as exc:
        logger.warning("cleanup_tmp failed for %s: %s", tmp_path, exc)

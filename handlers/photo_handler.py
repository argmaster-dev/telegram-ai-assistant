"""Handler for incoming photos.

Photos can be anything — a document scan, a receipt, a book cover, a meme.
Unlike PDFs (which are unambiguously documents), photos hit an extra
«What do you want me to do with it?» step before any saving happens.

Caption shortcut (§1.3): if the user attaches a caption containing
a trigger word («распознай», «ocr», «прочти», «прочитай», «извлеки текст»),
we skip the PhotoIntent menu and run inline OCR directly — no file is saved.
"""
from __future__ import annotations

import logging
from io import BytesIO

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from config import Settings
from handlers.confirm_handler import (
    run_inline_ocr_for_upload,
    start_photo_intent_flow,
)


logger = logging.getLogger(__name__)

router = Router(name="documents_photo")


_OCR_TRIGGERS: tuple[str, ...] = (
    "распознай",
    "распознать",
    "ocr",
    "прочти",
    "прочитай",
    "извлеки текст",
)


def _has_ocr_trigger(caption: str | None) -> bool:
    if not caption:
        return False
    lowered = caption.strip().lower()
    return any(trig in lowered for trig in _OCR_TRIGGERS)


@router.message(F.photo)
async def on_photo(
    message: Message,
    state: FSMContext,
    settings: Settings,
    db_path: str,
) -> None:
    # Telegram sends multiple sizes — the last one is the largest/original.
    photo = message.photo[-1]
    buf = BytesIO()
    await message.bot.download(photo, destination=buf)
    data = buf.getvalue()
    logger.info(
        "photo received: chat=%s size=%d bytes (file_id=%s, caption=%r)",
        message.chat.id, len(data), photo.file_id, message.caption,
    )

    if _has_ocr_trigger(message.caption):
        await run_inline_ocr_for_upload(
            message=message, data=data, mime="image/jpeg", settings=settings,
        )
        return

    await start_photo_intent_flow(
        message=message,
        state=state,
        settings=settings,
        db_path=db_path,
        data=data,
        mime="image/jpeg",
        suffix=".jpg",
    )

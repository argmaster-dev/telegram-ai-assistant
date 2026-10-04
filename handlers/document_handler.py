"""Handler for incoming PDF documents (Telegram ``document`` with application/pdf).

Caption shortcut (§1.3): if the user attaches the PDF with a caption
containing a trigger word («распознай», «ocr», «прочти», ...), we skip the
ConfirmType menu and run inline OCR directly — no file is saved.
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
    start_confirm_flow,
)
from handlers.photo_handler import _has_ocr_trigger


logger = logging.getLogger(__name__)

router = Router(name="documents_pdf")


@router.message(F.document & (F.document.mime_type == "application/pdf"))
async def on_pdf(
    message: Message,
    state: FSMContext,
    settings: Settings,
    db_path: str,
) -> None:
    doc = message.document
    buf = BytesIO()
    await message.bot.download(doc, destination=buf)
    data = buf.getvalue()
    logger.info(
        "pdf received: chat=%s size=%d bytes (file_id=%s, name=%s, caption=%r)",
        message.chat.id, len(data), doc.file_id, doc.file_name, message.caption,
    )

    if _has_ocr_trigger(message.caption):
        await run_inline_ocr_for_upload(
            message=message,
            data=data,
            mime="application/pdf",
            settings=settings,
        )
        return

    await start_confirm_flow(
        message=message,
        state=state,
        settings=settings,
        db_path=db_path,
        data=data,
        mime="application/pdf",
        suffix=".pdf",
    )

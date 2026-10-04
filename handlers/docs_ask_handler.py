"""``/doc_status`` command + free-text «как там документ» handler.

Both entry points route to the same ``_report_status`` which consults
``services.jobs.get_by_target`` and formats a short human reply.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from services import jobs, storage
from utils.messages import format_doc_status


logger = logging.getLogger(__name__)

router = Router(name="documents_ask")


_DOC_STATUS_TEXT_RE = re.compile(
    r"(?iu)(как там (мой )?документ|статус документ|что с документ)"
)


def _parse_int(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    for token in raw.split():
        if token.isdigit():
            return int(token)
    return None


async def _report_status(
    message: Message,
    db_path: str,
    doc_id: Optional[int],
) -> None:
    user_id = message.chat.id
    if doc_id is None:
        docs = await storage.find_last_documents(db_path, user_id, limit=1)
        if not docs:
            await message.answer("У тебя ещё нет документов. Отправь фото или PDF — разберу.")
            return
        doc_id = docs[0].id

    joblist = await jobs.get_by_target(db_path, "document", doc_id)
    await message.answer(format_doc_status(doc_id, joblist))


@router.message(Command("doc_status"))
async def on_doc_status(
    message: Message,
    command: CommandObject,
    db_path: str,
) -> None:
    doc_id = _parse_int(command.args)
    await _report_status(message, db_path, doc_id)


@router.message(F.text.regexp(_DOC_STATUS_TEXT_RE))
async def on_free_text_status(message: Message, db_path: str) -> None:
    # Try to extract trailing number; otherwise fall back to last document.
    doc_id = _parse_int(message.text)
    await _report_status(message, db_path, doc_id)

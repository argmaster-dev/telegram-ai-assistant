"""Tests for the 2-step document confirmation FSM.

We don't use an aiogram test harness — instead we exercise the handler
functions directly with stub ``CallbackQuery`` / ``Message`` / ``FSMContext``
objects. This is enough to verify the intended state transitions, the
encrypted file layout and the enqueued job payload.
"""
from __future__ import annotations

import base64
import hashlib
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio

from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage

from config import Settings
from handlers import confirm_handler
from services import jobs as jobs_module
from services import storage
from services.storage import init_db


def _make_key() -> str:
    return base64.b64encode(os.urandom(32)).decode()


@pytest_asyncio.fixture()
async def env(tmp_path: Path):
    db_path = str(tmp_path / "t.db")
    await init_db(db_path)
    docs_dir = tmp_path / "documents"
    stage_dir = tmp_path / "tmp"
    stage_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        TELEGRAM_BOT_TOKEN="fake",
        ALLOWED_CHAT_ID=999,
        DOCS_ENCRYPTION_KEY=_make_key(),
        GEMINI_API_KEY="fake-key",  # allow_public path reachable
        PRIVACY_STRICT=False,
        DB_PATH=db_path,
        DOCS_DIR=str(docs_dir),
        TMP_DIR=str(stage_dir),
    )

    # FSMContext bound to MemoryStorage (works without a real Bot).
    storage_ = MemoryStorage()
    key = StorageKey(bot_id=0, chat_id=999, user_id=999)
    state = FSMContext(storage=storage_, key=key)
    yield settings, state, db_path


class _MsgRecorder:
    """Stub that records ``.answer`` / ``.edit_text`` calls."""
    def __init__(self, chat_id: int = 999) -> None:
        self.chat = SimpleNamespace(id=chat_id)
        self.answers: list[str] = []

    async def answer(self, text, **_):
        self.answers.append(text)

    async def edit_text(self, text, **_):
        self.answers.append(text)


class _CbStub:
    def __init__(self, data: str, message: _MsgRecorder) -> None:
        self.data = data
        self.message = message
        self.answered = False

    async def answer(self, text="", show_alert=False):
        self.answered = True


@pytest.mark.asyncio
async def test_full_flow_local_only_enqueues_only_ocr(env):
    """Pick type=passport → mode=local_only: encrypted file exists, OCR job queued, no embed."""
    settings, state, db_path = env

    # Seed tmp file to simulate start_confirm_flow having downloaded + hashed.
    data = b"FAKE-PDF-CONTENT"
    await confirm_handler.start_confirm_flow(
        message=_MsgRecorder(chat_id=999),
        state=state,
        settings=settings,
        db_path=db_path,
        data=data,
        mime="application/pdf",
        suffix=".pdf",
    )

    current = await state.get_state()
    assert current == confirm_handler.ConfirmType.choosing.state

    # Step 1: pick 'passport'
    msg1 = _MsgRecorder(chat_id=999)
    cb1 = _CbStub("doctype:passport", msg1)
    await confirm_handler.on_type_chosen(cb1, state, settings)
    assert await state.get_state() == confirm_handler.ConfirmPiiAndMode.deciding.state

    # Step 2: pick 'local_only'
    msg2 = _MsgRecorder(chat_id=999)
    cb2 = _CbStub("docmode:local_only", msg2)
    await confirm_handler.on_mode_chosen(cb2, state, settings, db_path)
    assert await state.get_state() is None  # cleared

    # Exactly one document row with type=passport & storage_mode=local_only
    sha = hashlib.sha256(data).hexdigest()
    doc = await storage.find_document_by_sha256(db_path, 999, sha)
    assert doc is not None
    assert doc.type == "passport"
    assert doc.storage_mode == "local_only"
    assert doc.type_confirmed_by_user is True
    assert Path(doc.original_path).exists()
    # Tmp file must have been unlinked by encrypt_file.
    assert not any(Path(settings.TMP_DIR).iterdir())

    # One job enqueued, type=ocr, payload has storage_mode=local_only
    joblist = await jobs_module.get_by_target(db_path, "document", doc.id)
    assert len(joblist) == 1
    assert joblist[0].type == "ocr"
    assert joblist[0].payload.get("storage_mode") == "local_only"
    assert joblist[0].payload.get("mode") == "private"


@pytest.mark.asyncio
async def test_redacted_indexed_stays_as_ocr_only_until_worker_runs(env):
    """Handler itself enqueues only OCR; the follow-up embed is enqueued by the worker."""
    settings, state, db_path = env
    await confirm_handler.start_confirm_flow(
        message=_MsgRecorder(chat_id=999),
        state=state,
        settings=settings,
        db_path=db_path,
        data=b"PDF-REDACTED-INDEXED",
        mime="application/pdf",
        suffix=".pdf",
    )

    cb_type = _CbStub("doctype:contract", _MsgRecorder(chat_id=999))
    await confirm_handler.on_type_chosen(cb_type, state, settings)

    cb_mode = _CbStub("docmode:redacted_indexed", _MsgRecorder(chat_id=999))
    await confirm_handler.on_mode_chosen(cb_mode, state, settings, db_path)

    sha = hashlib.sha256(b"PDF-REDACTED-INDEXED").hexdigest()
    doc = await storage.find_document_by_sha256(db_path, 999, sha)
    assert doc is not None
    joblist = await jobs_module.get_by_target(db_path, "document", doc.id)
    assert [j.type for j in joblist] == ["ocr"]
    assert joblist[0].payload.get("storage_mode") == "redacted_indexed"


@pytest.mark.asyncio
async def test_abort_clears_state_and_removes_tmp(env):
    settings, state, db_path = env
    await confirm_handler.start_confirm_flow(
        message=_MsgRecorder(chat_id=999),
        state=state,
        settings=settings,
        db_path=db_path,
        data=b"SOMETHING",
        mime="application/pdf",
        suffix=".pdf",
    )
    assert any(Path(settings.TMP_DIR).iterdir())

    cb = _CbStub("doctype:abort", _MsgRecorder(chat_id=999))
    await confirm_handler.on_type_chosen(cb, state, settings)
    assert await state.get_state() is None
    assert not any(Path(settings.TMP_DIR).iterdir())


@pytest.mark.asyncio
async def test_photo_intent_menu_is_shown_first(env):
    """start_photo_intent_flow enters PhotoIntent.choosing, not ConfirmType."""
    settings, state, db_path = env
    msg = _MsgRecorder(chat_id=999)
    await confirm_handler.start_photo_intent_flow(
        message=msg,
        state=state,
        settings=settings,
        db_path=db_path,
        data=b"SOMEPHOTO",
        mime="image/jpeg",
        suffix=".jpg",
    )
    assert (await state.get_state()) == confirm_handler.PhotoIntent.choosing.state
    # The prompt mentions the 4-option menu.
    assert any("фото" in a.lower() for a in msg.answers), msg.answers


@pytest.mark.asyncio
async def test_photo_intent_save_transitions_to_confirm_type(env):
    """«Сохранить как документ» hands over to the existing ConfirmType flow."""
    settings, state, db_path = env
    await confirm_handler.start_photo_intent_flow(
        message=_MsgRecorder(chat_id=999),
        state=state, settings=settings, db_path=db_path,
        data=b"PHOTOSAVE", mime="image/jpeg", suffix=".jpg",
    )
    cb = _CbStub("photoact:save", _MsgRecorder(chat_id=999))
    await confirm_handler.on_photo_save_as_doc(cb, state)
    assert (await state.get_state()) == confirm_handler.ConfirmType.choosing.state


@pytest.mark.asyncio
async def test_photo_intent_cancel_cleans_up(env):
    """«Отмена» removes tmp file and clears FSM, no DB row."""
    settings, state, db_path = env
    await confirm_handler.start_photo_intent_flow(
        message=_MsgRecorder(chat_id=999),
        state=state, settings=settings, db_path=db_path,
        data=b"PHOTOCANCEL", mime="image/jpeg", suffix=".jpg",
    )
    assert any(Path(settings.TMP_DIR).iterdir())
    cb = _CbStub("photoact:cancel", _MsgRecorder(chat_id=999))
    await confirm_handler.on_photo_cancel(cb, state)
    assert (await state.get_state()) is None
    assert not any(Path(settings.TMP_DIR).iterdir())
    sha = hashlib.sha256(b"PHOTOCANCEL").hexdigest()
    assert await storage.find_document_by_sha256(db_path, 999, sha) is None


@pytest.mark.asyncio
async def test_photo_intent_ocr_returns_text_no_save(env):
    """«Распознать текст» runs OCR and replies with the text, nothing saved."""
    from unittest.mock import patch, AsyncMock
    settings, state, db_path = env
    await confirm_handler.start_photo_intent_flow(
        message=_MsgRecorder(chat_id=999),
        state=state, settings=settings, db_path=db_path,
        data=b"PHOTOOCR", mime="image/jpeg", suffix=".jpg",
    )
    mock_ocr = AsyncMock(return_value="Распознанный текст: привет мир")
    with patch("handlers.confirm_handler.extract_text", mock_ocr):
        cb = _CbStub("photoact:ocr", _MsgRecorder(chat_id=999))
        await confirm_handler.on_photo_ocr(cb, state, settings)
    # Тmp cleaned up, FSM cleared.
    assert (await state.get_state()) is None
    assert not any(Path(settings.TMP_DIR).iterdir())
    # No document row.
    sha = hashlib.sha256(b"PHOTOOCR").hexdigest()
    assert await storage.find_document_by_sha256(db_path, 999, sha) is None
    # OCR text is the final edit.
    final_answer = cb.message.answers[-1]
    assert "привет мир" in final_answer


@pytest.mark.asyncio
async def test_photo_intent_web_is_placeholder(env):
    """«Найти в интернете» cleans up and returns a placeholder message."""
    settings, state, db_path = env
    await confirm_handler.start_photo_intent_flow(
        message=_MsgRecorder(chat_id=999),
        state=state, settings=settings, db_path=db_path,
        data=b"PHOTOWEB", mime="image/jpeg", suffix=".jpg",
    )
    cb = _CbStub("photoact:web", _MsgRecorder(chat_id=999))
    await confirm_handler.on_photo_web_search(cb, state)
    assert (await state.get_state()) is None
    assert not any(Path(settings.TMP_DIR).iterdir())
    assert any("следующей версии" in a for a in cb.message.answers), cb.message.answers


@pytest.mark.asyncio
async def test_public_mode_blocked_when_privacy_strict(env):
    settings, state, db_path = env
    # Mutate settings: PRIVACY_STRICT=True
    strict = Settings(
        TELEGRAM_BOT_TOKEN=settings.TELEGRAM_BOT_TOKEN,
        ALLOWED_CHAT_ID=settings.ALLOWED_CHAT_ID,
        DOCS_ENCRYPTION_KEY=settings.DOCS_ENCRYPTION_KEY,
        GEMINI_API_KEY=settings.GEMINI_API_KEY,
        PRIVACY_STRICT=True,
        DB_PATH=settings.DB_PATH,
        DOCS_DIR=settings.DOCS_DIR,
        TMP_DIR=settings.TMP_DIR,
    )

    await confirm_handler.start_confirm_flow(
        message=_MsgRecorder(chat_id=999),
        state=state,
        settings=strict,
        db_path=db_path,
        data=b"PRIV-STRICT-TEST",
        mime="application/pdf",
        suffix=".pdf",
    )
    cb_type = _CbStub("doctype:contract", _MsgRecorder(chat_id=999))
    await confirm_handler.on_type_chosen(cb_type, state, strict)

    # Trying to pick PUBLIC mode must be rejected even if the callback data somehow arrives.
    cb_mode = _CbStub("docmode:public", _MsgRecorder(chat_id=999))
    await confirm_handler.on_mode_chosen(cb_mode, state, strict, db_path)

    # Nothing should have been saved / enqueued.
    sha = hashlib.sha256(b"PRIV-STRICT-TEST").hexdigest()
    doc = await storage.find_document_by_sha256(db_path, 999, sha)
    assert doc is None

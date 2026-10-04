"""Tests for services.worker.worker_loop — ocr_router / bot / encryption mocked."""
from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from config import Settings
from services import jobs as jobs_module
from services import storage
from services.intents import Document
from services.storage import init_db


def _make_key() -> str:
    return base64.b64encode(os.urandom(32)).decode()


@pytest_asyncio.fixture()
async def settings_and_db():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        db_path = str(tmp_dir / "w.db")
        docs_dir = tmp_dir / "documents"
        stage_dir = tmp_dir / "tmp"
        stage_dir.mkdir(parents=True, exist_ok=True)
        await init_db(db_path)
        settings = Settings(
            TELEGRAM_BOT_TOKEN="fake",
            ALLOWED_CHAT_ID=1,
            DOCS_ENCRYPTION_KEY=_make_key(),
            DB_PATH=db_path,
            DOCS_DIR=str(docs_dir),
            TMP_DIR=str(stage_dir),
        )
        yield settings


class _MockBot:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_message(self, chat_id, text, **_):
        self.sent.append(text)


@pytest.mark.asyncio
async def test_worker_processes_ocr_job_local_only(settings_and_db):
    """Pending OCR job with storage_mode=local_only → finalized done, push sent."""
    settings = settings_and_db
    bot = _MockBot()

    # Create encrypted file.
    from services.encryption import encrypt_file

    plain = settings.TMP_DIR + "/original.pdf"
    Path(plain).write_bytes(b"%PDF-1.4 fake content")
    doc_path = Path(settings.DOCS_DIR) / "1" / "7.bin"
    encrypt_file(Path(plain), doc_path, settings.DOCS_ENCRYPTION_KEY)

    # Save document row.
    doc_id = await storage.save_document(
        settings.DB_PATH,
        Document(
            user_id=1,
            type="other",
            storage_mode="local_only",
            original_path=str(doc_path),
            original_sha256="sha-test-1",
        ),
    )
    job_id = await jobs_module.enqueue(
        settings.DB_PATH,
        "ocr",
        "document",
        doc_id,
        payload={"mode": "private", "storage_mode": "local_only", "mime": "application/pdf"},
    )

    # Mock extract_text so we don't actually run EasyOCR or PyMuPDF.
    from services import worker

    async def fake_extract(data, mime, mode, **_):
        return "Fake OCR output for test"

    with patch.object(worker, "extract_text", side_effect=fake_extract):
        task = asyncio.create_task(worker.worker_loop(bot, settings))
        # Wait up to 5 sec for job to finish.
        for _ in range(50):
            j = await jobs_module.get_by_id(settings.DB_PATH, job_id)
            if j is not None and j.status in ("done", "failed"):
                break
            await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    j = await jobs_module.get_by_id(settings.DB_PATH, job_id)
    assert j is not None
    assert j.status == "done"

    doc = await storage.find_document_by_id(settings.DB_PATH, doc_id)
    assert doc.ocr_text == "Fake OCR output for test"
    # local_only → redacted_text stays NULL.
    assert doc.redacted_text is None

    # No follow-up embed job (local_only).
    joblist = await jobs_module.get_by_target(settings.DB_PATH, "document", doc_id)
    assert {j.type for j in joblist} == {"ocr"}

    assert any("✅" in m for m in bot.sent), f"expected a done push in {bot.sent}"


@pytest.mark.asyncio
async def test_worker_failure_sends_error_push(settings_and_db):
    """When extract_text raises, job is finalized failed and an error push is sent."""
    settings = settings_and_db
    bot = _MockBot()

    from services.encryption import encrypt_file

    plain = settings.TMP_DIR + "/orig2.pdf"
    Path(plain).write_bytes(b"%PDF-1.4 content")
    doc_path = Path(settings.DOCS_DIR) / "1" / "8.bin"
    encrypt_file(Path(plain), doc_path, settings.DOCS_ENCRYPTION_KEY)

    doc_id = await storage.save_document(
        settings.DB_PATH,
        Document(
            user_id=1,
            type="other",
            storage_mode="local_only",
            original_path=str(doc_path),
            original_sha256="sha-test-2",
        ),
    )
    job_id = await jobs_module.enqueue(
        settings.DB_PATH,
        "ocr",
        "document",
        doc_id,
        payload={"mode": "private", "storage_mode": "local_only"},
    )

    from services import worker

    async def fake_fail(*_, **__):
        raise RuntimeError("synthetic OCR boom")

    with patch.object(worker, "extract_text", side_effect=fake_fail):
        task = asyncio.create_task(worker.worker_loop(bot, settings))
        for _ in range(50):
            j = await jobs_module.get_by_id(settings.DB_PATH, job_id)
            if j is not None and j.status in ("done", "failed"):
                break
            await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    j = await jobs_module.get_by_id(settings.DB_PATH, job_id)
    assert j is not None
    assert j.status == "failed"
    assert "synthetic OCR boom" in (j.error or "")

    assert any("❌" in m for m in bot.sent), f"expected error push in {bot.sent}"

"""Background worker loop for OCR + embedding jobs.

Runs as a single long-lived asyncio task (launched in ``bot.py`` after
``requeue_orphans``). Concurrency = 1 (FIFO). Each iteration:

  1. ``jobs.fetch_next_pending`` atomically grabs the oldest pending job and
     flips it to ``running``.
  2. Dispatch by ``job.type``:
       - ``"ocr"``   → decrypt encrypted file, run ``ocr_router.extract_text``,
         mask PII (if redacted_indexed), write ``ocr_text``/``redacted_text``,
         enqueue follow-up ``"embed"`` job for redacted_indexed documents.
       - ``"embed"`` → load redacted_text from DB, upsert into Qdrant.
  3. ``calibrate_metric`` updates the ETA baseline.
  4. ``finalize`` marks the job ``done``/``failed`` and a push notification
     is sent to the user.

The loop never exits on its own — it's cancelled by ``bot.py`` shutdown.
All exceptions in a job are caught, finalised as ``failed``, and reported
to the user via ``bot.send_message`` with a short error message.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from aiogram import Bot

from config import Settings
from services import (
    embeddings,
    encryption,
    jobs,
    pii,
    qdrant_store,
    storage,
)
from services.ocr_errors import GeminiUnavailable, PrivacyStrictBlocked
from services.ocr_router import extract_text
from services.gemini_client import GeminiClient


logger = logging.getLogger(__name__)


_POLL_IDLE_SEC = 2.0
_PROGRESS_INITIAL = 10


async def worker_loop(bot: Bot, settings: Settings) -> None:
    """Main loop. Cancellation-safe — stops cleanly when the task is cancelled."""
    logger.info("worker_loop started (concurrency=1, FIFO)")
    while True:
        try:
            job = await jobs.fetch_next_pending(settings.DB_PATH)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("worker: fetch_next_pending crashed: %s", exc)
            await asyncio.sleep(_POLL_IDLE_SEC)
            continue

        if job is None:
            await asyncio.sleep(_POLL_IDLE_SEC)
            continue

        logger.info("worker: picked job %s type=%s target=%s",
                    job.id, job.type, job.target_id)
        t_start = time.monotonic()
        try:
            if job.type == "ocr":
                units, result = await _run_ocr(job, bot, settings)
                metric_key = f"ocr_{job.payload.get('mode', 'private')}_page"
            elif job.type == "embed":
                units, result = await _run_embed(job, bot, settings)
                metric_key = "embed_chunk"
            else:
                raise ValueError(f"unknown job.type: {job.type!r}")

            elapsed = time.monotonic() - t_start
            try:
                await jobs.calibrate_metric(
                    settings.DB_PATH, metric_key, elapsed, max(1, units)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("calibrate_metric failed: %s", exc)

            await jobs.finalize(
                settings.DB_PATH, job.id, status="done", result=result
            )
            await _send_push(
                bot,
                settings,
                _format_done(job, result, units, elapsed),
            )
        except asyncio.CancelledError:
            # Don't finalize — requeue_orphans on next boot will flip to pending.
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("worker job %s failed: %s", job.id, exc)
            await jobs.finalize(
                settings.DB_PATH,
                job.id,
                status="failed",
                error=str(exc)[:500],
            )
            await _send_push(
                bot,
                settings,
                f"❌ Документ #{job.target_id}: ошибка на стадии {job.type}: {exc}",
            )


# ---------------------------------------------------------------------------
# OCR job
# ---------------------------------------------------------------------------


async def _run_ocr(job: jobs.Job, bot: Bot, settings: Settings) -> tuple[int, dict]:
    """Decrypt → OCR → mask PII → persist. Returns (units_processed, result_dict)."""
    doc_id = job.target_id
    payload = job.payload or {}
    ocr_mode = payload.get("mode", "private")             # 'private' | 'public'
    storage_mode = payload.get("storage_mode", "local_only")
    mime = payload.get("mime", "application/pdf")

    doc = await storage.find_document_by_id(settings.DB_PATH, doc_id)
    if doc is None:
        raise RuntimeError(f"document #{doc_id} not found in DB")

    enc_path = Path(doc.original_path)
    if not enc_path.exists():
        raise RuntimeError(f"encrypted file missing: {enc_path}")

    await jobs.update_progress(settings.DB_PATH, job.id, _PROGRESS_INITIAL)

    # Decrypt to a temp file we own and unlink at the end.
    with tempfile.NamedTemporaryFile(
        dir=settings.TMP_DIR, delete=False, suffix=_suffix_for_mime(mime)
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        Path(settings.TMP_DIR).mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            encryption.decrypt_file, enc_path, tmp_path, settings.DOCS_ENCRYPTION_KEY
        )
        data = tmp_path.read_bytes()

        # Run OCR (private → local engines; public → Gemini Files API).
        gemini_client = None
        if ocr_mode == "public":
            if not settings.GEMINI_API_KEY or settings.PRIVACY_STRICT:
                raise PrivacyStrictBlocked(
                    "PUBLIC mode unavailable: GEMINI_API_KEY missing or PRIVACY_STRICT=true"
                )
            gemini_client = GeminiClient(
                api_key=settings.GEMINI_API_KEY,
                privacy_strict=settings.PRIVACY_STRICT,
            )

        await jobs.update_progress(settings.DB_PATH, job.id, 40)
        text = await extract_text(data, mime, ocr_mode, gemini_client=gemini_client)
        await jobs.update_progress(settings.DB_PATH, job.id, 70)

        # PII masking policy (patch — bug fix):
        #   * local_only            → no indexing, redacted_text stays NULL.
        #   * redacted_indexed+PRIVATE → mask_pii (classic defence-in-depth).
        #   * redacted_indexed+PUBLIC  → user explicitly declared «no ПД»
        #     when they picked the «☁️ Без ПД, Gemini» button. We trust the
        #     declaration: index the raw OCR text so exact-phrase search
        #     stays usable (masking false-positives from natasha on
        #     all-caps titles was degrading search quality).
        redacted = None
        if storage_mode == "redacted_indexed":
            if ocr_mode == "public":
                redacted = text
            else:
                redacted = await asyncio.to_thread(pii.mask_pii, text)

        await storage.update_document_ocr(
            settings.DB_PATH, doc_id, ocr_text=text, redacted_text=redacted
        )
        await jobs.update_progress(settings.DB_PATH, job.id, 90)

        # Chain: enqueue embed for redacted_indexed documents.
        if storage_mode == "redacted_indexed":
            await jobs.enqueue(
                settings.DB_PATH,
                type_="embed",
                target_kind="document",
                target_id=doc_id,
                payload={"user_id": doc.user_id},
            )

        return (1, {"chars": len(text), "storage_mode": storage_mode})
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _suffix_for_mime(mime: str) -> str:
    if mime.startswith("image/"):
        return ".jpg"
    if mime == "application/pdf":
        return ".pdf"
    return ".bin"


# ---------------------------------------------------------------------------
# Embed job
# ---------------------------------------------------------------------------


async def _run_embed(job: jobs.Job, bot: Bot, settings: Settings) -> tuple[int, dict]:
    """Load redacted_text from DB → chunk → embed → upsert Qdrant.

    reindex-path: when a document was originally stored as
    ``local_only`` (so ``redacted_text`` is NULL) and the user flips
    ``storage_mode → redacted_indexed`` via ``/doc_edit``, there is no OCR
    job to run — the text is already in ``ocr_text``. We detect this here,
    mask through the PRIVATE pipeline (``pii.mask_pii``), persist the
    redacted version, and then proceed with the normal upsert.
    """
    doc_id = job.target_id
    doc = await storage.find_document_by_id(settings.DB_PATH, doc_id)
    if doc is None:
        raise RuntimeError(f"document #{doc_id} not found in DB")

    # Reindex-path: no redacted_text but OCR exists → mask and persist.
    if not doc.redacted_text and doc.ocr_text:
        redacted = await asyncio.to_thread(pii.mask_pii, doc.ocr_text)
        await storage.update_document_ocr(
            settings.DB_PATH, doc_id,
            ocr_text=doc.ocr_text, redacted_text=redacted,
        )
        doc.redacted_text = redacted

    if not doc.redacted_text:
        # No text to index — nothing to do but still a successful no-op.
        return (0, {"points": 0, "reason": "no ocr_text or redacted_text"})

    await jobs.update_progress(settings.DB_PATH, job.id, 10)
    await qdrant_store.ensure_collection(settings.QDRANT_URL)
    await jobs.update_progress(settings.DB_PATH, job.id, 30)

    points = await qdrant_store.upsert(
        settings.QDRANT_URL,
        doc_id=doc_id,
        text=doc.redacted_text,
        metadata={"user_id": doc.user_id, "type": doc.type, "source": f"doc_{doc_id}"},
    )
    await jobs.update_progress(settings.DB_PATH, job.id, 90)
    return (max(1, points), {"points": points})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_done(job: jobs.Job, result: dict, units: int, elapsed: float) -> str:
    if job.type == "ocr":
        chars = result.get("chars", 0)
        mode = job.payload.get("mode", "private")
        return (
            f"✅ Документ #{job.target_id} обработан (OCR, режим {mode}, "
            f"{chars} симв., {elapsed:.1f} сек)."
        )
    if job.type == "embed":
        pts = result.get("points", 0)
        return (
            f"✅ Документ #{job.target_id} проиндексирован "
            f"({pts} точек в Qdrant, {elapsed:.1f} сек)."
        )
    return f"✅ Документ #{job.target_id}: {job.type} готов."


async def _send_push(bot: Bot, settings: Settings, text: str) -> None:
    try:
        await bot.send_message(settings.ALLOWED_CHAT_ID, text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("worker push send failed: %s", exc)


__all__ = ["worker_loop"]

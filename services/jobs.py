"""Background worker queue — SQLite-backed.

every heavy unit (OCR, embedding) is enqueued as a ``Job`` row in
the ``jobs`` table (schema) and picked up by
``services.worker.worker_loop`` in order.

Key invariants:
  * ``fetch_next_pending`` atomically transitions exactly one row from
    ``pending`` → ``running``. Concurrency safety is provided by (a) an
    ``asyncio.Lock`` for in-process coroutines and (b) ``BEGIN IMMEDIATE``
    for inter-process safety (currently unused — we have one worker — but
    cheap and future-proof).
  * ``requeue_orphans`` runs at bot startup to recover from crashes:
    everything still marked ``running`` is flipped back to ``pending``.
  * ``calibrate_metric`` maintains an exponential moving average of the
    observed time-per-unit per ``job_type`` so ``estimate_eta`` gives
    progressively better predictions.

Data shape: ``Job`` is a frozen dataclass built from the SQLite row;
``payload`` and ``result`` are stored as JSON text.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import aiosqlite


logger = logging.getLogger(__name__)


_FETCH_LOCK = asyncio.Lock()
_EMA_ALPHA = 0.3

ETA_DEFAULTS: dict[str, float] = {
    "ocr_private_page": 30.0,
    "ocr_public_page": 20.0,
    "embed_chunk": 5.0,
    "embed_default": 5.0,
    "ocr_default": 30.0,
}


@dataclass
class Job:
    id: int
    type: str
    target_kind: str
    target_id: int
    status: str
    progress_pct: int = 0
    eta_seconds: Optional[int] = None
    payload: dict[str, Any] = field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


def _row_to_job(row) -> Job:
    payload_raw = row[7]
    payload = json.loads(payload_raw) if payload_raw else {}
    result_raw = row[8]
    result = json.loads(result_raw) if result_raw else None
    return Job(
        id=row[0],
        type=row[1],
        target_kind=row[2],
        target_id=row[3],
        status=row[4],
        progress_pct=row[5] or 0,
        eta_seconds=row[6],
        payload=payload,
        result=result,
        error=row[9],
        created_at=row[10],
        started_at=row[11],
        finished_at=row[12],
    )


_JOB_COLUMNS = (
    "id, type, target_kind, target_id, status, progress_pct, eta_seconds, "
    "payload, result, error, created_at, started_at, finished_at"
)


# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------


async def enqueue(
    db_path: str,
    type_: str,
    target_kind: str,
    target_id: int,
    payload: Optional[dict] = None,
) -> int:
    """Insert a fresh ``pending`` job and return its id."""
    payload_json = json.dumps(payload or {}, ensure_ascii=False)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            """
            INSERT INTO jobs (type, target_kind, target_id, status, payload)
            VALUES (?, ?, ?, 'pending', ?)
            """,
            (type_, target_kind, target_id, payload_json),
        )
        await db.commit()
        return cur.lastrowid


async def fetch_next_pending(db_path: str) -> Optional[Job]:
    """Atomically pick the oldest pending job and mark it running."""
    async with _FETCH_LOCK:
        async with aiosqlite.connect(db_path) as db:
            await db.execute("BEGIN IMMEDIATE")
            cur = await db.execute(
                f"""
                SELECT {_JOB_COLUMNS} FROM jobs
                WHERE status = 'pending'
                ORDER BY id ASC
                LIMIT 1
                """
            )
            row = await cur.fetchone()
            if row is None:
                await db.execute("COMMIT")
                return None
            job_id = row[0]
            await db.execute(
                """
                UPDATE jobs
                SET status = 'running', started_at = datetime('now')
                WHERE id = ?
                """,
                (job_id,),
            )
            await db.execute("COMMIT")
            # Refetch so started_at is included and status reflects 'running'.
            cur2 = await db.execute(
                f"SELECT {_JOB_COLUMNS} FROM jobs WHERE id = ?", (job_id,)
            )
            refreshed = await cur2.fetchone()
            return _row_to_job(refreshed) if refreshed else None


async def update_progress(
    db_path: str,
    job_id: int,
    progress_pct: int,
    eta_seconds: Optional[int] = None,
) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            UPDATE jobs
            SET progress_pct = ?, eta_seconds = ?
            WHERE id = ?
            """,
            (int(progress_pct), eta_seconds, job_id),
        )
        await db.commit()


async def finalize(
    db_path: str,
    job_id: int,
    status: str,
    result: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    """Mark job as done/failed and stamp finished_at."""
    if status not in ("done", "failed"):
        raise ValueError(f"invalid finalize status: {status!r}")
    result_json = json.dumps(result, ensure_ascii=False) if result is not None else None
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            UPDATE jobs
            SET status = ?, result = ?, error = ?, finished_at = datetime('now'),
                progress_pct = CASE WHEN ? = 'done' THEN 100 ELSE progress_pct END
            WHERE id = ?
            """,
            (status, result_json, error, status, job_id),
        )
        await db.commit()


async def get_by_id(db_path: str, job_id: int) -> Optional[Job]:
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            f"SELECT {_JOB_COLUMNS} FROM jobs WHERE id = ?", (job_id,)
        )
        row = await cur.fetchone()
        return _row_to_job(row) if row else None


async def get_by_target(
    db_path: str, target_kind: str, target_id: int
) -> list[Job]:
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            f"""
            SELECT {_JOB_COLUMNS} FROM jobs
            WHERE target_kind = ? AND target_id = ?
            ORDER BY id ASC
            """,
            (target_kind, target_id),
        )
        rows = await cur.fetchall()
        return [_row_to_job(r) for r in rows]


async def requeue_orphans(db_path: str) -> int:
    """Flip any leftover ``running`` rows back to ``pending``. Returns count."""
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            """
            UPDATE jobs
            SET status = 'pending', started_at = NULL
            WHERE status = 'running'
            """
        )
        await db.commit()
        count = cur.rowcount or 0
        if count:
            logger.warning("requeued %d orphaned running jobs back to pending", count)
        return count


# ---------------------------------------------------------------------------
# Metrics / ETA
# ---------------------------------------------------------------------------


async def calibrate_metric(
    db_path: str, job_type: str, elapsed_sec: float, units: int
) -> None:
    """Update the EMA ``avg_seconds_per_unit`` for ``job_type``.

    EMA: ``new = old*(1-α) + sample*α`` with α=0.3. First sample seeds avg.
    """
    if units <= 0:
        units = 1
    sample = float(elapsed_sec) / float(units)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT avg_seconds_per_unit, sample_count FROM service_metrics "
            "WHERE job_type = ?",
            (job_type,),
        )
        row = await cur.fetchone()
        if row is None:
            await db.execute(
                """
                INSERT INTO service_metrics (job_type, avg_seconds_per_unit, sample_count)
                VALUES (?, ?, 1)
                """,
                (job_type, sample),
            )
        else:
            old_avg = float(row[0])
            new_avg = old_avg * (1.0 - _EMA_ALPHA) + sample * _EMA_ALPHA
            await db.execute(
                """
                UPDATE service_metrics
                SET avg_seconds_per_unit = ?,
                    sample_count = sample_count + 1,
                    last_updated_at = datetime('now')
                WHERE job_type = ?
                """,
                (new_avg, job_type),
            )
        await db.commit()


async def estimate_eta(db_path: str, job_type: str, units: int) -> int:
    """Return integer-seconds estimate based on metrics; default if untrained."""
    if units <= 0:
        units = 1
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT avg_seconds_per_unit, sample_count FROM service_metrics "
            "WHERE job_type = ?",
            (job_type,),
        )
        row = await cur.fetchone()
    if row is None or row[1] == 0:
        default = ETA_DEFAULTS.get(job_type, ETA_DEFAULTS["ocr_default"])
        logger.debug(
            "estimate_eta: no samples for %r, using default %.1f sec/unit",
            job_type,
            default,
        )
        return int(default * units)
    avg = float(row[0])
    return max(1, int(avg * units))


__all__ = [
    "Job",
    "enqueue",
    "fetch_next_pending",
    "update_progress",
    "finalize",
    "get_by_id",
    "get_by_target",
    "requeue_orphans",
    "calibrate_metric",
    "estimate_eta",
    "ETA_DEFAULTS",
]

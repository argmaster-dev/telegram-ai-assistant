"""Tests for services.jobs — queue CRUD, atomic fetch, EMA calibration."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from services import jobs as jobs_module
from services.storage import init_db


@pytest_asyncio.fixture()
async def db_path():
    with tempfile.TemporaryDirectory() as tmp:
        p = str(Path(tmp) / "jobs.db")
        await init_db(p)
        yield p


@pytest.mark.asyncio
async def test_enqueue_creates_pending_row(db_path):
    job_id = await jobs_module.enqueue(
        db_path, "ocr", "document", 42, payload={"mode": "private"}
    )
    assert job_id > 0
    job = await jobs_module.get_by_id(db_path, job_id)
    assert job is not None
    assert job.status == "pending"
    assert job.type == "ocr"
    assert job.payload == {"mode": "private"}


@pytest.mark.asyncio
async def test_fetch_next_pending_atomic_under_concurrency(db_path):
    """Two concurrent fetches must return two distinct jobs, not the same one twice."""
    id_a = await jobs_module.enqueue(db_path, "ocr", "document", 1)
    id_b = await jobs_module.enqueue(db_path, "ocr", "document", 2)

    fetched_a, fetched_b = await asyncio.gather(
        jobs_module.fetch_next_pending(db_path),
        jobs_module.fetch_next_pending(db_path),
    )
    ids = {j.id for j in (fetched_a, fetched_b) if j is not None}
    assert ids == {id_a, id_b}, f"got ids={ids}"
    assert fetched_a.status == "running"
    assert fetched_b.status == "running"

    # No pending left.
    assert await jobs_module.fetch_next_pending(db_path) is None


@pytest.mark.asyncio
async def test_requeue_orphans_flips_running_back(db_path):
    # Create two 'running' jobs manually by fetching them.
    await jobs_module.enqueue(db_path, "ocr", "document", 1)
    await jobs_module.enqueue(db_path, "ocr", "document", 2)
    await jobs_module.fetch_next_pending(db_path)
    await jobs_module.fetch_next_pending(db_path)

    count = await jobs_module.requeue_orphans(db_path)
    assert count == 2

    # Both should be pending again.
    j1 = await jobs_module.fetch_next_pending(db_path)
    j2 = await jobs_module.fetch_next_pending(db_path)
    assert j1 is not None and j1.status == "running"
    assert j2 is not None and j2.status == "running"


@pytest.mark.asyncio
async def test_calibrate_metric_ema_and_estimate(db_path):
    # No samples yet → default ETA.
    eta0 = await jobs_module.estimate_eta(db_path, "ocr_private_page", 3)
    assert eta0 == int(jobs_module.ETA_DEFAULTS["ocr_private_page"] * 3)

    # First sample seeds avg=10.0.
    await jobs_module.calibrate_metric(db_path, "ocr_private_page", 10.0, 1)
    eta1 = await jobs_module.estimate_eta(db_path, "ocr_private_page", 1)
    assert eta1 == 10

    # Second sample: avg = 10.0*(1-0.3) + 50.0*0.3 = 22.0
    await jobs_module.calibrate_metric(db_path, "ocr_private_page", 50.0, 1)
    eta2 = await jobs_module.estimate_eta(db_path, "ocr_private_page", 1)
    assert eta2 == 22


@pytest.mark.asyncio
async def test_finalize_marks_job_done(db_path):
    job_id = await jobs_module.enqueue(db_path, "ocr", "document", 7)
    await jobs_module.fetch_next_pending(db_path)
    await jobs_module.finalize(db_path, job_id, "done", result={"chars": 100})

    j = await jobs_module.get_by_id(db_path, job_id)
    assert j is not None
    assert j.status == "done"
    assert j.progress_pct == 100
    assert j.result == {"chars": 100}
    assert j.finished_at is not None


@pytest.mark.asyncio
async def test_get_by_target_returns_all_jobs(db_path):
    await jobs_module.enqueue(db_path, "ocr", "document", 33)
    await jobs_module.enqueue(db_path, "embed", "document", 33)
    joblist = await jobs_module.get_by_target(db_path, "document", 33)
    assert len(joblist) == 2
    assert {j.type for j in joblist} == {"ocr", "embed"}

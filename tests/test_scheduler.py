"""Tests for services.scheduler.SchedulerService.

These use tmp_path SQLite files so they don't touch the production
``tg_assist.db``. The fire function is never invoked in these tests — we only
exercise schedule/cancel/persistence mechanics.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from services import reminder_store
from services.intents import Reminder
from services.scheduler import JOB_ID_PREFIX, SchedulerService, _job_id
from services.storage import init_db


TZ = ZoneInfo("Europe/Moscow")


def _future_reminder(user_id: int = 1, hours: int = 1) -> Reminder:
    return Reminder(
        user_id=user_id,
        title="test reminder",
        when_at=datetime.now(tz=TZ) + timedelta(hours=hours),
    )


@pytest.mark.asyncio
async def test_schedule_reminder_creates_db_row_and_job(tmp_path):
    db_path = str(tmp_path / "test.db")
    await init_db(db_path)

    scheduler = SchedulerService(db_path)
    scheduler.start()
    try:
        reminder = _future_reminder(hours=3)
        job_id = await scheduler.schedule_reminder(reminder)

        assert job_id.startswith(JOB_ID_PREFIX)
        assert reminder.id is not None
        assert job_id == _job_id(reminder.id)

        stored = await reminder_store.get(db_path, reminder.id)
        assert stored is not None
        assert stored.title == "test reminder"
        assert stored.status == "scheduled"

        job = scheduler.scheduler.get_job(job_id)
        assert job is not None
    finally:
        scheduler.shutdown(wait=True)


@pytest.mark.asyncio
async def test_cancel_removes_job_and_marks_cancelled(tmp_path):
    db_path = str(tmp_path / "test.db")
    await init_db(db_path)

    scheduler = SchedulerService(db_path)
    scheduler.start()
    try:
        reminder = _future_reminder(hours=4)
        await scheduler.schedule_reminder(reminder)
        reminder_id = reminder.id
        assert reminder_id is not None

        ok = await scheduler.cancel(reminder_id)
        assert ok is True

        assert scheduler.scheduler.get_job(_job_id(reminder_id)) is None

        stored = await reminder_store.get(db_path, reminder_id)
        assert stored is not None
        assert stored.status == "cancelled"
    finally:
        scheduler.shutdown(wait=True)


@pytest.mark.asyncio
async def test_job_persists_across_scheduler_recreate(tmp_path):
    """Simulates a bot restart: a new SchedulerService over the same SQLite
    file must find the job persisted by the previous instance.
    """
    db_path = str(tmp_path / "test.db")
    await init_db(db_path)

    first = SchedulerService(db_path)
    first.start()
    reminder = _future_reminder(hours=5)
    await first.schedule_reminder(reminder)
    reminder_id = reminder.id
    assert reminder_id is not None
    first.shutdown(wait=True)

    second = SchedulerService(db_path)
    second.start()
    try:
        job = second.scheduler.get_job(_job_id(reminder_id))
        assert job is not None, "job should persist across scheduler recreate"
    finally:
        second.shutdown(wait=True)

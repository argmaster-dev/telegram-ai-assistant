"""CRUD over the ``reminders`` SQLite table.

Sits on top of ``services.storage`` (schema owner). Handler code talks to this
module rather than writing SQL inline. The same pattern will be reused for
``tasks``, ``subscriptions``, and ``documents`` later.

``create``, ``get``, ``list_by_user``, ``update_status``, ``delete``.
adds ``rrule``, ``exdates`` (JSON), ``exclude_holidays`` to the mapping
            and introduces ``update(reminder)`` for ``/remind_edit``.
"""
from __future__ import annotations

import json
from typing import Optional

import aiosqlite
from dateutil.parser import isoparse

from services.intents import Reminder


def _row_to_reminder(row: aiosqlite.Row) -> Reminder:
    when_at_raw = row["when_at"]
    when_at = isoparse(when_at_raw) if when_at_raw else None

    exdates_raw = row["exdates"]
    if exdates_raw:
        try:
            exdates = list(json.loads(exdates_raw))
        except (TypeError, ValueError):
            exdates = []
    else:
        exdates = []

    return Reminder(
        id=row["id"],
        user_id=row["user_id"],
        title=row["title"],
        when_at=when_at,
        rrule=row["rrule"] or None,
        exdates=exdates,
        exclude_holidays=row["exclude_holidays"] or "none",
        notify_before_sec=row["notify_before_sec"],
        source=row["source"],
        status=row["status"],
    )


async def create(db_path: str, reminder: Reminder) -> int:
    """Insert ``reminder`` and return the generated ``id``."""
    when_at_iso = reminder.when_at.isoformat() if reminder.when_at else None
    exdates_json = json.dumps(reminder.exdates) if reminder.exdates else None
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            INSERT INTO reminders
                (user_id, title, when_at, rrule, exdates, exclude_holidays,
                 notify_before_sec, source, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reminder.user_id,
                reminder.title,
                when_at_iso,
                reminder.rrule,
                exdates_json,
                reminder.exclude_holidays,
                reminder.notify_before_sec,
                reminder.source,
                reminder.status,
            ),
        )
        await db.commit()
        assert cursor.lastrowid is not None
        return cursor.lastrowid


async def get(db_path: str, reminder_id: int) -> Optional[Reminder]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM reminders WHERE id = ?", (reminder_id,)
        ) as cursor:
            row = await cursor.fetchone()
    return _row_to_reminder(row) if row else None


async def list_by_user(
    db_path: str,
    user_id: int,
    status: str = "scheduled",
) -> list[Reminder]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT * FROM reminders
            WHERE user_id = ? AND status = ?
            ORDER BY when_at ASC
            """,
            (user_id, status),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_reminder(row) for row in rows]


async def update_status(db_path: str, reminder_id: int, status: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            UPDATE reminders
            SET status = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (status, reminder_id),
        )
        await db.commit()


async def update(db_path: str, reminder: Reminder) -> None:
    """Full-record UPDATE (used by ``/remind_edit``). ``reminder.id`` required."""
    if reminder.id is None:
        raise ValueError("reminder.id is required for update")

    when_at_iso = reminder.when_at.isoformat() if reminder.when_at else None
    exdates_json = json.dumps(reminder.exdates) if reminder.exdates else None
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            UPDATE reminders SET
                title = ?,
                when_at = ?,
                rrule = ?,
                exdates = ?,
                exclude_holidays = ?,
                notify_before_sec = ?,
                status = ?,
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (
                reminder.title,
                when_at_iso,
                reminder.rrule,
                exdates_json,
                reminder.exclude_holidays,
                reminder.notify_before_sec,
                reminder.status,
                reminder.id,
            ),
        )
        await db.commit()


async def delete(db_path: str, reminder_id: int) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        await db.commit()

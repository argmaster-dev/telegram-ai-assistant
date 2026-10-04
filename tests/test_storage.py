"""Smoke tests for services.storage.init_db."""
from __future__ import annotations

import aiosqlite
import pytest

from services.storage import init_db


EXPECTED_TABLES = frozenset(
    {
        "documents",
        "reminders",
        "tasks",
        "subscriptions",
        "jobs",
        "service_metrics",
        "fsm_state",
    }
)


async def _list_tables(db_path: str) -> set[str]:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ) as cursor:
            rows = await cursor.fetchall()
    return {row[0] for row in rows}


@pytest.mark.asyncio
async def test_init_db_creates_all_expected_tables(tmp_path):
    db_path = str(tmp_path / "test.db")
    await init_db(db_path)

    tables = await _list_tables(db_path)
    missing = EXPECTED_TABLES - tables
    assert not missing, f"init_db did not create: {missing}"


@pytest.mark.asyncio
async def test_init_db_is_idempotent(tmp_path):
    db_path = str(tmp_path / "test.db")
    await init_db(db_path)
    await init_db(db_path)  # second call must not raise

    tables = await _list_tables(db_path)
    assert EXPECTED_TABLES.issubset(tables)

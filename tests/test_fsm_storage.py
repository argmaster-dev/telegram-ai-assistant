"""Tests for services.fsm_storage.SQLStorage (aiogram 3 BaseStorage on aiosqlite)."""
from __future__ import annotations

import pytest
from aiogram.fsm.storage.base import StorageKey

from services.fsm_storage import SQLStorage
from services.storage import init_db


@pytest.fixture
def key() -> StorageKey:
    return StorageKey(bot_id=1, chat_id=100, user_id=100)


@pytest.mark.asyncio
async def test_set_get_state_roundtrip(tmp_path, key):
    db_path = str(tmp_path / "test.db")
    await init_db(db_path)

    storage = SQLStorage(db_path)
    await storage.set_state(key, "awaiting_confirmation")
    assert await storage.get_state(key) == "awaiting_confirmation"

    # Clearing to None should be reflected
    await storage.set_state(key, None)
    assert await storage.get_state(key) is None


@pytest.mark.asyncio
async def test_state_and_data_survive_storage_recreate(tmp_path, key):
    """Simulates a container restart — the second ``SQLStorage`` opens the same
    SQLite file that the first one wrote to, and must see the persisted state.
    """
    db_path = str(tmp_path / "test.db")
    await init_db(db_path)

    first = SQLStorage(db_path)
    await first.set_state(key, "persisted_state")
    await first.set_data(key, {"field": "value", "n": 42})
    await first.close()

    second = SQLStorage(db_path)
    assert await second.get_state(key) == "persisted_state"
    assert await second.get_data(key) == {"field": "value", "n": 42}

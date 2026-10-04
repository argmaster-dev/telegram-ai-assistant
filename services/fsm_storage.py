"""aiogram 3 FSM storage backed by SQLite (aiosqlite).

Replaces the default `MemoryStorage` so FSM state survives container restarts.
Schema: the `fsm_state` table is created by `services.storage.init_db()`.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import aiosqlite

from aiogram.fsm.state import State
from aiogram.fsm.storage.base import BaseStorage, StorageKey


class SQLStorage(BaseStorage):
    """aiogram 3 ``BaseStorage`` implementation on top of aiosqlite.

    Opens a new connection per call — cheap for bot's low write rate (~1 per update)
    and avoids cross-coroutine contention on a single shared connection.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    # ---- key helpers ---------------------------------------------------

    @staticmethod
    def _key_tuple(key: StorageKey) -> tuple[int, int, int, int, str, str]:
        # SQLite PRIMARY KEY treats every NULL as unique, which breaks upsert.
        # Normalise optionals to concrete sentinels.
        return (
            key.bot_id,
            key.chat_id,
            key.user_id,
            key.thread_id if key.thread_id is not None else -1,
            key.business_connection_id or "",
            key.destiny or "default",
        )

    # ---- state ---------------------------------------------------------

    async def set_state(self, key: StorageKey, state: Optional[Any] = None) -> None:
        state_str: Optional[str]
        if state is None:
            state_str = None
        elif isinstance(state, State):
            state_str = state.state
        else:
            state_str = str(state)

        row = self._key_tuple(key)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO fsm_state (bot_id, chat_id, user_id, thread_id,
                                       business_connection_id, destiny, state, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, '{}')
                ON CONFLICT(bot_id, chat_id, user_id, thread_id,
                            business_connection_id, destiny)
                DO UPDATE SET state=excluded.state
                """,
                (*row, state_str),
            )
            await db.commit()

    async def get_state(self, key: StorageKey) -> Optional[str]:
        row = self._key_tuple(key)
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                """
                SELECT state FROM fsm_state
                WHERE bot_id=? AND chat_id=? AND user_id=? AND thread_id=?
                  AND business_connection_id=? AND destiny=?
                """,
                row,
            ) as cursor:
                result = await cursor.fetchone()
        return result[0] if result else None

    # ---- data ----------------------------------------------------------

    async def set_data(self, key: StorageKey, data: dict[str, Any]) -> None:
        data_str = json.dumps(data, ensure_ascii=False)
        row = self._key_tuple(key)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO fsm_state (bot_id, chat_id, user_id, thread_id,
                                       business_connection_id, destiny, state, data)
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(bot_id, chat_id, user_id, thread_id,
                            business_connection_id, destiny)
                DO UPDATE SET data=excluded.data
                """,
                (*row, data_str),
            )
            await db.commit()

    async def get_data(self, key: StorageKey) -> dict[str, Any]:
        row = self._key_tuple(key)
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                """
                SELECT data FROM fsm_state
                WHERE bot_id=? AND chat_id=? AND user_id=? AND thread_id=?
                  AND business_connection_id=? AND destiny=?
                """,
                row,
            ) as cursor:
                result = await cursor.fetchone()
        if result is None:
            return {}
        return json.loads(result[0]) if result[0] else {}

    async def update_data(self, key: StorageKey, data: dict[str, Any]) -> dict[str, Any]:
        current = await self.get_data(key)
        current.update(data)
        await self.set_data(key, current)
        return current

    # ---- lifecycle -----------------------------------------------------

    async def close(self) -> None:
        # No persistent connection to close — every call opens its own.
        return None

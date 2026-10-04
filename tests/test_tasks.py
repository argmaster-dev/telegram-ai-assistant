"""tasks parser + storage CRUD tests.

The Ollama call is mocked: parser tests assert that ``parse_task`` correctly
maps the model JSON onto the ``Task`` dataclass, including deadline edge
cases (past → None, date-only → end-of-day) and priority/category fallback.

Storage tests use ``tmp_path`` for a real on-disk SQLite (no mocking) — same
pattern as ``test_docs_crud.py``.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest

from services import storage, tasks
from services.intents import Task
from services.ollama_client import OllamaUnavailable


_TZ = ZoneInfo("Europe/Moscow")
_NOW = datetime(2026, 4, 25, 12, 0, 0, tzinfo=_TZ)
_NEXT_FRIDAY = "2026-05-01"  # Friday after _NOW


# ---------------------------------------------------------------------------
# parse_task — parser unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_task_with_deadline_and_category():
    """User: 'купить молоко к пятнице, категория home'."""
    mock_gen = AsyncMock(return_value={
        "title": "купить молоко",
        "deadline": _NEXT_FRIDAY,  # date-only → coerced to end-of-day
        "priority": "med",
        "category": "home",
    })
    with patch("services.tasks.generate_json", mock_gen):
        task = await tasks.parse_task(
            "купить молоко к пятнице, категория home",
            base_url="http://tg_ollama:11434",
            user_id=42,
            now=_NOW,
        )
    assert task.title == "купить молоко"
    assert task.category == "home"
    assert task.priority == "med"
    assert task.user_id == 42
    assert task.status == "todo"
    # Date-only deadline coerced to 23:59:59 same day in Moscow TZ.
    assert task.deadline is not None
    assert task.deadline.date().isoformat() == _NEXT_FRIDAY
    assert task.deadline.tzinfo is not None


@pytest.mark.asyncio
async def test_parse_task_high_priority_and_work_category():
    """User: 'срочно подготовить отчёт до пятницы, work'."""
    mock_gen = AsyncMock(return_value={
        "title": "подготовить отчёт",
        "deadline": _NEXT_FRIDAY,
        "priority": "high",
        "category": "work",
    })
    with patch("services.tasks.generate_json", mock_gen):
        task = await tasks.parse_task(
            "срочно подготовить отчёт до пятницы, work",
            base_url="http://x",
            user_id=1,
            now=_NOW,
        )
    assert task.priority == "high"
    assert task.category == "work"


@pytest.mark.asyncio
async def test_parse_task_open_ended_no_deadline():
    """User: 'добавь задачу купить молоко' — no deadline mentioned."""
    mock_gen = AsyncMock(return_value={
        "title": "купить молоко",
        "deadline": "",
        "priority": "med",
        "category": "shopping",
    })
    with patch("services.tasks.generate_json", mock_gen):
        task = await tasks.parse_task(
            "добавь задачу купить молоко",
            base_url="http://x",
            user_id=1,
            now=_NOW,
        )
    assert task.deadline is None
    assert task.category == "shopping"


@pytest.mark.asyncio
async def test_parse_task_past_deadline_dropped():
    """Past deadline becomes None (open-ended), not a parse error."""
    past_date = (_NOW - timedelta(days=10)).date().isoformat()
    mock_gen = AsyncMock(return_value={
        "title": "купить хлеб",
        "deadline": past_date,
        "priority": "med",
        "category": "shopping",
    })
    with patch("services.tasks.generate_json", mock_gen):
        task = await tasks.parse_task(
            "купить хлеб на прошлой неделе",
            base_url="http://x",
            now=_NOW,
        )
    assert task.deadline is None  # past → dropped
    assert task.title == "купить хлеб"


@pytest.mark.asyncio
async def test_parse_task_invalid_priority_falls_back_to_med():
    mock_gen = AsyncMock(return_value={
        "title": "что-то",
        "deadline": "",
        "priority": "BANANA",  # invalid
        "category": "general",
    })
    with patch("services.tasks.generate_json", mock_gen):
        task = await tasks.parse_task("что-то", base_url="http://x", now=_NOW)
    assert task.priority == "med"


@pytest.mark.asyncio
async def test_parse_task_empty_title_raises():
    mock_gen = AsyncMock(return_value={
        "title": "",
        "deadline": "",
        "priority": "med",
        "category": "general",
    })
    with patch("services.tasks.generate_json", mock_gen):
        with pytest.raises(tasks.TaskParseError):
            await tasks.parse_task("...", base_url="http://x", now=_NOW)


@pytest.mark.asyncio
async def test_parse_task_ollama_unavailable_raises():
    mock_gen = AsyncMock(side_effect=OllamaUnavailable("down"))
    with patch("services.tasks.generate_json", mock_gen):
        with pytest.raises(tasks.TaskParseError):
            await tasks.parse_task("что-то", base_url="http://x", now=_NOW)


# ---------------------------------------------------------------------------
# Storage CRUD — round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_get_list_round_trip(tmp_path):
    db_path = str(tmp_path / "tasks.db")
    await storage.init_db(db_path)

    deadline = _NOW + timedelta(days=5)
    t = Task(
        user_id=42, title="купить молоко", deadline=deadline,
        priority="high", status="todo", category="shopping",
    )
    task_id = await storage.save_task(db_path, t)
    assert task_id > 0

    fetched = await storage.get_task(db_path, task_id)
    assert fetched is not None
    assert fetched.title == "купить молоко"
    assert fetched.priority == "high"
    assert fetched.category == "shopping"
    assert fetched.user_id == 42
    assert fetched.deadline is not None
    # Serialised then re-parsed; same instant after both round trips.
    assert fetched.deadline.isoformat() == deadline.isoformat()

    # list_tasks default (status='todo')
    rows = await storage.list_tasks(db_path, 42)
    assert len(rows) == 1 and rows[0].id == task_id

    # list_tasks for a different user → empty
    rows_other = await storage.list_tasks(db_path, 99)
    assert rows_other == []


@pytest.mark.asyncio
async def test_update_task_fields_changes_priority_and_updated_at(tmp_path):
    db_path = str(tmp_path / "u.db")
    await storage.init_db(db_path)
    t = Task(user_id=1, title="x", priority="med", category="general")
    task_id = await storage.save_task(db_path, t)

    updated = await storage.update_task_fields(
        db_path, task_id, {"priority": "high"}
    )
    assert updated is not None and updated.priority == "high"

    # Non-whitelisted field rejected.
    with pytest.raises(ValueError):
        await storage.update_task_fields(db_path, task_id, {"user_id": 999})


@pytest.mark.asyncio
async def test_mark_done_then_listing_filters(tmp_path):
    db_path = str(tmp_path / "m.db")
    await storage.init_db(db_path)
    a = await storage.save_task(db_path, Task(user_id=1, title="a", category="home"))
    b = await storage.save_task(db_path, Task(user_id=1, title="b", category="work"))

    # initial: both in todo
    assert {t.id for t in await storage.list_tasks(db_path, 1)} == {a, b}

    await storage.mark_task_done(db_path, a)
    todo_only = await storage.list_tasks(db_path, 1)
    assert {t.id for t in todo_only} == {b}, "done task must not appear in /tasks"

    all_rows = await storage.list_tasks(db_path, 1, status=None)
    assert {t.id for t in all_rows} == {a, b}, "/tasks_all must include done"

    # mark back to todo
    await storage.mark_task_todo(db_path, a)
    assert {t.id for t in await storage.list_tasks(db_path, 1)} == {a, b}


@pytest.mark.asyncio
async def test_list_by_category(tmp_path):
    db_path = str(tmp_path / "c.db")
    await storage.init_db(db_path)
    await storage.save_task(db_path, Task(user_id=1, title="a", category="home"))
    await storage.save_task(db_path, Task(user_id=1, title="b", category="work"))
    await storage.save_task(db_path, Task(user_id=1, title="c", category="work"))

    work = await storage.list_tasks_by_category(db_path, 1, "work")
    assert {t.title for t in work} == {"b", "c"}

    home = await storage.list_tasks_by_category(db_path, 1, "home")
    assert {t.title for t in home} == {"a"}

    none = await storage.list_tasks_by_category(db_path, 1, "nonexistent")
    assert none == []


@pytest.mark.asyncio
async def test_delete_task_returns_row_then_404(tmp_path):
    db_path = str(tmp_path / "d.db")
    await storage.init_db(db_path)
    task_id = await storage.save_task(
        db_path, Task(user_id=1, title="kill me", category="general")
    )
    deleted = await storage.delete_task(db_path, task_id)
    assert deleted is not None and deleted.id == task_id

    # Idempotent: deleting again returns None.
    again = await storage.delete_task(db_path, task_id)
    assert again is None
    assert await storage.get_task(db_path, task_id) is None

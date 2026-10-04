"""SQLite storage — schema initialisation + document CRUD for tg_assist.

Tables match `00_CONCEPT.md §5`:
  - documents, reminders, tasks, subscriptions  — business entities
  - jobs, service_metrics                       — background worker queue (used)
  - fsm_state                                   — aiogram FSM persistence (used)
  - documents_fts                               — SQLite FTS5 virtual table

APScheduler creates its own `scheduler_jobs` table automatically on first use.

added async document CRUD; extends with tags CRUD,
``update_document``/``delete_document``, and FTS5-backed ``full_text_search``
with graceful LIKE fallback when FTS5 is not available in the sqlite3 build.

Pattern throughout: ``aiosqlite.connect`` per call (as in ``fsm_storage.py``).
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from services.intents import Document, Subscription, Task


logger = logging.getLogger(__name__)


# Module-level flag set by ``init_db`` after detecting FTS5 availability.
# ``full_text_search`` consults it to pick MATCH path or LIKE-fallback.
_FTS5_AVAILABLE: bool = False


_SCHEMA_STATEMENTS: tuple[str, ...] = (
    # --- documents ---
    """
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        type TEXT NOT NULL,
        type_confirmed_by_user INTEGER NOT NULL DEFAULT 0,
        has_pii INTEGER NOT NULL,
        storage_mode TEXT NOT NULL,
        original_path TEXT NOT NULL,
        original_sha256 TEXT NOT NULL,
        ocr_text TEXT,
        redacted_text TEXT,
        tags TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type)",
    "CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(user_id, original_sha256)",

    # --- reminders ---
    """
    CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        when_at TEXT,
        rrule TEXT,
        exdates TEXT,
        exclude_holidays TEXT NOT NULL DEFAULT 'none',
        notify_before_sec INTEGER NOT NULL DEFAULT 0,
        source TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'scheduled',
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status)",

    # --- tasks ---
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        deadline TEXT,
        priority TEXT NOT NULL DEFAULT 'med',
        status TEXT NOT NULL DEFAULT 'todo',
        category TEXT NOT NULL DEFAULT 'general',
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tasks_category ON tasks(category)",

    # --- subscriptions ---
    """
    CREATE TABLE IF NOT EXISTS subscriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        service TEXT NOT NULL,
        amount REAL NOT NULL,
        currency TEXT NOT NULL DEFAULT 'RUB',
        card TEXT,
        recurrence TEXT NOT NULL,
        day_of_month INTEGER,
        next_charge_at TEXT,
        notify_before_days INTEGER NOT NULL DEFAULT 2,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    # index for /subs listing + stats month-aggregation queries
    "CREATE INDEX IF NOT EXISTS idx_subscriptions_user_next ON subscriptions(user_id, next_charge_at)",
    # index for /tasks listing by user/status (category index already exists)
    "CREATE INDEX IF NOT EXISTS idx_tasks_user_status ON tasks(user_id, status)",

    # --- jobs (background worker queue) ---
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL,
        target_kind TEXT NOT NULL,
        target_id INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        progress_pct INTEGER NOT NULL DEFAULT 0,
        eta_seconds INTEGER,
        payload TEXT,
        result TEXT,
        error TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        started_at TEXT,
        finished_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_target ON jobs(target_kind, target_id)",

    # --- service_metrics (ETA calibration) ---
    """
    CREATE TABLE IF NOT EXISTS service_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_type TEXT NOT NULL,
        avg_seconds_per_unit REAL NOT NULL,
        sample_count INTEGER NOT NULL DEFAULT 0,
        last_updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_service_metrics_type ON service_metrics(job_type)",

    # --- fsm_state (aiogram 3 SQLStorage) ---
    """
    CREATE TABLE IF NOT EXISTS fsm_state (
        bot_id INTEGER NOT NULL,
        chat_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        thread_id INTEGER NOT NULL DEFAULT -1,
        business_connection_id TEXT NOT NULL DEFAULT '',
        destiny TEXT NOT NULL DEFAULT 'default',
        state TEXT,
        data TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY (bot_id, chat_id, user_id, thread_id, business_connection_id, destiny)
    )
    """,
)


# ---------------------------------------------------------------------------
# FTS5 virtual table + sync triggers
# ---------------------------------------------------------------------------
#
# External-content FTS5 — the virtual table does not store a copy of the
# text; it keeps only the tokenised inverted index and references rows in
# ``documents`` by ``rowid`` (= ``documents.id``). Three triggers keep the
# index in sync on INSERT/DELETE/UPDATE — the 'delete' contentless command
# is the standard SQLite idiom for external-content tables.
#
# ``tokenize='unicode61 remove_diacritics 2'`` normalises diacritics and
# splits Cyrillic correctly — required for Russian ocr text.
_FTS5_STATEMENTS: tuple[str, ...] = (
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
        ocr_text, redacted_text, tags,
        content='documents', content_rowid='id',
        tokenize='unicode61 remove_diacritics 2'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
        INSERT INTO documents_fts(rowid, ocr_text, redacted_text, tags)
        VALUES (new.id,
                coalesce(new.ocr_text, ''),
                coalesce(new.redacted_text, ''),
                coalesce(new.tags, ''));
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
        INSERT INTO documents_fts(documents_fts, rowid, ocr_text, redacted_text, tags)
        VALUES('delete', old.id,
               coalesce(old.ocr_text, ''),
               coalesce(old.redacted_text, ''),
               coalesce(old.tags, ''));
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
        INSERT INTO documents_fts(documents_fts, rowid, ocr_text, redacted_text, tags)
        VALUES('delete', old.id,
               coalesce(old.ocr_text, ''),
               coalesce(old.redacted_text, ''),
               coalesce(old.tags, ''));
        INSERT INTO documents_fts(rowid, ocr_text, redacted_text, tags)
        VALUES (new.id,
                coalesce(new.ocr_text, ''),
                coalesce(new.redacted_text, ''),
                coalesce(new.tags, ''));
    END
    """,
)


async def init_db(db_path: str) -> None:
    """Initialise the SQLite schema. Idempotent.

    Attempts to create the FTS5 virtual table and its sync triggers. If the
    local sqlite3 build has no FTS5 compiled in, the CREATE VIRTUAL TABLE
    call raises and we log the problem; the module-level ``_FTS5_AVAILABLE``
    flag stays False and ``full_text_search`` falls back to LIKE.
    """
    global _FTS5_AVAILABLE
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        for stmt in _SCHEMA_STATEMENTS:
            await db.execute(stmt)
        await db.commit()

        try:
            for stmt in _FTS5_STATEMENTS:
                await db.execute(stmt)
            await db.commit()
            # Backfill existing rows into FTS (idempotent — insert-or-replace via
            # external-content semantics). Only needed when upgrading a DB that
            # existed before 04B.
            await db.execute(
                "INSERT INTO documents_fts(documents_fts) VALUES('rebuild')"
            )
            await db.commit()
            _FTS5_AVAILABLE = True
            logger.info("FTS5 virtual table + triggers ready")
        except aiosqlite.OperationalError as exc:
            logger.error(
                "FTS5 unavailable (%s) — full_text_search will fall back to LIKE",
                exc,
            )
            _FTS5_AVAILABLE = False


def is_fts5_available() -> bool:
    """Whether the current ``init_db`` succeeded in building FTS5."""
    return _FTS5_AVAILABLE


# ---------------------------------------------------------------------------
# Documents CRUD
# ---------------------------------------------------------------------------


_DOC_COLUMNS = (
    "id, user_id, type, type_confirmed_by_user, has_pii, storage_mode, "
    "original_path, original_sha256, ocr_text, redacted_text, tags, "
    "created_at, updated_at"
)


def _row_to_document(row: Optional[aiosqlite.Row]) -> Optional[Document]:
    if row is None:
        return None

    def _parse_dt(raw):
        if raw is None:
            return None
        # SQLite returns ISO-ish strings like "2026-04-24 12:34:56"
        try:
            return datetime.fromisoformat(str(raw).replace(" ", "T"))
        except (TypeError, ValueError):
            return None

    return Document(
        id=row[0],
        user_id=row[1],
        type=row[2],
        type_confirmed_by_user=bool(row[3]),
        has_pii=bool(row[4]),
        storage_mode=row[5],
        original_path=row[6],
        original_sha256=row[7],
        ocr_text=row[8],
        redacted_text=row[9],
        tags=row[10],
        created_at=_parse_dt(row[11]),
        updated_at=_parse_dt(row[12]),
    )


async def save_document(db_path: str, doc: Document) -> int:
    """Insert ``doc``; return its id.

    Idempotent dedup on ``(user_id, original_sha256)``: if a row already
    exists for the same user+hash, return that existing id without creating
    a duplicate (the unique index on `documents(user_id, original_sha256)`
    makes this safe).
    """
    async with aiosqlite.connect(db_path) as db:
        # Fast path: dedup lookup first.
        cur = await db.execute(
            "SELECT id FROM documents WHERE user_id = ? AND original_sha256 = ?",
            (doc.user_id, doc.original_sha256),
        )
        row = await cur.fetchone()
        if row is not None:
            return row[0]

        cur = await db.execute(
            """
            INSERT INTO documents (
                user_id, type, type_confirmed_by_user, has_pii, storage_mode,
                original_path, original_sha256, ocr_text, redacted_text, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.user_id,
                doc.type,
                int(doc.type_confirmed_by_user),
                int(doc.has_pii),
                doc.storage_mode,
                doc.original_path,
                doc.original_sha256,
                doc.ocr_text,
                doc.redacted_text,
                doc.tags,
            ),
        )
        await db.commit()
        return cur.lastrowid


async def find_document_by_sha256(
    db_path: str, user_id: int, sha256: str
) -> Optional[Document]:
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            f"SELECT {_DOC_COLUMNS} FROM documents "
            "WHERE user_id = ? AND original_sha256 = ?",
            (user_id, sha256),
        )
        return _row_to_document(await cur.fetchone())


async def find_document_by_id(db_path: str, doc_id: int) -> Optional[Document]:
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            f"SELECT {_DOC_COLUMNS} FROM documents WHERE id = ?",
            (doc_id,),
        )
        return _row_to_document(await cur.fetchone())


async def update_document_ocr(
    db_path: str,
    doc_id: int,
    ocr_text: Optional[str],
    redacted_text: Optional[str],
) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            UPDATE documents
            SET ocr_text = ?, redacted_text = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (ocr_text, redacted_text, doc_id),
        )
        await db.commit()


async def update_document_path(
    db_path: str, doc_id: int, original_path: str
) -> None:
    """Used right after encryption to write the final encrypted-file path."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            UPDATE documents
            SET original_path = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (original_path, doc_id),
        )
        await db.commit()


async def find_document_by_type(
    db_path: str, user_id: int, doc_type: str
) -> list[Document]:
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            f"SELECT {_DOC_COLUMNS} FROM documents "
            "WHERE user_id = ? AND type = ? ORDER BY created_at DESC",
            (user_id, doc_type),
        )
        rows = await cur.fetchall()
        return [d for d in (_row_to_document(r) for r in rows) if d is not None]


async def find_last_documents(
    db_path: str, user_id: int, limit: int = 1
) -> list[Document]:
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            f"SELECT {_DOC_COLUMNS} FROM documents "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cur.fetchall()
        return [d for d in (_row_to_document(r) for r in rows) if d is not None]


async def search_documents_full_text(
    db_path: str, user_id: int, query: str, limit: int = 10
) -> list[Document]:
    """LIKE-fallback full-text search — kept as a safety net.

    Callers should prefer ``full_text_search`` (FTS5). This function
    is used both as an internal fallback and as an explicit opt-out path.
    """
    like = f"%{query}%"
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            f"""
            SELECT {_DOC_COLUMNS} FROM documents
            WHERE user_id = ?
              AND (ocr_text LIKE ? OR redacted_text LIKE ?)
            ORDER BY created_at DESC LIMIT ?
            """,
            (user_id, like, like, limit),
        )
        rows = await cur.fetchall()
        return [d for d in (_row_to_document(r) for r in rows) if d is not None]


# ---------------------------------------------------------------------------
# Documents CRUD
# ---------------------------------------------------------------------------


_UPDATABLE_FIELDS: frozenset[str] = frozenset(
    {"type", "type_confirmed_by_user", "has_pii", "storage_mode", "tags"}
)


def _parse_tags_csv(raw: Optional[str]) -> list[str]:
    """CSV → list with empties stripped. Returns [] for None/empty."""
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _format_tags_csv(tags: list[str]) -> str:
    return ",".join(tags)


def _validate_tag(tag: str) -> str:
    """Normalise + validate one tag. Raises ValueError on bad input."""
    t = tag.strip()
    if not t:
        raise ValueError("tag is empty after stripping")
    if "," in t:
        raise ValueError(f"tag must not contain comma: {tag!r}")
    if len(t) > 50:
        raise ValueError(f"tag longer than 50 chars: {tag!r}")
    return t


async def update_document(
    db_path: str, doc_id: int, fields: dict[str, Any]
) -> Optional[Document]:
    """Update whitelisted fields and return the refreshed document.

    Whitelist: ``type``, ``type_confirmed_by_user``, ``has_pii``,
    ``storage_mode``, ``tags``. Any other key raises ``ValueError``.
    Boolean-ish fields are coerced to ``int`` for SQLite.
    """
    if not fields:
        return await find_document_by_id(db_path, doc_id)

    bad = set(fields.keys()) - _UPDATABLE_FIELDS
    if bad:
        raise ValueError(f"non-updatable fields: {sorted(bad)}")

    coerced: dict[str, Any] = {}
    for k, v in fields.items():
        if k in ("type_confirmed_by_user", "has_pii"):
            coerced[k] = int(bool(v))
        else:
            coerced[k] = v

    set_clause = ", ".join(f"{k} = ?" for k in coerced)
    params = list(coerced.values()) + [doc_id]
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            f"UPDATE documents SET {set_clause}, updated_at = datetime('now') "
            "WHERE id = ?",
            params,
        )
        await db.commit()
    return await find_document_by_id(db_path, doc_id)


async def delete_document(
    db_path: str, doc_id: int
) -> Optional[Document]:
    """Fetch-then-delete. Returns the Document that was removed (or None).

    Caller is responsible for cleaning up the encrypted file on disk and
    Qdrant points — ``delete_document`` only removes the SQLite row. The
    AFTER DELETE trigger syncs ``documents_fts`` automatically.
    """
    doc = await find_document_by_id(db_path, doc_id)
    if doc is None:
        return None
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        await db.commit()
    return doc


async def add_tags(
    db_path: str, doc_id: int, tags: list[str]
) -> list[str]:
    """Merge ``tags`` into the document's CSV, return the final list.

    Validation: no comma, no empty, ≤50 chars per tag. Raises ``ValueError``
    on invalid input. Dedup-by-value preserves insertion order.
    """
    clean = [_validate_tag(t) for t in tags]
    doc = await find_document_by_id(db_path, doc_id)
    if doc is None:
        raise LookupError(f"document #{doc_id} not found")

    existing = _parse_tags_csv(doc.tags)
    merged: list[str] = list(existing)
    for t in clean:
        if t not in merged:
            merged.append(t)

    csv = _format_tags_csv(merged)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE documents SET tags = ?, updated_at = datetime('now') WHERE id = ?",
            (csv, doc_id),
        )
        await db.commit()
    return merged


async def remove_tags(
    db_path: str, doc_id: int, tags: list[str]
) -> list[str]:
    """Remove ``tags`` from the document's CSV, return the final list."""
    to_remove = {_validate_tag(t) for t in tags}
    doc = await find_document_by_id(db_path, doc_id)
    if doc is None:
        raise LookupError(f"document #{doc_id} not found")

    remaining = [t for t in _parse_tags_csv(doc.tags) if t not in to_remove]
    csv = _format_tags_csv(remaining)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE documents SET tags = ?, updated_at = datetime('now') WHERE id = ?",
            (csv, doc_id),
        )
        await db.commit()
    return remaining


async def find_by_tag(
    db_path: str, user_id: int, tag: str
) -> list[Document]:
    """Exact-match tag lookup via the CSV-wrapping trick."""
    t = _validate_tag(tag)
    pattern = f"%,{t},%"
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            f"SELECT {_DOC_COLUMNS} FROM documents "
            "WHERE user_id = ? "
            "AND ',' || coalesce(tags, '') || ',' LIKE ? "
            "ORDER BY created_at DESC",
            (user_id, pattern),
        )
        rows = await cur.fetchall()
        return [d for d in (_row_to_document(r) for r in rows) if d is not None]


async def full_text_search(
    db_path: str, user_id: int, query: str, limit: int = 20
) -> list[tuple[Document, float]]:
    """FTS5-backed search; falls back to LIKE when FTS5 is unavailable.

    Returns ``(Document, relevance)`` pairs sorted by FTS rank (relevance
    ∈ [0, 1], higher = better match). When falling back to LIKE, relevance
    is set to 0.5 for every match (no useful ranking is available).
    """
    if not query.strip():
        return []

    if not _FTS5_AVAILABLE:
        docs = await search_documents_full_text(db_path, user_id, query, limit)
        return [(d, 0.5) for d in docs]

    # FTS5 ``rank`` is negative bm25; we convert to a positive [0,1]-like score
    # via 1/(1+|rank|). Good enough for a single-user merge ranking.
    sql = f"""
        SELECT {', '.join(f'd.{c.strip()}' for c in _DOC_COLUMNS.split(','))},
               documents_fts.rank AS fts_rank
        FROM documents_fts
        JOIN documents d ON d.id = documents_fts.rowid
        WHERE d.user_id = ?
          AND documents_fts MATCH ?
        ORDER BY documents_fts.rank
        LIMIT ?
    """
    async with aiosqlite.connect(db_path) as db:
        try:
            cur = await db.execute(sql, (user_id, query, limit))
            rows = await cur.fetchall()
        except aiosqlite.OperationalError as exc:
            logger.warning("FTS5 MATCH failed (%s) — LIKE fallback", exc)
            docs = await search_documents_full_text(db_path, user_id, query, limit)
            return [(d, 0.5) for d in docs]

    out: list[tuple[Document, float]] = []
    for row in rows:
        doc = _row_to_document(row[:-1])
        if doc is None:
            continue
        rank = float(row[-1]) if row[-1] is not None else 0.0
        relevance = 1.0 / (1.0 + abs(rank))
        out.append((doc, relevance))
    return out


# ---------------------------------------------------------------------------
# Tasks CRUD
# ---------------------------------------------------------------------------


_TASK_COLUMNS = (
    "id, user_id, title, deadline, priority, status, category, "
    "created_at, updated_at"
)


_TASK_UPDATABLE: frozenset[str] = frozenset(
    {"title", "deadline", "priority", "status", "category"}
)


def _row_to_task(row: Optional[aiosqlite.Row]) -> Optional[Task]:
    if row is None:
        return None

    def _parse_dt(raw):
        if raw is None:
            return None
        try:
            return datetime.fromisoformat(str(raw).replace(" ", "T"))
        except (TypeError, ValueError):
            return None

    return Task(
        id=row[0],
        user_id=row[1],
        title=row[2],
        deadline=_parse_dt(row[3]),
        priority=row[4],
        status=row[5],
        category=row[6],
        created_at=_parse_dt(row[7]),
        updated_at=_parse_dt(row[8]),
    )


async def save_task(db_path: str, task: Task) -> int:
    """Insert ``task``; return new id."""
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            """
            INSERT INTO tasks (
                user_id, title, deadline, priority, status, category
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                task.user_id,
                task.title,
                task.deadline.isoformat() if task.deadline else None,
                task.priority,
                task.status,
                task.category,
            ),
        )
        await db.commit()
        return cur.lastrowid


async def get_task(db_path: str, task_id: int) -> Optional[Task]:
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            f"SELECT {_TASK_COLUMNS} FROM tasks WHERE id = ?", (task_id,)
        )
        return _row_to_task(await cur.fetchone())


async def list_tasks(
    db_path: str,
    user_id: int,
    *,
    status: Optional[str] = "todo",
    category: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 100,
) -> list[Task]:
    """List tasks for a user with optional filters.

    ``status=None`` means "any status" (used by ``/tasks_all``).
    Default ``status="todo"`` matches ``/tasks`` (active list).
    """
    where = ["user_id = ?"]
    params: list[Any] = [user_id]
    if status is not None:
        where.append("status = ?")
        params.append(status)
    if category is not None:
        where.append("category = ?")
        params.append(category)
    if priority is not None:
        where.append("priority = ?")
        params.append(priority)

    sql = (
        f"SELECT {_TASK_COLUMNS} FROM tasks "
        f"WHERE {' AND '.join(where)} "
        # Order: open tasks by deadline (NULLs last), done tasks by updated_at desc.
        "ORDER BY (deadline IS NULL), deadline ASC, updated_at DESC "
        "LIMIT ?"
    )
    params.append(limit)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(sql, params)
        rows = await cur.fetchall()
        return [t for t in (_row_to_task(r) for r in rows) if t is not None]


async def update_task_fields(
    db_path: str, task_id: int, fields: dict[str, Any]
) -> Optional[Task]:
    """Update whitelisted fields and return the refreshed task.

    Whitelist: ``title``, ``deadline``, ``priority``, ``status``, ``category``.
    Pass ``deadline`` as ``datetime`` or ``None``; it is normalised to ISO.
    """
    if not fields:
        return await get_task(db_path, task_id)

    bad = set(fields.keys()) - _TASK_UPDATABLE
    if bad:
        raise ValueError(f"non-updatable task fields: {sorted(bad)}")

    coerced: dict[str, Any] = {}
    for k, v in fields.items():
        if k == "deadline":
            coerced[k] = v.isoformat() if isinstance(v, datetime) else v
        else:
            coerced[k] = v

    set_clause = ", ".join(f"{k} = ?" for k in coerced)
    params = list(coerced.values()) + [task_id]
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            f"UPDATE tasks SET {set_clause}, updated_at = datetime('now') "
            "WHERE id = ?",
            params,
        )
        await db.commit()
    return await get_task(db_path, task_id)


async def delete_task(db_path: str, task_id: int) -> Optional[Task]:
    """Fetch-then-delete. Returns the Task that was removed (or None)."""
    task = await get_task(db_path, task_id)
    if task is None:
        return None
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        await db.commit()
    return task


async def mark_task_done(db_path: str, task_id: int) -> Optional[Task]:
    return await update_task_fields(db_path, task_id, {"status": "done"})


async def mark_task_todo(db_path: str, task_id: int) -> Optional[Task]:
    return await update_task_fields(db_path, task_id, {"status": "todo"})


async def list_tasks_by_category(
    db_path: str, user_id: int, category: str, *, status: Optional[str] = "todo"
) -> list[Task]:
    return await list_tasks(db_path, user_id, status=status, category=category)


async def list_tasks_by_priority(
    db_path: str, user_id: int, priority: str, *, status: Optional[str] = "todo"
) -> list[Task]:
    return await list_tasks(db_path, user_id, status=status, priority=priority)


# ---------------------------------------------------------------------------
# Subscriptions CRUD
# ---------------------------------------------------------------------------


_SUB_COLUMNS = (
    "id, user_id, service, amount, currency, card, recurrence, "
    "day_of_month, next_charge_at, notify_before_days, created_at"
)


_SUB_UPDATABLE: frozenset[str] = frozenset(
    {
        "service", "amount", "currency", "card", "recurrence",
        "day_of_month", "next_charge_at", "notify_before_days",
    }
)


def _row_to_subscription(row: Optional[aiosqlite.Row]) -> Optional[Subscription]:
    if row is None:
        return None

    def _parse_dt(raw):
        if raw is None:
            return None
        try:
            return datetime.fromisoformat(str(raw).replace(" ", "T"))
        except (TypeError, ValueError):
            return None

    return Subscription(
        id=row[0],
        user_id=row[1],
        service=row[2],
        amount=float(row[3]),
        currency=row[4],
        card=row[5],
        recurrence=row[6],
        day_of_month=row[7],
        next_charge_at=_parse_dt(row[8]),
        notify_before_days=int(row[9]),
        created_at=_parse_dt(row[10]),
    )


async def save_subscription(db_path: str, sub: Subscription) -> int:
    """Insert ``sub``; return new id."""
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            """
            INSERT INTO subscriptions (
                user_id, service, amount, currency, card, recurrence,
                day_of_month, next_charge_at, notify_before_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sub.user_id,
                sub.service,
                sub.amount,
                sub.currency,
                sub.card,
                sub.recurrence,
                sub.day_of_month,
                sub.next_charge_at.isoformat() if sub.next_charge_at else None,
                sub.notify_before_days,
            ),
        )
        await db.commit()
        return cur.lastrowid


async def get_subscription(db_path: str, sub_id: int) -> Optional[Subscription]:
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            f"SELECT {_SUB_COLUMNS} FROM subscriptions WHERE id = ?", (sub_id,)
        )
        return _row_to_subscription(await cur.fetchone())


async def list_subscriptions(
    db_path: str, user_id: int, *, limit: int = 100
) -> list[Subscription]:
    """List user's subscriptions ordered by upcoming next_charge_at (NULLs last)."""
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            f"SELECT {_SUB_COLUMNS} FROM subscriptions "
            "WHERE user_id = ? "
            "ORDER BY (next_charge_at IS NULL), next_charge_at ASC "
            "LIMIT ?",
            (user_id, limit),
        )
        rows = await cur.fetchall()
        return [s for s in (_row_to_subscription(r) for r in rows) if s is not None]


async def update_subscription_fields(
    db_path: str, sub_id: int, fields: dict[str, Any]
) -> Optional[Subscription]:
    """Update whitelisted fields. Returns refreshed subscription."""
    if not fields:
        return await get_subscription(db_path, sub_id)

    bad = set(fields.keys()) - _SUB_UPDATABLE
    if bad:
        raise ValueError(f"non-updatable subscription fields: {sorted(bad)}")

    coerced: dict[str, Any] = {}
    for k, v in fields.items():
        if k == "next_charge_at":
            coerced[k] = v.isoformat() if isinstance(v, datetime) else v
        else:
            coerced[k] = v

    set_clause = ", ".join(f"{k} = ?" for k in coerced)
    params = list(coerced.values()) + [sub_id]
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            f"UPDATE subscriptions SET {set_clause} WHERE id = ?",
            params,
        )
        await db.commit()
    return await get_subscription(db_path, sub_id)


async def delete_subscription(db_path: str, sub_id: int) -> Optional[Subscription]:
    """Fetch-then-delete. Returns the Subscription that was removed (or None)."""
    sub = await get_subscription(db_path, sub_id)
    if sub is None:
        return None
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM subscriptions WHERE id = ?", (sub_id,))
        await db.commit()
    return sub


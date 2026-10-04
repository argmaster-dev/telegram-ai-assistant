"""Tests for SQLite FTS5 virtual table + triggers + full_text_search.

These check:
 1. ``init_db`` creates ``documents_fts`` and flips ``is_fts5_available`` → True
    (on a sqlite3 build with FTS5 compiled in).
 2. INSERT/UPDATE/DELETE triggers keep the FTS5 index in sync.
 3. ``full_text_search`` returns relevance-ranked Documents, not LIKE hits.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

from services import storage
from services.intents import Document
from services.storage import (
    add_tags,
    delete_document,
    find_by_tag,
    find_document_by_id,
    full_text_search,
    init_db,
    is_fts5_available,
    save_document,
    update_document_ocr,
)


@pytest_asyncio.fixture()
async def db_path():
    with tempfile.TemporaryDirectory() as tmp:
        p = str(Path(tmp) / "fts.db")
        await init_db(p)
        yield p


@pytest.mark.asyncio
async def test_fts5_is_available_after_init_db(db_path):
    """Require FTS5 in the sqlite3 build used by the docker image."""
    assert is_fts5_available(), (
        "FTS5 is expected to be compiled into sqlite3 in the pytorch docker image"
    )


@pytest.mark.asyncio
async def test_insert_trigger_populates_fts_index(db_path):
    doc_id = await save_document(
        db_path,
        Document(
            user_id=1, type="passport", has_pii=True,
            storage_mode="redacted_indexed",
            original_path="/tmp/x.bin", original_sha256="sha-fts-1",
            ocr_text="Паспорт Российской Федерации, выдан МВД.",
            redacted_text=None,
        ),
    )
    hits = await full_text_search(db_path, user_id=1, query="паспорт")
    assert any(d.id == doc_id for d, _score in hits), (
        f"INSERT trigger did not sync FTS: {[(d.id, s) for d, s in hits]}"
    )


@pytest.mark.asyncio
async def test_update_trigger_reflects_new_text(db_path):
    doc_id = await save_document(
        db_path,
        Document(
            user_id=2, type="other", has_pii=False,
            storage_mode="redacted_indexed",
            original_path="/tmp/y.bin", original_sha256="sha-fts-2",
            ocr_text="первоначальный текст документа",
        ),
    )
    # UPDATE (via update_document_ocr) must retrigger FTS sync.
    await update_document_ocr(
        db_path, doc_id,
        ocr_text="совершенно новое содержание — фотокнига CLEVER",
        redacted_text=None,
    )

    # Old content should NOT match; new content should.
    assert not await full_text_search(db_path, 2, "первоначальный")
    new_hits = await full_text_search(db_path, 2, "фотокнига")
    assert any(d.id == doc_id for d, _ in new_hits)


@pytest.mark.asyncio
async def test_delete_trigger_removes_from_fts(db_path):
    doc_id = await save_document(
        db_path,
        Document(
            user_id=3, type="other", has_pii=False,
            storage_mode="redacted_indexed",
            original_path="/tmp/z.bin", original_sha256="sha-fts-3",
            ocr_text="удалимый документ с уникальным маркером FTSDELETEKEY",
        ),
    )
    assert await full_text_search(db_path, 3, "FTSDELETEKEY")

    deleted = await delete_document(db_path, doc_id)
    assert deleted is not None and deleted.id == doc_id

    hits = await full_text_search(db_path, 3, "FTSDELETEKEY")
    assert hits == [], f"FTS still returns deleted doc: {hits}"


@pytest.mark.asyncio
async def test_full_text_search_uses_match_not_like(db_path):
    """Sanity: the MATCH path returns rank scores, not hardcoded 0.5 (which is the LIKE fallback)."""
    await save_document(
        db_path,
        Document(
            user_id=5, type="contract", has_pii=False,
            storage_mode="redacted_indexed",
            original_path="/tmp/c.bin", original_sha256="sha-fts-5",
            ocr_text="договор аренды квартиры между сторонами Иванов и Петров",
        ),
    )
    hits = await full_text_search(db_path, 5, "договор")
    assert hits, "FTS returned no results for a word actually in ocr_text"
    score = hits[0][1]
    # FTS MATCH gives 1/(1+|rank|) — will never be exactly 0.5; LIKE-fallback would.
    assert score != 0.5, "got the LIKE-fallback score; expected real FTS rank"


@pytest.mark.asyncio
async def test_explain_query_plan_uses_fts_virtual_index(db_path):
    """Direct EXPLAIN probe — verify MATCH path hits the FTS5 virtual index."""
    await save_document(
        db_path,
        Document(
            user_id=6, type="other", has_pii=False,
            storage_mode="redacted_indexed",
            original_path="/tmp/q.bin", original_sha256="sha-fts-6",
            ocr_text="exploration test for EXPLAIN QUERY PLAN",
        ),
    )
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT d.id FROM documents_fts "
            "JOIN documents d ON d.id = documents_fts.rowid "
            "WHERE documents_fts MATCH ?",
            ("exploration",),
        )
        plan = await cur.fetchall()
    plan_text = "\n".join(str(row) for row in plan)
    assert "documents_fts" in plan_text.lower(), (
        f"EXPLAIN QUERY PLAN did not reference the FTS virtual table:\n{plan_text}"
    )

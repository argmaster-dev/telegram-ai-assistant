"""Tests for /docs_find merge ranking (FTS + Qdrant).

Qdrant is mocked — we're testing the merge/label/scoring logic, not the
vector backend (that's covered by ``test_qdrant_store.py``).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from handlers.docs_handler import _search_merged
from services.intents import Document
from services.storage import init_db, save_document


@pytest_asyncio.fixture()
async def db_path():
    with tempfile.TemporaryDirectory() as tmp:
        p = str(Path(tmp) / "merge.db")
        await init_db(p)
        yield p


async def _seed(db_path: str, *, sha: str, ocr: str, user_id: int = 1) -> int:
    return await save_document(
        db_path,
        Document(
            user_id=user_id, type="other", has_pii=False,
            storage_mode="redacted_indexed",
            original_path="/tmp/x.bin", original_sha256=sha,
            ocr_text=ocr,
        ),
    )


@pytest.mark.asyncio
async def test_merge_fts_only_when_qdrant_empty(db_path):
    id_a = await _seed(db_path, sha="m-a", ocr="Паспорт РФ выдан в 2014 году")
    id_b = await _seed(db_path, sha="m-b", ocr="Договор аренды")

    mock_search = AsyncMock(return_value=[])
    with patch("handlers.docs_handler.qdrant_store.search", mock_search):
        hits = await _search_merged(
            db_path, "http://nope", user_id=1, query="паспорт", limit=10
        )

    assert len(hits) == 1
    assert hits[0]["doc"].id == id_a
    assert hits[0]["source"] == "fts"
    assert hits[0]["fts_score"] > 0
    assert hits[0]["qdrant_score"] == 0


@pytest.mark.asyncio
async def test_merge_labels_both_when_doc_in_fts_and_qdrant(db_path):
    doc_id = await _seed(db_path, sha="m-both", ocr="Договор аренды квартиры")

    mock_search = AsyncMock(
        return_value=[
            {"id": "p-1", "score": 0.8, "payload": {"doc_id": doc_id, "user_id": 1}},
        ]
    )
    with patch("handlers.docs_handler.qdrant_store.search", mock_search):
        hits = await _search_merged(
            db_path, "http://nope", user_id=1, query="договор", limit=10
        )

    assert len(hits) == 1
    h = hits[0]
    assert h["doc"].id == doc_id
    assert h["source"] == "both"
    assert h["fts_score"] > 0
    assert h["qdrant_score"] == pytest.approx(0.8)
    assert h["score"] > 0.8  # sum of fts + qdrant


@pytest.mark.asyncio
async def test_merge_qdrant_only_when_fts_misses(db_path):
    """Docs only in Qdrant (e.g. embedded after FTS text trimmed) still surface."""
    doc_id = await _seed(db_path, sha="m-q", ocr="abcdef", user_id=1)

    mock_search = AsyncMock(
        return_value=[
            {"id": "p-1", "score": 0.7, "payload": {"doc_id": doc_id, "user_id": 1}},
        ]
    )
    with patch("handlers.docs_handler.qdrant_store.search", mock_search):
        hits = await _search_merged(
            db_path, "http://nope", user_id=1, query="нетвтексте", limit=10
        )

    assert len(hits) == 1
    assert hits[0]["source"] == "qdrant"
    assert hits[0]["fts_score"] == 0
    assert hits[0]["qdrant_score"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_merge_handles_qdrant_exception_gracefully(db_path):
    await _seed(db_path, sha="m-resil", ocr="текст про налоги")

    async def _boom(*_a, **_kw):
        raise RuntimeError("qdrant down")

    with patch("handlers.docs_handler.qdrant_store.search", side_effect=_boom):
        # FTS5 unicode61 doesn't do stemming — use exact token from the text.
        hits = await _search_merged(
            db_path, "http://nope", user_id=1, query="налоги", limit=10
        )

    # FTS still returns result despite Qdrant being down.
    assert len(hits) == 1
    assert hits[0]["source"] == "fts"


@pytest.mark.asyncio
async def test_merge_empty_query_returns_empty(db_path):
    hits = await _search_merged(
        db_path, "http://nope", user_id=1, query="", limit=10
    )
    assert hits == []

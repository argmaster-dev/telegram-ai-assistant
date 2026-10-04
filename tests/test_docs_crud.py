"""Tests for document CRUD: update, delete, tags, find_by_tag."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from services.intents import Document
from services.storage import (
    add_tags,
    delete_document,
    find_by_tag,
    find_document_by_id,
    init_db,
    remove_tags,
    save_document,
    update_document,
)


@pytest_asyncio.fixture()
async def db_path():
    with tempfile.TemporaryDirectory() as tmp:
        p = str(Path(tmp) / "crud.db")
        await init_db(p)
        yield p


async def _make_doc(db_path: str, *, sha: str = "sha-crud-1", **kw) -> int:
    base = dict(
        user_id=1, type="other", has_pii=False,
        storage_mode="local_only",
        original_path="/tmp/none", original_sha256=sha,
    )
    base.update(kw)
    return await save_document(db_path, Document(**base))


@pytest.mark.asyncio
async def test_update_document_whitelisted_fields(db_path):
    doc_id = await _make_doc(db_path)
    updated = await update_document(
        db_path, doc_id,
        {"type": "passport", "has_pii": True, "storage_mode": "redacted_indexed"},
    )
    assert updated is not None
    assert updated.type == "passport"
    assert updated.has_pii is True
    assert updated.storage_mode == "redacted_indexed"


@pytest.mark.asyncio
async def test_update_document_rejects_non_whitelisted_field(db_path):
    doc_id = await _make_doc(db_path, sha="sha-crud-reject")
    with pytest.raises(ValueError, match="non-updatable"):
        await update_document(db_path, doc_id, {"ocr_text": "hacked"})


@pytest.mark.asyncio
async def test_delete_document_returns_removed_and_erases_row(db_path):
    doc_id = await _make_doc(db_path, sha="sha-del-1")
    deleted = await delete_document(db_path, doc_id)
    assert deleted is not None
    assert deleted.id == doc_id
    assert await find_document_by_id(db_path, doc_id) is None
    # Second delete is a no-op (returns None), not an error.
    assert await delete_document(db_path, doc_id) is None


@pytest.mark.asyncio
async def test_add_tags_dedupes_and_preserves_order(db_path):
    doc_id = await _make_doc(db_path, sha="sha-tag-1")
    first = await add_tags(db_path, doc_id, ["важное", "семья"])
    assert first == ["важное", "семья"]
    # Adding with duplicates + new one.
    second = await add_tags(db_path, doc_id, ["важное", "работа"])
    assert second == ["важное", "семья", "работа"]


@pytest.mark.asyncio
async def test_add_tags_rejects_bad_tags(db_path):
    doc_id = await _make_doc(db_path, sha="sha-tag-bad")
    with pytest.raises(ValueError, match="comma"):
        await add_tags(db_path, doc_id, ["two,commas"])
    with pytest.raises(ValueError, match="empty"):
        await add_tags(db_path, doc_id, [""])


@pytest.mark.asyncio
async def test_remove_tags(db_path):
    doc_id = await _make_doc(db_path, sha="sha-tag-rm")
    await add_tags(db_path, doc_id, ["a", "b", "c"])
    remaining = await remove_tags(db_path, doc_id, ["b"])
    assert remaining == ["a", "c"]


@pytest.mark.asyncio
async def test_find_by_tag_exact_match(db_path):
    id_a = await _make_doc(db_path, sha="sha-tag-find-a")
    id_b = await _make_doc(db_path, sha="sha-tag-find-b")
    id_c = await _make_doc(db_path, sha="sha-tag-find-c")
    await add_tags(db_path, id_a, ["ВАЖ"])
    await add_tags(db_path, id_b, ["ВАЖНОЕ"])  # different tag
    await add_tags(db_path, id_c, ["ВАЖ", "семья"])

    hits = await find_by_tag(db_path, user_id=1, tag="ВАЖ")
    found_ids = {d.id for d in hits}
    assert found_ids == {id_a, id_c}, f"expected exact match only: got {found_ids}"

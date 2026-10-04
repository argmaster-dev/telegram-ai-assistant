"""Integration tests for services.qdrant_store — real tg_qdrant container.

Uses a dedicated test collection ``tg_assist_documents_test`` so the prod
collection is never touched. Each test function gets a clean collection
through the ``_fresh_collection`` fixture and tears it down afterwards.

Expects ``tg_qdrant`` reachable on the docker-compose internal network via
``http://tg_qdrant:6333``.
"""
from __future__ import annotations

import os

import pytest
import pytest_asyncio

from services import qdrant_store


_TEST_COLLECTION = "tg_assist_documents_test"
_QDRANT_URL = os.environ.get("QDRANT_URL", "http://tg_qdrant:6333")


@pytest_asyncio.fixture()
async def _fresh_collection():
    """Ensure a clean test collection exists for every test."""
    # Defensive: reset any stale client from a previous session.
    await qdrant_store.reset_client_for_tests()
    await qdrant_store.drop_collection(_QDRANT_URL, collection=_TEST_COLLECTION)
    await qdrant_store.ensure_collection(_QDRANT_URL, collection=_TEST_COLLECTION)
    try:
        yield _TEST_COLLECTION
    finally:
        await qdrant_store.drop_collection(_QDRANT_URL, collection=_TEST_COLLECTION)


@pytest.mark.asyncio
async def test_ensure_collection_is_idempotent(_fresh_collection):
    # Second ensure must not raise (no-op when collection already exists).
    await qdrant_store.ensure_collection(_QDRANT_URL, collection=_fresh_collection)


@pytest.mark.asyncio
async def test_upsert_and_search_roundtrip(_fresh_collection):
    text = (
        "Иванов Иван Иванович подал заявление о выдаче "
        "нового паспорта взамен утраченного."
    )
    points_written = await qdrant_store.upsert(
        _QDRANT_URL,
        doc_id=42,
        text=text,
        metadata={"user_id": 999, "source": "test.pdf"},
        collection=_fresh_collection,
    )
    assert points_written >= 1

    hits = await qdrant_store.search(
        _QDRANT_URL,
        query="заявление о паспорте",
        top_k=3,
        filter_={"user_id": 999},
        collection=_fresh_collection,
    )
    assert len(hits) >= 1
    top = hits[0]
    assert top["payload"].get("doc_id") == 42
    assert top["payload"].get("user_id") == 999


@pytest.mark.asyncio
async def test_delete_by_doc_id_removes_all_points(_fresh_collection):
    await qdrant_store.upsert(
        _QDRANT_URL,
        doc_id=7,
        text="тестовый документ с разными предложениями. " * 50,
        metadata={"user_id": 1},
        collection=_fresh_collection,
    )
    # Sanity: at least one point exists.
    hits_before = await qdrant_store.search(
        _QDRANT_URL,
        query="документ",
        top_k=10,
        filter_={"doc_id": 7},
        collection=_fresh_collection,
    )
    assert len(hits_before) >= 1

    await qdrant_store.delete(_QDRANT_URL, doc_id=7, collection=_fresh_collection)

    hits_after = await qdrant_store.search(
        _QDRANT_URL,
        query="документ",
        top_k=10,
        filter_={"doc_id": 7},
        collection=_fresh_collection,
    )
    assert hits_after == []

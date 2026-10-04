"""Qdrant-backed vector store for tg_assist documents.

One collection (``tg_assist_documents``, 1024D cosine) holds every
``redacted_indexed`` document's parent + child chunks. A single
``AsyncQdrantClient`` is lazily instantiated once per process.

Dependencies:
  * ``services.embeddings.embed_texts`` — turns text into vectors.
  * ``services.chunker.ParentChildChunker`` — splits OCR text into parent +
    child dicts that become Qdrant points.

All public functions take ``url`` as a parameter (taken from
``Settings.QDRANT_URL``) so tests can point at an alternate collection name
via the ``collection`` keyword.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient, models

from services.chunker import ParentChildChunker
from services.embeddings import embed_texts, embedding_dimension


logger = logging.getLogger(__name__)


DEFAULT_COLLECTION = "tg_assist_documents"


_client: Optional[AsyncQdrantClient] = None
_client_lock = asyncio.Lock()


async def _get_client(url: str) -> AsyncQdrantClient:
    """Return a singleton ``AsyncQdrantClient`` bound to ``url``."""
    global _client
    if _client is not None:
        return _client
    async with _client_lock:
        if _client is None:
            _client = AsyncQdrantClient(url=url, timeout=30.0)
            logger.info("AsyncQdrantClient created (url=%s)", url)
    return _client


async def reset_client_for_tests() -> None:
    """Drop the cached client — used by test fixtures after teardown."""
    global _client
    if _client is not None:
        try:
            await _client.close()
        except Exception:  # noqa: BLE001
            pass
    _client = None


async def ensure_collection(
    url: str, collection: str = DEFAULT_COLLECTION
) -> None:
    """Create the collection with ``embedding_dimension()`` dense cosine vectors if missing."""
    client = await _get_client(url)
    exists = await client.collection_exists(collection)
    if exists:
        return
    await client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )
    logger.info("created qdrant collection %r", collection)


async def upsert(
    url: str,
    doc_id: int,
    text: str,
    metadata: Optional[dict] = None,
    collection: str = DEFAULT_COLLECTION,
) -> int:
    """Chunk ``text`` (parent + child), embed, upsert into Qdrant.

    ``metadata`` is merged into every point payload so filters like
    ``{"user_id": X}`` work in ``search``.
    Returns the number of points written.
    """
    if not text or not text.strip():
        return 0
    chunks = ParentChildChunker().chunk_text(text, source=(metadata or {}).get("source", ""))
    if not chunks:
        return 0

    vecs = await embed_texts([c["content"] for c in chunks])
    if len(vecs) != len(chunks):
        raise RuntimeError(
            f"embed returned {len(vecs)} vectors for {len(chunks)} chunks"
        )

    base_meta = dict(metadata or {})
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={**base_meta, **chunk, "doc_id": doc_id},
        )
        for chunk, vec in zip(chunks, vecs)
    ]

    client = await _get_client(url)
    await client.upsert(collection_name=collection, points=points)
    return len(points)


async def search(
    url: str,
    query: str,
    top_k: int = 5,
    filter_: Optional[dict] = None,
    collection: str = DEFAULT_COLLECTION,
) -> list[dict]:
    """Return ``top_k`` point payloads for the query text.

    ``filter_`` is a simple dict ``{"payload_key": value}`` — translated to
    a Qdrant ``must`` filter on exact match. For more complex filters a
    caller can reach the client via ``_get_client`` directly (not recommended).
    """
    if not query.strip():
        return []

    vecs = await embed_texts([query])
    client = await _get_client(url)
    result = await client.query_points(
        collection_name=collection,
        query=vecs[0],
        limit=top_k,
        query_filter=_build_filter(filter_),
        with_payload=True,
    )
    return [
        {"id": p.id, "score": float(p.score), "payload": p.payload or {}}
        for p in result.points
    ]


async def delete(
    url: str, doc_id: int, collection: str = DEFAULT_COLLECTION
) -> None:
    """Delete every point whose payload.doc_id == ``doc_id``."""
    client = await _get_client(url)
    await client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            )
        ),
    )


async def drop_collection(
    url: str, collection: str = DEFAULT_COLLECTION
) -> None:
    """Drop the whole collection — only intended for tests / rebuild scripts."""
    client = await _get_client(url)
    if await client.collection_exists(collection):
        await client.delete_collection(collection_name=collection)


def _build_filter(filter_: Optional[dict]) -> Optional[models.Filter]:
    """Build a Qdrant ``must``-filter from a simple dict.

    Supports:
      * scalar value → ``MatchValue`` (exact)  — ``{"doc_id": 42}``
      * list value   → ``MatchAny``            — ``{"type": ["passport", "snils"]}``
      * bool value   → ``MatchValue``          — ``{"has_pii": True}`` (Qdrant stores bool natively)
    """
    if not filter_:
        return None
    conds = []
    for k, v in filter_.items():
        if isinstance(v, (list, tuple, set)):
            conds.append(
                models.FieldCondition(key=k, match=models.MatchAny(any=list(v)))
            )
        else:
            conds.append(
                models.FieldCondition(key=k, match=models.MatchValue(value=v))
            )
    return models.Filter(must=conds)


# Alias required by the contract §1.2 — same semantics as ``delete``.
delete_point = delete


__all__ = [
    "DEFAULT_COLLECTION",
    "ensure_collection",
    "upsert",
    "search",
    "delete",
    "delete_point",
    "drop_collection",
    "reset_client_for_tests",
]

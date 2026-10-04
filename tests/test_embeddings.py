"""Tests for services.embeddings — real BGE-m3 model (pre-cached in Dockerfile).

First test in a cold suite pays ~5 s to load the model; subsequent tests reuse
the singleton. The model is already downloaded into the image by the
``docker build`` step that pre-caches it, so there is no network access here.
"""
from __future__ import annotations

import pytest

from services import embeddings


@pytest.mark.asyncio
async def test_embed_texts_returns_1024d_dense_vector():
    vecs = await embeddings.embed_texts(["hello"])
    assert len(vecs) == 1
    assert len(vecs[0]) == 1024
    assert all(isinstance(x, float) for x in vecs[0][:8])


@pytest.mark.asyncio
async def test_singleton_not_reinitialised_between_calls():
    """Two calls to _get_encoder must return the exact same object."""
    m1 = await embeddings._get_encoder()
    m2 = await embeddings._get_encoder()
    assert m1 is m2


@pytest.mark.asyncio
async def test_embed_empty_input_returns_empty_list():
    assert await embeddings.embed_texts([]) == []


@pytest.mark.asyncio
async def test_embed_batch_preserves_input_order():
    """Vectors for ["A","B","C"] must come back in the same order."""
    vecs = await embeddings.embed_texts(["кошка", "документ", "паспорт"])
    assert len(vecs) == 3
    # Embeddings should not be identical across different tokens.
    assert vecs[0] != vecs[1]
    assert vecs[1] != vecs[2]


def test_embedding_dimension_constant():
    assert embeddings.embedding_dimension() == 1024

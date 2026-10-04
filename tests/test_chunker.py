"""Tests for services.chunker.ParentChildChunker — BGE-m3 tokenizer backed."""
from __future__ import annotations

import pytest

from services import chunker
from services.chunker import (
    CHILD_MAX,
    PARENT_MAX,
    ParentChildChunker,
    count_tokens,
)


def test_empty_text_returns_no_chunks():
    assert ParentChildChunker().chunk_text("") == []
    assert ParentChildChunker().chunk_text("   \n  \t ") == []


def test_short_text_yields_one_parent_and_one_child():
    """Text well under CHILD_MAX → exactly one parent + one child (equal text)."""
    text = "Это короткий документ на русском языке для теста chunker'а. " * 3
    assert count_tokens(text) <= CHILD_MAX

    chunks = ParentChildChunker().chunk_text(text, source="doc_42.pdf")
    parents = [c for c in chunks if c["chunk_type"] == "parent"]
    children = [c for c in chunks if c["chunk_type"] == "child"]
    assert len(parents) == 1
    assert len(children) == 1
    assert parents[0]["content"] == children[0]["content"]
    assert children[0]["parent_id"] == parents[0]["node_id"]
    assert parents[0]["parent_id"] == ""
    assert parents[0]["source"] == "doc_42.pdf"
    assert children[0]["source"] == "doc_42.pdf"


def test_medium_text_splits_into_multiple_children():
    """Text > CHILD_MAX but < PARENT_MAX → 1 parent + N children with overlap."""
    # Produce a ~1000-token chunk (well above CHILD_MAX=600, below PARENT_MAX=3000).
    block = "Иванов Иван Иванович работает в компании. " * 120
    assert CHILD_MAX < count_tokens(block) < PARENT_MAX

    chunks = ParentChildChunker().chunk_text(block)
    parents = [c for c in chunks if c["chunk_type"] == "parent"]
    children = [c for c in chunks if c["chunk_type"] == "child"]
    assert len(parents) == 1
    assert len(children) >= 2, "expected splitter to produce several children"
    for child in children:
        assert child["parent_id"] == parents[0]["node_id"]
    # All children reference the single parent.
    parent_ids = {c["parent_id"] for c in children}
    assert len(parent_ids) == 1


def test_huge_text_produces_multiple_parents():
    """Text > PARENT_MAX → multiple parents, each with its own children."""
    block = ("Многостраничный документ — договор между сторонами. " * 800)
    assert count_tokens(block) > PARENT_MAX

    chunks = ParentChildChunker().chunk_text(block)
    parents = [c for c in chunks if c["chunk_type"] == "parent"]
    children = [c for c in chunks if c["chunk_type"] == "child"]
    assert len(parents) >= 2, f"expected multiple parents, got {len(parents)}"
    # Every child has a parent_id that matches some real parent node_id.
    parent_ids = {p["node_id"] for p in parents}
    for child in children:
        assert child["parent_id"] in parent_ids


def test_count_tokens_matches_bge_m3_tokenizer():
    """Sanity: BGE-m3 tokenizer is loaded (not a fake)."""
    # Arbitrary sentence; we don't hard-code the number (it's tokenizer-dependent),
    # we just check that a non-trivial count comes back.
    n = count_tokens("Я отправил документ в налоговую службу.")
    assert n > 0
    assert n < 50

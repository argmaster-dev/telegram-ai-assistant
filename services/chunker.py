"""Parent-child text chunker for tg_assist documents.

Chunker adapted for BGE-m3 with the
Domain-specific logic stripped out (no section detection, no
``node_builder`` dependency, no table handling, no summary chunks).

Two-level hierarchy:
  * **parent chunks** (~2000 tokens, hard cap 3000) — larger passages used to
    return context around a matched child.
  * **child chunks** (~400 tokens, cap 600, overlap 50) — the points actually
    embedded and written to Qdrant for semantic search.

Tokens are counted via the BGE-m3 tokenizer (``BAAI/bge-m3``) — same model we
embed with, so the token budget in the chunker matches what the model sees.

Chunk metadata (per dict):
  * ``content`` — the text itself
  * ``chunk_type`` — ``"parent"`` or ``"child"``
  * ``node_id`` — short deterministic string for dedup / trace
  * ``parent_id`` — for children: the parent's ``node_id``; for parents: ``""``
  * ``source`` — free-form tag (e.g. the original filename), propagated from caller
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BGE-m3 tokenizer — lazy singleton
# ---------------------------------------------------------------------------


_tokenizer = None


def _get_tokenizer():
    """Lazy import and load the BGE-m3 tokenizer once per process."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count BGE-m3 tokens (without special tokens) in ``text``."""
    tok = _get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Constants (calibrated for BGE-m3)
# ---------------------------------------------------------------------------


PARENT_TARGET = 2000   # tokens — desired parent size
PARENT_MAX = 3000      # tokens — split parent when exceeded
CHILD_TARGET = 400     # tokens — desired child size
CHILD_MAX = 600        # tokens — single child if parent is this small
CHILD_OVERLAP = 50     # tokens — overlap between sibling children

# Approximate chars-per-token for Russian text with BGE-m3 — used by
# RecursiveCharacterTextSplitter (which works in chars, not tokens).
_CHARS_PER_TOKEN = 2.5

_SEPARATORS = ["\n\n", "\n", ". ", "; ", " "]


# ---------------------------------------------------------------------------
# ParentChildChunker
# ---------------------------------------------------------------------------


class ParentChildChunker:
    """Two-level text chunker producing parent and child dicts.

    Usage:

        chunker = ParentChildChunker()
        chunks = chunker.chunk_text(ocr_output_text, source="doc_42.pdf")
        # chunks: list of dicts with chunk_type, node_id, parent_id, content, source
    """

    def __init__(
        self,
        parent_target: int = PARENT_TARGET,
        parent_max: int = PARENT_MAX,
        child_target: int = CHILD_TARGET,
        child_max: int = CHILD_MAX,
        child_overlap: int = CHILD_OVERLAP,
    ) -> None:
        self.parent_target = parent_target
        self.parent_max = parent_max
        self.child_target = child_target
        self.child_max = child_max
        self.child_overlap = child_overlap

        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(parent_target * _CHARS_PER_TOKEN),
            chunk_overlap=int(child_overlap * _CHARS_PER_TOKEN),
            separators=_SEPARATORS,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(child_target * _CHARS_PER_TOKEN),
            chunk_overlap=int(child_overlap * _CHARS_PER_TOKEN),
            separators=_SEPARATORS,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(self, text: str, source: str = "") -> list[dict]:
        """Chunk a plain text blob into a list of parent + child dicts.

        Empty or whitespace-only ``text`` → empty list.
        """
        if not text or not text.strip():
            return []

        parents = self._build_parents(text, source)

        all_chunks: list[dict] = []
        for parent in parents:
            all_chunks.append(parent)
            all_chunks.extend(self._split_into_children(parent, source))
        return all_chunks

    # ------------------------------------------------------------------
    # Parent-level split
    # ------------------------------------------------------------------

    def _build_parents(self, text: str, source: str) -> list[dict]:
        tokens = count_tokens(text)
        if tokens <= self.parent_max:
            return [self._make_parent(text, source)]
        sub_texts = self._parent_splitter.split_text(text)
        return [self._make_parent(t, source) for t in sub_texts if t.strip()]

    def _make_parent(self, text: str, source: str) -> dict:
        return {
            "content": text,
            "chunk_type": "parent",
            "node_id": _short_node_id("p"),
            "parent_id": "",
            "source": source,
        }

    # ------------------------------------------------------------------
    # Child-level split
    # ------------------------------------------------------------------

    def _split_into_children(self, parent: dict, source: str) -> list[dict]:
        text = parent["content"]
        tokens = count_tokens(text)

        if tokens <= self.child_max:
            # Small parent: single child == parent content (verbatim).
            return [self._make_child(text, parent, source, seq=0)]

        sub_texts = self._child_splitter.split_text(text)
        children: list[dict] = []
        for i, sub_text in enumerate(sub_texts):
            if not sub_text.strip():
                continue
            children.append(self._make_child(sub_text, parent, source, seq=i))
        return children

    @staticmethod
    def _make_child(text: str, parent: dict, source: str, seq: int) -> dict:
        return {
            "content": text,
            "chunk_type": "child",
            "node_id": _short_node_id(f"c{seq}"),
            "parent_id": parent["node_id"],
            "source": source,
        }


def _short_node_id(prefix: str) -> str:
    """Short deterministic-ish node id: ``<prefix>_<8hex>``."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


__all__ = [
    "PARENT_TARGET",
    "PARENT_MAX",
    "CHILD_TARGET",
    "CHILD_MAX",
    "CHILD_OVERLAP",
    "ParentChildChunker",
    "count_tokens",
]

"""BGE-m3 CPU-only embedding singleton.

single entry point ``embed_texts(list[str]) -> list[list[float]]``
used by the worker's ``_run_embed`` stage before Qdrant upsert. The model is
kept on CPU intentionally (§4.3 concept p.1 — only the intent router runs on
GPU). First load pulls ~2.2 GB from HuggingFace if not pre-cached — the
Dockerfile pre-downloads it, so the in-container first call is fast.

Design:
  * Lazy singleton guarded by an ``asyncio.Lock`` — multiple concurrent
    ``embed_texts`` calls wait on the same init.
  * Model load and every ``model.encode`` run in a worker thread via
    ``asyncio.to_thread`` so the bot's event loop stays responsive.
  * Returns dense 1024D vectors only (BGE-m3 also produces sparse + ColBERT,
    but uses dense-only for simplicity — can enable
    sparse/hybrid if needed).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional


logger = logging.getLogger(__name__)


_model: Any = None
_lock = asyncio.Lock()


_MODEL_NAME = "BAAI/bge-m3"
_EMBED_DIM = 1024


async def _get_encoder() -> Any:
    """Return the BGEM3FlagModel singleton, loading it on first call."""
    global _model
    if _model is not None:
        return _model
    async with _lock:
        if _model is None:
            logger.info("loading BGE-m3 on CPU (first call; may take ~5-10 s)")
            _model = await asyncio.to_thread(_load_model)
            logger.info("BGE-m3 ready on CPU")
    return _model


def _load_model() -> Any:
    """Blocking model-load helper (runs inside ``asyncio.to_thread``)."""
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(_MODEL_NAME, use_fp16=False, devices="cpu")


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return dense 1024D embedding per input text.

    Empty input → empty output. The model call happens in a worker thread.
    """
    if not texts:
        return []
    model = await _get_encoder()
    out = await asyncio.to_thread(model.encode, texts)
    dense = out["dense_vecs"]
    # BGE-m3 returns numpy arrays; convert to native Python list[float] so the
    # Qdrant client happily passes them to the HTTP layer as JSON numbers.
    return [list(map(float, v)) for v in dense]


async def warm() -> None:
    """Eagerly load the singleton (used during bot startup pre-warm)."""
    await _get_encoder()


def embedding_dimension() -> int:
    """Return the vector dimension used for Qdrant collection config."""
    return _EMBED_DIM


__all__ = [
    "embed_texts",
    "warm",
    "embedding_dimension",
]

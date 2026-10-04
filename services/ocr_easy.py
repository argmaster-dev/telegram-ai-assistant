"""EasyOCR wrapper — CPU-only async interface with module-level singleton.

The Reader is initialised lazily on first call (model download in cold start
takes ~30s; subsequent calls reuse the singleton). Initialisation and every
``readtext`` happen in a worker thread (``asyncio.to_thread``) so the bot's
event loop stays responsive.

OCR only — no PII, no FSM, no scheduling. Background worker queue will call these functions and own the user-facing latency.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import easyocr
import fitz  # PyMuPDF


logger = logging.getLogger(__name__)


_reader: Optional[easyocr.Reader] = None
_init_lock = asyncio.Lock()


async def _get_reader() -> easyocr.Reader:
    """Lazy singleton accessor for the EasyOCR Reader."""
    global _reader
    if _reader is not None:
        return _reader
    async with _init_lock:
        if _reader is None:
            logger.info(
                "initializing EasyOCR Reader (CPU, ru+en) — first call may "
                "block for several seconds while models load"
            )
            _reader = await asyncio.to_thread(
                easyocr.Reader, ["ru", "en"], gpu=False
            )
            logger.info("EasyOCR Reader ready")
    return _reader


def _readtext_sync(reader: easyocr.Reader, data: bytes) -> str:
    """Blocking helper: run reader.readtext and join results into one string."""
    results = reader.readtext(data, detail=1, paragraph=False)
    return " ".join(text for (_bbox, text, _conf) in results)


async def extract_from_image(data: bytes) -> str:
    """OCR a single raster image (PNG/JPEG bytes). Returns combined text."""
    reader = await _get_reader()
    return await asyncio.to_thread(_readtext_sync, reader, data)


def _render_pdf_pages(
    pdf_bytes: bytes,
    page_indexes: Optional[list[int]] = None,
    dpi: int = 150,
) -> list[bytes]:
    """Render selected (or all) PDF pages to PNG bytes via PyMuPDF."""
    out: list[bytes] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        indexes = (
            page_indexes
            if page_indexes is not None
            else list(range(len(doc)))
        )
        for idx in indexes:
            page = doc.load_page(idx)
            pix = page.get_pixmap(dpi=dpi)
            out.append(pix.tobytes("png"))
    return out


async def extract_from_pdf_pages(
    pdf_bytes: bytes,
    page_indexes: Optional[list[int]] = None,
) -> list[str]:
    """Render PDF pages and OCR each one. Returns one string per rendered page."""
    rendered = await asyncio.to_thread(
        _render_pdf_pages, pdf_bytes, page_indexes
    )
    reader = await _get_reader()
    pages: list[str] = []
    for page_png in rendered:
        pages.append(await asyncio.to_thread(_readtext_sync, reader, page_png))
    return pages

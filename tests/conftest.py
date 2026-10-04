"""Pytest session-scoped fixtures: synthetic OCR samples (regenerated on first use).

We don't commit binary fixtures. Instead, ``conftest.py`` generates them via
Pillow + PyMuPDF on first session run, into ``tests/fixtures/samples/``.
The folder is git-ignored (see ``samples/.gitignore``).
"""
from __future__ import annotations

import io
from pathlib import Path

import fitz
import pytest
from PIL import Image, ImageDraw, ImageFont


_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "samples"
_DIGITAL_PDF_KEYWORD = "DigitalTestMarker"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try common system fonts; fall back to PIL default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _generate_passport_jpg(out_path: Path) -> None:
    img = Image.new("RGB", (1200, 800), color="white")
    draw = ImageDraw.Draw(img)
    big = _font(60)
    med = _font(40)
    draw.text((50, 50), "РОССИЙСКАЯ ФЕДЕРАЦИЯ", fill="black", font=big)
    draw.text((50, 150), "ПАСПОРТ", fill="black", font=big)
    draw.text((50, 280), "Серия 1234 Номер 567890", fill="black", font=med)
    draw.text((50, 360), "Иванов Иван Иванович", fill="black", font=med)
    img.save(out_path, "JPEG", quality=85)


def _generate_digital_pdf(out_path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    text = (
        " OCR fixture\n"
        f"{_DIGITAL_PDF_KEYWORD}\n"
        "This is a digital PDF with embedded text.\n"
        "PyMuPDF page.get_text() returns non-empty for this file."
    )
    page.insert_text((72, 100), text, fontsize=14)
    doc.save(str(out_path))
    doc.close()


def _generate_scan_pdf(out_path: Path) -> None:
    """A PDF wrapping a rasterised image — no embedded text."""
    img = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(img)
    font = _font(36)
    draw.text((50, 50), "Scan only document", fill="black", font=font)
    draw.text((50, 150), "no embedded text", fill="black", font=font)

    img_buf = io.BytesIO()
    img.save(img_buf, "PNG")

    doc = fitz.open()
    page = doc.new_page(width=img.width, height=img.height)
    page.insert_image(page.rect, stream=img_buf.getvalue())
    doc.save(str(out_path))
    doc.close()


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return _FIXTURES_DIR


@pytest.fixture(scope="session")
def passport_jpg_path(fixtures_dir: Path) -> Path:
    p = fixtures_dir / "passport.jpg"
    if not p.exists():
        _generate_passport_jpg(p)
    return p


@pytest.fixture(scope="session")
def passport_jpg_bytes(passport_jpg_path: Path) -> bytes:
    return passport_jpg_path.read_bytes()


@pytest.fixture(scope="session")
def digital_pdf_path(fixtures_dir: Path) -> Path:
    p = fixtures_dir / "digital.pdf"
    if not p.exists():
        _generate_digital_pdf(p)
    return p


@pytest.fixture(scope="session")
def digital_pdf_bytes(digital_pdf_path: Path) -> bytes:
    return digital_pdf_path.read_bytes()


@pytest.fixture(scope="session")
def digital_pdf_keyword() -> str:
    return _DIGITAL_PDF_KEYWORD


@pytest.fixture(scope="session")
def scan_pdf_path(fixtures_dir: Path) -> Path:
    p = fixtures_dir / "scan.pdf"
    if not p.exists():
        _generate_scan_pdf(p)
    return p


@pytest.fixture(scope="session")
def scan_pdf_bytes(scan_pdf_path: Path) -> bytes:
    return scan_pdf_path.read_bytes()

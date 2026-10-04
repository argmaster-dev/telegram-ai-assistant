"""Tests for services.ocr_detect.is_digital_pdf."""
from __future__ import annotations

from services.ocr_detect import is_digital_pdf


def test_digital_pdf_returns_true(digital_pdf_bytes):
    assert is_digital_pdf(digital_pdf_bytes) is True


def test_scan_pdf_returns_false(scan_pdf_bytes):
    assert is_digital_pdf(scan_pdf_bytes) is False


def test_threshold_can_be_raised(digital_pdf_bytes):
    """Raising threshold above the fixture's text length flips detection to scan."""
    # Synthetic digital fixture has < 1000 chars on page 1 (clearly bigger than 100)
    assert is_digital_pdf(digital_pdf_bytes, threshold_chars=100) is True
    assert is_digital_pdf(digital_pdf_bytes, threshold_chars=10_000) is False

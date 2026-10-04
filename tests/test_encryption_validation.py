"""Tests for services.encryption.validate_key."""
from __future__ import annotations

import base64
import os

import pytest

from services.encryption import validate_key


def test_none_raises():
    with pytest.raises(ValueError, match="required"):
        validate_key(None)


def test_empty_string_raises():
    with pytest.raises(ValueError, match="required"):
        validate_key("")


def test_invalid_base64_raises():
    with pytest.raises(ValueError, match="base64"):
        validate_key("not*valid*base64")


def test_too_short_key_raises():
    # 10 bytes after base64-decode, not 32
    short = base64.b64encode(b"0123456789").decode()
    with pytest.raises(ValueError, match="32 bytes"):
        validate_key(short)


def test_valid_32_byte_key_passes():
    # Generated the same way the .env.example docs show
    valid = base64.b64encode(os.urandom(32)).decode()
    # Must not raise
    validate_key(valid)

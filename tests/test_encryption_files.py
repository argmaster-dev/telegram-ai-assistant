"""Tests for services.encryption.encrypt_file / decrypt_file — AES-256-GCM."""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidTag

from services.encryption import (
    KEY_BYTES,
    NONCE_BYTES,
    decrypt_file,
    encrypt_file,
)


def _make_key() -> str:
    return base64.b64encode(os.urandom(KEY_BYTES)).decode()


def test_encrypt_decrypt_identity(tmp_path: Path):
    """encrypt → decrypt round-trip returns the original bytes (sha256-checked)."""
    key = _make_key()
    plain = os.urandom(1024 * 1024 + 37)  # ~1 MB + tail
    src = tmp_path / "orig.bin"
    enc = tmp_path / "enc.bin"
    dec = tmp_path / "dec.bin"
    src.write_bytes(plain)
    sha_before = hashlib.sha256(plain).hexdigest()

    encrypt_file(src, enc, key)
    decrypt_file(enc, dec, key)

    sha_after = hashlib.sha256(dec.read_bytes()).hexdigest()
    assert sha_before == sha_after


def test_encrypt_deletes_source(tmp_path: Path):
    """encrypt_file removes the plaintext source after a successful write."""
    key = _make_key()
    src = tmp_path / "plain.txt"
    src.write_bytes(b"secret content")
    enc = tmp_path / "enc.bin"

    encrypt_file(src, enc, key)
    assert not src.exists(), "source plaintext must be unlinked after encrypt"
    assert enc.exists() and enc.stat().st_size > 0


def test_encrypted_bytes_differ_from_plain(tmp_path: Path):
    """Ciphertext must not contain the plaintext verbatim."""
    key = _make_key()
    plain = b"THIS_IS_A_RECOGNIZABLE_MARKER_STRING_DEADBEEF"
    src = tmp_path / "p.txt"
    src.write_bytes(plain)
    enc = tmp_path / "c.bin"

    encrypt_file(src, enc, key)
    blob = enc.read_bytes()
    assert plain not in blob, "plaintext marker leaked into ciphertext"
    assert len(blob) == NONCE_BYTES + len(plain) + 16  # GCM tag = 16 bytes


def test_wrong_key_raises_invalid_tag(tmp_path: Path):
    """Decrypting with a different key must raise InvalidTag (not return garbage)."""
    key_a = _make_key()
    key_b = _make_key()
    assert key_a != key_b
    src = tmp_path / "p.txt"
    src.write_bytes(b"top secret")
    enc = tmp_path / "c.bin"
    dec = tmp_path / "d.bin"
    encrypt_file(src, enc, key_a)

    with pytest.raises(InvalidTag):
        decrypt_file(enc, dec, key_b)


def test_nonce_is_unique_per_encryption(tmp_path: Path):
    """Encrypting the same plaintext twice produces different blobs (fresh nonce)."""
    key = _make_key()
    payload = b"deterministic_content_blob_" * 100

    src1 = tmp_path / "a.txt"
    src1.write_bytes(payload)
    enc1 = tmp_path / "a.bin"
    encrypt_file(src1, enc1, key)

    src2 = tmp_path / "b.txt"
    src2.write_bytes(payload)
    enc2 = tmp_path / "b.bin"
    encrypt_file(src2, enc2, key)

    assert enc1.read_bytes() != enc2.read_bytes(), (
        "two encryptions of the same plaintext must differ (fresh nonce)"
    )

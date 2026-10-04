"""Encryption helpers — AES-256-GCM for documents at rest.

``validate_key`` only (start-up config check).
``encrypt_file`` / ``decrypt_file`` — authenticated encryption for
every document on disk.

On-disk format: ``nonce(12) || ciphertext+tag``. AESGCM from
``cryptography.hazmat`` returns ciphertext with the 16-byte GCM tag already
appended, so layout on disk is exactly 12 + len(plaintext) + 16 bytes.

``encrypt_file`` **unlinks the source** after a successful write, enforcing
the «no plaintext left behind» invariant of §8.1 of the concept.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


KEY_BYTES = 32   # AES-256
NONCE_BYTES = 12  # GCM recommended size


def validate_key(raw: Optional[str]) -> None:
    """Raise ``ValueError`` if ``raw`` is not a valid base64-encoded 32-byte key.

    Called from ``bot.py`` at start-up. On failure the bot exits with a clear
    message, preventing documents from being stored without encryption-at-rest.
    """
    if not raw:
        raise ValueError("DOCS_ENCRYPTION_KEY is required")

    try:
        decoded = base64.b64decode(raw, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"DOCS_ENCRYPTION_KEY must be valid base64: {exc}") from exc

    if len(decoded) != KEY_BYTES:
        raise ValueError(
            f"DOCS_ENCRYPTION_KEY must be base64-encoded 32 bytes, got {len(decoded)} bytes"
        )


def _key_bytes(key_b64: str) -> bytes:
    """Decode the base64 key. Assumes ``validate_key`` already passed."""
    return base64.b64decode(key_b64)


def encrypt_file(src_path: Path, dst_path: Path, key_b64: str) -> None:
    """Encrypt ``src_path`` with AES-256-GCM and write to ``dst_path``.

    On-disk layout: ``nonce(12) || ciphertext+tag``.
    Deletes ``src_path`` after a successful write (plaintext must not persist).
    ``dst_path`` parent is created if missing.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    aes = AESGCM(_key_bytes(key_b64))
    nonce = os.urandom(NONCE_BYTES)
    ct_with_tag = aes.encrypt(nonce, src_path.read_bytes(), associated_data=None)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_bytes(nonce + ct_with_tag)
    src_path.unlink()


def decrypt_file(src_path: Path, dst_path: Path, key_b64: str) -> None:
    """Decrypt an ``encrypt_file``-produced blob and write plaintext to ``dst_path``.

    Raises ``cryptography.exceptions.InvalidTag`` if the ciphertext has been
    tampered with or the wrong key was supplied.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    raw = src_path.read_bytes()
    if len(raw) < NONCE_BYTES + 16:  # nonce + smallest possible tag-only
        raise ValueError(
            f"{src_path}: file too small to be an AES-GCM blob ({len(raw)} bytes)"
        )
    nonce, ct_with_tag = raw[:NONCE_BYTES], raw[NONCE_BYTES:]
    aes = AESGCM(_key_bytes(key_b64))
    plaintext = aes.decrypt(nonce, ct_with_tag, associated_data=None)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_bytes(plaintext)


__all__ = [
    "KEY_BYTES",
    "NONCE_BYTES",
    "validate_key",
    "encrypt_file",
    "decrypt_file",
]

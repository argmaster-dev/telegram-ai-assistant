"""Runtime configuration loaded from ``.env`` + process environment."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Settings:
    # Required
    TELEGRAM_BOT_TOKEN: str
    ALLOWED_CHAT_ID: int
    DOCS_ENCRYPTION_KEY: str  # validated by services.encryption.validate_key at startup

    # Optional cloud services
    GEMINI_API_KEY: Optional[str] = None
    AUDD_API_KEY: Optional[str] = None

    # Internal URLs (DNS names within tg_assist_internal docker network)
    OLLAMA_URL: str = "http://tg_ollama:11434"
    QDRANT_URL: str = "http://tg_qdrant:6333"

    # Filesystem (paths inside the container, mapped from host ./data/)
    DB_PATH: str = "/app/data/tg_assist.db"
    DOCS_DIR: str = "/app/data/documents"
    TMP_DIR: str = "/app/data/tmp"  # pre-encryption upload staging

    # Privacy kill-switch — if true, disables all cloud calls (Gemini, AudD)
    PRIVACY_STRICT: bool = False

    @classmethod
    def load_from_env(cls) -> "Settings":
        """Load settings from ``.env`` file + process environment.

        Raises ``ValueError`` on missing required variables. The encryption
        key itself is passed through unchanged and validated separately by
        ``services.encryption.validate_key`` so that messaging is specific.
        """
        load_dotenv()

        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        raw_chat_id = os.getenv("ALLOWED_CHAT_ID", "").strip()
        if not raw_chat_id:
            raise ValueError("ALLOWED_CHAT_ID is required")
        try:
            chat_id = int(raw_chat_id)
        except ValueError as exc:
            raise ValueError(
                f"ALLOWED_CHAT_ID must be an integer, got {raw_chat_id!r}"
            ) from exc

        def _opt(key: str) -> Optional[str]:
            val = os.getenv(key, "").strip()
            return val or None

        def _bool(key: str, default: bool = False) -> bool:
            val = os.getenv(key)
            if val is None:
                return default
            return val.strip().lower() in ("1", "true", "yes", "on")

        return cls(
            TELEGRAM_BOT_TOKEN=token,
            ALLOWED_CHAT_ID=chat_id,
            DOCS_ENCRYPTION_KEY=os.getenv("DOCS_ENCRYPTION_KEY", ""),
            GEMINI_API_KEY=_opt("GEMINI_API_KEY"),
            AUDD_API_KEY=_opt("AUDD_API_KEY"),
            OLLAMA_URL=os.getenv("OLLAMA_URL", cls.OLLAMA_URL).strip() or cls.OLLAMA_URL,
            QDRANT_URL=os.getenv("QDRANT_URL", cls.QDRANT_URL).strip() or cls.QDRANT_URL,
            DB_PATH=os.getenv("DB_PATH", cls.DB_PATH).strip() or cls.DB_PATH,
            DOCS_DIR=os.getenv("DOCS_DIR", cls.DOCS_DIR).strip() or cls.DOCS_DIR,
            TMP_DIR=os.getenv("TMP_DIR", cls.TMP_DIR).strip() or cls.TMP_DIR,
            PRIVACY_STRICT=_bool("PRIVACY_STRICT"),
        )

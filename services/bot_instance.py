"""Module-level singleton holding the active aiogram ``Bot`` instance.

APScheduler with ``SQLAlchemyJobStore`` serialises job callables — closures
capturing a ``bot`` object cannot be persisted. The fire function registered
with the scheduler therefore has to be a module-level symbol that looks up
the current ``Bot`` via this singleton at fire time.
"""
from __future__ import annotations

from typing import Optional

from aiogram import Bot

_bot: Optional[Bot] = None


def set_bot(bot: Bot) -> None:
    """Store the active ``Bot`` for later lookup from scheduler fire functions."""
    global _bot
    _bot = bot


def get_bot() -> Bot:
    """Return the active ``Bot``. Raises if called before ``set_bot``."""
    if _bot is None:
        raise RuntimeError(
            "Bot instance not set; bot.py::_main must call set_bot() before "
            "starting the scheduler or polling"
        )
    return _bot

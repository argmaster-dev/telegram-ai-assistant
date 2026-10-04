"""Module-level singleton for the active worker task.

Used by shutdown hooks in ``bot.py`` to cancel the worker cleanly and by
future maintenance handlers (e.g. ``/worker_status``) to read the
running task's metadata. Same pattern as ``services.bot_instance`` /
``services.scheduler_instance``.
"""
from __future__ import annotations

import asyncio
from typing import Optional


_worker_task: Optional[asyncio.Task] = None


def set_worker(task: asyncio.Task) -> None:
    """Register the background worker ``Task`` for later lookup."""
    global _worker_task
    _worker_task = task


def get_worker() -> Optional[asyncio.Task]:
    """Return the registered worker ``Task`` or ``None`` if not started yet."""
    return _worker_task


def clear_worker() -> None:
    """Drop the singleton reference — used after clean shutdown."""
    global _worker_task
    _worker_task = None

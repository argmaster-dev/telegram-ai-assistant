"""Module-level singleton for the active ``SchedulerService``.

Needed because ``_fire_recurring`` (module-level, persisted by
``SQLAlchemyJobStore``) has to re-schedule the next job on fire — it can't
close over the scheduler instance created in ``bot.py`` because closures can't
be pickled by the job store. Same pattern as ``services.bot_instance``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from services.scheduler import SchedulerService


_scheduler: Optional["SchedulerService"] = None


def set_scheduler(scheduler: "SchedulerService") -> None:
    """Register the active ``SchedulerService`` for later lookup."""
    global _scheduler
    _scheduler = scheduler


def get_scheduler() -> "SchedulerService":
    """Return the active scheduler. Raises if called before ``set_scheduler``."""
    if _scheduler is None:
        raise RuntimeError(
            "SchedulerService not set; bot.py::_main must call set_scheduler() "
            "right after scheduler.start()"
        )
    return _scheduler

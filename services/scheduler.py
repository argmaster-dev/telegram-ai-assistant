"""APScheduler wrapper for persistent reminder jobs (single-shot + recurring).

``AsyncIOScheduler`` with ``SQLAlchemyJobStore`` on the shared ``tg_assist.db``
SQLite file. Scheduled reminders survive container restarts. The database has
``PRAGMA journal_mode=WAL`` (set by ``services.storage.init_db``) so APScheduler's
sync SQLAlchemy writer coexists with our async ``aiosqlite`` readers/writers
without lock contention.

Fire callables must be module-level symbols because ``SQLAlchemyJobStore``
persists jobs by importing the callable by its fully qualified name. Renaming
or moving ``_fire_reminder`` / ``_fire_recurring`` silently breaks jobs stored
from prior bot versions until the user reschedules them.

Recurring path:
- ``_fire_recurring(reminder_id, db_path)`` — sends push, computes
  ``next_fire_at`` from ``now+1min`` and re-schedules itself.
- ``SchedulerService.schedule_reminder`` dispatches to ``_schedule_single`` or
  ``_schedule_recurring`` based on ``reminder.rrule``.
- ``SchedulerService.start()`` registers the instance in
  ``services.scheduler_instance`` so ``_fire_recurring`` can find it.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from services import reminder_store, storage
from services.bot_instance import get_bot
from services.intents import Reminder, Subscription
from services.reminders import next_fire_at
from services.scheduler_instance import get_scheduler, set_scheduler


logger = logging.getLogger(__name__)


TZ = "Europe/Moscow"
_TZ = ZoneInfo(TZ)
JOB_ID_PREFIX = "reminder:"
SUB_JOB_ID_PREFIX = "subscription:"


def _job_id(reminder_id: int) -> str:
    return f"{JOB_ID_PREFIX}{reminder_id}"


def _sub_job_id(sub_id: int) -> str:
    return f"{SUB_JOB_ID_PREFIX}{sub_id}"


def parse_reminder_id(job_id: str) -> Optional[int]:
    """Inverse of ``_job_id``. Returns None if ``job_id`` is not ours."""
    if not job_id.startswith(JOB_ID_PREFIX):
        return None
    try:
        return int(job_id[len(JOB_ID_PREFIX):])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Module-level fire callables (SQLAlchemyJobStore imports them by FQN on fire)
# ---------------------------------------------------------------------------

async def _fire_reminder(reminder_id: int, db_path: str) -> None:
    """Fire a single-shot reminder. MUST stay at ``services.scheduler._fire_reminder``."""
    bot = get_bot()
    reminder = await reminder_store.get(db_path, reminder_id)
    if reminder is None:
        logger.warning(
            "fire: reminder #%d not found in DB (deleted after scheduling?)",
            reminder_id,
        )
        return
    if reminder.status != "scheduled":
        logger.info(
            "fire: reminder #%d status=%s, skipping push",
            reminder_id,
            reminder.status,
        )
        return

    try:
        await bot.send_message(
            chat_id=reminder.user_id,
            text=f"🔔 {reminder.title}",
        )
    except Exception as exc:
        logger.error(
            "fire: send_message failed for reminder #%d: %s", reminder_id, exc
        )
        await reminder_store.update_status(db_path, reminder_id, "failed")
        return

    await reminder_store.update_status(db_path, reminder_id, "done")


async def _fire_recurring(reminder_id: int, db_path: str) -> None:
    """Fire a recurring reminder and self-reschedule to the next occurrence.

    MUST stay at ``services.scheduler._fire_recurring``.

    Flow on each invocation:
      1. Load reminder; skip silently if deleted or not ``scheduled``.
      2. Send the push; log but ignore send errors (recurring series continues).
      3. Compute ``next_fire_at(now + 1min)`` to avoid immediate re-fire.
      4. If None → mark ``done``. Otherwise update cached ``when_at`` in DB
         and enqueue the next ``date``-triggered job under the same job_id.
    """
    bot = get_bot()
    reminder = await reminder_store.get(db_path, reminder_id)
    if reminder is None:
        logger.warning(
            "fire_recurring: reminder #%d not found in DB", reminder_id
        )
        return
    if reminder.status != "scheduled":
        logger.info(
            "fire_recurring: reminder #%d status=%s, skipping",
            reminder_id, reminder.status,
        )
        return
    if not reminder.rrule:
        logger.warning(
            "fire_recurring: reminder #%d has no rrule — marking failed",
            reminder_id,
        )
        await reminder_store.update_status(db_path, reminder_id, "failed")
        return

    # 1. Push — non-fatal if it fails, series keeps going.
    try:
        await bot.send_message(
            chat_id=reminder.user_id,
            text=f"🔔 {reminder.title}",
        )
    except Exception as exc:
        logger.error(
            "fire_recurring: send_message failed for #%d: %s (continuing series)",
            reminder_id, exc,
        )

    # 2. Compute the next fire strictly in the future.
    try:
        now_next = datetime.now(tz=_TZ) + timedelta(minutes=1)
        next_dt = next_fire_at(
            reminder.rrule,
            reminder.exdates,
            reminder.exclude_holidays,
            from_dt=now_next,
        )
    except Exception as exc:
        logger.error(
            "fire_recurring: next_fire_at crashed for #%d: %s", reminder_id, exc
        )
        await reminder_store.update_status(db_path, reminder_id, "failed")
        return

    if next_dt is None:
        logger.info(
            "fire_recurring: reminder #%d has no more occurrences — marking done",
            reminder_id,
        )
        await reminder_store.update_status(db_path, reminder_id, "done")
        return

    # 3. Cache next_dt in DB so /remind_list shows it correctly.
    reminder.when_at = next_dt
    await reminder_store.update(db_path, reminder)

    # 4. Enqueue next job under the same id.
    scheduler = get_scheduler()
    scheduler.scheduler.add_job(
        _fire_recurring,
        trigger="date",
        run_date=next_dt,
        args=(reminder_id, db_path),
        id=_job_id(reminder_id),
        replace_existing=True,
    )
    logger.info(
        "fire_recurring: rescheduled #%d to %s",
        reminder_id, next_dt.isoformat(),
    )


async def _fire_subscription(sub_id: int, db_path: str) -> None:
    """Fire a subscription pre-charge notification and self-reschedule.

    MUST stay at ``services.scheduler._fire_subscription`` — APScheduler
    persists this FQN in ``scheduler_jobs``; renaming silently breaks
    pending notifications across container restarts.

    Flow on each invocation:
      1. Load the subscription. If deleted → skip silently.
      2. Send the push (non-fatal — series continues if Telegram is down).
      3. Compute the next charge date, push notification 'notify_before_days'
         before it, and re-enqueue under the same job_id.
    """
    bot = get_bot()
    sub = await storage.get_subscription(db_path, sub_id)
    if sub is None:
        logger.warning(
            "fire_subscription: sub #%d not found in DB (deleted?)", sub_id
        )
        return

    # Currency symbol for the push text. Plain dict to avoid reaching into
    # utils.messages from a module-level fire callable (keeps import graph
    # tight; APScheduler imports this file at fire-time on cold restart).
    sym = {"RUB": "₽", "USD": "$", "EUR": "€", "GBP": "£"}.get(sub.currency, sub.currency)
    next_date_str = sub.next_charge_at.strftime("%d.%m") if sub.next_charge_at else "—"
    title = (
        f"💳 Через {sub.notify_before_days} дн. — {sub.service}: "
        f"{sub.amount:g} {sym}"
        + (f" ({sub.card})" if sub.card else "")
        + f"\n⏭ Списание: {next_date_str}"
    )
    try:
        await bot.send_message(chat_id=sub.user_id, text=title)
    except Exception as exc:
        logger.error(
            "fire_subscription: send_message failed for sub #%d: %s "
            "(continuing series)",
            sub_id, exc,
        )

    # Compute next-month charge from the just-fired one.
    # Lazy import to avoid a circular: services.subscriptions imports
    # services.storage, storage already imported here is fine, but
    # subscriptions also imports utils.messages → handlers chain.
    from services.subscriptions import compute_next_charge_at

    try:
        new_next = compute_next_charge_at(
            sub,
            from_dt=(sub.next_charge_at or datetime.now(tz=_TZ))
            + timedelta(days=1),
        )
    except Exception as exc:
        logger.error(
            "fire_subscription: compute_next_charge_at crashed for #%d: %s",
            sub_id, exc,
        )
        return

    if new_next is None:
        logger.warning(
            "fire_subscription: no next charge for sub #%d — leaving without reschedule",
            sub_id,
        )
        return

    # Persist the new next_charge_at on the subscription row.
    await storage.update_subscription_fields(
        db_path, sub_id, {"next_charge_at": new_next}
    )

    # Re-enqueue under the same job_id, firing 'notify_before_days' before new_next.
    fire_at = new_next - timedelta(days=sub.notify_before_days)
    if fire_at <= datetime.now(tz=_TZ):
        # If notify-before window already elapsed, fire ASAP next minute so we
        # don't lose the notification entirely.
        fire_at = datetime.now(tz=_TZ) + timedelta(minutes=1)

    scheduler = get_scheduler()
    scheduler.scheduler.add_job(
        _fire_subscription,
        trigger="date",
        run_date=fire_at,
        args=(sub_id, db_path),
        id=_sub_job_id(sub_id),
        replace_existing=True,
    )
    logger.info(
        "fire_subscription: rescheduled sub #%d notification to %s "
        "(next charge %s)",
        sub_id, fire_at.isoformat(), new_next.isoformat(),
    )


# ---------------------------------------------------------------------------
# SchedulerService
# ---------------------------------------------------------------------------

class SchedulerService:
    """Thin wrapper around ``AsyncIOScheduler`` with reminder helpers."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        sa_url = f"sqlite:///{db_path}"
        jobstores = {
            "default": SQLAlchemyJobStore(url=sa_url, tablename="scheduler_jobs")
        }
        self.scheduler = AsyncIOScheduler(jobstores=jobstores, timezone=TZ)

    def start(self) -> None:
        self.scheduler.start()
        set_scheduler(self)  # register for _fire_recurring to look us up
        logger.info("scheduler started (jobstore=%s)", self._db_path)

    def shutdown(self, wait: bool = False) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("scheduler stopped")

    # ---- public dispatchers ------------------------------------------------

    async def schedule_reminder(self, reminder: Reminder) -> str:
        """Schedule a single-shot OR recurring reminder. Returns job_id."""
        if reminder.rrule:
            return await self._schedule_recurring(reminder)
        return await self._schedule_single(reminder)

    async def list_scheduled(self, user_id: int) -> list[Reminder]:
        return await reminder_store.list_by_user(
            self._db_path, user_id, status="scheduled"
        )

    async def cancel(self, reminder_id: int) -> bool:
        """Remove the APScheduler job and mark reminder ``cancelled``."""
        job_id = _job_id(reminder_id)
        try:
            self.scheduler.remove_job(job_id)
        except Exception as exc:
            logger.debug("cancel: job %s not in scheduler: %s", job_id, exc)

        existing = await reminder_store.get(self._db_path, reminder_id)
        if existing is None:
            return False
        await reminder_store.update_status(self._db_path, reminder_id, "cancelled")
        return True

    async def reschedule(self, reminder: Reminder) -> str:
        """Re-schedule an existing reminder (used by ``/remind_edit``).

        Removes the old APScheduler job (if any) and re-adds based on current
        fields. ``reminder.id`` must be set. ``reminder_store`` row should
        already be updated by the caller.
        """
        if reminder.id is None:
            raise ValueError("reminder.id is required for reschedule")

        job_id = _job_id(reminder.id)
        try:
            self.scheduler.remove_job(job_id)
        except Exception as exc:
            logger.debug("reschedule: old job %s not present: %s", job_id, exc)

        if reminder.rrule:
            return await self._add_recurring_job(reminder)
        return await self._add_single_job(reminder)

    # ---- internal single-shot path ----------------------------------------

    async def _schedule_single(self, reminder: Reminder) -> str:
        if reminder.when_at is None:
            raise ValueError("reminder.when_at is required for single-shot")

        reminder_id = await reminder_store.create(self._db_path, reminder)
        reminder.id = reminder_id
        return await self._add_single_job(reminder)

    async def _add_single_job(self, reminder: Reminder) -> str:
        assert reminder.id is not None and reminder.when_at is not None
        job_id = _job_id(reminder.id)
        self.scheduler.add_job(
            _fire_reminder,
            trigger="date",
            run_date=reminder.when_at,
            args=(reminder.id, self._db_path),
            id=job_id,
            replace_existing=True,
        )
        logger.info(
            "scheduled single reminder #%d at %s (job_id=%s)",
            reminder.id, reminder.when_at.isoformat(), job_id,
        )
        return job_id

    # ---- internal recurring path ------------------------------------------

    async def _schedule_recurring(self, reminder: Reminder) -> str:
        if not reminder.rrule:
            raise ValueError("reminder.rrule is required for recurring")

        # Compute initial next fire if caller didn't set it.
        if reminder.when_at is None:
            now = datetime.now(tz=_TZ)
            next_dt = next_fire_at(
                reminder.rrule,
                reminder.exdates,
                reminder.exclude_holidays,
                from_dt=now,
            )
            if next_dt is None:
                raise ValueError(
                    f"rrule {reminder.rrule!r} has no valid next fire within 1 year"
                )
            reminder.when_at = next_dt

        reminder_id = await reminder_store.create(self._db_path, reminder)
        reminder.id = reminder_id
        return await self._add_recurring_job(reminder)

    async def _add_recurring_job(self, reminder: Reminder) -> str:
        assert reminder.id is not None and reminder.when_at is not None
        job_id = _job_id(reminder.id)
        self.scheduler.add_job(
            _fire_recurring,
            trigger="date",
            run_date=reminder.when_at,
            args=(reminder.id, self._db_path),
            id=job_id,
            replace_existing=True,
        )
        logger.info(
            "scheduled recurring reminder #%d at %s (rrule=%s, job_id=%s)",
            reminder.id, reminder.when_at.isoformat(), reminder.rrule, job_id,
        )
        return job_id

    # ---- subscription jobs -----------------------------------

    async def schedule_subscription(self, sub: Subscription) -> Optional[str]:
        """Schedule a pre-charge notification for ``sub``.

        Returns the APScheduler job_id, or ``None`` when there is no future
        charge to notify about (already-deleted sub, or recurrence='yearly'
        without a computable next date).

        Caller is expected to set ``sub.id`` + ``sub.next_charge_at`` before
        calling this method.
        """
        if sub.id is None:
            raise ValueError("subscription.id is required for schedule_subscription")
        if sub.next_charge_at is None:
            logger.info(
                "schedule_subscription: sub #%d has no next_charge_at — skipping",
                sub.id,
            )
            return None

        fire_at = sub.next_charge_at - timedelta(days=sub.notify_before_days)
        if fire_at <= datetime.now(tz=_TZ):
            # Notification window passed (e.g. saving a subscription where
            # the next charge is in 1 day but notify_before_days=2).
            # Fire one minute from now so the user still gets the heads-up.
            fire_at = datetime.now(tz=_TZ) + timedelta(minutes=1)

        job_id = _sub_job_id(sub.id)
        self.scheduler.add_job(
            _fire_subscription,
            trigger="date",
            run_date=fire_at,
            args=(sub.id, self._db_path),
            id=job_id,
            replace_existing=True,
        )
        logger.info(
            "scheduled subscription #%d notification at %s (charge %s, job_id=%s)",
            sub.id, fire_at.isoformat(),
            sub.next_charge_at.isoformat(), job_id,
        )
        return job_id

    async def cancel_subscription(self, sub_id: int) -> bool:
        """Remove the APScheduler job for a subscription. Idempotent.

        Returns True if a job was actually removed, False otherwise (already
        cancelled or never scheduled).
        """
        job_id = _sub_job_id(sub_id)
        try:
            self.scheduler.remove_job(job_id)
            return True
        except Exception as exc:
            logger.debug(
                "cancel_subscription: job %s not in scheduler: %s", job_id, exc
            )
            return False

    async def reschedule_subscription(self, sub: Subscription) -> Optional[str]:
        """Cancel + re-schedule a subscription notification (used on edit)."""
        if sub.id is None:
            raise ValueError("subscription.id is required for reschedule")
        await self.cancel_subscription(sub.id)
        return await self.schedule_subscription(sub)

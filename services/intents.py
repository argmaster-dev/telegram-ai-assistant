"""Dataclass definitions for the four core entities.

``Reminder`` supports single-shot (``when_at`` set, ``rrule`` is None) and
recurring (``rrule`` set, ``when_at`` stores the *next* fire moment — updated
on each self-reschedule).

populated ``Document``, ``Task`` (full CRUD
with category/priority/deadline/status), ``Subscription``
(full CRUD with recurrence/day_of_month/next_charge_at).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Reminder:
    """Reminder entity, single-shot OR recurring.

    Single-shot: ``when_at`` set, ``rrule`` is None.
    Recurring:   ``rrule`` set (RFC 5545, without DTSTART), ``when_at`` stores
                 the *next* scheduled fire moment (refreshed by the scheduler on
                 each self-reschedule so ``/remind_list`` shows correct "next").
    """
    id: Optional[int] = None
    user_id: int = 0
    title: str = ""
    when_at: Optional[datetime] = None
    rrule: Optional[str] = None                 # RFC 5545 RRULE without DTSTART
    exdates: list[str] = field(default_factory=list)  # ISO dates (YYYY-MM-DD) to skip
    exclude_holidays: str = "none"              # 'none' | 'ru' | 'ru+custom'
    notify_before_sec: int = 0
    source: str = "text"  # 'text' | 'voice' | 'subscription'
    status: str = "scheduled"  # 'scheduled' | 'done' | 'cancelled' | 'failed'


@dataclass
class Task:
    """Open-ended TODO entity with optional deadline + category/priority.

    full CRUD via ``services.storage`` (save/get/list/update/delete
    + mark_done/mark_todo + by-category/by-priority filters). Status starts as
    ``"todo"`` and transitions to ``"done"`` (and back). ``deadline`` is an
    optional ISO datetime — None means open-ended.
    """
    id: Optional[int] = None
    user_id: int = 0
    title: str = ""
    deadline: Optional[datetime] = None
    priority: str = "med"  # 'low' | 'med' | 'high'
    status: str = "todo"   # 'todo' | 'done'
    category: str = "general"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Subscription:
    """Recurring paid subscription (Spotify, Netflix, ChatGPT, ...).

    full CRUD via ``services.storage`` + ``compute_next_charge_at``
    handles "31 февраля" → last day of month via ``calendar.monthrange``.
    Each subscription is paired with an APScheduler reminder (created via
    ``services.scheduler.schedule_subscription``) that fires
    ``notify_before_days`` before the next charge.
    """
    id: Optional[int] = None
    user_id: int = 0
    service: str = ""
    amount: float = 0.0
    currency: str = "RUB"          # 'RUB' | 'USD' | 'EUR' | 'GBP'
    card: Optional[str] = None
    recurrence: str = "monthly"    # 'monthly' | 'yearly' | 'weekly'
    day_of_month: Optional[int] = None
    next_charge_at: Optional[datetime] = None
    notify_before_days: int = 2
    created_at: Optional[datetime] = None


@dataclass
class Document:
    """Document entity — file uploaded by the user, OCR + optional embedding.

    populates every field except ``ocr_text``/``redacted_text``
    at upload-time; the worker fills them after OCR. ``storage_mode`` is
    chosen by the user in the ``ConfirmPiiAndMode`` FSM state.
    """
    id: Optional[int] = None
    user_id: int = 0
    type: str = "other"
    type_confirmed_by_user: bool = False
    has_pii: bool = False
    storage_mode: str = "local_only"   # 'local_only' | 'redacted_indexed'
    original_path: str = ""
    original_sha256: str = ""
    ocr_text: Optional[str] = None
    redacted_text: Optional[str] = None
    tags: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

"""Formatted user-facing strings for document handlers.

minimal formatters plus list/edit/delete/tag/search formatters.
Can be expanded / localised later.
"""
from __future__ import annotations

from typing import Iterable, Optional

from services.intents import Document, Subscription, Task
from services.jobs import Job


_DOC_TYPE_LABELS: dict[str, str] = {
    "passport": "Паспорт РФ",
    "snils": "СНИЛС",
    "driving": "Водительское удостоверение",
    "inn": "ИНН",
    "policy": "Страховой полис",
    "contract": "Договор",
    "other": "Другое",
}


def doc_type_label(doc_type: str) -> str:
    return _DOC_TYPE_LABELS.get(doc_type, doc_type)


def format_duplicate_reply(doc_id: int) -> str:
    return f"Этот документ уже есть у меня, #{doc_id}. Сохранять ещё раз не буду."


def format_accepted_reply(doc_id: int, eta_seconds: int, ocr_mode: str) -> str:
    mode_ru = "Gemini Files API" if ocr_mode == "public" else "локальный OCR"
    return (
        f"✅ Принял, документ #{doc_id}.\n"
        f"Очередь обработки: {mode_ru}, ETA ≈ {eta_seconds} сек.\n"
        f"Проверить статус: /doc_status {doc_id}"
    )


def format_aborted_reply() -> str:
    return "❌ Документ отменён, временные файлы удалены."


def format_doc_status(doc_id: int, joblist: Iterable[Job]) -> str:
    lines = [f"Документ #{doc_id}:"]
    joblist = list(joblist)
    if not joblist:
        return f"Документ #{doc_id} — нет активных задач."
    for j in joblist:
        icon = {
            "pending": "⏳",
            "running": "⚙️",
            "done": "✅",
            "failed": "❌",
        }.get(j.status, "•")
        eta_part = f", ETA ~{j.eta_seconds} сек" if j.eta_seconds else ""
        err_part = f" ({j.error})" if j.status == "failed" and j.error else ""
        lines.append(
            f"{icon} {j.type} — {j.status} ({j.progress_pct}%{eta_part}){err_part}"
        )
    return "\n".join(lines)


def format_done_push(doc_id: int, stage: str, summary: str) -> str:
    return f"✅ Документ #{doc_id}: {stage} готов. {summary}"


def format_fail_push(doc_id: int, stage: str, error: str) -> str:
    return f"❌ Документ #{doc_id}: {stage} упал. {error}"


# ---------------------------------------------------------------------------
# list / edit / delete / search formatters
# ---------------------------------------------------------------------------


_STORAGE_MODE_LABEL: dict[str, str] = {
    "local_only": "🔒 local-only",
    "redacted_indexed": "🎭 indexed",
}


def _short_dt(dt) -> str:
    return dt.strftime("%Y-%m-%d") if dt is not None else "—"


def format_docs_page(
    docs: list[Document], page: int, total_pages: int, total_count: int
) -> str:
    if not docs:
        return "У тебя пока нет документов. Отправь фото или PDF."
    lines = [f"📎 Документы (стр. {page + 1}/{total_pages}, всего {total_count}):"]
    for d in docs:
        tags = f" [{d.tags}]" if d.tags else ""
        mode = _STORAGE_MODE_LABEL.get(d.storage_mode, d.storage_mode)
        lines.append(
            f"#{d.id} · {doc_type_label(d.type)} · {mode} · {_short_dt(d.created_at)}{tags}"
        )
    return "\n".join(lines)


def format_doc_detail(doc: Document, joblist: Iterable[Job]) -> str:
    lines = [
        f"📎 Документ #{doc.id}",
        f"Тип: {doc_type_label(doc.type)}"
        + (" (подтверждён)" if doc.type_confirmed_by_user else " (не подтверждён)"),
        f"Режим: {_STORAGE_MODE_LABEL.get(doc.storage_mode, doc.storage_mode)}",
        f"PII: {'есть' if doc.has_pii else 'нет'}",
        f"Теги: {doc.tags or '—'}",
        f"Загружен: {_short_dt(doc.created_at)}",
        f"Обновлён: {_short_dt(doc.updated_at)}",
    ]
    ocr_len = len(doc.ocr_text or "")
    red_len = len(doc.redacted_text or "")
    lines.append(f"OCR: {ocr_len} симв. · Redacted: {red_len} симв.")
    jl = list(joblist)
    if jl:
        lines.append("")
        lines.append("Задачи:")
        lines.append(format_doc_status(doc.id, jl).split("\n", 1)[1])
    return "\n".join(lines)


def format_edit_menu(doc: Document) -> str:
    return (
        f"✏️ Правка документа #{doc.id}. Что изменить?"
        f"\n\nТекущее:"
        f"\n• Тип: {doc_type_label(doc.type)}"
        f"\n• PII: {'есть' if doc.has_pii else 'нет'}"
        f"\n• Режим: {_STORAGE_MODE_LABEL.get(doc.storage_mode, doc.storage_mode)}"
        f"\n• Теги: {doc.tags or '—'}"
    )


def format_delete_confirm(doc: Document) -> str:
    return (
        f"⚠️ Удалить документ #{doc.id} ({doc_type_label(doc.type)})?\n\n"
        f"Это удалит: зашифрованный файл, запись в БД и {'точки в Qdrant' if doc.storage_mode == 'redacted_indexed' else 'только файл'}.\n"
        f"Отменить невозможно."
    )


def format_tag_result(doc_id: int, tags: list[str], action: str = "обновлены") -> str:
    if not tags:
        return f"Теги документа #{doc_id} очищены."
    return f"Теги документа #{doc_id} {action}: {', '.join(tags)}"


def format_search_result(hits: list[dict], query: str) -> str:
    """Format merge-search results with source labels.

    ``hits`` is a list of dicts: ``{doc, source: 'fts'|'qdrant'|'both', score}``.
    """
    if not hits:
        return f"По запросу «{query}» ничего не найдено."
    _LABELS = {"fts": "📝 FTS", "qdrant": "🧠 Qdrant", "both": "🔀 Both"}
    lines = [f"🔍 Найдено {len(hits)} по запросу «{query}»:"]
    for h in hits:
        d: Document = h["doc"]
        src = _LABELS.get(h["source"], h["source"])
        score = h.get("score", 0.0)
        snippet = (d.ocr_text or d.redacted_text or "")[:80].replace("\n", " ")
        lines.append(
            f"{src} #{d.id} · {doc_type_label(d.type)} · score={score:.2f}"
            + (f"\n   {snippet}…" if snippet else "")
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tasks formatters
# ---------------------------------------------------------------------------

_PRIORITY_LABEL: dict[str, str] = {
    "low": "🟢 low",
    "med": "🟡 med",
    "high": "🔴 high",
}

_TASK_STATUS_LABEL: dict[str, str] = {
    "todo": "⬜ todo",
    "done": "✅ done",
}


def _short_dt_with_time(dt) -> str:
    if dt is None:
        return "—"
    return dt.strftime("%d.%m.%Y %H:%M") if (dt.hour or dt.minute) else dt.strftime("%d.%m.%Y")


def format_task_line(t: Task) -> str:
    """Single-line task summary used in ``/tasks`` listing."""
    deadline = _short_dt_with_time(t.deadline)
    return (
        f"#{t.id} · {_PRIORITY_LABEL.get(t.priority, t.priority)} · "
        f"📁 {t.category} · ⏰ {deadline} · «{t.title}»"
    )


def format_tasks_page(
    tasks: list[Task], *, scope: str = "todo"
) -> str:
    """Multi-line block for ``/tasks``, ``/tasks_all``, ``/tasks_by_cat``."""
    if not tasks:
        empty_messages = {
            "todo": "🎉 Активных задач нет.",
            "all": "📭 Задач пока нет — отправь «добавь задачу …».",
            "by_cat": "📭 В этой категории задач нет.",
        }
        return empty_messages.get(scope, "📭 Пусто.")
    header = {
        "todo": f"🎯 Активные задачи ({len(tasks)}):",
        "all": f"📋 Все задачи ({len(tasks)}):",
        "by_cat": f"📁 Задачи в категории ({len(tasks)}):",
    }.get(scope, f"📋 Задачи ({len(tasks)}):")
    lines = [header]
    lines.extend(format_task_line(t) for t in tasks)
    return "\n".join(lines)


def format_task_detail(t: Task) -> str:
    """Detailed view used in confirmation reply + after edit."""
    return (
        f"🎯 Задача #{t.id}\n"
        f"«{t.title}»\n"
        f"📁 Категория: {t.category}\n"
        f"🏷 Приоритет: {_PRIORITY_LABEL.get(t.priority, t.priority)}\n"
        f"⏰ Дедлайн: {_short_dt_with_time(t.deadline)}\n"
        f"Статус: {_TASK_STATUS_LABEL.get(t.status, t.status)}"
    )


def format_task_confirm_added(t: Task) -> str:
    """Reply after a task is created from free text."""
    return f"✅ Задача добавлена.\n\n{format_task_detail(t)}"


def format_task_delete_confirm(t: Task) -> str:
    return (
        f"⚠️ Удалить задачу #{t.id}?\n\n«{t.title}»\n\n"
        f"Отменить нельзя — но всегда можно создать заново."
    )


# ---------------------------------------------------------------------------
# Subscriptions formatters
# ---------------------------------------------------------------------------

_CURRENCY_SYMBOL: dict[str, str] = {
    "RUB": "₽",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
}

_RECURRENCE_LABEL: dict[str, str] = {
    "monthly": "ежемесячно",
    "yearly": "ежегодно",
    "weekly": "еженедельно",
}


def format_amount(amount: float, currency: str) -> str:
    """``169 ₽``, ``19.99 $``. Trims trailing .0 for whole numbers."""
    sym = _CURRENCY_SYMBOL.get(currency, currency)
    if abs(amount - round(amount)) < 0.005:
        return f"{int(round(amount))} {sym}"
    return f"{amount:.2f} {sym}"


def format_subscription_line(s: Subscription) -> str:
    """Single-line summary for ``/subs`` listing."""
    next_at = _short_dt_with_time(s.next_charge_at)
    card_part = f" · 💳 {s.card}" if s.card else ""
    day_part = f" · 🗓 {s.day_of_month}-го" if s.day_of_month else ""
    return (
        f"#{s.id} · {s.service} · 💰 {format_amount(s.amount, s.currency)}"
        f"{card_part}{day_part} · ⏭ {next_at}"
    )


def format_subscriptions_page(subs: list[Subscription]) -> str:
    if not subs:
        return "📭 Подписок пока нет — отправь «Spotify 169₽ Сбер 15-го»."
    lines = [f"💳 Подписки ({len(subs)}):"]
    lines.extend(format_subscription_line(s) for s in subs)
    return "\n".join(lines)


def format_subscription_detail(s: Subscription) -> str:
    next_at = _short_dt_with_time(s.next_charge_at)
    return (
        f"💳 Подписка #{s.id}\n"
        f"Сервис: {s.service}\n"
        f"💰 Сумма: {format_amount(s.amount, s.currency)}\n"
        f"💳 Карта: {s.card or '—'}\n"
        f"🔁 Периодичность: {_RECURRENCE_LABEL.get(s.recurrence, s.recurrence)}"
        + (f" (день: {s.day_of_month})" if s.day_of_month else "")
        + f"\n🔔 Уведомление за: {s.notify_before_days} дн.\n"
        f"⏭ Следующее списание: {next_at}"
    )


def format_subscription_confirm_added(s: Subscription, *, reminder_active: bool) -> str:
    note = (
        "🔔 Уведомление поставлено."
        if reminder_active
        else "⚠️ Уведомление не создано (нет даты следующего списания)."
    )
    return f"✅ Подписка добавлена.\n\n{format_subscription_detail(s)}\n\n{note}"


def format_subscription_delete_confirm(s: Subscription) -> str:
    return (
        f"⚠️ Удалить подписку #{s.id} ({s.service})?\n\n"
        f"Связанное напоминание тоже будет отменено."
    )


def format_subs_stats(stats: dict) -> str:
    """Format the dict returned by ``services.subscriptions.compute_stats``.

    Schema:
      {
        "months": [{"month": "2026-05", "total_by_currency": {"RUB": 668, "USD": 20}, "count": 2}, ...],
        "year_forecast": {"RUB": 8016, "USD": 240},
      }
    """
    months: list[dict] = stats.get("months", [])
    forecast: dict = stats.get("year_forecast", {})
    if not months and not forecast:
        return "📭 Подписок пока нет — статистика будет позже."

    lines = ["📊 Статистика по подпискам:"]
    if months:
        lines.append("")
        for m in months:
            parts = ", ".join(
                format_amount(amt, ccy) for ccy, amt in m["total_by_currency"].items()
            )
            lines.append(f"🗓 {m['month']}: {parts} ({m['count']} подписк.)")

    if forecast:
        lines.append("")
        forecast_parts = ", ".join(
            format_amount(amt, ccy) for ccy, amt in forecast.items()
        )
        lines.append(f"🔮 Прогноз на 12 мес.: {forecast_parts}")

    return "\n".join(lines)


__all__ = [
    "doc_type_label",
    "format_duplicate_reply",
    "format_accepted_reply",
    "format_aborted_reply",
    "format_doc_status",
    "format_done_push",
    "format_fail_push",
    "format_docs_page",
    "format_doc_detail",
    "format_edit_menu",
    "format_delete_confirm",
    "format_tag_result",
    "format_search_result",
    # Tasks
    "format_task_line",
    "format_tasks_page",
    "format_task_detail",
    "format_task_confirm_added",
    "format_task_delete_confirm",
    # Subscriptions
    "format_amount",
    "format_subscription_line",
    "format_subscriptions_page",
    "format_subscription_detail",
    "format_subscription_confirm_added",
    "format_subscription_delete_confirm",
    "format_subs_stats",
]

"""tg_assist — aiogram 3 entry point (documents end-to-end).

Startup order is fixed:
  1. load settings
  2. validate encryption key (fail-closed)
  3. init DB schema
  4. probe peer containers (best-effort)
  5. create Bot, register it in bot_instance
  6. create Dispatcher with SQLStorage (FSM survives restart)
  7. attach AllowedChatMiddleware
  8. register /start handler + inject scheduler/settings/db_path
  9. include routers (docs_ask → photo → document → confirm → reminders)
 10. create + start scheduler (persistent reminder jobs resume here)
 11. requeue orphan jobs (worker crashed mid-task on previous boot)
 12. kick off background BGE-m3 pre-warm + worker_loop
 13. start polling; on shutdown cancel worker, stop scheduler, close bot session
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from typing import Any, Awaitable, Callable

import httpx
from aiogram import BaseMiddleware, Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, TelegramObject

from config import Settings
from handlers import (
    confirm_handler,
    docs_ask_handler,
    docs_handler,
    document_handler,
    photo_handler,
    subs_handler,
    tasks_handler,
    text_dispatch,
)
from handlers import reminders as reminders_handlers
from services import embeddings, jobs
from services.bot_instance import set_bot
from services.encryption import validate_key
from services.fsm_storage import SQLStorage
from services.scheduler import SchedulerService
from services.storage import init_db
from services.worker import worker_loop
from services.worker_instance import clear_worker, set_worker


logger = logging.getLogger(__name__)


class AllowedChatMiddleware(BaseMiddleware):
    """Outer middleware: drops messages from any chat other than ``allowed_chat_id``.

    Unauthorised events are silently ignored (no reply back to the sender) and
    logged as WARNING so the bot owner can spot probing.
    """

    def __init__(self, allowed_chat_id: int) -> None:
        self._allowed = allowed_chat_id

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        chat_id = getattr(getattr(event, "chat", None), "id", None)
        if chat_id is not None and chat_id != self._allowed:
            logger.warning(
                "unauthorized chat_id=%s (dropping %s silently)",
                chat_id,
                type(event).__name__,
            )
            return None
        return await handler(event, data)


async def ensure_dependencies_ok(settings: Settings) -> None:
    """Best-effort connectivity check for peer containers.

    Does NOT hard-fail on unavailability: the bot can still serve /start and
    manual commands while a dependency is recovering. Writes a WARNING so the
    operator knows.
    """
    targets = (
        ("tg_ollama", f"{settings.OLLAMA_URL}/api/version"),
        ("tg_qdrant", f"{settings.QDRANT_URL}/readyz"),
    )
    async with httpx.AsyncClient(timeout=3.0) as client:
        for name, url in targets:
            ok = False
            for attempt in range(1, 4):
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        logger.info("dependency OK: %s (%s)", name, url)
                        ok = True
                        break
                    logger.debug(
                        "dependency %s attempt %d returned HTTP %s",
                        name,
                        attempt,
                        resp.status_code,
                    )
                except httpx.RequestError as exc:
                    logger.debug(
                        "dependency %s attempt %d failed: %s", name, attempt, exc
                    )
                if attempt < 3:
                    await asyncio.sleep(2)
            if not ok:
                logger.warning(
                    "dependency %s unreachable at %s — bot will start but related "
                    "features may degrade until the service comes up",
                    name,
                    url,
                )


def _start_message_text() -> str:
    return (
        "Привет! Это <b>tg_assist</b>.\n\n"
        "Что я умею:\n"
        "• <b>Напоминания</b> — <i>«напомни завтра в 14 позвонить»</i>\n"
        "• <b>Задачи</b> — <i>«добавь задачу купить молоко к пятнице, home»</i>\n"
        "• <b>Подписки</b> — <i>«Spotify 169₽ Сбер каждое 15 число»</i>\n"
        "• <b>Документы</b> — пришли фото или PDF; можно сохранить с шифрованием+OCR "
        "или просто распознать текст (caption «распознай»).\n\n"
        "Команды:\n"
        "• /reminders — активные напоминания\n"
        "• /subs — мои подписки\n"
        "• /subs_stats — статистика по месяцам + прогноз\n"
        "• /sub_edit &lt;id&gt; — изменить подписку\n"
        "• /sub_delete &lt;id&gt; — удалить подписку\n"
        "• /tasks — активные задачи\n"
        "• /tasks_all — все задачи (включая done)\n"
        "• /tasks_by_cat &lt;cat&gt; — задачи по категории\n"
        "• /task_done &lt;id&gt; — закрыть задачу\n"
        "• /task_edit &lt;id&gt; — изменить задачу\n"
        "• /task_delete &lt;id&gt; — удалить задачу\n"
        "• /docs — мои документы\n"
        "• /docs_find &lt;запрос&gt; — поиск по документам (FTS + семантика)\n"
        "• /docs_by_type &lt;тип&gt; — фильтр по типу\n"
        "• /docs_by_tag &lt;тег&gt; — фильтр по тегу\n"
        "• /doc_status [id] — статус обработки\n"
        "• /doc_edit &lt;id&gt; — изменить тип / теги / режим / PII\n"
        "• /doc_tag &lt;id&gt; tag1,tag2 — добавить теги\n"
        "• /doc_untag &lt;id&gt; &lt;tag&gt; — убрать тег\n"
        "• /doc_delete &lt;id&gt; — удалить документ полностью\n"
        "• /start — эта справка"
    )


async def _warm_embeddings() -> None:
    """Pre-load BGE-m3 in the background so first upload doesn't wait."""
    try:
        await embeddings.warm()
        logger.info("BGE-m3 pre-warm complete")
    except Exception as exc:  # noqa: BLE001
        logger.exception("BGE-m3 pre-warm failed: %s — first embed will retry", exc)


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("tg_assist starting")

    try:
        settings = Settings.load_from_env()
    except ValueError as exc:
        logger.error("configuration error: %s", exc)
        sys.exit(2)

    try:
        validate_key(settings.DOCS_ENCRYPTION_KEY)
    except ValueError as exc:
        logger.error("encryption key invalid: %s", exc)
        logger.error(
            "generate with: "
            'python -c "import os, base64; print(base64.b64encode(os.urandom(32)).decode())"'
        )
        sys.exit(3)

    await init_db(settings.DB_PATH)
    logger.info("sqlite schema initialised at %s", settings.DB_PATH)

    await ensure_dependencies_ok(settings)

    bot = Bot(
        token=settings.TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    set_bot(bot)  # required before scheduler starts — _fire_reminder looks it up

    dp = Dispatcher(storage=SQLStorage(settings.DB_PATH))
    dp.message.outer_middleware(AllowedChatMiddleware(settings.ALLOWED_CHAT_ID))

    @dp.message(CommandStart())
    async def _start_handler(message: Message) -> None:
        await message.answer(_start_message_text())

    scheduler = SchedulerService(settings.DB_PATH)

    # Inject scheduler + settings + db_path for handlers that receive them as kwargs.
    dp["scheduler"] = scheduler
    dp["settings"] = settings
    dp["db_path"] = settings.DB_PATH

    # Order matters: docs_ask first (specific /doc_status + regex), then the
    # file-upload handlers, then confirm callbacks, then per-domain slash +
    # callback handlers (reminders / tasks / subs), and finally text_dispatch
    # which is the catch-all for ``F.text & ~startswith("/")`` and routes
    # the message to the right per-intent parser via ``services.router``.
    dp.include_router(docs_ask_handler.router)
    dp.include_router(docs_handler.router)  # /docs, /docs_find, /doc_edit, ...
    dp.include_router(photo_handler.router)
    dp.include_router(document_handler.router)
    dp.include_router(confirm_handler.router)
    dp.include_router(reminders_handlers.router)
    dp.include_router(tasks_handler.router)
    dp.include_router(subs_handler.router)
    dp.include_router(text_dispatch.router)  # MUST be last — catch-all for free text

    scheduler.start()  # resumes persistent reminder jobs

    # Document-worker queue: flush orphans from a prior crashed boot, then
    # start the loop that will pick them up along with fresh jobs.
    orphans = await jobs.requeue_orphans(settings.DB_PATH)
    if orphans:
        logger.info("requeued %d orphan jobs from previous boot", orphans)

    warm_task = asyncio.create_task(_warm_embeddings())
    worker_task = asyncio.create_task(worker_loop(bot, settings))
    set_worker(worker_task)

    logger.info(
        "polling started (ALLOWED_CHAT_ID=%s, privacy_strict=%s)",
        settings.ALLOWED_CHAT_ID,
        settings.PRIVACY_STRICT,
    )
    try:
        await dp.start_polling(bot)
    finally:
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
        clear_worker()
        warm_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await warm_task
        scheduler.shutdown(wait=False)
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("tg_assist stopped by KeyboardInterrupt")

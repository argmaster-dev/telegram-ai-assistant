"""Async client for Google Gemini API.

Files API methods only (``upload_file``, ``extract_text_from_file``,
``delete_file``) for OCR of large PUBLIC documents. Later versions may extend this
class with ``text``, ``text_with_grounding``, ``vision``, ``image_gen``, ``stt``.

Tests pass a ``transport=httpx.MockTransport(...)`` to intercept HTTP without
hitting the real Gemini service.

PRIVACY_STRICT guard: when ``privacy_strict=True`` is set on the client, every
public method raises ``PrivacyStrictBlocked`` BEFORE any network call.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import httpx

from services.ocr_errors import GeminiUnavailable, PrivacyStrictBlocked


logger = logging.getLogger(__name__)


_API_BASE = "https://generativelanguage.googleapis.com"
_DEFAULT_MODEL = "gemini-3-flash-preview"
_RATE_LIMIT_SEC = 6.5  # Free-tier Flash quota
_DEFAULT_TIMEOUT = 60.0

_OCR_PROMPT = (
    "Извлеки полный текст и таблицы из приложенного документа. "
    "Сохрани структуру: используй markdown для заголовков, списков и таблиц. "
    "Не добавляй никаких комментариев, не оборачивай ответ в кодовые блоки. "
    "Верни только сам markdown с извлечённым содержимым."
)


class GeminiClient:
    """Async Gemini client with rate-limiting and a mock-friendly transport."""

    def __init__(
        self,
        *,
        api_key: str,
        privacy_strict: bool = False,
        model: str = _DEFAULT_MODEL,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        rate_limit_sec: float = _RATE_LIMIT_SEC,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._privacy_strict = privacy_strict
        self._model = model
        self._transport = transport
        self._rate_limit_sec = rate_limit_sec
        self._timeout = timeout
        self._last_call: float = 0.0
        self._rate_lock = asyncio.Lock()

    # ---- guards ---------------------------------------------------------

    def _check_privacy(self) -> None:
        if self._privacy_strict:
            raise PrivacyStrictBlocked(
                "PRIVACY_STRICT=true — Gemini calls are disabled by configuration"
            )

    def _check_api_key(self) -> None:
        if not self._api_key:
            raise GeminiUnavailable("Gemini API key is not configured")

    async def _throttle(self) -> None:
        async with self._rate_lock:
            since = time.monotonic() - self._last_call
            wait = self._rate_limit_sec - since
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

    def _client(self) -> httpx.AsyncClient:
        if self._transport is not None:
            return httpx.AsyncClient(
                transport=self._transport, timeout=self._timeout
            )
        return httpx.AsyncClient(timeout=self._timeout)

    # ---- public API -----------------------------------------------------

    async def upload_file(self, data: bytes, mime_type: str) -> str:
        """Upload raw bytes to Gemini Files API. Returns the file URI."""
        self._check_privacy()
        self._check_api_key()
        await self._throttle()

        url = f"{_API_BASE}/upload/v1beta/files?key={self._api_key}"
        headers = {
            "X-Goog-Upload-Protocol": "raw",
            "Content-Type": mime_type,
        }

        try:
            async with self._client() as client:
                resp = await client.post(url, headers=headers, content=data)
        except httpx.RequestError as exc:
            raise GeminiUnavailable(f"upload_file network error: {exc}") from exc

        if resp.status_code >= 400:
            raise GeminiUnavailable(
                f"upload_file HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise GeminiUnavailable(f"upload_file: invalid JSON: {exc}") from exc

        file_obj = payload.get("file", {})
        uri = file_obj.get("uri") or file_obj.get("name")
        if not uri:
            raise GeminiUnavailable(
                f"upload_file: response missing file.uri: {payload!r}"
            )
        return uri

    async def extract_text_from_file(
        self,
        file_uri: str,
        mime_type: str,
        *,
        prompt: str = _OCR_PROMPT,
    ) -> str:
        """Run generateContent over a file already uploaded via ``upload_file``.

        Returns the model's reply (markdown, per the prompt).
        """
        self._check_privacy()
        self._check_api_key()
        await self._throttle()

        url = (
            f"{_API_BASE}/v1beta/models/{self._model}:generateContent"
            f"?key={self._api_key}"
        )
        body: dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {
                            "file_data": {
                                "file_uri": file_uri,
                                "mime_type": mime_type,
                            }
                        },
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0},
        }

        try:
            async with self._client() as client:
                resp = await client.post(url, json=body)
        except httpx.RequestError as exc:
            raise GeminiUnavailable(f"extract_text network error: {exc}") from exc

        if resp.status_code >= 400:
            raise GeminiUnavailable(
                f"extract_text HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise GeminiUnavailable(f"extract_text: invalid JSON: {exc}") from exc

        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GeminiUnavailable(
                f"extract_text: unexpected response shape: {payload!r}"
            ) from exc

        return text

    async def delete_file(self, file_uri_or_name: str) -> None:
        """Best-effort cleanup; logs warning on failure (Files API auto-expires in 48h)."""
        self._check_privacy()
        self._check_api_key()

        # The DELETE endpoint expects the resource name "files/<id>". Accept both
        # the bare name and a full URI.
        if "/files/" in file_uri_or_name:
            name = "files/" + file_uri_or_name.split("/files/", 1)[1].rstrip("/")
        elif file_uri_or_name.startswith("files/"):
            name = file_uri_or_name
        else:
            name = f"files/{file_uri_or_name}"

        url = f"{_API_BASE}/v1beta/{name}?key={self._api_key}"

        try:
            async with self._client() as client:
                resp = await client.delete(url)
        except httpx.RequestError as exc:
            logger.warning("delete_file network error (ignored): %s", exc)
            return

        if resp.status_code >= 400:
            logger.warning(
                "delete_file HTTP %s (ignored, Files API auto-expires): %s",
                resp.status_code,
                resp.text[:200],
            )

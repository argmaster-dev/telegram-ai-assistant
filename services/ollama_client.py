"""Async client for the local Ollama container (``tg_ollama``).

Uses Ollama's OpenAI-compatible endpoint ``/v1/chat/completions``. JSON output
is forced both via ``response_format={"type": "json_object"}`` and an explicit
system-prompt instruction, because ``qwen3:8b`` is a thinking model that
occasionally emits ``<think>...</think>`` blocks — those are stripped before
JSON parsing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional

import httpx


logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OllamaError(Exception):
    """Base class for ollama_client errors."""


class OllamaUnavailable(OllamaError):
    """Raised when Ollama is unreachable or consistently 5xx after retries."""


class OllamaInvalidJSON(OllamaError):
    """Raised when the response content cannot be parsed as JSON."""


def _strip_think(content: str) -> str:
    return _THINK_BLOCK_RE.sub("", content).strip()


async def generate_json(
    prompt: str,
    *,
    base_url: str,
    model: str = "qwen3:8b",
    timeout: float = 30.0,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Send ``prompt`` and return the parsed JSON response.

    Retries once on transient network errors or HTTP 5xx (total attempts
    = ``max_retries + 1``). Non-JSON output raises ``OllamaInvalidJSON`` without
    retrying — the model is expected to follow the schema on the first try and
    retrying usually costs more than it helps.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You always reply with a single valid JSON object. "
                    "Never include prose, markdown, or code fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
        "temperature": 0.0,
    }

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 2):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
            if resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"ollama 5xx: {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
            resp.raise_for_status()
            data = resp.json()
            content: str = data["choices"][0]["message"]["content"]
            cleaned = _strip_think(content)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as exc:
                raise OllamaInvalidJSON(
                    f"non-JSON response from {model}: {cleaned[:200]!r}"
                ) from exc
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            logger.debug("ollama attempt %d failed: %s", attempt, exc)
            if attempt <= max_retries:
                await asyncio.sleep(0.5 * attempt)

    raise OllamaUnavailable(
        f"ollama unreachable after {max_retries + 1} attempts: {last_exc}"
    )


async def healthcheck(base_url: str, timeout: float = 3.0) -> bool:
    """Return True if ``GET /v1/models`` answers HTTP 200 within ``timeout``."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/v1/models")
        return resp.status_code == 200
    except httpx.RequestError as exc:
        logger.debug("ollama healthcheck failed: %s", exc)
        return False

"""Intent router — first stage of the free-text pipeline.

The router extends the intent set from ``reminder | unknown`` to
``reminder | task | subscription | unknown``. The router is intentionally
thin: it returns only ``intent`` + ``confidence`` so callers can decide
which per-intent parser (reminders pipeline, ``tasks.parse_task``,
``subscriptions.parse_subscription``) to invoke next.
"""
from __future__ import annotations

import logging
from typing import Literal, TypedDict

from services.ollama_client import OllamaError, generate_json
from utils.prompts import INTENT_ROUTER_PROMPT


logger = logging.getLogger(__name__)


Intent = Literal["reminder", "task", "subscription", "unknown"]

_VALID_INTENTS: frozenset[str] = frozenset(
    {"reminder", "task", "subscription", "unknown"}
)


class ClassifyResult(TypedDict):
    intent: Intent
    confidence: float


async def classify(text: str, *, base_url: str) -> ClassifyResult:
    """Return ``{"intent", "confidence"}`` for ``text``.

    Defensive: any Ollama error or malformed output returns
    ``{"intent": "unknown", "confidence": 0.0}`` so callers have a single
    degradation path. Unknown intent labels (anything outside
    ``_VALID_INTENTS``) are coerced to ``"unknown"``.
    """
    prompt = INTENT_ROUTER_PROMPT.format(text=text)
    try:
        raw = await generate_json(prompt, base_url=base_url)
    except OllamaError as exc:
        logger.warning("router classify failed (ollama): %s", exc)
        return {"intent": "unknown", "confidence": 0.0}

    intent_raw = str(raw.get("intent", "unknown")).strip()
    intent: Intent = intent_raw if intent_raw in _VALID_INTENTS else "unknown"  # type: ignore[assignment]

    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {"intent": intent, "confidence": confidence}

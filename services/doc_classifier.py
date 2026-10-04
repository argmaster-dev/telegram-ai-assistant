"""Document type classifier — rules first, Ollama LLM fallback.

Returns one of seven closed types: ``passport`` / ``snils`` / ``driving`` /
``inn`` / ``policy`` / ``contract`` / ``other``. Used by the
``ConfirmType`` FSM state to propose a default choice that the user can
override with inline-keyboard buttons.

Flow:
  1. ``classify_by_rules(text)`` computes a per-rule score
     ``(hits_keywords − hits_anti) / len(keywords)``, clamped to ``[0, 1]``.
  2. If the best score ≥ 0.5 — done, source = "rules".
  3. Otherwise call ``classify_llm(text, base_url=…)`` which asks
     ``qwen3:8b`` to pick a type from the closed list; source = "llm".
  4. If LLM is disabled or unreachable — return the rules result (usually
     ``("other", <low_score>)``).

DoD requirement: ``classify_llm`` only forwards the first 1500 chars of
``text`` to the model, never the full document.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

from services.ollama_client import OllamaError, generate_json
from utils.prompts import DOC_CLASSIFIER_PROMPT


logger = logging.getLogger(__name__)


DocType = Literal[
    "passport", "snils", "driving", "inn", "policy", "contract", "other"
]

_VALID_TYPES: frozenset[str] = frozenset(
    {"passport", "snils", "driving", "inn", "policy", "contract", "other"}
)


# Keywords are matched as substrings on the lower-cased text — cheap and
# good enough for the current test corpus. Anti-keywords demote a rule
# when the text is adjacent but distinct (e.g. "заграничный паспорт" ≠
# internal passport).
DOC_TYPE_RULES: tuple[dict, ...] = (
    {
        "type": "passport",
        "keywords": ("паспорт", "серия", "выдан", "российская федерация"),
        "anti": ("заграничный", "международный"),
    },
    {
        "type": "snils",
        "keywords": (
            "снилс",
            "страховое свидетельство",
            "пенсионного страхования",
        ),
        "anti": (),
    },
    {
        "type": "driving",
        "keywords": (
            "водительское удостоверение",
            "категория",
            "правила дорожного движения",
        ),
        "anti": (),
    },
    {
        "type": "inn",
        "keywords": ("инн", "идентификационный номер налогоплательщика"),
        "anti": (),
    },
    {
        "type": "policy",
        "keywords": ("полис", "страхование", "омс", "дмс"),
        "anti": (),
    },
    {
        "type": "contract",
        "keywords": (
            "договор",
            "стороны",
            "предмет договора",
            "приложение к договору",
        ),
        "anti": (),
    },
)


_RULES_THRESHOLD = 0.5
_LLM_CHAR_LIMIT = 1500
_LLM_CONFIDENCE = 0.5  # constant confidence reported for LLM answers


def classify_by_rules(text: str) -> tuple[str, float]:
    """Return the best-scoring rule type and its confidence in ``[0, 1]``.

    If no rule exceeds 0.3, falls back to ``("other", best_score)``.
    """
    lowered = text.lower()
    best_type = "other"
    best_score = 0.0
    for rule in DOC_TYPE_RULES:
        keywords = rule["keywords"]
        anti = rule["anti"]
        hits = sum(1 for kw in keywords if kw in lowered)
        anti_hits = sum(1 for kw in anti if kw in lowered)
        score = max(0.0, min(1.0, (hits - anti_hits) / len(keywords)))
        if score > best_score:
            best_score = score
            best_type = rule["type"]
    if best_score < 0.3:
        return ("other", best_score)
    return (best_type, best_score)


async def classify_llm(text: str, *, base_url: str) -> str:
    """Ask ``qwen3:8b`` to pick a type from the closed 7-element list.

    Only the first ``_LLM_CHAR_LIMIT`` characters of ``text`` are forwarded
    — any value outside the closed list collapses to ``"other"``.
    """
    trimmed = text[:_LLM_CHAR_LIMIT]
    prompt = DOC_CLASSIFIER_PROMPT.format(text=trimmed)
    try:
        raw = await generate_json(prompt, base_url=base_url)
    except OllamaError as exc:
        logger.warning("classify_llm failed (ollama): %s", exc)
        return "other"

    value = str(raw.get("type", "other")).strip().lower()
    if value not in _VALID_TYPES:
        logger.debug("classify_llm returned unknown type %r — coerced to other", value)
        return "other"
    return value


async def classify(
    text: str,
    *,
    base_url: Optional[str] = None,
    use_llm_fallback: bool = True,
) -> tuple[str, float, str]:
    """Dispatch rules → LLM fallback, returning ``(type, confidence, source)``.

    * source ``"rules"`` when rules are confident enough (score ≥ 0.5) or
      when LLM fallback is disabled / unavailable.
    * source ``"llm"`` when rules were low-confidence and LLM was consulted.
    """
    rule_type, rule_conf = classify_by_rules(text)
    if rule_conf >= _RULES_THRESHOLD:
        return (rule_type, rule_conf, "rules")

    if use_llm_fallback and base_url:
        llm_type = await classify_llm(text, base_url=base_url)
        return (llm_type, _LLM_CONFIDENCE, "llm")

    return (rule_type, rule_conf, "rules")


__all__ = [
    "DOC_TYPE_RULES",
    "DocType",
    "classify",
    "classify_by_rules",
    "classify_llm",
]

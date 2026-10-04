"""PII detection and masking — local, defence-in-depth before any upload.

Detects eight structured categories via regex (``phone``, ``passport_rf``,
``snils``, ``inn_10``, ``inn_12``, ``email``, ``card_16``, ``card_20``) plus a
ninth free-form category ``name`` via ``natasha.NamesExtractor`` backed by a
top-5000 surname dictionary (``data/surnames_ru.txt``).

Used by ``_run_ocr`` and ``confirm_handler`` to decide whether
PUBLIC Gemini flow is allowed and what text to store in Qdrant.

Design notes (by the contract §1, §6, §8):
  * ``passport_rf``, ``inn_10``, ``inn_12`` need **contextual** regex — the
    raw number pattern alone is too permissive. A match is kept only if one
    of the trigger words (``паспорт|серия|номер`` / ``инн``) is within 30
    characters of it.
  * ``mask_pii`` is idempotent: re-running it on an already masked string
    leaves the string unchanged.
  * ``natasha`` runs lazily — ``MorphVocab`` init takes ~1–2 s and should
    not happen at import time.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PiiMatch:
    type: str
    value: str
    span: tuple[int, int]


@dataclass
class PiiReport:
    has_pii: bool
    categories: list[str] = field(default_factory=list)
    matches: list[PiiMatch] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------


# Russian mobile: +7 / 8 prefix, then 9XX XXX XX XX with flexible separators.
# (?<!\d) / (?!\d) — word boundary that also rejects "sitting inside a longer
# digit string" (e.g. avoids partial match inside a 16-digit card number).
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+7|8)[\s\-\(\)]*9\d{2}[\s\-\(\)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)"
)

# Passport: 4 digits (series) + 6 digits (number), optional separator.
_PASSPORT_RE = re.compile(r"\b\d{4}\s?\d{6}\b")

# SNILS formatted: XXX-XXX-XXX XX or XXX-XXX-XXX-XX.
_SNILS_RE = re.compile(r"\b\d{3}-\d{3}-\d{3}[\s\-]\d{2}\b")

# INN: exactly 10 (legal) or 12 (individual). Standalone 10/12-digit tokens
# are too easy to false-positive, so they are context-gated.
_INN12_RE = re.compile(r"\b\d{12}\b")
_INN10_RE = re.compile(r"\b\d{10}\b")

# Email — RFC-lite.
_EMAIL_RE = re.compile(r"\b[\w\.\-]+@[\w\.\-]+\.\w{2,}\b")

# Card numbers: 16 or 20 digits in groups of 4, optional space/dash.
_CARD20_RE = re.compile(
    r"\b\d{4}[ \-]?\d{4}[ \-]?\d{4}[ \-]?\d{4}[ \-]?\d{4}\b"
)
_CARD16_RE = re.compile(r"\b\d{4}[ \-]?\d{4}[ \-]?\d{4}[ \-]?\d{4}\b")


@dataclass(frozen=True)
class _Pattern:
    name: str
    regex: re.Pattern
    context_words: tuple[str, ...] = ()


# Order matters for overlap dedup: longer / more specific patterns first so
# ``_dedupe_overlaps`` keeps them when ``card_20`` and ``card_16`` overlap on
# the same 16-digit prefix of a 20-digit number.
_PATTERNS: tuple[_Pattern, ...] = (
    _Pattern("card_20", _CARD20_RE),
    _Pattern("card_16", _CARD16_RE),
    _Pattern("phone", _PHONE_RE),
    _Pattern("passport_rf", _PASSPORT_RE, context_words=("паспорт", "серия", "номер")),
    _Pattern("snils", _SNILS_RE),
    _Pattern("inn_12", _INN12_RE, context_words=("инн",)),
    _Pattern("inn_10", _INN10_RE, context_words=("инн",)),
    _Pattern("email", _EMAIL_RE),
)


PII_PATTERNS: dict[str, re.Pattern] = {p.name: p.regex for p in _PATTERNS}


_PLACEHOLDERS: dict[str, str] = {
    "phone": "[PHONE]",
    "passport_rf": "[PASSPORT]",
    "snils": "[SNILS]",
    "inn_10": "[INN]",
    "inn_12": "[INN]",
    "email": "[EMAIL]",
    "card_16": "[CARD]",
    "card_20": "[CARD]",
    "name": "[NAME]",
}


_CONTEXT_RADIUS = 30


def _has_context(
    text: str, span: tuple[int, int], words: tuple[str, ...]
) -> bool:
    """True iff at least one of ``words`` occurs within ±``_CONTEXT_RADIUS`` of ``span``."""
    if not words:
        return True
    start, end = span
    win_start = max(0, start - _CONTEXT_RADIUS)
    win_end = min(len(text), end + _CONTEXT_RADIUS)
    window = text[win_start:win_end].lower()
    return any(w in window for w in words)


# ---------------------------------------------------------------------------
# ФИО detector — natasha + surname dictionary (complementary signals)
# ---------------------------------------------------------------------------


_SURNAMES_PATH = Path(__file__).resolve().parent.parent / "data" / "surnames_ru.txt"
_surnames: Optional[frozenset[str]] = None
_names_extractor = None  # type: ignore[var-annotated]


def _load_surnames() -> frozenset[str]:
    """Lazy-load and cache the lowercase top-5000 surnames frozenset."""
    global _surnames
    if _surnames is not None:
        return _surnames
    try:
        with _SURNAMES_PATH.open(encoding="utf-8") as f:
            _surnames = frozenset(
                line.strip().lower() for line in f if line.strip()
            )
        logger.info("loaded %d surnames from %s", len(_surnames), _SURNAMES_PATH)
    except FileNotFoundError:
        logger.warning(
            "surnames dictionary not found at %s — ФИО detection degraded to natasha-only",
            _SURNAMES_PATH,
        )
        _surnames = frozenset()
    return _surnames


def _get_names_extractor():
    """Lazy-build the natasha NamesExtractor (expensive init: ~1–2 s)."""
    global _names_extractor
    if _names_extractor is not None:
        return _names_extractor
    from natasha import MorphVocab, NamesExtractor

    _names_extractor = NamesExtractor(MorphVocab())
    logger.info("natasha NamesExtractor initialised")
    return _names_extractor


_SURNAME_WORD_RE = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\b")


def _natasha_match_accepted(match, surnames: frozenset[str]) -> bool:
    """Accept a natasha match only if BOTH filters pass:

    (a) ≥ 2 non-empty fact components (first/last/middle) — rules out
        bare capitalised nouns ("Прогноз", "Москве", "В");
    (b) at least one component (lower-cased, stripped of trailing '.')
        is present in the surname dictionary — rules out all-caps
        headline pairs like "ПЕРВАЯ фотокнига" where natasha returns
        first='ПЕРВАЯ', last='фотокнига' (2 components, but neither is
        actually a surname).

    Coverage for real ФИО:
      * "Иванов Иван Иванович" → {first,last,middle}, 'иванов' ∈ dict → accept
      * "Козлов А.В."          → {first='А',last='Козлов',middle='В'}; 'козлов' ∈ dict → accept
      * "А.В. Козлов"          → same shape → accept
      * "Алексей Козлов"       → {first,last}, 'козлов' ∈ dict → accept
      * "ВОЛКОВ ИВАН ИВАНОВИЧ" (all-caps passport) → 'волков' ∈ dict → accept
      * "ПЕРВАЯ фотокнига"     → neither word ∈ dict → reject
      * "Москве"               → 1 component → reject at step (a)

    If ``surnames`` is empty (very degraded config — dict file missing)
    we fall back to the original "≥2 components" rule so we don't silently
    drop every ФИО.
    """
    fact = match.fact
    non_empty = [
        getattr(fact, attr, None)
        for attr in ("first", "last", "middle")
        if getattr(fact, attr, None)
    ]
    if len(non_empty) < 2:
        return False
    if not surnames:
        return True
    for comp in non_empty:
        token = str(comp).lower().strip(" .")
        if token and token in surnames:
            return True
    return False


def _detect_names(text: str) -> list[PiiMatch]:
    """Collect ``name`` matches from natasha + surname-dictionary lookups.

    natasha is tightened via ``_natasha_match_accepted`` — only multi-
    component matches qualify. A separate dictionary pass afterwards
    catches single bare surnames that natasha dismissed, using a capital-
    letter / ≥3-char regex to avoid noisy short or lowercase tokens.
    """
    matches: list[PiiMatch] = []
    seen: set[tuple[int, int]] = set()
    surnames = _load_surnames()

    try:
        extractor = _get_names_extractor()
        for m in extractor(text):
            if not _natasha_match_accepted(m, surnames):
                continue
            span = (m.start, m.stop)
            if span in seen:
                continue
            matches.append(
                PiiMatch(type="name", value=text[m.start : m.stop], span=span)
            )
            seen.add(span)
    except Exception as exc:  # noqa: BLE001
        # natasha import or runtime error must not break PII detection;
        # fall back to dictionary only.
        logger.warning("natasha extractor failed (%s) — surname-only fallback", exc)

    if surnames:
        for word_match in _SURNAME_WORD_RE.finditer(text):
            span = word_match.span()
            word_lower = word_match.group().lower()
            if word_lower in surnames and span not in seen:
                matches.append(
                    PiiMatch(type="name", value=word_match.group(), span=span)
                )
                seen.add(span)

    return matches


# ---------------------------------------------------------------------------
# Overlap dedup
# ---------------------------------------------------------------------------


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _dedupe_overlaps(matches: Iterable[PiiMatch]) -> list[PiiMatch]:
    """Keep the longest match when two matches overlap."""
    # Prefer longer spans; stable order for equal lengths by earlier start.
    ordered = sorted(
        matches,
        key=lambda m: (-(m.span[1] - m.span[0]), m.span[0]),
    )
    kept: list[PiiMatch] = []
    for m in ordered:
        if any(_overlaps(m.span, k.span) for k in kept):
            continue
        kept.append(m)
    kept.sort(key=lambda m: m.span[0])
    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_pii(text: str) -> PiiReport:
    """Scan ``text`` for all supported PII categories and return a report."""
    raw: list[PiiMatch] = []

    for pat in _PATTERNS:
        for m in pat.regex.finditer(text):
            if not _has_context(text, m.span(), pat.context_words):
                continue
            raw.append(
                PiiMatch(type=pat.name, value=m.group(), span=m.span())
            )

    raw.extend(_detect_names(text))

    merged = _dedupe_overlaps(raw)
    categories = sorted({m.type for m in merged})
    return PiiReport(has_pii=bool(merged), categories=categories, matches=merged)


def mask_pii(text: str, categories: Optional[Iterable[str]] = None) -> str:
    """Return ``text`` with every matched span replaced by its placeholder.

    ``categories`` — optional iterable restricting which categories are
    masked; ``None`` (default) masks every detected category. The function is
    idempotent: a second call on the output yields the same string.
    """
    report = detect_pii(text)
    wanted: Optional[set[str]] = set(categories) if categories is not None else None

    targets = [
        m for m in report.matches if wanted is None or m.type in wanted
    ]
    # Apply from right to left so earlier spans remain valid after each splice.
    targets.sort(key=lambda m: m.span[0], reverse=True)
    out = text
    for m in targets:
        placeholder = _PLACEHOLDERS.get(m.type, "[PII]")
        out = out[: m.span[0]] + placeholder + out[m.span[1] :]
    return out


__all__ = [
    "PII_PATTERNS",
    "PiiMatch",
    "PiiReport",
    "detect_pii",
    "mask_pii",
]

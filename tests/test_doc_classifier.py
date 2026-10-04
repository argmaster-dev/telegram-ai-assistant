"""Tests for services.doc_classifier — rules + Ollama LLM fallback (mocked)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from services import doc_classifier


_FIXTURES = Path(__file__).parent / "fixtures" / "sample_texts"


def _read(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# classify_by_rules
# ---------------------------------------------------------------------------


def test_rules_recognise_passport_fixture():
    type_, conf = doc_classifier.classify_by_rules(_read("passport.txt"))
    assert type_ == "passport"
    assert conf >= 0.7


def test_rules_recognise_snils_fixture():
    type_, conf = doc_classifier.classify_by_rules(_read("snils.txt"))
    assert type_ == "snils"
    assert conf >= 0.7


def test_rules_recognise_contract_fixture():
    type_, conf = doc_classifier.classify_by_rules(_read("contract.txt"))
    assert type_ == "contract"
    assert conf >= 0.5


def test_rules_clean_text_falls_through_to_other():
    type_, conf = doc_classifier.classify_by_rules(_read("clean_no_pii.txt"))
    assert type_ == "other"
    assert conf < 0.5


def test_rules_anti_keywords_demote_rule():
    """Text about a foreign passport should NOT be classified as passport."""
    text = (
        "Заграничный паспорт — международный документ для выезда. "
        "Содержит сведения о гражданине для путешествий за рубеж."
    )
    type_, _conf = doc_classifier.classify_by_rules(text)
    assert type_ != "passport"


# ---------------------------------------------------------------------------
# classify() dispatcher — rules path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_rules_path_short_circuits_llm():
    """When rules score ≥ 0.5 we return early with source='rules' and no LLM call."""
    mock_gen = AsyncMock(return_value={"type": "other"})
    with patch("services.doc_classifier.generate_json", mock_gen):
        type_, conf, source = await doc_classifier.classify(
            _read("passport.txt"),
            base_url="http://tg_ollama:11434",
            use_llm_fallback=True,
        )
    assert type_ == "passport"
    assert conf >= 0.7
    assert source == "rules"
    mock_gen.assert_not_called()


# ---------------------------------------------------------------------------
# classify() dispatcher — LLM fallback path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_llm_fallback_invoked_on_low_confidence():
    """Unclassifiable text triggers classify_llm and reports source='llm'."""
    mock_gen = AsyncMock(return_value={"type": "policy"})
    with patch("services.doc_classifier.generate_json", mock_gen):
        type_, _conf, source = await doc_classifier.classify(
            "Какой-то нейтральный текст без ключевых слов о документах.",
            base_url="http://tg_ollama:11434",
            use_llm_fallback=True,
        )
    assert type_ == "policy"
    assert source == "llm"
    mock_gen.assert_awaited_once()


@pytest.mark.asyncio
async def test_classify_llm_output_out_of_enum_coerced_to_other():
    """If the LLM returns an off-list value we collapse to 'other'."""
    mock_gen = AsyncMock(return_value={"type": "random-unseen-type"})
    with patch("services.doc_classifier.generate_json", mock_gen):
        type_, _conf, source = await doc_classifier.classify(
            "Нейтральный текст.",
            base_url="http://x",
            use_llm_fallback=True,
        )
    assert type_ == "other"
    assert source == "llm"


@pytest.mark.asyncio
async def test_classify_without_fallback_returns_rules_even_on_low_score():
    """use_llm_fallback=False keeps the classifier offline."""
    type_, _conf, source = await doc_classifier.classify(
        "Нейтральный текст.",
        base_url=None,
        use_llm_fallback=False,
    )
    assert source == "rules"


@pytest.mark.asyncio
async def test_classify_llm_trims_input_to_1500_chars():
    """DoD: the model must not see the full document (only first 1500 chars)."""
    long_text = "без ключевых слов " * 500  # ~9000 chars
    captured: dict[str, str] = {}

    async def fake_gen(prompt, **kwargs):
        captured["prompt"] = prompt
        return {"type": "other"}

    with patch("services.doc_classifier.generate_json", side_effect=fake_gen):
        await doc_classifier.classify(
            long_text, base_url="http://x", use_llm_fallback=True
        )

    # The template wraps the text, so the prompt is a bit longer than 1500 —
    # but the inserted document slice must be exactly 1500 chars.
    assert "prompt" in captured
    assert long_text[:1500] in captured["prompt"]
    assert long_text[:1501] not in captured["prompt"]

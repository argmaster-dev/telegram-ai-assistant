"""Tests for services.pii — regex detectors + natasha-backed ФИО detection."""
from __future__ import annotations

from pathlib import Path

import pytest

from services import pii


_FIXTURES = Path(__file__).parent / "fixtures" / "sample_texts"


# ---------------------------------------------------------------------------
# Phone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Звоните по номеру +7 (915) 123-45-67 в будни.",
        "мой телефон 8-915-123-45-67",
        "contact 89151234567",
    ],
)
def test_phone_regex_variants_detected(text: str):
    report = pii.detect_pii(text)
    assert report.has_pii is True
    assert "phone" in report.categories


# ---------------------------------------------------------------------------
# Passport (context-gated)
# ---------------------------------------------------------------------------


def test_passport_with_context_detected():
    report = pii.detect_pii("Мой паспорт 4510 123456, выдан в Москве.")
    assert "passport_rf" in report.categories
    assert any(m.value == "4510 123456" for m in report.matches)


def test_passport_without_context_not_detected():
    # 10-digit sequence split as 4+6 but no trigger word within 30 chars.
    report = pii.detect_pii(
        "Случайная цепочка чисел 1234 567890 просто так в тексте."
    )
    # Without the 'паспорт/серия/номер' context we explicitly ignore the raw
    # 4+6 pattern. The text also has no other PII.
    assert "passport_rf" not in report.categories


# ---------------------------------------------------------------------------
# SNILS / INN / email / card
# ---------------------------------------------------------------------------


def test_snils_formatted_detected():
    report = pii.detect_pii("СНИЛС: 112-233-445 95, оформлен в 2014.")
    assert "snils" in report.categories
    assert any(m.value == "112-233-445 95" for m in report.matches)


def test_inn_10_and_inn_12_differentiated_by_length():
    text_10 = "Моя организация, ИНН 7707083893 — главный офис."
    text_12 = "Мой ИНН 123456789012 как физлица."
    r10 = pii.detect_pii(text_10)
    r12 = pii.detect_pii(text_12)
    assert "inn_10" in r10.categories
    assert "inn_12" not in r10.categories
    assert "inn_12" in r12.categories
    assert "inn_10" not in r12.categories


def test_email_detected():
    report = pii.detect_pii("пиши мне на ivan.ivanov@mail.ru когда сможешь")
    assert "email" in report.categories
    assert any(m.value == "ivan.ivanov@mail.ru" for m in report.matches)


def test_card_16_detected():
    report = pii.detect_pii("номер карты 1234 5678 9012 3456 запомни")
    assert "card_16" in report.categories


# ---------------------------------------------------------------------------
# mask_pii
# ---------------------------------------------------------------------------


def test_mask_pii_replaces_phone_and_removes_original():
    original = "Звонить +7 (915) 123-45-67 в понедельник."
    masked = pii.mask_pii(original)
    assert "[PHONE]" in masked
    assert "915" not in masked
    assert "123-45-67" not in masked


def test_mask_pii_idempotent():
    original = (
        "тел +7 915 123-45-67, паспорт 4510 123456, "
        "email ivan@mail.ru, карта 1234 5678 9012 3456"
    )
    once = pii.mask_pii(original)
    twice = pii.mask_pii(once)
    assert once == twice
    # Also sanity-check placeholders are present.
    for placeholder in ("[PHONE]", "[PASSPORT]", "[EMAIL]", "[CARD]"):
        assert placeholder in once


def test_mask_pii_restricted_to_categories():
    text = "email ivan@mail.ru and паспорт 4510 123456"
    masked = pii.mask_pii(text, categories={"email"})
    assert "[EMAIL]" in masked
    # Passport must remain untouched because it's not in the category set.
    assert "4510 123456" in masked


# ---------------------------------------------------------------------------
# Clean text — zero matches
# ---------------------------------------------------------------------------


def test_clean_no_pii_fixture_has_no_pii():
    text = (_FIXTURES / "clean_no_pii.txt").read_text(encoding="utf-8")
    report = pii.detect_pii(text)
    # Weather report must NOT trigger any structured-PII category.
    structured = {
        "phone",
        "passport_rf",
        "snils",
        "inn_10",
        "inn_12",
        "email",
        "card_16",
        "card_20",
    }
    assert not (set(report.categories) & structured), (
        f"unexpected structured PII in weather text: {report.categories}"
    )


# ---------------------------------------------------------------------------
# ФИО detector — natasha + surname dictionary
# ---------------------------------------------------------------------------


def test_names_detected_in_full_triple():
    # Full ФИО triple — natasha is very confident here.
    report = pii.detect_pii("Иванов Иван Иванович работает в отделе продаж.")
    assert "name" in report.categories


def test_mask_pii_hides_name_placeholder():
    text = "Встречайте гостя: Петров Пётр Петрович приедет завтра."
    masked = pii.mask_pii(text)
    assert "[NAME]" in masked
    # Surname must no longer be in the masked text.
    assert "Петров" not in masked


def test_names_surname_with_initials_detected():
    """Сокращённые форматы ФИО (фамилия + инициалы) должны ловиться."""
    for text in (
        "Документ подписал Козлов А.В. от 15.04.2026",
        "А.В. Козлов приходил вчера.",
        "подпись: Петров П.П.",
    ):
        report = pii.detect_pii(text)
        assert "name" in report.categories, f"missed ФИО in: {text!r}"


def test_names_bare_capitalised_common_nouns_rejected():
    """Капитализированные общие слова не должны улетать в категорию name.

    Это главный выхлоп ужесточения natasha: одиночные matches принимаются
    только если токен есть в словаре фамилий. «Прогноз», «Москве», «погода»
    — не в словаре → не name.
    """
    text = (
        "Прогноз погоды на выходные. В субботу в Москве ожидается дождь. "
        "Столбик термометра поднимется до двадцати градусов тепла."
    )
    report = pii.detect_pii(text)
    for m in report.matches:
        if m.type != "name":
            continue
        assert m.value.lower() not in {
            "прогноз",
            "погоды",
            "погода",
            "в",
            "субботу",
            "выходные",
            "москве",
            "градусов",
            "столбик",
            "до",
        }, f"unexpected name FP on common noun: {m.value!r}"


def test_names_single_surname_accepted_via_dictionary():
    """Одиночная фамилия без имени/отчества должна ловиться через словарь."""
    report = pii.detect_pii("Козлов приходил вчера за документами.")
    assert "name" in report.categories

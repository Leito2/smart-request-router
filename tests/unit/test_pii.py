import pytest

pytest.importorskip("presidio_analyzer")

from routercore.adapters.pii import STRONG, PiiRedactor  # noqa: E402
from scripts.pii_scan import scan  # noqa: E402


@pytest.fixture(scope="module")
def redactor():
    return PiiRedactor()


def test_redacts_card_email_and_phone(redactor):
    text = "My card 4111 1111 1111 1111 was stolen, call +34 612 345 678 or ana@example.com"
    out = redactor.redact(text)
    assert out == "My card <CREDIT_CARD> was stolen, call <PHONE_NUMBER> or <EMAIL_ADDRESS>"


def test_redacts_spanish_iban_and_nif(redactor):
    out = redactor.redact("Mi IBAN es ES9121000418450200051332 y mi DNI 12345678Z")
    assert out == "Mi IBAN es <IBAN_CODE> y mi DNI <ES_NIF>"


def test_leaves_support_text_amounts_and_ids_alone(redactor):
    for text in ["Why was my payment pay_8842 blocked? I paid 400 euros on 12/03",
                 "What is the exchange rate fee when I pay abroad?", "Transfer 1234.56 EUR, ref 2026-11-19"]:
        assert redactor.redact(text) == text


def test_a_card_number_that_fails_luhn_is_not_a_card(redactor):
    assert "<CREDIT_CARD>" not in redactor.redact("order 4111 1111 1111 1112")


def test_mask_walks_nested_payloads_and_keeps_other_types(redactor):
    payload = {"message": {"text": "mail ana@example.com"}, "context": ["IBAN ES9121000418450200051332", 7],
               "n": 3, "ok": None}
    assert redactor.mask(data=payload) == {"message": {"text": "mail <EMAIL_ADDRESS>"},
                                           "context": ["IBAN <IBAN_CODE>", 7], "n": 3, "ok": None}


def test_scan_reports_ids_and_counts_never_the_text(redactor):
    rows = [{"id": "a", "text": "hello"}, {"id": "b", "text": "card 4111111111111111"},
            {"id": "c", "text": "write to ana@example.com"}]
    report = scan(rows, redactor)
    assert report["rows"] == 3 and report["rows_flagged"] == 2
    assert report["strong"] == {"CREDIT_CARD": 1, "EMAIL_ADDRESS": 1}
    assert [f["id"] for f in report["flagged"]] == ["b", "c"]
    assert "4111" not in str(report) and "ana@" not in str(report)
    assert STRONG == 0.8

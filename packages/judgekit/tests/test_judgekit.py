import json

import pytest

from judgekit.calibration import cohen_kappa
from judgekit.cli import main
from judgekit.metrics import citation_precision, exact_match, token_f1


def test_exact_match_ignores_case_accents_punctuation_and_articles():
    assert exact_match("La comisión es 2%.", "comision es 2%") == 1.0
    assert exact_match("2%", "3%") == 0.0


def test_token_f1_partial_overlap():
    assert token_f1("transfer fee is two dollars", "the fee is two dollars") == pytest.approx(0.889, abs=1e-3)
    assert token_f1("", "") == 1.0
    assert token_f1("abc", "") == 0.0


def test_citation_precision_counts_only_retrieved_sources():
    assert citation_precision("Fee is 2% [1][4].", {1, 2, 3}) == 0.5
    assert citation_precision("No citations.", {1}) == 1.0


def test_cohen_kappa():
    assert cohen_kappa(["y", "n", "y", "n"], ["y", "n", "y", "n"]) == 1.0
    assert cohen_kappa(["y", "y", "n", "n"], ["y", "n", "y", "n"]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cohen_kappa(["y"], [])


def test_gate_cli_exit_codes(tmp_path):
    rules = tmp_path / "rules.yaml"
    rules.write_text("gates:\n  macro_f1: {min: 0.80}\n  ece: {max: 0.05}\n", encoding="utf-8")
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps({"macro_f1": 0.85, "ece": 0.03}), encoding="utf-8")
    bad.write_text(json.dumps({"macro_f1": 0.70}), encoding="utf-8")
    assert main(["gate", str(rules), str(good)]) == 0
    assert main(["gate", str(rules), str(bad)]) == 1

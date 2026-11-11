from pathlib import Path

from ci_gate import gate_failures, gate_metrics, select_ci_subset

from judgekit.gate import load_rules

RULES = load_rules(Path(__file__).resolve().parents[2] / "config" / "gates.yaml")
BASELINE = {"macro_f1": 0.90, "ece": 0.04}


def rows(n=300, urgent_every=10):
    return [{"id": f"r{i}", "route": "a", "lang": "en", "urgency": 2 if i % urgent_every == 0 else 0}
            for i in range(n)]


def test_subset_is_fixed_size_deterministic_and_keeps_every_urgent_row():
    data = rows()
    subset = select_ci_subset(data, 100)
    assert len(subset) == 100 and subset == select_ci_subset(data, 100)
    assert {r["id"] for r in data if r["urgency"] >= 2} <= {r["id"] for r in subset}


def test_no_regression_passes_the_gate():
    current = {"macro_f1": 0.905, "urgent_recall": 0.99, "ece": 0.04}
    assert gate_failures(RULES, gate_metrics(current, BASELINE)) == []


def test_macro_f1_drop_of_more_than_one_point_fails():
    current = {"macro_f1": 0.87, "urgent_recall": 0.99, "ece": 0.04}
    assert any("macro_f1_drop" in f for f in gate_failures(RULES, gate_metrics(current, BASELINE)))


def test_urgent_recall_below_98_fails():
    current = {"macro_f1": 0.90, "urgent_recall": 0.95, "ece": 0.04}
    assert any("urgent_recall" in f for f in gate_failures(RULES, gate_metrics(current, BASELINE)))


def test_ece_rise_of_more_than_two_points_fails():
    current = {"macro_f1": 0.90, "urgent_recall": 0.99, "ece": 0.07}
    assert any("ece_rise" in f for f in gate_failures(RULES, gate_metrics(current, BASELINE)))

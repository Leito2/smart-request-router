import pytest
from harness import Prediction, evaluate

# Hand-computed (not derived by the code under test):
#   route:  truth [a, a, b, b]  predicted [a, b, b, b]
#   F1(a) = 2*1*0.5/1.5 = 2/3, F1(b) = 2*(2/3)*1/(5/3) = 0.8  ->  macro-F1 = 11/15
ROWS = [
    {"route": "a", "urgency": 3, "lang": "en"},
    {"route": "a", "urgency": 0, "lang": "en"},
    {"route": "b", "urgency": 2, "lang": "es"},
    {"route": "b", "urgency": 0, "lang": "es"},
]
PREDICTIONS = [
    Prediction("a", urgency=3, latency_ms=10),
    Prediction("b", urgency=0, latency_ms=20),
    Prediction("b", urgency=0, escalated=True, llm_calls=1, tokens=300, latency_ms=900),
    Prediction("b", urgency=0, latency_ms=30),
]


def test_route_quality_and_selective_metrics_match_hand_computed_values():
    metrics = evaluate(ROWS, PREDICTIONS)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(11 / 15)
    assert metrics["coverage"] == pytest.approx(0.75)               # 3 of 4 not escalated
    assert metrics["selective_accuracy"] == pytest.approx(2 / 3)    # of those 3, rows 0 and 3 are right
    assert metrics["urgent_recall"] == pytest.approx(1.0)   # row 0 flagged urgent, row 2 escalated
    assert metrics["llm_calls_per_1k"] == pytest.approx(250.0)
    assert metrics["macro_f1_en"] == pytest.approx(1 / 3)           # en: [a, a] vs [a, b] -> (2/3+0)/2
    assert metrics["macro_f1_es"] == pytest.approx(1.0)             # es: truth [b, b] vs [b, b]


def test_ece_is_reported_when_predictions_carry_confidence():
    rows = [{"route": "a", "lang": "en", "urgency": 0} for _ in range(10)]
    predictions = [Prediction(route="a" if i < 5 else "b", confidence=0.9) for i in range(10)]
    metrics = evaluate(rows, predictions)
    assert abs(metrics["ece"] - 0.4) < 1e-9


def test_no_ece_without_confidence():
    rows = [{"route": "a", "lang": "en", "urgency": 0}]
    assert "ece" not in evaluate(rows, [Prediction(route="a")])

from harness import Prediction
from run_system2 import combine, escalation_payload, score_escalated


def row(i, route="cards", text="t"):
    return {"id": f"r{i}", "text": text, "route": route, "lang": "en", "urgency": 0}


def test_escalation_payload_follows_the_contract():
    payload = escalation_payload(row(1), {"intent": {"cards": 0.5}, "urgency": {"0": 1.0}}, "multi_intent")
    assert payload["escalation"]["reason"] == "multi_intent"
    assert payload["escalation"]["message"]["message_id"] == "r1" and payload["escalation"]["laya"]["intent"]


def test_combine_replaces_escalated_predictions_with_system2_and_counts_calls():
    laya = [Prediction(route="cards", confidence=0.4, escalated=True),
            Prediction(route="fees", confidence=0.9)]
    out = combine(laya, {0: ("disputes", 7)})
    assert out[0].route == "disputes" and out[0].llm_calls == 1 and out[0].tokens == 7
    assert out[1].route == "fees" and out[1].llm_calls == 0


def test_score_escalated_reports_both_accuracies_and_significance():
    rows = [row(i) for i in range(40)]
    laya_routes = {i: "fees" for i in range(40)}
    s2_routes = {i: "cards" for i in range(40)}
    result = score_escalated(rows, laya_routes, s2_routes)
    assert result["laya_accuracy"] == 0.0 and result["system2_accuracy"] == 1.0
    assert result["significant"] is True and result["n"] == 40

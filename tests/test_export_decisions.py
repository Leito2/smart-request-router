from scripts.export_decisions import build_decisions


def test_decisions_join_language_and_label_the_kind():
    inbound = [{"message_id": "a", "lang": "es"}, {"message_id": "b"}]
    routed = [{"message_id": "a", "route": "cards", "confidence": 0.9}]
    escalations = [{"message": {"message_id": "b"}, "laya": {"route": {"fees": 0.4, "cards": 0.3}}}]
    assert build_decisions(inbound, routed, escalations) == [
        {"route": "cards", "lang": "es", "confidence": 0.9, "kind": "routed"},
        {"route": "fees", "lang": "unknown", "confidence": 0.4, "kind": "escalation"}]


def test_escalation_without_a_route_distribution_is_still_exported():
    result = build_decisions([], [], [{"message": {"message_id": "x"}, "laya": {}}])
    assert result[0]["route"] == "unknown" and result[0]["kind"] == "escalation"

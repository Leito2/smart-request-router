from routercore.application.route_messages import RouteMessages
from routercore.domain.decision import Assessment
from routercore.domain.policy import Policy
from services.dataflow.pipeline import process


class FakeModel:
    def assess(self, messages):
        return [Assessment({"intent": {"cards": 0.95, "general": 0.05} if "card" in m.text
                            else {"cards": 0.5, "disputes": 0.45, "general": 0.05},
                            "urgency": {"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0}}, True, "fake")
                for m in messages]


ROUTE = RouteMessages(FakeModel(), Policy(), "t1", now_ns=lambda: 1)


def raw(text="my card", message_id="m1", at=1_000_000_000):
    return {"message_id": message_id, "customer_id": "c1", "channel": "chat", "text": text, "ts_sent_ns": at}


def test_confident_message_goes_to_routed():
    [(topic, key, value)] = process(raw(), {}, ROUTE)
    assert (topic, key, value["route"], value["decided_by"]) == ("routed", "m1", "cards", "system1")


def test_unsure_message_goes_to_escalations_with_context():
    state = {}
    process(raw("my card", "m1", 1_000_000_000), state, ROUTE)
    [(topic, key, value)] = process(raw("money thing", "m2", 2_000_000_000), state, ROUTE)
    assert topic == "escalations" and key == "m2"
    assert value["reason"] == "multi_intent" and value["context"] == ["my card"]


def test_invalid_message_goes_to_the_dlq():
    [(topic, _, value)] = process({"text": ""}, {}, ROUTE)
    assert topic == "dlq" and "error" in value


def test_duplicate_is_dropped():
    state = {}
    process(raw(), state, ROUTE)
    assert process(raw(message_id="m2", at=2_000_000_000), state, ROUTE) == []

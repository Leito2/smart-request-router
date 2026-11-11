from routercore.application.contracts import Routed
from routercore.application.dispatch import Dispatch


class FakeLog:
    def __init__(self):
        self.recorded = {}

    def record(self, routed: Routed) -> bool:
        if routed.message_id in self.recorded:
            return False
        self.recorded[routed.message_id] = routed
        return True


class FakeQueues:
    def __init__(self):
        self.items = []

    def enqueue(self, team, routed):
        self.items.append((team, routed.message_id))


class FakeKnowledge:
    def answer(self, question: str) -> str:
        return f"answer to {question}"


class FakePayments:
    def decision_for(self, text: str) -> str:
        return "blocked: velocity"


def routed(route="cards", message_id="m1", text="hello"):
    return Routed(message_id=message_id, route=route, priority="normal", decided_by="system1", confidence=0.9,
                  model_version="v", threshold_set="t", ts_routed_ns=0, text=text)


def build():
    log, queues = FakeLog(), FakeQueues()
    return Dispatch(log, queues, FakeKnowledge(), FakePayments()), log, queues


def test_human_route_is_queued_once():
    dispatch, log, queues = build()
    assert dispatch(routed("security")) == "queued:security"
    assert queues.items == [("security", "m1")]


def test_duplicate_message_has_no_second_effect():
    dispatch, log, queues = build()
    dispatch(routed("cards"))
    assert dispatch(routed("cards")) == "duplicate"
    assert len(queues.items) == 1


def test_knowledge_route_is_answered_from_the_kb_without_queue():
    dispatch, log, queues = build()
    assert dispatch(routed("knowledge", text="how do I top up?")) == "answered:answer to how do I top up?"
    assert queues.items == []


def test_fraud_inquiry_consults_the_payment_decision_api():
    dispatch, log, queues = build()
    assert dispatch(routed("fraud_inquiry", text="why was my payment blocked")) == "payment:blocked: velocity"


def test_general_goes_to_the_general_queue():
    dispatch, log, queues = build()
    assert dispatch(routed("general")) == "queued:general"

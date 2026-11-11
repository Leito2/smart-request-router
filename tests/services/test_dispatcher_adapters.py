import json

import httpx
from fastapi.testclient import TestClient

from routercore.application.contracts import Routed
from services.dispatcher.adapters import (
    HttpKnowledgeBase,
    HttpPaymentDecisions,
    RedisTeamQueues,
    assemble_answer,
    parse_sse,
)
from services.mocks.app import create_mock_app


def routed():
    return Routed(message_id="m1", route="security", priority="critical", decided_by="system1",
                  confidence=0.9, model_version="v", threshold_set="t", ts_routed_ns=0, text="card stolen")


def sse(*events):
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def kb_with(body: str, status: int = 200) -> HttpKnowledgeBase:
    transport = httpx.MockTransport(lambda request: httpx.Response(
        status, text=body, headers={"content-type": "text/event-stream"}))
    return HttpKnowledgeBase("http://p3", client=httpx.Client(transport=transport))


def test_sse_tokens_are_joined_into_the_answer():
    body = sse(("status", {"state": "generating"}), ("token", {"text": "You can "}),
               ("token", {"text": "top up in the app."}), ("done", {"abstained": False}))
    assert kb_with(body).answer("how do I top up?") == "You can top up in the app."


def test_retract_discards_what_was_streamed_before():
    stream = sse(("token", {"text": "wrong"}), ("retract", {}), ("token", {"text": "right"}))
    events = parse_sse(stream.splitlines())
    assert assemble_answer(events) == "right"


def test_stream_error_event_degrades_to_a_marker():
    assert kb_with(sse(("error", {"code": "internal_error"}))).answer("x y").startswith("unavailable")


def test_knowledge_base_falls_back_when_the_service_is_down():
    def boom(request):
        raise httpx.ConnectError("down")
    kb = HttpKnowledgeBase("http://p3", client=httpx.Client(transport=httpx.MockTransport(boom)))
    assert kb.answer("anything").startswith("unavailable")


def payments_with(handler) -> HttpPaymentDecisions:
    return HttpPaymentDecisions("http://p1", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_payment_decision_uses_the_payment_id_and_the_explanation():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"decision": "BLOCK", "label_is_fraud": True,
                                         "explanation": "BLOCK because of velocity"})
    answer = payments_with(handler).decision_for("why was pay_42 blocked?")
    assert answer == "BLOCK: BLOCK because of velocity"
    assert seen == ["/decisions/pay_42"]


def test_unknown_payment_is_reported_not_raised():
    result = payments_with(lambda r: httpx.Response(404)).decision_for("what about pay_9?")
    assert result == "no decision found for that payment"


def test_payment_decision_without_payment_id_asks_for_it():
    payments = payments_with(lambda r: httpx.Response(500))
    assert payments.decision_for("my payment failed") == "needs payment id"


def test_adapters_work_against_the_mock_services_with_the_real_contracts():
    mock = TestClient(create_mock_app())
    kb = HttpKnowledgeBase("http://mock", client=mock)
    assert kb.answer("how do I top up?").startswith("(mock) Help-center answer")
    payments = HttpPaymentDecisions("http://mock", client=mock)
    assert payments.decision_for("pay_1 was blocked").split(":")[0] in {"ALLOW", "BLOCK", "REVIEW"}
    assert payments.decision_for("pay_unknown?") == "no decision found for that payment"


def test_redis_queue_adds_to_the_team_stream():
    class FakeRedis:
        def __init__(self):
            self.calls = []

        def xadd(self, name, fields):
            self.calls.append((name, fields))

    fake = FakeRedis()
    RedisTeamQueues(fake).enqueue("security", routed())
    name, fields = fake.calls[0]
    assert name == "queue:security" and fields["message_id"] == "m1" and fields["priority"] == "critical"

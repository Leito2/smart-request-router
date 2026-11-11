import httpx
from fastapi.testclient import TestClient

from routercore.application.contracts import Routed
from services.dispatcher.adapters import HttpKnowledgeBase, HttpPaymentDecisions, RedisTeamQueues
from services.mocks.app import create_mock_app


def routed():
    return Routed(message_id="m1", route="security", priority="critical", decided_by="system1",
                  confidence=0.9,
                  model_version="v", threshold_set="t", ts_routed_ns=0, text="card stolen")


def test_knowledge_base_client_parses_the_ask_response():
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, json={"answer": "You can top up in the app [1]", "sources": []}))
    kb = HttpKnowledgeBase("http://p3", client=httpx.Client(transport=transport))
    assert "top up" in kb.answer("how do I top up?")


def test_knowledge_base_falls_back_when_the_service_is_down():
    def boom(request):
        raise httpx.ConnectError("down")
    kb = HttpKnowledgeBase("http://p3", client=httpx.Client(transport=httpx.MockTransport(boom)))
    assert kb.answer("anything").startswith("unavailable")


def test_payment_decisions_client_uses_the_payment_id_in_the_text():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"decision": "BLOCK", "reason": "velocity"})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert HttpPaymentDecisions("http://p1", client=client).decision_for("why was pay_42 blocked?") == \
        "BLOCK: velocity"
    assert seen == ["/decisions/pay_42"]


def test_payment_decisions_without_payment_id_asks_for_it():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert HttpPaymentDecisions("http://p1", client=client).decision_for("my payment failed") == \
        "needs payment id"


def test_mock_services_follow_the_real_contracts():
    mock = TestClient(create_mock_app())
    assert "answer" in mock.post("/ask", json={"question": "how do I top up?"}).json()
    assert mock.get("/decisions/pay_1").json()["decision"] in {"ALLOW", "BLOCK", "REVIEW"}


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

from fastapi.testclient import TestClient

from routercore.application.route_messages import RouteMessages
from routercore.domain.decision import Assessment
from routercore.domain.policy import Policy
from services.api.app import create_app


class FakeModel:
    def assess(self, messages):
        out = []
        for m in messages:
            sure = "card" in m.text
            route = ({"cards": 0.95, "general": 0.05} if sure
                     else {"cards": 0.5, "disputes": 0.45, "general": 0.05})
            out.append(Assessment({"intent": route, "urgency": {"0": 0.9, "1": 0.1, "2": 0.0, "3": 0.0}},
                                  True, "fake"))
        return out


def client(published):
    route = RouteMessages(FakeModel(), Policy(), "t1", now_ns=lambda: 1)
    return TestClient(create_app(route, publish=published.append))


def test_route_endpoint_returns_system1_decision():
    response = client([]).post("/route", json={"text": "I lost my card"})
    body = response.json()
    assert response.status_code == 200
    assert body["decision"] == "system1" and body["route"] == "cards" and body["priority"] == "low"


def test_route_endpoint_reports_escalation_without_calling_system2():
    body = client([]).post("/route", json={"text": "money thing"}).json()
    assert body["decision"] == "escalate" and body["reason"] == "multi_intent"


def test_messages_endpoint_publishes_and_accepts():
    published = []
    response = client(published).post("/messages", json={"text": "hello", "customer_id": "c_1"})
    assert response.status_code == 202
    assert published[0].customer_id == "c_1" and published[0].text == "hello"
    assert response.json()["message_id"] == published[0].message_id


def test_empty_text_is_rejected():
    assert client([]).post("/route", json={"text": ""}).status_code == 422


def test_healthz():
    assert client([]).get("/healthz").json() == {"status": "ok"}

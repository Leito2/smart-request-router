import json

from services.system2.graph import Tools, build_agent

ROUTES = {"disputes": "duplicate charges", "cards": "card issues", "security": "fraud", "general": "other"}
LAYA = {"intent": {"disputes": 0.48, "cards": 0.44, "general": 0.08},
        "urgency": {"0": 0.7, "1": 0.25, "2": 0.04, "3": 0.01}}


class ScriptedLlm:
    """Replays canned answers in order and records the prompts it was given."""

    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def __call__(self, system, user):
        self.prompts.append(user)
        return self.answers.pop(0) if self.answers else "{}"


class FakeTools(Tools):
    def __init__(self):
        self.calls = []

    def get_customer_history(self, customer_id):
        self.calls.append(("history", customer_id))
        return ["I was charged twice yesterday"]

    def lookup_payment_decision(self, text):
        self.calls.append(("payment", text))
        return "BLOCK: velocity"

    def search_kb(self, question):
        self.calls.append(("kb", question))
        return "Top-ups are free up to 500 EUR."


def escalation(reason="multi_intent", text="charged twice on my card", laya=LAYA):
    return {"escalation": {"message": {"message_id": "m1", "customer_id": "c1", "channel": "chat",
                                       "text": text, "ts_sent_ns": 0},
                           "reason": reason, "laya": laya, "context": []}}


def decision(route="disputes", priority="normal"):
    return json.dumps({"route": route, "priority": priority, "rationale": "duplicate charge",
                       "evidence": ["history: charged twice yesterday"]})


def test_agent_uses_a_tool_then_decides():
    llm = ScriptedLlm('{"tool": "get_customer_history"}', '{"tool": null}', decision())
    tools = FakeTools()
    out = build_agent(llm, tools, ROUTES).invoke(escalation())
    assert tools.calls == [("history", "c1")]
    assert out["routed"]["route"] == "disputes" and out["routed"]["decided_by"] == "system2"
    assert "charged twice yesterday" in llm.prompts[-1]


def test_agent_can_decide_without_tools():
    llm = ScriptedLlm('{"tool": null}', decision("cards"))
    out = build_agent(llm, FakeTools(), ROUTES).invoke(escalation())
    assert out["routed"]["route"] == "cards"


def test_invalid_route_is_retried_then_falls_back_to_triage():
    llm = ScriptedLlm('{"tool": null}', decision("nonsense"), decision("nonsense"), decision("nonsense"))
    out = build_agent(llm, FakeTools(), ROUTES).invoke(escalation())
    assert out["routed"]["route"] == "triage" and out["routed"]["decided_by"] == "system2"


def test_unparseable_answers_fall_back_to_triage_never_crash():
    llm = ScriptedLlm("not json", "still not json", "nope", "nope")
    out = build_agent(llm, FakeTools(), ROUTES).invoke(escalation())
    assert out["routed"]["route"] == "triage"


def test_tool_loop_is_bounded():
    llm = ScriptedLlm(*(['{"tool": "search_kb"}'] * 10), decision())
    tools = FakeTools()
    build_agent(llm, tools, ROUTES, max_steps=3).invoke(escalation())
    assert len(tools.calls) <= 3


def test_safety_net_escalation_never_ends_below_high_priority():
    llm = ScriptedLlm('{"tool": null}', decision("security", "low"))
    out = build_agent(llm, FakeTools(), ROUTES).invoke(escalation(reason="urgency_safety_net"))
    assert out["routed"]["priority"] in {"high", "critical"}


def test_history_tool_falls_back_to_the_context_carried_by_the_escalation():
    class NoHistory(FakeTools):
        def get_customer_history(self, customer_id):
            return []
    llm = ScriptedLlm('{"tool": "get_customer_history"}', '{"tool": null}', decision())
    payload = escalation()
    payload["escalation"]["context"] = ["my card was blocked"]
    build_agent(llm, NoHistory(), ROUTES).invoke(payload)
    assert "my card was blocked" in llm.prompts[-1]

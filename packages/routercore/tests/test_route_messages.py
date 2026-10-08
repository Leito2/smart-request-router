from pathlib import Path

from routercore.adapters.policy_config import load_policy
from routercore.application.contracts import Escalation, InboundMessage, Routed
from routercore.application.route_messages import RouteMessages
from routercore.domain.decision import Assessment
from routercore.domain.policy import Policy

CALM = {"0": 0.7, "1": 0.25, "2": 0.04, "3": 0.01}
REPO_ROOT = Path(__file__).resolve().parents[3]


class FakeModel:
    def __init__(self, *route_ps: dict[str, float]):
        self._route_ps = route_ps

    def assess(self, messages):
        return [Assessment({"intent": p, "urgency": CALM}, lang_supported=True, model_version="fake-1")
                for p in self._route_ps]


def message(i: int) -> InboundMessage:
    return InboundMessage(message_id=f"m{i}", customer_id="c_1", channel="chat", text="hola", ts_sent_ns=0)


def test_confident_message_is_routed_and_unsure_one_escalated():
    model = FakeModel({"disputes": 0.93, "general": 0.07}, {"disputes": 0.48, "cards": 0.44, "general": 0.08})
    route = RouteMessages(model, Policy(), threshold_set="t1", now_ns=lambda: 42)
    routed, escalated = route([message(1), message(2)])

    assert isinstance(routed, Routed)
    assert (routed.route, routed.priority, routed.confidence) == ("disputes", "low", 0.93)
    assert (routed.threshold_set, routed.ts_routed_ns) == ("t1", 42)
    assert isinstance(escalated, Escalation)
    assert (escalated.message.message_id, escalated.reason) == ("m2", "multi_intent")


def test_load_policy_reads_the_versioned_config():
    policy, threshold_set = load_policy(REPO_ROOT / "config" / "policy.yaml")
    assert threshold_set == "v0-placeholder"
    assert policy == Policy()

"""Use case: route a micro-batch of inbound messages with System 1, escalating what it should not decide.
Returns the events instead of publishing them, so the stream framework (Quix) or the API owns delivery."""
import time
from collections.abc import Callable, Sequence

from routercore.application.contracts import Escalation, InboundMessage, Routed
from routercore.application.ports import DecisionModel
from routercore.domain.decision import Assessment, priority_from_urgency
from routercore.domain.policy import Policy, decide


class RouteMessages:
    def __init__(self, model: DecisionModel, policy: Policy, threshold_set: str,
                 now_ns: Callable[[], int] = time.time_ns):
        self._model = model
        self._policy = policy
        self._threshold_set = threshold_set
        self._now_ns = now_ns

    def __call__(self, messages: Sequence[InboundMessage]) -> list[Routed | Escalation]:
        assessments = self._model.assess(messages)
        return [self._route(m, a) for m, a in zip(messages, assessments, strict=True)]

    def _route(self, message: InboundMessage, a: Assessment) -> Routed | Escalation:
        target, reason = decide(a.route_p, a.urgency_p, a.lang_supported, self._policy)
        if reason != "system1":
            return Escalation(message=message, reason=reason, laya=a.distributions)
        return Routed(message_id=message.message_id, route=target,
                      priority=priority_from_urgency(a.urgency_p), decided_by="system1",
                      confidence=a.route_p[target], model_version=a.model_version,
                      threshold_set=self._threshold_set, ts_routed_ns=self._now_ns())

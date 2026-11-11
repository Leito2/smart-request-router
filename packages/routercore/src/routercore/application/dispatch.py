"""Use case: execute a routing decision (PLAN §3.8). Idempotent by message_id (ADR-8): the decision log's
unique key makes processing a message twice equal to processing it once, whatever the broker redelivers."""
from routercore.application.contracts import Routed
from routercore.application.ports import DecisionLog, KnowledgeBase, PaymentDecisions, TeamQueues


class Dispatch:
    def __init__(self, log: DecisionLog, queues: TeamQueues, knowledge: KnowledgeBase,
                 payments: PaymentDecisions):
        self._log, self._queues, self._knowledge, self._payments = log, queues, knowledge, payments

    def __call__(self, routed: Routed) -> str:
        if not self._log.record(routed):
            return "duplicate"
        text = routed.text or ""
        if routed.route == "knowledge":
            return f"answered:{self._knowledge.answer(text)}"
        if routed.route == "fraud_inquiry":
            return f"payment:{self._payments.decision_for(text)}"
        self._queues.enqueue(routed.route, routed)
        return f"queued:{routed.route}"

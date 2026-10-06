"""Event contracts v1 (PLAN.md §2.4): inbound-messages → routed | escalations."""
from typing import Literal

from pydantic import BaseModel, Field

DecidedBy = Literal["system1", "system2", "safety_net", "human"]
EscalationReason = Literal["low_confidence", "multi_intent", "urgency_safety_net", "ood"]


class InboundMessage(BaseModel):
    message_id: str
    customer_id: str
    channel: Literal["chat", "email", "app"]
    text: str = Field(min_length=1, max_length=4000)
    lang: str | None = None
    ts_sent_ns: int = Field(ge=0)


class Routed(BaseModel):
    message_id: str
    route: str
    priority: Literal["low", "normal", "high", "critical"]
    decided_by: DecidedBy
    confidence: float = Field(ge=0, le=1)
    model_version: str
    threshold_set: str
    ts_routed_ns: int


class Escalation(BaseModel):
    message: InboundMessage
    reason: EscalationReason
    laya: dict                              # raw calibrated distributions per question

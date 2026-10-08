"""Event contracts v1 (PLAN.md §2.4): inbound-messages → routed | escalations.
They are the use cases' input/output models; adapters serialize them to topics and HTTP."""
from typing import Literal

from pydantic import BaseModel, Field

from routercore.domain.decision import DecidedBy, EscalationReason, Priority


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
    priority: Priority
    decided_by: DecidedBy
    confidence: float = Field(ge=0, le=1)
    model_version: str
    threshold_set: str
    ts_routed_ns: int


class Escalation(BaseModel):
    message: InboundMessage
    reason: EscalationReason
    laya: dict                              # raw calibrated distributions per question

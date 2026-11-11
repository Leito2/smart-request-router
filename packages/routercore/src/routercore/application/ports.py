"""Interfaces the use cases need from the outside world; adapters implement them.
New ports are added with the use case that needs them."""
from collections.abc import Sequence
from typing import Protocol

from routercore.application.contracts import InboundMessage, Routed
from routercore.domain.decision import Assessment


class DecisionModel(Protocol):
    """System 1 model (Laya; plan B Von or ModernBERT, PLAN §11 R1). Takes a micro-batch (ADR-3)."""

    def assess(self, messages: Sequence[InboundMessage]) -> list[Assessment]: ...


class DecisionLog(Protocol):
    """Append-only record of decisions; the unique message_id is what makes the dispatcher idempotent."""

    def record(self, routed: Routed) -> bool:
        """True if the decision is new, False if this message_id was already recorded."""
        ...


class TeamQueues(Protocol):
    """Work queues for human teams (Redis Streams, ADR-7)."""

    def enqueue(self, team: str, routed: Routed) -> None: ...


class KnowledgeBase(Protocol):
    """Help-center RAG (P3 `/ask`)."""

    def answer(self, question: str) -> str: ...


class PaymentDecisions(Protocol):
    """Why was a payment blocked (P1 `/decisions/{payment_id}`)."""

    def decision_for(self, text: str) -> str: ...

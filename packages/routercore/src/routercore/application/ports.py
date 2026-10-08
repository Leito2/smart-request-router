"""Interfaces the use cases need from the outside world; adapters implement them.
New ports are added with the use case that needs them (customer context, destinations, ... in M4–M5)."""
from collections.abc import Sequence
from typing import Protocol

from routercore.application.contracts import InboundMessage
from routercore.domain.decision import Assessment


class DecisionModel(Protocol):
    """System 1 model (Laya; plan B Von or ModernBERT, PLAN §11 R1). Takes a micro-batch (ADR-3)."""

    def assess(self, messages: Sequence[InboundMessage]) -> list[Assessment]: ...

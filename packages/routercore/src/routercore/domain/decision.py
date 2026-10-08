"""Routing vocabulary and the model's assessment of one message. Pure Python: no I/O, no frameworks."""
from dataclasses import dataclass
from typing import Literal

DecidedBy = Literal["system1", "system2", "safety_net", "human"]
EscalationReason = Literal["low_confidence", "multi_intent", "urgency_safety_net", "ood"]
Reason = Literal["system1"] | EscalationReason
Priority = Literal["low", "normal", "high", "critical"]

_PRIORITY_BY_URGENCY: dict[str, Priority] = {"0": "low", "1": "normal", "2": "high", "3": "critical"}


@dataclass(frozen=True)
class Assessment:
    """Calibrated answers of the decision model for one message (PLAN §2.2 ADR-5)."""
    distributions: dict[str, dict[str, float]]   # question → option → probability
    lang_supported: bool
    model_version: str

    @property
    def route_p(self) -> dict[str, float]:
        return self.distributions["intent"]

    @property
    def urgency_p(self) -> dict[str, float]:
        return self.distributions["urgency"]


def priority_from_urgency(urgency_p: dict[str, float]) -> Priority:
    return _PRIORITY_BY_URGENCY[max(urgency_p, key=urgency_p.__getitem__)]

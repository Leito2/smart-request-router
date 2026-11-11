"""Laya (System 1 decision model) behind the DecisionModel port. The `laya` package is imported lazily:
it is a heavy optional dependency (extra `ml`), and the agent is injected so tests need no weights."""
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from routercore.application.contracts import InboundMessage
from routercore.domain.calibration import Calibration, apply_temperature
from routercore.domain.decision import Assessment

URGENCY_CRITERIA = {
    "0": "informational question, nothing is wrong",
    "1": "money is stuck or an action failed",
    "2": "unrecognised charge or possible fraud",
    "3": "card or phone lost, stolen or compromised",
}
CHECKPOINT = "convaiinnovations/laya-multilingual"


class LayaAgent(Protocol):
    def predict_batch(self, states: list, questions: dict, **kwargs: Any) -> list[dict]: ...


def build_questions(routes: dict[str, str]) -> dict[str, dict]:
    """The two typed questions asked of every message, both `choice`: `score` is Laya's weakest type."""
    return {
        "route": {"type": "choice", "instructions": "Which team should handle this customer message?",
                  "criteria": dict(routes)},
        "urgency": {"type": "choice", "instructions": "How urgent is this customer message?",
                    "criteria": dict(URGENCY_CRITERIA)},
    }


def _normalize(distribution: dict[str, float]) -> dict[str, float]:
    total = sum(distribution.values())
    return {k: v / total for k, v in distribution.items()} if total > 0 else dict(distribution)


class LayaModel:
    """Implements DecisionModel. Laya rounds probabilities to 4 decimals, so they are renormalized here."""

    def __init__(self, agent: LayaAgent, routes: dict[str, str], *, model_version: str,
                 calibration: Calibration | None = None, supported_langs: set[str] | None = None,
                 batch_size: int = 16):
        self._agent, self._questions = agent, build_questions(routes)
        self._calibration = calibration or Calibration()
        self._supported, self._batch_size = supported_langs, batch_size
        self.model_version = model_version

    @classmethod
    def load(cls, routes: dict[str, str], *, checkpoint: str = CHECKPOINT, model_version: str | None = None,
             device: str = "cpu", **kwargs: Any) -> "LayaModel":
        import laya

        return cls(laya.load(checkpoint, device=device), routes,
                   model_version=model_version or checkpoint, **kwargs)

    def assess(self, messages: Sequence[InboundMessage]) -> list[Assessment]:
        results = self._agent.predict_batch([m.text for m in messages], self._questions,
                                            batch_size=self._batch_size)
        pairs = zip(messages, results, strict=True)
        return [self._to_assessment(message, result) for message, result in pairs]

    def _to_assessment(self, message: InboundMessage, result: dict) -> Assessment:
        answers = result["answers"]
        route = _normalize(answers["route"]["probabilities"])
        urgency = _normalize(answers["urgency"]["probabilities"])
        route = apply_temperature(route, self._calibration.temperature("intent", message.lang))
        urgency = apply_temperature(urgency, self._calibration.temperature("urgency", message.lang))
        supported = self._supported is None or message.lang is None or message.lang in self._supported
        return Assessment(distributions={"intent": route, "urgency": urgency}, lang_supported=supported,
                          model_version=self.model_version)


def load_calibration(path: Path) -> Calibration:
    """Read the temperatures fitted by training/calibrate.py: {"temperatures": {question: {lang: T}}}."""
    return Calibration(json.loads(path.read_text(encoding="utf-8"))["temperatures"])

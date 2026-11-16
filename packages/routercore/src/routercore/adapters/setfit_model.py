"""B1: SetFit few-shot router, the modern classical baseline (PLAN §4.3). CPU only.

Two SetFit models, one per question (route, urgency), because a SetFit model answers one question. That is the
cost the decision model (Laya) avoids. Implements the DecisionModel port. Needs the `ml` dependency group.
"""
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from routercore.application.contracts import InboundMessage
from routercore.domain.decision import Assessment

URGENCY_LABELS = ["0", "1", "2", "3"]
DEFAULT_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _rows(probabilities) -> list[list[float]]:
    if hasattr(probabilities, "tolist"):
        return probabilities.tolist()
    return [list(row) for row in probabilities]


class SetFitRouter:
    def __init__(self, *, intent_head, intent_labels: list[str], urgency_head,
                 model_version: str = "b1-setfit"):
        self._intent_head, self._intent_labels = intent_head, intent_labels
        self._urgency_head = urgency_head
        self.model_version = model_version

    def assess(self, messages: Sequence[InboundMessage]) -> list[Assessment]:
        texts = [m.text for m in messages]
        intent_rows, urgency_rows = _rows(self._intent_head.predict_proba(texts)), \
            _rows(self._urgency_head.predict_proba(texts))
        return [
            Assessment(
                distributions={"intent": dict(zip(self._intent_labels, intent, strict=True)),
                               "urgency": dict(zip(URGENCY_LABELS, urgency, strict=True))},
                lang_supported=True, model_version=self.model_version)
            for intent, urgency in zip(intent_rows, urgency_rows, strict=True)]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._intent_head.save_pretrained(str(directory / "intent"))
        self._urgency_head.save_pretrained(str(directory / "urgency"))
        (directory / "labels.json").write_text(
            json.dumps({"intent": self._intent_labels, "version": self.model_version}), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "SetFitRouter":
        from setfit import SetFitModel

        meta = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
        return cls(intent_head=SetFitModel.from_pretrained(str(directory / "intent")),
                   intent_labels=meta["intent"],
                   urgency_head=SetFitModel.from_pretrained(str(directory / "urgency")),
                   model_version=meta["version"])

    @classmethod
    def train(cls, records: list[dict], *, per_class: int = 16, seed: int = 0, iterations: int = 10,
              batch_size: int = 16, encoder: str = DEFAULT_ENCODER) -> "SetFitRouter":
        """Few-shot fit from `per_class` examples of each route and each urgency level (train split only)."""
        train = [r for r in records if r["split"] == "train"]
        intent_labels = sorted({r["route"] for r in train})
        intent_head = _fit(_sample(train, lambda r: r["route"], per_class, seed),
                           intent_labels, "route", encoder, iterations, batch_size, seed)
        urgency_head = _fit(_sample(train, lambda r: str(r["urgency"]), per_class, seed),
                            URGENCY_LABELS, "urgency", encoder, iterations, batch_size, seed)
        return cls(intent_head=intent_head, intent_labels=intent_labels, urgency_head=urgency_head)


def _sample(rows: list[dict], key, per_class: int, seed: int) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    rng = random.Random(seed)
    return [r for _, group in sorted(groups.items()) for r in rng.sample(group, min(per_class, len(group)))]


def _fit(shots: list[dict], labels: list[str], field: str, encoder: str, iterations: int, batch_size: int,
         seed: int):
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    ids = {label: i for i, label in enumerate(labels)}
    to_label = (lambda r: r["route"]) if field == "route" else (lambda r: str(r["urgency"]))
    data = Dataset.from_dict({"text": [r["text"] for r in shots], "label": [ids[to_label(r)] for r in shots]})
    model = SetFitModel.from_pretrained(encoder)
    args = TrainingArguments(batch_size=batch_size, num_epochs=1, num_iterations=iterations, seed=seed)
    Trainer(model=model, args=args, train_dataset=data).train()
    return model

"""B1: SetFit few-shot router, the modern classical baseline (PLAN §4.3). CPU only.

Two heads, one per question (route, urgency), because a SetFit model answers one question. That is the
cost the decision model (Laya) avoids. The route head is a few-shot SetFit model. The urgency head is
`IntentUrgencyHead` (see below); checkpoints saved with the older few-shot SetFit urgency head still load.
Implements the DecisionModel port. Needs the `ml` dependency group.
"""
import json
import random
import re
import shutil
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from routercore.application.contracts import InboundMessage
from routercore.domain.decision import Assessment

URGENCY_LABELS = ["0", "1", "2", "3"]
DEFAULT_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
URGENCY_HEAD_FILE = "urgency_lr.json"
SENTENCE_END = re.compile(r"(?<=[.?!;])\s+")
MIN_CLAUSE_CHARS = 8


def clauses(text: str) -> list[str]:
    """The sentences of a message with more than one, else []. Short fragments ("Thanks.") are dropped."""
    parts = [p.strip() for p in SENTENCE_END.split(text) if len(p.strip()) >= MIN_CLAUSE_CHARS]
    return parts if len(parts) > 1 else []


def _rows(probabilities) -> list[list[float]]:
    if hasattr(probabilities, "tolist"):
        return probabilities.tolist()
    return [list(row) for row in probabilities]


def intent_label(row: dict) -> str:
    """Banking77 intent of a row. Hard cases and greetings have none: one class per urgency level for them."""
    return row["intent"] or f"none_u{row['urgency']}"


class IntentUrgencyHead:
    """Urgency as a function of the Banking77 intent (PLAN §4.2: urgency is rule-based per intent).

    A logistic regression over frozen sentence embeddings predicts the 77 intents (plus one `none_uN` class
    per urgency level for rows without an intent); P(urgency = u) is the sum of the probabilities of the
    intents whose rule gives u. It trains on the whole train split, where the few-shot SetFit head saw 16 rows
    per level, and it flags far fewer harmless messages at the same urgent recall. `predict_proba` returns
    [P0, P1, P2, P3] per row, so it plugs in where the SetFit urgency head did.

    A message with several sentences is scored as a whole and sentence by sentence, and the answer is that of
    the part with the highest P(urgent): "Why is my top up pending? Also, someone stole my card" has one
    intent as a whole (top up) but an urgent second sentence, and a safety net must not miss that.
    """

    def __init__(self, *, encoder, classifier, urgency_of_class: dict[str, int], encoder_name: str):
        self._encoder, self._classifier = encoder, classifier
        self._urgency_of_class, self.encoder_name = urgency_of_class, encoder_name

    def predict_proba(self, texts: Sequence[str]):
        parts, owner = [], []
        for index, text in enumerate(texts):
            for part in [text, *clauses(text)]:
                parts.append(part)
                owner.append(index)
        embeddings = self._encoder.encode(parts, normalize_embeddings=True, show_progress_bar=False)
        rows = self._fold(self._classifier.predict_proba(embeddings), self._classifier.classes_,
                          self._urgency_of_class)
        best: dict[int, int] = {}
        for row, index in enumerate(owner):
            if index not in best or rows[row, 2] + rows[row, 3] > rows[best[index], 2] + rows[best[index], 3]:
                best[index] = row
        return rows[[best[i] for i in range(len(texts))]]

    @staticmethod
    def _fold(probabilities, classes, urgency_of_class: dict[str, int]):
        import numpy as np

        out = np.zeros((len(probabilities), len(URGENCY_LABELS)))
        for column, name in enumerate(classes):
            out[:, urgency_of_class[name]] += probabilities[:, column]
        return out

    def save_pretrained(self, directory: str) -> None:
        import joblib

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._encoder.save(str(path / "encoder"))
        joblib.dump(self._classifier, path / "intent_lr.joblib")
        (path / URGENCY_HEAD_FILE).write_text(json.dumps(
            {"urgency_of_class": self._urgency_of_class, "encoder": self.encoder_name}), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, *, device: str = "cpu") -> "IntentUrgencyHead":
        import joblib
        from sentence_transformers import SentenceTransformer

        meta = json.loads((directory / URGENCY_HEAD_FILE).read_text(encoding="utf-8"))
        return cls(encoder=SentenceTransformer(str(directory / "encoder"), device=device),
                   classifier=joblib.load(directory / "intent_lr.joblib"),
                   urgency_of_class=meta["urgency_of_class"], encoder_name=meta["encoder"])

    @classmethod
    def fit(cls, records: list[dict], *, encoder_name: str = DEFAULT_ENCODER, C: float = 10.0,
            device: str | None = None, folds: int = 0) -> tuple["IntentUrgencyHead", list[float]]:
        """Fits on the train split. With `folds` > 0 also returns the out-of-fold P(urgency >= 2) of every
        urgent train row (folds grouped by `origin_id`, so a sentence and its translation never straddle
        them): a much larger sample than the validation split to choose the safety-net threshold."""
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold

        train = [r for r in records if r["split"] == "train"]
        labels = np.array([intent_label(r) for r in train])
        urgency = np.array([r["urgency"] for r in train])
        urgency_of_class = {}
        for name in sorted(set(labels)):
            levels = set(urgency[labels == name].tolist())
            if len(levels) != 1:
                raise ValueError(f"intent {name!r} maps to several urgency levels {sorted(levels)}")
            urgency_of_class[name] = levels.pop()
        encoder = SentenceTransformer(encoder_name, device=device)
        embeddings = encoder.encode([r["text"] for r in train], batch_size=128, normalize_embeddings=True,
                                    show_progress_bar=False)
        out_of_fold: list[float] = []
        if folds:
            groups = np.array([r.get("origin_id") or r["id"] for r in train])
            masses = np.zeros(len(train))
            for fit_rows, held_out in GroupKFold(folds).split(embeddings, labels, groups):
                fold = LogisticRegression(C=C, max_iter=3000).fit(embeddings[fit_rows], labels[fit_rows])
                proba = cls._fold(fold.predict_proba(embeddings[held_out]), fold.classes_, urgency_of_class)
                masses[held_out] = proba[:, 2] + proba[:, 3]
            out_of_fold = masses[urgency >= 2].tolist()
        classifier = LogisticRegression(C=C, max_iter=3000).fit(embeddings, labels)
        return cls(encoder=encoder, classifier=classifier, urgency_of_class=urgency_of_class,
                   encoder_name=encoder_name), out_of_fold


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
        shutil.rmtree(directory / "urgency", ignore_errors=True)    # no leftovers
        self._intent_head.save_pretrained(str(directory / "intent"))
        self._urgency_head.save_pretrained(str(directory / "urgency"))
        (directory / "labels.json").write_text(
            json.dumps({"intent": self._intent_labels, "version": self.model_version}), encoding="utf-8")

    def with_urgency_head(self, urgency_head, *, model_version: str) -> "SetFitRouter":
        """Same route head, another urgency head: the route answers do not change."""
        return SetFitRouter(intent_head=self._intent_head, intent_labels=self._intent_labels,
                            urgency_head=urgency_head, model_version=model_version)

    @classmethod
    def load(cls, directory: Path) -> "SetFitRouter":
        from setfit import SetFitModel

        meta = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
        urgency_dir = directory / "urgency"
        urgency_head = (IntentUrgencyHead.load(urgency_dir) if (urgency_dir / URGENCY_HEAD_FILE).exists()
                        else SetFitModel.from_pretrained(str(urgency_dir)))
        return cls(intent_head=SetFitModel.from_pretrained(str(directory / "intent")),
                   intent_labels=meta["intent"], urgency_head=urgency_head, model_version=meta["version"])

    @classmethod
    def train(cls, records: list[dict], *, per_class: int = 16, seed: int = 0, iterations: int = 10,
              batch_size: int = 16, encoder: str = DEFAULT_ENCODER) -> "SetFitRouter":
        """Few-shot route head (`per_class` rows per route); the urgency head uses the whole train split."""
        train = [r for r in records if r["split"] == "train"]
        intent_labels = sorted({r["route"] for r in train})
        intent_head = _fit(_sample(train, lambda r: r["route"], per_class, seed),
                           intent_labels, "route", encoder, iterations, batch_size, seed)
        urgency_head, _ = IntentUrgencyHead.fit(records, encoder_name=encoder)
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

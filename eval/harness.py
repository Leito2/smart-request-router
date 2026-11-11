"""Router evaluation harness (PLAN §5): the same metrics for every router B0-B5.

`evaluate(rows, predictions)` is pure: rows are golden-set records (route, urgency, lang), predictions are
what a router answered. Stdlib only so it can run in CI. Urgent means truth urgency >= 2 (PLAN §1.7);
an urgent message counts as caught if the router flagged it urgent or escalated it (System 2 or a human
decides).
"""
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from judgekit.probabilistic import expected_calibration_error

URGENT = 2


@dataclass(frozen=True)
class Prediction:
    route: str
    urgency: int = 0
    escalated: bool = False
    llm_calls: int = 0
    tokens: int = 0
    latency_ms: float = 0.0
    confidence: float | None = None       # top route probability; enables ECE


def macro_f1(truth: Sequence[str], predicted: Sequence[str]) -> float:
    labels = set(truth) | set(predicted)
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truth, predicted, strict=True))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted, strict=True))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted, strict=True))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)] if ordered else 0.0


def evaluate(rows: Sequence[dict], predictions: Sequence[Prediction]) -> dict[str, float]:
    if len(rows) != len(predictions):
        raise ValueError("one prediction per row is required")
    truth = [r["route"] for r in rows]
    predicted = [p.route for p in predictions]
    n = len(rows)
    metrics = {
        "accuracy": sum(t == p for t, p in zip(truth, predicted, strict=True)) / n,
        "macro_f1": macro_f1(truth, predicted),
    }
    for lang in sorted(Counter(r["lang"] for r in rows)):
        idx = [i for i, r in enumerate(rows) if r["lang"] == lang]
        metrics[f"macro_f1_{lang}"] = macro_f1([truth[i] for i in idx], [predicted[i] for i in idx])

    kept = [i for i, p in enumerate(predictions) if not p.escalated]
    metrics["coverage"] = len(kept) / n
    metrics["selective_accuracy"] = (
        sum(truth[i] == predicted[i] for i in kept) / len(kept) if kept else 0.0)

    urgent = [i for i, r in enumerate(rows) if r["urgency"] >= URGENT]
    caught = [i for i in urgent if predictions[i].urgency >= URGENT or predictions[i].escalated]
    metrics["urgent_recall"] = len(caught) / len(urgent) if urgent else 1.0

    if all(p.confidence is not None for p in predictions):
        metrics["ece"] = expected_calibration_error(
            [p.confidence for p in predictions], [t == q for t, q in zip(truth, predicted, strict=True)])
    metrics["llm_calls_per_1k"] = 1000 * sum(p.llm_calls for p in predictions) / n
    metrics["tokens_per_1k"] = 1000 * sum(p.tokens for p in predictions) / n
    latencies = [p.latency_ms for p in predictions]
    metrics["latency_p50_ms"] = percentile(latencies, 0.50)
    metrics["latency_p95_ms"] = percentile(latencies, 0.95)
    return metrics

"""Calibration metrics and temperature fitting for probabilistic classifiers (ECE, Brier). Pure Python."""
import math
from collections.abc import Sequence


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool],
                               bins: int = 10) -> float:
    """ECE: bin by confidence, average |accuracy - confidence| weighted by bin size."""
    n = len(confidences)
    if n == 0:
        return 0.0
    totals = [[0, 0.0, 0.0] for _ in range(bins)]      # count, sum confidence, sum correct
    for conf, ok in zip(confidences, correct, strict=True):
        slot = totals[min(int(conf * bins), bins - 1)]
        slot[0] += 1
        slot[1] += conf
        slot[2] += float(ok)
    return sum(abs(c_ok - c_conf) for count, c_conf, c_ok in totals if count) / n


def brier_score(distributions: Sequence[dict[str, float]], truth: Sequence[str]) -> float:
    """Multiclass Brier score: mean squared distance to the one-hot truth."""
    total = 0.0
    for dist, label in zip(distributions, truth, strict=True):
        total += sum((p - (1.0 if k == label else 0.0)) ** 2 for k, p in dist.items())
    return total / len(truth)


def _nll(distributions: Sequence[dict[str, float]], truth: Sequence[str], temperature: float) -> float:
    total = 0.0
    for dist, label in zip(distributions, truth, strict=True):
        logits = {k: math.log(max(p, 1e-12)) / temperature for k, p in dist.items()}
        peak = max(logits.values())
        log_norm = peak + math.log(sum(math.exp(v - peak) for v in logits.values()))
        total -= logits.get(label, -1e12) - log_norm
    return total / len(truth)


def fit_temperature(distributions: Sequence[dict[str, float]], truth: Sequence[str],
                    low: float = 0.25, high: float = 8.0) -> float:
    """Temperature minimizing the negative log-likelihood (golden-section search; NLL is unimodal in T)."""
    ratio = (math.sqrt(5) - 1) / 2
    a, b = low, high
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    for _ in range(60):
        if _nll(distributions, truth, c) < _nll(distributions, truth, d):
            b = d
        else:
            a = c
        c, d = b - ratio * (b - a), a + ratio * (b - a)
    return (a + b) / 2

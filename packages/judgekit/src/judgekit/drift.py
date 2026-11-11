"""Drift on prediction signals: Population Stability Index between a reference and a current distribution."""
import math
from collections.abc import Mapping


def _shares(counts: Mapping[str, float], keys: set[str], epsilon: float) -> dict[str, float]:
    total = sum(counts.values()) or 1.0
    return {k: max(counts.get(k, 0) / total, epsilon) for k in keys}


def psi(expected: Mapping[str, float], actual: Mapping[str, float], epsilon: float = 1e-4) -> float:
    """PSI over categorical counts (< 0.1 stable, 0.1-0.25 moderate, > 0.25 major shift)."""
    keys = set(expected) | set(actual)
    e, a = _shares(expected, keys, epsilon), _shares(actual, keys, epsilon)
    return sum((a[k] - e[k]) * math.log(a[k] / e[k]) for k in keys)


def drift_report(reference: Mapping[str, Mapping[str, float]], current: Mapping[str, Mapping[str, float]],
                 threshold: float = 0.25) -> dict[str, dict]:
    """One PSI per feature (e.g. route, language, confidence bucket) and whether it crossed the threshold."""
    report = {}
    for feature, ref_counts in reference.items():
        value = psi(ref_counts, current[feature])
        report[feature] = {"psi": round(value, 4), "drifted": value > threshold}
    return report

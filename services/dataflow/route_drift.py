"""Route-mix drift for the dashboards (PLAN §3.10 R2): PSI of the last N decisions against the training mix.

A rolling window, not the whole history, so a shift that ends also shows up as ending. Nothing is reported
until the window has `min_samples` decisions: with 12 routes, 100 samples already give a PSI of ~0.1 by
chance alone, 250 about 0.04.
"""
import json
from collections import Counter, deque
from collections.abc import Mapping
from pathlib import Path

from judgekit.drift import psi


def load_reference(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))


class RouteDrift:
    def __init__(self, reference: Mapping[str, float], *, window: int = 500, min_samples: int = 250):
        self._reference, self._min_samples = dict(reference), min_samples
        self._recent: deque[str] = deque(maxlen=window)
        self._counts: Counter[str] = Counter()

    def observe(self, route: str) -> float | None:
        """Adds a decision; returns the PSI of the window, or None while it is still too small."""
        if len(self._recent) == self._recent.maxlen:
            self._counts[self._recent[0]] -= 1
        self._recent.append(route)
        self._counts[route] += 1
        if len(self._recent) < self._min_samples:
            return None
        return psi(self._reference, {r: n for r, n in self._counts.items() if n > 0})

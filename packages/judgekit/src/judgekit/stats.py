"""Paired bootstrap confidence interval for the difference between two systems on the same items."""
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Bootstrap:
    mean_diff: float
    low: float
    high: float

    @property
    def significant(self) -> bool:
        """The interval excludes zero on the positive side: A is better than B."""
        return self.low > 0


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, samples: int = 5000, seed: int = 0,
                     confidence: float = 0.95) -> Bootstrap:
    """Resample items with replacement; interval of mean(a - b). a, b: per-item scores (1/0 correct)."""
    if len(a) != len(b) or not a:
        raise ValueError("a and b must score the same non-empty set of items")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n, rng = len(diffs), random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(samples))
    tail = (1 - confidence) / 2
    low, high = means[int(tail * samples)], means[int((1 - tail) * samples) - 1]
    return Bootstrap(round(sum(diffs) / n, 10), low, high)

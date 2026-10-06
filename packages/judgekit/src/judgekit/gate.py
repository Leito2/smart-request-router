"""Quality gates: metric thresholds (YAML) → pass/fail, used in CI and before promoting a model or prompt."""
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Rule:
    metric: str
    min: float | None = None
    max: float | None = None

    def check(self, value: float) -> bool:
        return (self.min is None or value >= self.min) and (self.max is None or value <= self.max)


def load_rules(path: Path) -> list[Rule]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Rule(metric=name, **bounds) for name, bounds in raw["gates"].items()]


def evaluate(rules: list[Rule], metrics: dict[str, float]) -> list[str]:
    """Return human-readable failures; an empty list means the gate passes. Missing metrics fail."""
    failures = []
    for rule in rules:
        if rule.metric not in metrics:
            failures.append(f"{rule.metric}: missing")
        elif not rule.check(metrics[rule.metric]):
            failures.append(f"{rule.metric}={metrics[rule.metric]:.4f} outside [{rule.min}, {rule.max}]")
    return failures

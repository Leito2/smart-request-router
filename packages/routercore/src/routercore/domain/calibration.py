"""Temperature scaling of a model's probabilities (PLAN §4.4). Pure Python.

The fitting happens offline (training/calibrate.py); this module only applies the fitted temperatures.
"""
from dataclasses import dataclass, field


def apply_temperature(distribution: dict[str, float], temperature: float) -> dict[str, float]:
    """Soften (T > 1) or sharpen (T < 1) a distribution: p_i^(1/T), renormalized. The argmax never changes."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if temperature == 1.0:
        return dict(distribution)
    scaled = {k: (p ** (1.0 / temperature) if p > 0 else 0.0) for k, p in distribution.items()}
    total = sum(scaled.values())
    return {k: v / total for k, v in scaled.items()} if total > 0 else dict(distribution)


@dataclass(frozen=True)
class Calibration:
    """Temperatures per question, with an optional override per language ("default" otherwise)."""
    temperatures: dict[str, dict[str, float]] = field(default_factory=dict)

    def temperature(self, question: str, lang: str | None) -> float:
        by_lang = self.temperatures.get(question, {})
        return by_lang.get(lang, by_lang.get("default", 1.0)) if lang else by_lang.get("default", 1.0)

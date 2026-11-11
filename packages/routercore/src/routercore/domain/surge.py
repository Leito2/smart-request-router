"""Surge detection: per-intent volume anomalies against an EWMA baseline (PLAN §3.9). Pure Python.

SurgeDetector judges one closed window at a time; SurgeMonitor buckets a stream of events into tumbling
windows and feeds the detector, so Quix (or a test) only has to call `add`.
"""
import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SurgeConfig:
    window_s: int = 60
    ewma_alpha: float = 0.2
    z_threshold: float = 4.0
    consecutive_windows: int = 2
    warmup_windows: int = 10


@dataclass(frozen=True)
class Alert:
    intent: str
    window: int
    volume: int
    baseline: float
    zscore: float


@dataclass
class _Series:
    mean: float = 0.0
    variance: float = 0.0
    seen: int = 0
    streak: int = 0


@dataclass
class SurgeDetector:
    config: SurgeConfig
    _series: dict[str, _Series] = field(default_factory=dict)

    def observe(self, intent: str, window: int, count: int) -> Alert | None:
        cfg, s = self.config, self._series.setdefault(intent, _Series())
        if s.seen == 0:
            s.mean, s.seen = float(count), 1
            return None
        std = max(math.sqrt(s.variance), math.sqrt(max(s.mean, 1.0)))     # Poisson floor
        z = (count - s.mean) / std
        warm = s.seen >= cfg.warmup_windows
        anomalous = warm and z > cfg.z_threshold
        s.streak = s.streak + 1 if anomalous else 0
        if not anomalous:                       # never learn the surge as the new normal
            delta = count - s.mean
            s.mean += cfg.ewma_alpha * delta
            s.variance = (1 - cfg.ewma_alpha) * (s.variance + cfg.ewma_alpha * delta * delta)
        s.seen += 1
        if s.streak == cfg.consecutive_windows:          # one alert per surge, not one per window
            return Alert(intent, window, count, round(s.mean, 2), round(z, 2))
        return None


class SurgeMonitor:
    def __init__(self, config: SurgeConfig):
        self._config = config
        self._detector = SurgeDetector(config)
        self._counts: dict[str, dict[int, int]] = {}
        self._last_closed: dict[str, int] = {}

    def add(self, intent: str, ts_ns: int) -> list[Alert]:
        """Count one event; closes (and judges) every window of this intent that ended before it."""
        window = ts_ns // (self._config.window_s * 1_000_000_000)
        buckets = self._counts.setdefault(intent, {})
        alerts: list[Alert] = []
        for closed in sorted(w for w in buckets if w < window):
            start = self._last_closed.get(intent, closed - 1) + 1
            for w in range(start, closed + 1):                  # quiet windows count as zero
                alert = self._detector.observe(intent, w, buckets.get(w, 0))
                if alert:
                    alerts.append(alert)
            self._last_closed[intent] = closed
            del buckets[closed]
        buckets[window] = buckets.get(window, 0) + 1
        return alerts

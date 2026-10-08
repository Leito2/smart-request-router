"""System 1 / System 2 policy (course 06/34 note 05). Thresholds come from config/policy.yaml
(loaded by routercore.adapters.policy_config); the defaults here mirror it for tests."""
import math
from dataclasses import dataclass

from routercore.domain.decision import Reason


@dataclass(frozen=True)
class Policy:
    tau: float = 0.80          # min calibrated route confidence
    margin: float = 0.15       # min gap between top-2 routes
    tau_urgent: float = 0.15   # asymmetric safety net on P(urgency >= high)
    max_entropy: float = 0.85  # normalized entropy above this → OOD


def normalized_entropy(p: dict[str, float]) -> float:
    h = -sum(v * math.log(v) for v in p.values() if v > 0)
    return h / math.log(len(p)) if len(p) > 1 else 0.0


def decide(route_p: dict[str, float], urgency_p: dict[str, float], lang_supported: bool,
           pol: Policy | None = None) -> tuple[str, Reason]:
    """Return (target, reason): target is a route name or 'system2'."""
    pol = pol or Policy()
    ranked = sorted(route_p.items(), key=lambda kv: kv[1], reverse=True)
    (top, p1), (_, p2) = ranked[0], ranked[1] if len(ranked) > 1 else ("", 0.0)
    if urgency_p.get("2", 0.0) + urgency_p.get("3", 0.0) > pol.tau_urgent:
        return "system2", "urgency_safety_net"
    if not lang_supported or normalized_entropy(route_p) > pol.max_entropy:
        return "system2", "ood"
    if p1 - p2 < pol.margin:
        return "system2", "multi_intent"
    if p1 < pol.tau:
        return "system2", "low_confidence"
    return top, "system1"

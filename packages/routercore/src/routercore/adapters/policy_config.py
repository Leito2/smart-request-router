"""Loads the versioned thresholds in config/policy.yaml into the domain Policy."""
from pathlib import Path

import yaml

from routercore.domain.policy import Policy


def load_policy(path: Path) -> tuple[Policy, str]:
    """Return (policy, threshold_set id)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    policy = Policy(tau=raw["tau"], margin=raw["margin"], tau_urgent=raw["tau_urgent"],
                    max_entropy=raw["max_entropy"])
    return policy, raw["threshold_set"]

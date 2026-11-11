"""Load config/surge.yaml into the domain's SurgeConfig."""
from pathlib import Path

import yaml

from routercore.domain.surge import SurgeConfig


def load_surge_config(path: Path) -> SurgeConfig:
    return SurgeConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))

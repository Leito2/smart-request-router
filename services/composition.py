"""Composition root shared by the API and the dataflow: config files + the chosen model → RouteMessages.

MODEL=setfit (default: the shipping System 1, PLAN §11 R1 plan B; B1 SetFit with policy.setfit.yaml)
| laya (calibrated frozen-encoder fine-tune, policy.yaml) | rules (B0, no weights; smoke/CI).
"""
import os
from pathlib import Path

from routercore.adapters.model_factory import build_model
from routercore.adapters.policy_config import load_policy, policy_path
from routercore.adapters.route_map_config import load_route_map
from routercore.application.route_messages import RouteMessages

ROOT = Path(__file__).resolve().parent.parent


def build_route_messages(model_name: str | None = None) -> RouteMessages:
    model_name = model_name or os.environ.get("MODEL", "setfit")
    route_map = load_route_map(ROOT / "config" / "route_map.yaml")
    policy, threshold_set = load_policy(policy_path(ROOT / "config", model_name))
    model = build_model(model_name, route_map, ROOT / "models", device=os.environ.get("DEVICE", "cpu"),
                        model_version=os.environ.get("MODEL_VERSION"))
    return RouteMessages(model, policy, threshold_set)

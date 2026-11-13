"""Composition root shared by the API and the dataflow: config files + the chosen model → RouteMessages.

MODEL=laya (default; calibrated checkpoint from models/ when present) | setfit (plan B, PLAN §11 R1: B1
SetFit with its own policy.setfit.yaml) | rules (B0, no weights; smoke/CI).
"""
import os
import sys
from pathlib import Path

from routercore.adapters.policy_config import load_policy, policy_path
from routercore.adapters.route_map_config import load_route_map
from routercore.application.route_messages import RouteMessages

ROOT = Path(__file__).resolve().parent.parent


def build_route_messages(model_name: str | None = None) -> RouteMessages:
    model_name = model_name or os.environ.get("MODEL", "laya")
    route_map = load_route_map(ROOT / "config" / "route_map.yaml")
    policy, threshold_set = load_policy(policy_path(ROOT / "config", model_name))
    if model_name == "rules":
        sys.path.insert(0, str(ROOT / "training"))
        from baselines.rules import RulesModel
        model = RulesModel(route_map)
    elif model_name == "setfit":
        sys.path.insert(0, str(ROOT / "training"))
        from baselines.setfit_model import SetFitRouter
        model = SetFitRouter.load(ROOT / "models" / "b1_setfit")
    else:
        from routercore.adapters.laya_model import CHECKPOINT, LayaModel, load_calibration
        fine_tuned, calibration = ROOT / "models" / "laya_ft", ROOT / "models" / "calibration.json"
        model = LayaModel.load(
            route_map.routes, checkpoint=str(fine_tuned) if fine_tuned.exists() else CHECKPOINT,
            model_version=os.environ.get("MODEL_VERSION", "laya"),
            calibration=load_calibration(calibration) if calibration.exists() else None,
            supported_langs={"en", "es"}, device=os.environ.get("DEVICE", "cpu"))
    return RouteMessages(model, policy, threshold_set)

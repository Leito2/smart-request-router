"""The one place that knows how each System 1 model is built from `models/`; services, eval and
training all use it.

rules     keyword baseline, no weights
setfit    models/b1_setfit (extra `setfit`)
laya      models/laya_ft + models/calibration.json when they exist, else the base checkpoint (extra `laya`)
laya-base the zero-shot base checkpoint, uncalibrated (the B2 baseline)
"""
from pathlib import Path

from routercore.adapters.laya_model import CHECKPOINT, LayaModel, load_calibration
from routercore.adapters.rules_model import RulesModel
from routercore.adapters.setfit_model import SetFitRouter
from routercore.application.ports import DecisionModel
from routercore.domain.routes import RouteMap

MODEL_NAMES = ("rules", "setfit", "laya", "laya-base")


def build_model(name: str, route_map: RouteMap, models_dir: Path, *, device: str = "cpu",
                model_version: str | None = None, checkpoint: str | None = None,
                calibrated: bool = True) -> DecisionModel:
    """`checkpoint` overrides the Laya weights; `calibrated=False` skips the fitted temperatures
    (what calibration and threshold selection need)."""
    if name == "rules":
        return RulesModel(route_map)
    if name == "setfit":
        return SetFitRouter.load(models_dir / "b1_setfit")
    if name in ("laya", "laya-base"):
        fine_tuned, calibration = models_dir / "laya_ft", models_dir / "calibration.json"
        if checkpoint is None:
            checkpoint = str(fine_tuned) if name == "laya" and fine_tuned.exists() else CHECKPOINT
        use_calibration = calibrated and name == "laya" and calibration.exists()
        return LayaModel.load(
            route_map.routes, checkpoint=checkpoint, model_version=model_version or name,
            calibration=load_calibration(calibration) if use_calibration else None,
            supported_langs={"en", "es"}, device=device)
    raise ValueError(f"unknown model {name!r}; expected one of {' | '.join(MODEL_NAMES)}")

import json

import pytest

from routercore.adapters import model_factory
from routercore.adapters.laya_model import CHECKPOINT
from routercore.adapters.model_factory import build_model
from routercore.adapters.rules_model import RulesModel
from routercore.domain.routes import RouteMap

ROUTE_MAP = RouteMap(routes={"cards": "cards", "general": "general"}, intents={})


class RecordingLaya:
    """Stands in for LayaModel.load: records how the factory called it, loads no weights."""

    calls: list[dict] = []

    @classmethod
    def load(cls, routes, **kwargs):
        cls.calls.append({"routes": routes, **kwargs})
        return "laya-model"


@pytest.fixture
def laya(monkeypatch):
    RecordingLaya.calls = []
    monkeypatch.setattr(model_factory, "LayaModel", RecordingLaya)
    return RecordingLaya


def test_rules_needs_no_weights(tmp_path):
    assert isinstance(build_model("rules", ROUTE_MAP, tmp_path), RulesModel)


def test_unknown_model_name_lists_the_valid_ones(tmp_path):
    with pytest.raises(ValueError, match="rules.*setfit.*laya"):
        build_model("gpt", ROUTE_MAP, tmp_path)


def test_laya_uses_the_fine_tuned_checkpoint_and_calibration_when_they_exist(tmp_path, laya):
    (tmp_path / "laya_ft").mkdir()
    (tmp_path / "calibration.json").write_text(
        json.dumps({"temperatures": {"intent": {"default": 2.0}}}), encoding="utf-8")

    build_model("laya", ROUTE_MAP, tmp_path, device="cuda", model_version="v7")

    [call] = laya.calls
    assert call["checkpoint"] == str(tmp_path / "laya_ft")
    assert call["calibration"].temperature("intent", "en") == 2.0
    assert call["device"] == "cuda" and call["model_version"] == "v7"
    assert call["supported_langs"] == {"en", "es"}


def test_laya_falls_back_to_the_base_checkpoint_without_a_fine_tune(tmp_path, laya):
    build_model("laya", ROUTE_MAP, tmp_path)

    [call] = laya.calls
    assert call["checkpoint"] == CHECKPOINT and call["calibration"] is None


def test_laya_base_is_the_zero_shot_model_even_if_a_fine_tune_exists(tmp_path, laya):
    (tmp_path / "laya_ft").mkdir()
    (tmp_path / "calibration.json").write_text(json.dumps({"temperatures": {}}), encoding="utf-8")

    build_model("laya-base", ROUTE_MAP, tmp_path)

    [call] = laya.calls
    assert call["checkpoint"] == CHECKPOINT and call["calibration"] is None


def test_an_explicit_checkpoint_and_uncalibrated_request_are_honoured(tmp_path, laya):
    (tmp_path / "calibration.json").write_text(json.dumps({"temperatures": {}}), encoding="utf-8")

    build_model("laya", ROUTE_MAP, tmp_path, checkpoint="models/other", calibrated=False)

    [call] = laya.calls
    assert call["checkpoint"] == "models/other" and call["calibration"] is None

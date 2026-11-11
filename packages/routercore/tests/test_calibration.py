import math

import pytest

from routercore.domain.calibration import Calibration, apply_temperature


def test_temperature_one_keeps_distribution():
    p = {"a": 0.7, "b": 0.2, "c": 0.1}
    out = apply_temperature(p, 1.0)
    assert all(math.isclose(out[k], p[k]) for k in p)


def test_higher_temperature_flattens_and_keeps_order():
    p = {"a": 0.9, "b": 0.07, "c": 0.03}
    out = apply_temperature(p, 3.0)
    assert math.isclose(sum(out.values()), 1.0)
    assert out["a"] < p["a"] and out["c"] > p["c"]
    assert max(out, key=out.get) == "a"


def test_zero_probabilities_do_not_break():
    out = apply_temperature({"a": 1.0, "b": 0.0}, 2.0)
    assert out == {"a": 1.0, "b": 0.0}


def test_nonpositive_temperature_is_rejected():
    with pytest.raises(ValueError):
        apply_temperature({"a": 1.0}, 0.0)


def test_calibration_prefers_language_temperature_then_question_default():
    cal = Calibration(temperatures={"intent": {"default": 2.0, "es": 4.0}, "urgency": {"default": 1.0}})
    assert cal.temperature("intent", "es") == 4.0
    assert cal.temperature("intent", "en") == 2.0
    assert cal.temperature("intent", None) == 2.0
    assert cal.temperature("urgency", "es") == 1.0
    assert cal.temperature("missing", "es") == 1.0

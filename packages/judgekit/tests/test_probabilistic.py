import math

import pytest

from judgekit.probabilistic import brier_score, expected_calibration_error, fit_temperature


def test_perfectly_calibrated_has_zero_ece():
    confidences = [1.0, 1.0, 0.5, 0.5]
    correct = [True, True, True, False]
    assert expected_calibration_error(confidences, correct, bins=2) == pytest.approx(0.0)


def test_overconfident_ece_is_the_gap():
    assert expected_calibration_error([0.9] * 10, [True] * 5 + [False] * 5, bins=10) == pytest.approx(0.4)


def test_brier_multiclass():
    probs = [{"a": 1.0, "b": 0.0}, {"a": 0.5, "b": 0.5}]
    assert brier_score(probs, ["a", "b"]) == pytest.approx((0.0 + 0.5) / 2)


def test_fit_temperature_softens_overconfident_model():
    # 90% confident but right only 60% of the time -> T > 1
    probs = [{"a": 0.9, "b": 0.1}] * 10
    truth = ["a"] * 6 + ["b"] * 4
    assert fit_temperature(probs, truth) > 1.0


def test_fit_temperature_sharpens_underconfident_model():
    probs = [{"a": 0.6, "b": 0.4}] * 10
    truth = ["a"] * 10
    assert fit_temperature(probs, truth) < 1.0


def test_fit_temperature_leaves_calibrated_model_alone():
    probs = [{"a": 0.8, "b": 0.2}] * 10
    truth = ["a"] * 8 + ["b"] * 2
    assert math.isclose(fit_temperature(probs, truth), 1.0, abs_tol=0.05)

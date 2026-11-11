from calibrate import fit_calibration


def record(lang, conf, truth_is_top):
    top, other = ("a", "b") if truth_is_top else ("b", "a")
    return {"lang": lang, "intent": ({"a": conf, "b": 1 - conf}, top), "urgency": ({"0": 1.0, "1": 0.0}, "0")}


def test_overconfident_language_gets_its_own_temperature_and_lower_ece():
    records = ([record("en", 0.9, True) for _ in range(90)] + [record("en", 0.9, False) for _ in range(10)]
               + [record("es", 0.9, True) for _ in range(60)] + [record("es", 0.9, False) for _ in range(40)])
    report = fit_calibration(records, min_per_language=50)
    temperatures = report["temperatures"]["intent"]
    assert temperatures["es"] > temperatures["en"] >= 1.0 - 0.05
    assert report["ece_after"]["intent"] < report["ece_before"]["intent"]


def test_small_language_falls_back_to_default():
    records = [record("en", 0.9, i < 8) for i in range(100)] + [record("es", 0.9, False) for _ in range(5)]
    report = fit_calibration(records, min_per_language=50)
    assert "es" not in report["temperatures"]["intent"] and "default" in report["temperatures"]["intent"]

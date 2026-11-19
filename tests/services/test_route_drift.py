import json

from services.dataflow.route_drift import RouteDrift, load_reference

REFERENCE = {"cards": 50, "fees": 50}


def test_nothing_is_reported_until_the_window_has_enough_decisions():
    drift = RouteDrift(REFERENCE, window=10, min_samples=4)
    assert [drift.observe("cards") for _ in range(3)] == [None, None, None]
    assert drift.observe("fees") is not None


def test_same_mix_as_training_is_stable_and_a_one_route_flood_is_a_major_shift():
    stable = RouteDrift(REFERENCE, window=100, min_samples=100)
    values = [stable.observe("cards" if i % 2 else "fees") for i in range(100)]
    assert values[-1] < 0.01
    flood = RouteDrift(REFERENCE, window=100, min_samples=100)
    assert [flood.observe("cards") for _ in range(100)][-1] > 0.25


def test_the_window_forgets_old_decisions_so_a_shift_that_ends_shows_as_ending():
    drift = RouteDrift(REFERENCE, window=40, min_samples=40)
    for _ in range(40):
        drift.observe("cards")
    for i in range(40):
        last = drift.observe("cards" if i % 2 else "fees")
    assert last < 0.01


def test_a_route_unseen_in_training_raises_the_psi(tmp_path):
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(REFERENCE), encoding="utf-8")
    drift = RouteDrift(load_reference(path), window=20, min_samples=20)
    assert [drift.observe("security") for _ in range(20)][-1] > 1.0

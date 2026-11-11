from pathlib import Path

import pytest

from routercore.adapters.route_map_config import load_route_map

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_every_banking77_intent_maps_to_a_declared_route():
    route_map = load_route_map(REPO_ROOT / "config" / "route_map.yaml")

    assert len(route_map.intents) == 77
    assert set(route_map.route_of(i) for i in route_map.intents) <= set(route_map.routes)
    assert route_map.route_of("lost_or_stolen_card") == "security"
    assert route_map.urgency_of("lost_or_stolen_card") == 3
    assert route_map.urgency_of("exchange_rate") == 0


def test_an_intent_pointing_at_an_undeclared_route_is_rejected(tmp_path):
    bad = tmp_path / "route_map.yaml"
    bad.write_text("routes: {cards: x}\nintents: {lost_card: [securty, 3]}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="securty"):
        load_route_map(bad)

"""Loads config/route_map.yaml into the domain RouteMap, rejecting intents that point at undeclared routes."""
from pathlib import Path

import yaml

from routercore.domain.routes import RouteMap


def load_route_map(path: Path) -> RouteMap:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    routes = dict(raw["routes"])
    intents = {name: (route, int(urgency)) for name, (route, urgency) in raw["intents"].items()}
    undeclared = sorted({route for route, _ in intents.values()} - set(routes))
    if undeclared:
        raise ValueError(f"intents use undeclared routes: {undeclared}")
    return RouteMap(routes=routes, intents=intents)

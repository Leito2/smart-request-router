"""The router's destinations and how Banking77 intents map onto them. Pure Python."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteMap:
    routes: dict[str, str]                       # route → description (the choices Laya sees)
    intents: dict[str, tuple[str, int]]          # source intent → (route, urgency 0-3)

    def route_of(self, intent: str) -> str:
        return self.intents[intent][0]

    def urgency_of(self, intent: str) -> int:
        return self.intents[intent][1]

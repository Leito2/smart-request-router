"""B0: keyword rules, the floor every other router must beat (PLAN §4.3).

Implements the DecisionModel port. Probabilities are keyword hit shares, so they are not calibrated:
that is the point of the comparison. Known weakness: negation ("my card was NOT stolen") fools it.
"""
from collections.abc import Sequence

from routercore.application.contracts import InboundMessage
from routercore.domain.decision import Assessment
from routercore.domain.routes import RouteMap

# route -> (keywords in English and Spanish, weight). Security weighs more: a missed theft is the worst error.
KEYWORDS: dict[str, tuple[tuple[str, ...], float]] = {
    "security": (("stolen", "lost my", "hacked", "fraud", "scam", "unauthori", "compromised", "not recogni",
                  "don't recogni", "robaron", "robo", "perdí", "perdi", "hackeado", "fraude", "estafa",
                  "no reconozco", "no autoricé"), 3.0),
    "disputes": (("charged twice", "refund", "double charge", "wrong amount", "cobraron dos", "reembolso",
                  "cobro doble", "contracargo"), 2.0),
    "fraud_inquiry": (("declined", "blocked", "rechazad", "bloquearon", "bloqueado"), 2.0),
    "kyc": (("verify my identity", "identity", "verificar mi identidad", "identidad", "documento"), 2.0),
    "topup": (("top up", "top-up", "topping", "deposit", "recarg", "depósito", "deposito"), 1.0),
    "transfers": (("transfer", "beneficiar", "transferencia", "destinatario"), 1.0),
    "fees": (("fee", "charge", "exchange rate", "commission", "comisión", "comision", "tipo de cambio"), 1.0),
    "payments": (("payment", "pending", "apple pay", "google pay", "pago", "pendiente"), 1.0),
    "cards": (("card", "pin", "activate", "tarjeta", "activar", "virtual"), 1.0),
    "account": (("account", "passcode", "personal details", "cuenta", "clave", "datos personales"), 1.0),
    "knowledge": (("how do i", "how to", "supported", "which countries", "cómo puedo", "cómo hago",
                   "soportad", "qué países"), 0.5),
}
URGENCY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "3": ("stolen", "lost my", "compromised", "hacked", "robaron", "robo", "perdí", "perdi", "hackeado"),
    "2": ("not recogni", "don't recogni", "unauthori", "fraud", "scam", "no reconozco", "fraude", "estafa"),
    "1": ("pending", "failed", "declined", "blocked", "twice", "pendiente", "falló", "rechazad",
          "bloquearon"),
}


def _normalised(scores: dict[str, float], fallback: str) -> dict[str, float]:
    total = sum(scores.values())
    if total == 0:
        return {key: float(key == fallback) for key in scores}
    return {key: value / total for key, value in scores.items()}


class RulesModel:
    model_version = "rules-v1"

    def __init__(self, route_map: RouteMap):
        self._routes = list(route_map.routes)

    def assess(self, messages: Sequence[InboundMessage]) -> list[Assessment]:
        return [self._one(m.text.casefold()) for m in messages]

    def _one(self, text: str) -> Assessment:
        route_scores = {route: 0.0 for route in self._routes}
        for route, (words, weight) in KEYWORDS.items():
            if route in route_scores:
                route_scores[route] = weight * sum(word in text for word in words)
        urgency_scores = {level: 0.0 for level in ("0", "1", "2", "3")}
        for level, words in URGENCY_KEYWORDS.items():
            urgency_scores[level] = float(any(word in text for word in words))
        highest = max((lvl for lvl in ("3", "2", "1") if urgency_scores[lvl]), default="0")
        urgency = {level: float(level == highest) for level in urgency_scores}
        return Assessment(
            distributions={"intent": _normalised(route_scores, "general"), "urgency": urgency},
            lang_supported=True, model_version=self.model_version)

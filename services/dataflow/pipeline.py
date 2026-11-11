"""One message through the System 1 dataflow (PLAN §2.5): parse → dedupe → context → classify → decide.

Pure function over a customer's keyed state; main.py wraps it in Quix Streams operators, so the logic is
testable without a broker. Returns (topic, key, value) triples to publish.
"""
from collections.abc import MutableMapping
from typing import Any

from pydantic import ValidationError

from routercore.application.contracts import Escalation, InboundMessage
from routercore.application.customer_context import register
from routercore.application.route_messages import RouteMessages


def process(raw: dict[str, Any], state: MutableMapping[str, Any],
            route: RouteMessages) -> list[tuple[str, str, dict]]:
    try:
        message = InboundMessage.model_validate(raw)
    except ValidationError as error:
        return [("dlq", str(raw.get("message_id", "")), {"error": error.errors()[0]["msg"], "raw": raw})]
    duplicate, context = register(state, message)
    if duplicate:
        return []
    [result] = route([message])
    if isinstance(result, Escalation):
        result = result.model_copy(update={"context": context})
        return [("escalations", message.message_id, result.model_dump())]
    return [("routed", message.message_id, result.model_dump())]

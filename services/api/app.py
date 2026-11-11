"""HTTP API (PLAN §3.9): `/route` answers synchronously with System 1 only, `/messages` feeds the stream."""
import time
import uuid
from collections.abc import Callable

from fastapi import FastAPI
from pydantic import BaseModel, Field

from routercore.application.contracts import Escalation, InboundMessage
from routercore.application.route_messages import RouteMessages


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    customer_id: str = "anonymous"
    channel: str = "chat"
    lang: str | None = None


def _inbound(body: MessageIn) -> InboundMessage:
    return InboundMessage(message_id=str(uuid.uuid4()), customer_id=body.customer_id, channel=body.channel,
                          text=body.text, lang=body.lang, ts_sent_ns=time.time_ns())


def create_app(route: RouteMessages, publish: Callable[[InboundMessage], None]) -> FastAPI:
    app = FastAPI(title="smart-request-router")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/route")
    def route_one(body: MessageIn) -> dict:
        [result] = route([_inbound(body)])
        if isinstance(result, Escalation):
            return {"decision": "escalate", "reason": result.reason, "message_id": result.message.message_id}
        return {"decision": "system1", **result.model_dump(exclude={"text"})}

    @app.post("/messages", status_code=202)
    def accept(body: MessageIn) -> dict:
        message = _inbound(body)
        publish(message)
        return {"message_id": message.message_id}

    return app

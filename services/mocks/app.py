"""Mock P1 and P3 with their real contracts, for CI and offline demos (PLAN §3.12).

P3 `POST /ask` streams server-sent events (status, token, done); P1 `GET /decisions/{id}` returns the decision
with its label and explanation, 404 when unknown.
"""
import hashlib
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

DECISIONS = ["ALLOW", "BLOCK", "REVIEW"]


class Ask(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def create_mock_app() -> FastAPI:
    app = FastAPI(title="mock P1 + P3")

    @app.post("/ask")
    def ask(body: Ask) -> StreamingResponse:
        def stream():
            yield _event("status", {"state": "generating"})
            yield _event("token", {"text": f"(mock) Help-center answer for: {body.question}"})
            yield _event("done", {"abstained": False})
        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/decisions/{payment_id}")
    def decision(payment_id: str) -> dict:
        if payment_id == "pay_unknown":
            raise HTTPException(404, f"no decision for payment {payment_id}")
        verdict = DECISIONS[int(hashlib.sha256(payment_id.encode()).hexdigest(), 16) % len(DECISIONS)]
        return {"payment_id": payment_id, "decision": verdict, "label_is_fraud": verdict == "BLOCK",
                "explanation": f"{verdict} because of the mock velocity rule"}

    return app

"""Mock P1 and P3 with the real contracts, for CI and offline demos (PLAN §3.12)."""
import hashlib

from fastapi import FastAPI
from pydantic import BaseModel

DECISIONS = ["ALLOW", "BLOCK", "REVIEW"]


class Ask(BaseModel):
    question: str


def create_mock_app() -> FastAPI:
    app = FastAPI(title="mock P1 + P3")

    @app.post("/ask")
    def ask(body: Ask) -> dict:
        return {"answer": f"(mock) Help-center answer for: {body.question} [1]",
                "sources": [{"id": 1, "title": "Help center"}]}

    @app.get("/decisions/{payment_id}")
    def decision(payment_id: str) -> dict:
        index = int(hashlib.sha256(payment_id.encode()).hexdigest(), 16) % len(DECISIONS)
        return {"payment_id": payment_id, "decision": DECISIONS[index], "reason": "mock rule"}

    return app

"""The real P1 `api.app.create_app` with in-memory stores. Run with P1's own interpreter:
`<P1>/.venv/.../python -m uvicorn --app-dir tests/contract --factory p1_server:build`."""
from api.app import create_app

DECISIONS = {
    "pay_explained": {"payment_id": "pay_explained", "decision": "block", "score": 0.97,
                      "explanation": "velocity spike on a new device"},
    "pay_pending": {"payment_id": "pay_pending", "decision": "review", "score": 0.61, "explanation": None},
}


class Decisions:
    def decision(self, payment_id: str) -> dict | None:
        return DECISIONS.get(payment_id)


class NoProfiles:
    def features(self, user_id: str) -> dict | None:
        return None


class NoModel:
    def champion(self) -> dict | None:
        return None


def build():
    return create_app(Decisions(), NoProfiles(), NoModel())

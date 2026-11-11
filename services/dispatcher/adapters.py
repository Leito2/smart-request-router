"""Outbound adapters of the dispatcher: Postgres decision log, Redis Streams queues, P3/P1 HTTP clients."""
import json
import re

import httpx

from routercore.application.contracts import Routed

PAYMENT_ID = re.compile(r"\b(pay_[A-Za-z0-9]+)\b")


class PostgresDecisionLog:
    """`record` is the idempotency gate (ADR-8): the primary key rejects a message_id seen before."""

    SQL = ("INSERT INTO decisions (message_id, route, priority, decided_by, confidence, model_version, "
           "threshold_set) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (message_id) DO NOTHING")

    def __init__(self, connection):
        self._connection = connection

    def record(self, routed: Routed) -> bool:
        with self._connection.cursor() as cursor:
            cursor.execute(self.SQL, (routed.message_id, routed.route, routed.priority, routed.decided_by,
                                      routed.confidence, routed.model_version, routed.threshold_set))
            inserted = cursor.rowcount == 1
        self._connection.commit()
        return inserted


class RedisTeamQueues:
    def __init__(self, redis_client):
        self._redis = redis_client

    def enqueue(self, team: str, routed: Routed) -> None:
        self._redis.xadd(f"queue:{team}", {"message_id": routed.message_id, "priority": routed.priority,
                                           "text": routed.text or ""})


def parse_sse(lines) -> list[tuple[str, dict]]:
    """Server-sent events as (name, data) pairs."""
    events, name = [], ""
    for line in lines:
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            events.append((name, json.loads(line[5:])))
    return events


def assemble_answer(events: list[tuple[str, dict]]) -> str:
    """Join the `token` events; a `retract` discards what was streamed so far (the answer was withdrawn)."""
    text = ""
    for name, data in events:
        if name == "token":
            text += data.get("text", "")
        elif name == "retract":
            text = ""
        elif name == "error":
            return f"unavailable ({data.get('code', 'error')})"
    return text.strip()


class HttpKnowledgeBase:
    """P3 live-rag-platform `POST /ask` (server-sent events). A dead service degrades to a marker."""

    def __init__(self, base_url: str, client: httpx.Client | None = None, timeout_s: float = 30):
        self._url, self._client = base_url.rstrip("/"), client or httpx.Client(timeout=timeout_s)

    def answer(self, question: str) -> str:
        try:
            with self._client.stream("POST", f"{self._url}/ask", json={"question": question}) as response:
                response.raise_for_status()
                return assemble_answer(parse_sse(response.iter_lines())) or "unavailable (empty answer)"
        except (httpx.HTTPError, ValueError) as error:
            return f"unavailable ({type(error).__name__})"


class HttpPaymentDecisions:
    """P1 realtime-fraud-detection `GET /decisions/{payment_id}`; the id is read from the message text."""

    def __init__(self, base_url: str, client: httpx.Client | None = None, timeout_s: float = 10):
        self._url, self._client = base_url.rstrip("/"), client or httpx.Client(timeout=timeout_s)

    def decision_for(self, text: str) -> str:
        match = PAYMENT_ID.search(text)
        if match is None:
            return "needs payment id"
        try:
            response = self._client.get(f"{self._url}/decisions/{match.group(1)}")
            if response.status_code == 404:
                return "no decision found for that payment"
            response.raise_for_status()
            body = response.json()
            return f"{body['decision']}: {body.get('explanation', '')}"
        except (httpx.HTTPError, KeyError, ValueError) as error:
            return f"unavailable ({type(error).__name__})"

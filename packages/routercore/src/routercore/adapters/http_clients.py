"""HTTP clients of the other platforms (P3 live-rag-platform, P1 realtime-fraud-detection), shared by the
dispatcher and System 2. Needs `httpx` (routercore extra `http`)."""
import json
import re

import httpx

PAYMENT_ID = re.compile(r"\b(pay_[A-Za-z0-9]+)\b")


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
            explanation = body.get("explanation")      # null until the explainer has run (LEFT JOIN in P1)
            return f"{body['decision']}: {explanation}" if explanation else str(body["decision"])
        except (httpx.HTTPError, KeyError, ValueError) as error:
            return f"unavailable ({type(error).__name__})"

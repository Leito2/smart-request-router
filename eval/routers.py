"""Routers under evaluation (PLAN §4.3), all answering with harness Predictions.

ModelRouter wraps any DecisionModel (B0 rules, B1 SetFit, B3 Laya), with or without the escalation policy.
LlmOnlyRouter (B4) asks an LLM for every message through the gateway.
"""
import asyncio
import json
import time

from harness import Prediction

from routercore.application.contracts import InboundMessage
from routercore.application.ports import DecisionModel
from routercore.domain.policy import Policy, decide

BATCH = 16     # micro-batch size (ADR-3)


def _argmax(distribution: dict[str, float]) -> str:
    return max(distribution, key=distribution.__getitem__)


class ModelRouter:
    def __init__(self, model: DecisionModel, policy: Policy | None):
        self._model, self._policy = model, policy

    def predict(self, rows: list[dict]) -> list[Prediction]:
        predictions = []
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            messages = [InboundMessage(message_id=r["id"], customer_id="eval", channel="chat",
                                       text=r["text"], lang=r.get("lang"), ts_sent_ns=0) for r in batch]
            began = time.perf_counter()
            assessments = self._model.assess(messages)
            per_message_ms = (time.perf_counter() - began) * 1000 / len(batch)
            for a in assessments:
                best_route = _argmax(a.route_p)
                escalated = False
                if self._policy is not None:
                    target, _ = decide(a.route_p, a.urgency_p, a.lang_supported, self._policy)
                    escalated = target == "system2"
                predictions.append(Prediction(route=best_route, urgency=int(_argmax(a.urgency_p)),
                                              escalated=escalated, latency_ms=per_message_ms))
        return predictions


SYSTEM_PROMPT = (
    "You route messages sent to a fintech's customer support. Pick the single best route and the urgency "
    "(0 informational, 1 money stuck or an action failed, 2 unrecognised charge or possible fraud, "
    "3 card or phone lost, stolen or compromised). Routes:\n{routes}\n"
    'Answer with JSON only: {{"route": "<route>", "urgency": <0-3>}}.'
)


def parse_answer(text: str, routes: set[str]) -> tuple[str, int]:
    """Parse the LLM's JSON; an invalid answer falls back to `general`, urgency 0."""
    try:
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        route, urgency = data["route"], int(data["urgency"])
    except (ValueError, KeyError, TypeError):
        return "general", 0
    return (route if route in routes else "general"), min(max(urgency, 0), 3)


class LlmOnlyRouter:
    """B4: one LLM call per message, structured JSON output. Needs the `runner` extra (aiohttp)."""

    def __init__(self, routes: dict[str, str], *, base_url: str, model: str, api_key: str | None = None,
                 concurrency: int = 2, extra_body: dict | None = None):
        self._routes, self._base_url, self._model = routes, base_url, model
        self._api_key, self._concurrency, self._extra_body = api_key, concurrency, extra_body

    def predict(self, rows: list[dict]) -> list[Prediction]:
        return asyncio.run(self._predict(rows))

    async def _predict(self, rows: list[dict]) -> list[Prediction]:
        from judgekit.clients import ChatClient
        from judgekit.runner import run

        listing = "\n".join(f"- {name}: {desc}" for name, desc in self._routes.items())
        system = SYSTEM_PROMPT.format(routes=listing)
        async with ChatClient(self._base_url, model=self._model, api_key=self._api_key,
                              extra_body=self._extra_body) as client:
            async def ask(text: str) -> dict:
                done = await client.complete(
                    [{"role": "system", "content": system}, {"role": "user", "content": text}], max_tokens=40)
                return {"text": done.text, "tokens": done.prompt_tokens + done.completion_tokens,
                        "ms": done.latency_ms}

            result = await run([(r["id"], r["text"]) for r in rows], ask, concurrency=self._concurrency,
                               retries=2)
        predictions = []
        for r in rows:
            answer = result.outputs.get(r["id"])
            route, urgency = parse_answer(answer["text"], set(self._routes)) if answer else ("general", 0)
            predictions.append(Prediction(route=route, urgency=urgency, llm_calls=1,
                                          tokens=answer["tokens"] if answer else 0,
                                          latency_ms=answer["ms"] if answer else 0.0))
        return predictions

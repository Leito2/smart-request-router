"""System 2 adapters: LLM through the gateway, and the agent's tools over the existing ports."""
import json
import os
import time

import httpx

from routercore.application.ports import KnowledgeBase, PaymentDecisions

NO_THINKING = {"reasoning_effort": "none"}      # qwen3 on Ollama otherwise spends the budget thinking
# the gateway validates reasoning_effort (low|medium|high), so talking to it sends nothing extra by default
GATEWAY_EXTRA_BODY: dict = {}


def extra_body_from_env(default: dict) -> dict:
    """`LLM_EXTRA_BODY` (a JSON object merged into every request) overrides the default for the endpoint."""
    raw = os.environ.get("LLM_EXTRA_BODY")
    return json.loads(raw) if raw else default


class GatewayLlm:
    """OpenAI-compatible chat call (llm-gateway alias `smart_router`, or Ollama directly) -> text."""

    def __init__(self, base_url: str, *, model: str = "smart_router", api_key: str | None = None,
                 client: httpx.Client | None = None, timeout_s: float = 20, max_tokens: int = 400,
                 extra_body: dict | None = None):
        self._url, self._model, self._max_tokens = base_url.rstrip("/"), model, max_tokens
        self._extra = NO_THINKING if extra_body is None else extra_body
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout_s)
        self.tokens = 0
        self.providers: list[str] = []         # who answered each call (x-gateway-provider), for audits

    def __call__(self, system: str, user: str) -> str:
        response = self._client.post(
            f"{self._url}/v1/chat/completions", headers=self._headers,
            json={"model": self._model, "temperature": 0, "max_tokens": self._max_tokens, **self._extra,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        response.raise_for_status()
        body = response.json()
        self.tokens += body.get("usage", {}).get("total_tokens", 0)
        self.providers.append(response.headers.get("x-gateway-provider", self._model))
        return body["choices"][0]["message"]["content"] or ""


class QuotaPacedLlm:
    """Eval-time wrapper: the gateway enforces the free tier's outbound quotas, so a 429/503 means "wait",
    not "fail". Waits for Retry-After (else 10 s) and retries; other errors propagate to the agent."""

    def __init__(self, llm, *, retries: int = 90, default_wait_s: float = 10):
        self._llm, self._retries, self._wait = llm, retries, default_wait_s
        self.waited_s = 0.0

    @property
    def tokens(self) -> int:
        return self._llm.tokens

    @property
    def providers(self) -> list[str]:
        return self._llm.providers

    def __call__(self, system: str, user: str) -> str:
        for attempt in range(self._retries + 1):
            try:
                return self._llm(system, user)
            except httpx.HTTPStatusError as error:
                if error.response.status_code not in (429, 503) or attempt == self._retries:
                    raise
                wait = float(error.response.headers.get("retry-after", self._wait))
                self.waited_s += wait
                time.sleep(wait)
        raise AssertionError("unreachable")


class RedisHistory:
    """Recent messages of a customer, kept in a Redis list per customer."""

    def __init__(self, redis_client, key_prefix: str = "history:"):
        self._redis, self._prefix = redis_client, key_prefix

    def get(self, customer_id: str) -> list[str]:
        return [item.decode() if isinstance(item, bytes) else item
                for item in self._redis.lrange(f"{self._prefix}{customer_id}", 0, 9)]


class InstrumentedTools:
    """Wraps the agent's tools and reports each call by tool name (the R4 dashboard's steps per case)."""

    def __init__(self, inner, on_call):
        self._inner, self._on_call = inner, on_call

    def __getattr__(self, name):
        method = getattr(self._inner, name)

        def call(*args, **kwargs):
            self._on_call(name)
            return method(*args, **kwargs)
        return call


class LiveTools:
    def __init__(self, history: RedisHistory, payments: PaymentDecisions, knowledge: KnowledgeBase):
        self._history, self._payments, self._knowledge = history, payments, knowledge

    def get_customer_history(self, customer_id: str) -> list[str]:
        return self._history.get(customer_id)

    def lookup_payment_decision(self, text: str) -> str:
        return self._payments.decision_for(text)

    def search_kb(self, question: str) -> str:
        return self._knowledge.answer(question)

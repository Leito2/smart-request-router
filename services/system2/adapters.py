"""System 2 adapters: LLM through the gateway, and the agent's tools over the existing ports."""
import httpx

from routercore.application.ports import KnowledgeBase, PaymentDecisions

NO_THINKING = {"reasoning_effort": "none"}      # qwen3 on Ollama otherwise spends the budget thinking


class GatewayLlm:
    """OpenAI-compatible chat call (llm-gateway alias `smart_router`, or Ollama directly) -> text."""

    def __init__(self, base_url: str, *, model: str = "smart_router", api_key: str | None = None,
                 client: httpx.Client | None = None, timeout_s: float = 20, max_tokens: int = 400):
        self._url, self._model, self._max_tokens = base_url.rstrip("/"), model, max_tokens
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout_s)
        self.tokens = 0

    def __call__(self, system: str, user: str) -> str:
        response = self._client.post(
            f"{self._url}/v1/chat/completions", headers=self._headers,
            json={"model": self._model, "temperature": 0, "max_tokens": self._max_tokens, **NO_THINKING,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        response.raise_for_status()
        body = response.json()
        self.tokens += body.get("usage", {}).get("total_tokens", 0)
        return body["choices"][0]["message"]["content"] or ""


class RedisHistory:
    """Recent messages of a customer, kept in a Redis list per customer."""

    def __init__(self, redis_client, key_prefix: str = "history:"):
        self._redis, self._prefix = redis_client, key_prefix

    def get(self, customer_id: str) -> list[str]:
        return [item.decode() if isinstance(item, bytes) else item
                for item in self._redis.lrange(f"{self._prefix}{customer_id}", 0, 9)]


class LiveTools:
    def __init__(self, history: RedisHistory, payments: PaymentDecisions, knowledge: KnowledgeBase):
        self._history, self._payments, self._knowledge = history, payments, knowledge

    def get_customer_history(self, customer_id: str) -> list[str]:
        return self._history.get(customer_id)

    def lookup_payment_decision(self, text: str) -> str:
        return self._payments.decision_for(text)

    def search_kb(self, question: str) -> str:
        return self._knowledge.answer(question)

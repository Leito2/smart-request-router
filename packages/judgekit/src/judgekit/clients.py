"""aiohttp client for OpenAI-compatible chat endpoints (LLM gateway, Ollama). Needs the `runner` extra."""
import time
from dataclasses import dataclass
from typing import Any

import aiohttp


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class ChatClient:
    def __init__(self, base_url: str, *, model: str, api_key: str | None = None, timeout_s: float = 120,
                 extra_body: dict[str, Any] | None = None):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._extra_body = extra_body or {}
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "ChatClient":
        self._session = aiohttp.ClientSession(headers=self._headers, timeout=self._timeout)
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._session is not None:
            await self._session.close()

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int = 256, temperature: float = 0,
                       response_format: dict[str, Any] | None = None) -> Completion:
        body: dict[str, Any] = {"model": self._model, "messages": messages, "max_tokens": max_tokens,
                                "temperature": temperature,
                                **self._extra_body}
        if response_format is not None:
            body["response_format"] = response_format
        started = time.perf_counter()
        async with self._session.post(f"{self._base_url}/v1/chat/completions", json=body) as response:
            response.raise_for_status()
            data = await response.json()
        usage = data.get("usage", {})
        return Completion(
            text=data["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=(time.perf_counter() - started) * 1000)

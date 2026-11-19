"""Tracing of System 2 runs in Langfuse (PLAN §3.12). Optional: without LANGFUSE_* keys it is a no-op, so the
agent never depends on the observability stack to route a message.

Langfuse Cloud (free tier) until M10; self-hosted Langfuse needs more RAM than this machine has.

Everything sent to Langfuse goes through the Presidio redactor (card numbers, IBAN, email, phone, IP, NIF/NIE
are replaced by `<ENTITY>`). If it cannot be loaded, tracing is OFF rather than leaking the message text.
"""
import logging
import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol

log = logging.getLogger("system2.tracing")


class Span(Protocol):
    def update(self, output: Any = None, metadata: dict | None = None) -> None: ...


class Tracer(Protocol):
    def span(self, name: str, input: Any) -> AbstractContextManager[Span]: ...


class _NoopSpan:
    def update(self, output: Any = None, metadata: dict | None = None) -> None:
        return None


class NoopTracer:
    def span(self, name: str, input: Any) -> AbstractContextManager[Span]:
        return nullcontext(_NoopSpan())


class LangfuseTracer:
    """Langfuse SDK: one observation per agent run, with input, output and metadata, PII redacted."""

    def __init__(self):
        from langfuse import Langfuse, get_client

        from routercore.adapters.pii import PiiRedactor

        Langfuse(mask=PiiRedactor().mask)      # registers the masked client that get_client() returns below
        self._client = get_client()

    def span(self, name: str, input: Any) -> AbstractContextManager[Span]:
        return self._client.start_as_current_observation(name=name, as_type="span", input=input)


def from_env() -> Tracer:
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        try:
            return LangfuseTracer()
        except Exception as error:             # a broken tracer must never stop routing, nor leak text
            log.warning("tracing disabled: %s: %s", type(error).__name__, error)
            return NoopTracer()
    return NoopTracer()


def trace_run(tracer: Tracer, name: str, input: Any, run: Callable[[], Any], *,
              metadata: Callable[[Any], dict] | None = None) -> Any:
    with tracer.span(name, input) as span:
        output = run()
        span.update(output=output, metadata=metadata(output) if metadata else None)
        return output

"""Tracing of System 2 runs in Langfuse (PLAN §3.12). Optional: without LANGFUSE_* keys it is a no-op, so the
agent never depends on the observability stack to route a message.

Langfuse Cloud (free tier) until M10; self-hosted Langfuse needs more RAM than this machine has.
"""
import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol


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
    """Langfuse SDK v3: one observation per agent run, with input, output and metadata."""

    def __init__(self):
        from langfuse import get_client

        self._client = get_client()

    def span(self, name: str, input: Any) -> AbstractContextManager[Span]:
        return self._client.start_as_current_observation(name=name, as_type="span", input=input)


def from_env() -> Tracer:
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        try:
            return LangfuseTracer()
        except Exception:                      # a broken tracer must never stop routing
            return NoopTracer()
    return NoopTracer()


def trace_run(tracer: Tracer, name: str, input: Any, run: Callable[[], Any], *,
              metadata: Callable[[Any], dict] | None = None) -> Any:
    with tracer.span(name, input) as span:
        output = run()
        span.update(output=output, metadata=metadata(output) if metadata else None)
        return output

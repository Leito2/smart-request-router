"""Async runner: bounded concurrency over items, used for judge calls and the LLM-only router."""
import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    outputs: dict[str, Any] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)


async def run(items: Sequence[tuple[str, Any]], fn: Callable[[Any], Awaitable[Any]], *,
              concurrency: int = 8, retries: int = 0, backoff_s: float = 0.5,
              checkpoint: Path | None = None) -> RunResult:
    """Call `fn` on every (id, item) with at most `concurrency` calls in flight.

    A failing call is retried `retries` times with exponential backoff. If it still fails, the error is
    recorded in `failures` and the other items carry on. With a `checkpoint` file, finished outputs are
    appended as they arrive and a rerun skips them (outputs must be JSON). Failures are never checkpointed.
    """
    result = RunResult()
    gate = asyncio.Semaphore(concurrency)
    if checkpoint is not None and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            done = json.loads(line)
            result.outputs[done["id"]] = done["output"]
    sink = checkpoint.open("a", encoding="utf-8") if checkpoint is not None else None

    async def one(item_id: str, item: Any) -> None:
        async with gate:
            for attempt in range(retries + 1):
                try:
                    result.outputs[item_id] = await fn(item)
                    if sink is not None:
                        line = json.dumps({"id": item_id, "output": result.outputs[item_id]})
                        sink.write(line + "\n")
                        sink.flush()
                    return
                except Exception as error:
                    if attempt == retries:
                        result.failures[item_id] = f"{type(error).__name__}: {error}"
                    else:
                        await asyncio.sleep(backoff_s * 2**attempt)

    try:
        async with asyncio.TaskGroup() as group:
            for item_id, item in items:
                if item_id not in result.outputs:
                    group.create_task(one(item_id, item))
    finally:
        if sink is not None:
            sink.close()
    return result


def estimate_cost(n_items: int, *, tokens_per_item: int, usd_per_1m_tokens: float) -> float:
    """Dollar estimate for a run, computed before spending anything."""
    return n_items * tokens_per_item * usd_per_1m_tokens / 1_000_000

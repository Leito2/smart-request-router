"""Use case: dedupe and per-customer context (PLAN §2.2 ADR-2, §3.3).

The state is a plain dict, so Quix Streams' keyed state (JSON-serializable, one per customer_id) can be
passed straight in; this module knows nothing of Quix.
"""
from collections.abc import MutableMapping
from typing import Any

from routercore.application.contracts import InboundMessage

SECOND_NS = 1_000_000_000
DEDUPE_NS = 60 * SECOND_NS
CONTEXT_NS = 300 * SECOND_NS
MAX_RECENT = 10


def _key(text: str) -> str:
    return " ".join(text.lower().split())


def register(state: MutableMapping[str, Any], message: InboundMessage) -> tuple[bool, list[str]]:
    """Returns (is_duplicate, recent texts of this customer within 5 min) and updates the state in place."""
    now = message.ts_sent_ns
    recent = [r for r in state.get("recent", []) if now - r[0] <= CONTEXT_NS]
    key = _key(message.text)
    duplicate = any(r[1] == key and now - r[0] <= DEDUPE_NS for r in recent)
    context = [r[2] for r in recent]
    if not duplicate:
        recent.append([now, key, message.text])
    state["recent"] = recent[-MAX_RECENT:]
    return duplicate, context

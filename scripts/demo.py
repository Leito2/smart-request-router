# ruff: noqa: E501
"""Five-message demo of the whole router, no Docker: System 1 (SetFit + policy) decides what it can, and what
it escalates goes to the System 2 agent (LangGraph) through the llm-gateway alias `smart_router`.

    uv run --no-sync python scripts/demo.py            # make demo
    uv run --no-sync python scripts/demo.py --no-s2    # System 1 only (no gateway needed)

Needs the SetFit model in models/b1_setfit (download it from the Hub, see README "Try it in 5 minutes").
System 2 reads GATEWAY_URL / GATEWAY_KEY / LLM_MODEL from `.env`. If the gateway is not reachable the escalated
messages show what would happen in production: they go to the human triage queue, never lost. With LANGFUSE_*
keys in `.env` every System 2 run is traced. The payment and help-center tools are in-memory stand-ins unless
`--p1` / `--p3` point at real services. Writes docs/results/demo.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CASES = [
    ("S1 resolves it (English)", "What is the exchange rate fee when I pay abroad?"),
    ("S1 resolves it (Spanish)", "Hola, quiero saber cuánto cuesta una transferencia internacional"),
    ("Urgent: safety net, then System 2",
     "My card was stolen and someone just spent 400 euros with it, please block everything now"),
    ("Needs a tool: P1 payment decision", "Why was my payment pay_8842 blocked? I need it to go through today"),
    ("Vague message", "Can you help?"),
]


class DemoTools:
    """In-memory stand-ins for P1 (payment decisions), P3 (help center) and the customer history."""

    def __init__(self):
        self.calls: list[str] = []

    def get_customer_history(self, customer_id: str) -> list[str]:
        self.calls.append("get_customer_history")
        return []

    def lookup_payment_decision(self, text: str) -> str:
        self.calls.append("lookup_payment_decision")
        return "block: velocity spike on a new device" if "pay_8842" in text else "needs payment id"

    def search_kb(self, question: str) -> str:
        self.calls.append("search_kb")
        return "Card payments are declined when the issuer flags unusual activity; support can lift the block."


class RecordingTools:
    """Wraps real or stand-in tools and remembers which ones the agent called."""

    def __init__(self, inner):
        self._inner, self.calls = inner, []

    def __getattr__(self, name):
        method = getattr(self._inner, name)

        def call(*args):
            self.calls.append(name)
            return method(*args)
        return call


def build_tools(p1: str | None, p3: str | None):
    if not (p1 or p3):
        return DemoTools()
    from routercore.adapters.http_clients import HttpKnowledgeBase, HttpPaymentDecisions
    from services.system2.adapters import LiveTools

    class NoHistory:
        def get(self, customer_id: str) -> list[str]:
            return []
    return LiveTools(NoHistory(), HttpPaymentDecisions(p1 or "http://localhost:9"),
                     HttpKnowledgeBase(p3 or "http://localhost:9"))


def main() -> None:
    from routercore.application.contracts import Escalation, InboundMessage
    from services.composition import build_route_messages
    from services.envfile import load_env

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-s2", action="store_true", help="System 1 only")
    parser.add_argument("--model", default="setfit")
    parser.add_argument("--p1", help="real P1 base URL (GET /decisions/{payment_id})")
    parser.add_argument("--p3", help="real P3 base URL (POST /ask)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_env(ROOT / ".env")

    route_messages = build_route_messages(args.model)
    agent = llm = tracer = None
    tools = RecordingTools(build_tools(args.p1, args.p3))
    if not args.no_s2:
        from routercore.adapters.route_map_config import load_route_map
        from services.system2.adapters import (
            GATEWAY_EXTRA_BODY,
            GatewayLlm,
            QuotaPacedLlm,
            extra_body_from_env,
        )
        from services.system2.graph import build_agent
        from services.system2.tracing import from_env

        routes = load_route_map(ROOT / "config" / "route_map.yaml").routes
        llm = GatewayLlm(os.environ.get("GATEWAY_URL", "http://localhost:8080"),
                         model=os.environ.get("LLM_MODEL", "smart_router"), api_key=os.environ.get("GATEWAY_KEY"),
                         timeout_s=60, extra_body=extra_body_from_env(GATEWAY_EXTRA_BODY))
        llm = QuotaPacedLlm(llm)        # the free tier's quota says "wait", it is not a failure
        agent, tracer = build_agent(llm, tools, routes), from_env()

    results = []
    for number, (label, text) in enumerate(CASES, 1):
        message = InboundMessage(message_id=f"demo-{number}", customer_id="demo", channel="chat", text=text,
                                 lang=None, ts_sent_ns=0)
        began = time.perf_counter()
        event = route_messages([message])[0]
        s1_ms = (time.perf_counter() - began) * 1000
        row = {"case": label, "text": text, "system1_ms": round(s1_ms, 1)}
        if isinstance(event, Escalation):
            row |= {"escalated_because": event.reason}
            if agent is None:
                row |= {"decided_by": "-", "route": "(System 2 off)"}
            else:
                tools.calls.clear()
                providers_before = len(llm.providers)
                began = time.perf_counter()
                try:
                    from services.system2.tracing import trace_run

                    payload = event.model_dump()
                    routed = trace_run(tracer, "demo.system2", payload,
                                       lambda: agent.invoke({"escalation": payload})["routed"])  # noqa: B023
                    row |= {"decided_by": "system2", "route": routed["route"], "priority": routed["priority"],
                            "rationale": routed.get("rationale", ""), "tools_called": list(tools.calls),
                            "served_by": sorted(set(llm.providers[providers_before:])),
                            "system2_s": round(time.perf_counter() - began, 1)}
                except Exception as error:      # in production: queue:triage, a human takes it
                    row |= {"decided_by": "triage queue", "route": "triage", "priority": "high",
                            "rationale": f"System 2 unavailable ({type(error).__name__})"}
        else:
            row |= {"decided_by": "system1", "route": event.route, "priority": event.priority,
                    "confidence": round(event.confidence, 2)}
        results.append(row)
        print(f"\n[{number}] {label}\n    \"{text}\"")
        print(f"    System 1 ({s1_ms:.0f} ms): "
              + (f"escalated, reason = {row['escalated_because']}" if "escalated_because" in row
                 else f"route = {row['route']}, priority = {row['priority']}, confidence = {row['confidence']}"))
        if "escalated_because" in row:
            print(f"    -> {row['decided_by']}: route = {row['route']}, priority = {row.get('priority', '-')}")
            if row.get("rationale"):
                print(f"       why: {row['rationale']}")
            if row.get("tools_called"):
                print(f"       tools: {', '.join(row['tools_called'])}")
            if row.get("served_by"):
                print(f"       LLM: {', '.join(row['served_by'])} in {row['system2_s']} s")

    if tracer is not None and os.environ.get("LANGFUSE_PUBLIC_KEY"):
        try:
            from langfuse import get_client
            get_client().flush()
            print("\nTraces sent to Langfuse.")
        except Exception as error:
            print(f"\nLangfuse flush failed: {type(error).__name__}")
    out = ROOT / "docs" / "results" / "demo.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{len(results)} cases -> {out}")


if __name__ == "__main__":
    main()

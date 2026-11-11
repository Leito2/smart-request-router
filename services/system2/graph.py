"""System 2: a bounded LangGraph agent that makes ONE better routing decision for an escalated message
(PLAN §3.6).  analyze <-> tools -> decide -> validate -> emit; max 2 decision retries; fallback queue:triage.

The LLM is a plain callable (system, user) -> text, so the gateway, a local model or a scripted fake plug in.
"""
import json
import time
from collections.abc import Callable
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, StateGraph

from routercore.application.contracts import InboundMessage, Routed

Llm = Callable[[str, str], str]
PRIORITIES = ("low", "normal", "high", "critical")
TOOL_NAMES = ("get_customer_history", "lookup_payment_decision", "search_kb")
FALLBACK_ROUTE = "triage"
MAX_DECISION_ATTEMPTS = 3          # first try + 2 retries


class Tools(Protocol):
    def get_customer_history(self, customer_id: str) -> list[str]: ...
    def lookup_payment_decision(self, text: str) -> str: ...
    def search_kb(self, question: str) -> str: ...


class State(TypedDict, total=False):
    escalation: dict
    observations: list[str]
    steps: int
    next_tool: str | None
    decision: dict | None
    attempts: int
    routed: dict


def _json(text: str) -> dict:
    try:
        value = json.loads(text[text.index("{"):text.rindex("}") + 1])
        return value if isinstance(value, dict) else {}
    except ValueError:
        return {}


ANALYZE_PROMPT = (
    "You help route a fintech support message. Decide if you need one more tool. Tools: "
    "get_customer_history (the customer's recent messages), lookup_payment_decision (why a payment was "
    'blocked), search_kb (help-center answer). Reply JSON only: {"tool": "<name>"|null}.')
DECIDE_PROMPT = (
    "Pick the single best route and priority. Routes:\n{routes}\nPriorities: low, normal, high, critical. "
    'Reply JSON only: {{"route": "...", "priority": "...", "rationale": "...", "evidence": ["..."]}}.')


def build_agent(llm: Llm, tools: Tools, routes: dict[str, str], *, max_steps: int = 6,
                model_version: str = "system2", threshold_set: str = "s2",
                now_ns: Callable[[], int] = time.time_ns):
    route_list = "\n".join(f"- {name}: {desc}" for name, desc in routes.items())

    def facts(state: State) -> str:
        e = state["escalation"]
        scores = {q: {k: round(v, 2) for k, v in d.items() if v >= 0.03} for q, d in e["laya"].items()}
        seen = "\n".join(state.get("observations", [])) or "(none)"
        return (f"Message: {e['message']['text']}\nEscalated because: {e['reason']}\n"
                f"System 1 scores: {json.dumps(scores)}\nRecent messages: {e.get('context', [])}\n"
                f"Tool observations:\n{seen}")

    def analyze(state: State) -> State:
        if state.get("steps", 0) >= max_steps:
            return {"next_tool": None}
        tool = _json(llm(ANALYZE_PROMPT, facts(state))).get("tool")
        return {"next_tool": tool if tool in TOOL_NAMES else None}

    def run_tool(state: State) -> State:
        message = state["escalation"]["message"]
        tool = state["next_tool"]
        try:
            if tool == "get_customer_history":
                history = (tools.get_customer_history(message["customer_id"])
                           or state["escalation"].get("context", []))
                result = " | ".join(history) or "no recent messages"
            elif tool == "lookup_payment_decision":
                result = tools.lookup_payment_decision(message["text"])
            else:
                result = tools.search_kb(message["text"])
        except Exception as error:              # a broken tool is an observation, not a crash
            result = f"tool failed: {type(error).__name__}"
        return {"observations": [*state.get("observations", []), f"{tool}: {result}"],
                "steps": state.get("steps", 0) + 1}

    def decide(state: State) -> State:
        answer = _json(llm(DECIDE_PROMPT.format(routes=route_list), facts(state)))
        return {"decision": answer, "attempts": state.get("attempts", 0) + 1}

    def validate(state: State) -> State:
        d = state.get("decision") or {}
        if d.get("route") in routes and d.get("priority") in PRIORITIES and d.get("rationale"):
            if state["escalation"]["reason"] == "urgency_safety_net" and PRIORITIES.index(d["priority"]) < 2:
                d = {**d, "priority": "high"}          # the safety net never ends below high (ADR-4)
            return {"decision": d}
        return {"decision": None}

    def after_validate(state: State) -> str:
        if state.get("decision"):
            return "emit"
        return "decide" if state.get("attempts", 0) < MAX_DECISION_ATTEMPTS else "emit"

    def emit(state: State) -> State:
        d = state.get("decision")
        message = InboundMessage.model_validate(state["escalation"]["message"])
        routed = Routed(
            message_id=message.message_id, route=d["route"] if d else FALLBACK_ROUTE,
            priority=d["priority"] if d else "high", decided_by="system2", confidence=1.0 if d else 0.0,
            model_version=model_version, threshold_set=threshold_set, ts_routed_ns=now_ns(),
            text=message.text)
        out: dict[str, Any] = routed.model_dump()
        out["rationale"] = d["rationale"] if d else "fallback: no valid decision"
        out["evidence"] = d.get("evidence", []) if d else []
        return {"routed": out}

    graph = StateGraph(State)
    graph.add_node("analyze", analyze)
    graph.add_node("tools", run_tool)
    graph.add_node("decide", decide)
    graph.add_node("validate", validate)
    graph.add_node("emit", emit)
    graph.set_entry_point("analyze")
    graph.add_conditional_edges("analyze", lambda s: "tools" if s.get("next_tool") else "decide",
                                {"tools": "tools", "decide": "decide"})
    graph.add_edge("tools", "analyze")
    graph.add_edge("decide", "validate")
    graph.add_conditional_edges("validate", after_validate, {"emit": "emit", "decide": "decide"})
    graph.add_edge("emit", END)
    return graph.compile()

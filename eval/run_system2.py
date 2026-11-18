"""B5 = System 1 + System 2 (PLAN §4.3 / §3.6 gate): does the agent beat Laya on the escalated cases?

Usage: python eval/run_system2.py --model b1|laya --set golden|hard|test [--limit N] [--no-tools] [--tag NAME]
LLM: LLM_BASE_URL / LLM_MODEL / LLM_API_KEY (default local Ollama qwen3:1.7b; the gateway alias is
`smart_router`; against the gateway set LLM_EXTRA_BODY='{}' or '{"reasoning_effort": "low"}', and LLM_PACED=1
to wait out 429/503 quota responses instead of falling back).
Tools are mocks here; the real P1/P3 adapters are used in the demos (M8).
Writes docs/results/b5-<set>.json including the paired-bootstrap interval of (System 2 - Laya) accuracy.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harness import Prediction, evaluate
from run_eval import SETS, build_router, load_rows

from judgekit.stats import paired_bootstrap
from routercore.adapters.policy_config import load_policy, policy_path
from routercore.adapters.route_map_config import load_route_map
from routercore.application.contracts import InboundMessage
from routercore.domain.policy import decide

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def escalation_payload(row: dict, laya: dict, reason: str) -> dict:
    message = InboundMessage(message_id=row["id"], customer_id="eval", channel="chat", text=row["text"],
                             lang=row.get("lang"), ts_sent_ns=0).model_dump()
    return {"escalation": {"message": message, "reason": reason, "laya": laya, "context": []}}


URGENCY_OF_PRIORITY = {"low": 0, "normal": 1, "high": 2, "critical": 3}


def combine(laya: list[Prediction], system2: dict[int, tuple[str, int, str]]) -> list[Prediction]:
    """System 2's route and priority replace Laya's on escalated rows (one LLM call each).
    A `triage` fallback is a human queue: the row stays escalated."""
    out = []
    for i, p in enumerate(laya):
        if i not in system2:
            out.append(p)
            continue
        route, tokens, priority = system2[i]
        if route == "triage":
            out.append(Prediction(route=p.route, urgency=p.urgency, escalated=True, llm_calls=1,
                                  tokens=tokens, latency_ms=p.latency_ms))
        else:
            out.append(Prediction(route=route, urgency=URGENCY_OF_PRIORITY[priority], escalated=False,
                                  llm_calls=1, tokens=tokens, latency_ms=p.latency_ms))
    return out


def score_escalated(rows: list[dict], laya_routes: dict[int, str], s2_routes: dict[int, str]) -> dict:
    ids = sorted(laya_routes)
    laya_ok = [float(rows[i]["route"] == laya_routes[i]) for i in ids]
    s2_ok = [float(rows[i]["route"] == s2_routes[i]) for i in ids]
    boot = paired_bootstrap(s2_ok, laya_ok)
    return {"n": len(ids), "laya_accuracy": sum(laya_ok) / len(ids),
            "system2_accuracy": sum(s2_ok) / len(ids),
            "diff": boot.mean_diff, "ci95": [round(boot.low, 4), round(boot.high, 4)],
            "significant": boot.significant}


class MockTools:
    def get_customer_history(self, customer_id: str) -> list[str]:
        return []

    def lookup_payment_decision(self, text: str) -> str:
        return "REVIEW: mock rule"

    def search_kb(self, question: str) -> str:
        return "(mock) help-center article"


def main() -> None:
    from services.system2.adapters import NO_THINKING, GatewayLlm, QuotaPacedLlm, extra_body_from_env
    from services.system2.graph import build_agent

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="b1")
    parser.add_argument("--set", dest="dataset", choices=SETS, default="golden")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-tools", action="store_true", help="single-call baseline: no tool steps")
    parser.add_argument("--tag", default="", help="suffix of the result file, e.g. the LLM used")
    args = parser.parse_args()

    route_map = load_route_map(ROOT / "config" / "route_map.yaml")
    policy, threshold_set = load_policy(policy_path(ROOT / "config", args.model))
    rows = load_rows(args.dataset, args.limit)
    router = build_router(args.model, route_map, policy)
    laya = router.predict(rows)

    # recompute the distributions of the escalated rows (reason + scores travel to the agent)
    model = router._model
    messages = [InboundMessage(message_id=r["id"], customer_id="eval", channel="chat", text=r["text"],
                               lang=r.get("lang"), ts_sent_ns=0) for r in rows]
    assessments = model.assess(messages)
    escalated = {}
    for i, a in enumerate(assessments):
        target, reason = decide(a.route_p, a.urgency_p, a.lang_supported, policy)
        if target == "system2":
            escalated[i] = (a, reason)

    llm = GatewayLlm(os.environ.get("LLM_BASE_URL", "http://localhost:11434"),
                     model=os.environ.get("LLM_MODEL", "qwen3:1.7b"), api_key=os.environ.get("LLM_API_KEY"),
                     extra_body=extra_body_from_env(NO_THINKING))
    paced = QuotaPacedLlm(llm) if os.environ.get("LLM_PACED") else llm
    agent = build_agent(paced, MockTools(), route_map.routes, max_steps=0 if args.no_tools else 6)

    def resolve(i: int) -> tuple[int, str, str, float]:
        a, reason = escalated[i]
        began = time.perf_counter()
        try:
            routed = agent.invoke(escalation_payload(rows[i], a.distributions, reason))["routed"]
            route, priority = routed["route"], routed["priority"]
        except Exception:
            route, priority = "triage", "high"
        return i, route, priority, (time.perf_counter() - began) * 1000

    with ThreadPoolExecutor(max_workers=2) as pool:
        resolved = list(pool.map(resolve, sorted(escalated)))
    tokens_per_call = llm.tokens / max(len(resolved), 1)
    s2 = {i: (route, int(tokens_per_call), priority) for i, route, priority, _ in resolved}

    combined = combine(laya, s2)
    latency_p50 = sorted(ms for *_, ms in resolved)[len(resolved) // 2] if resolved else 0
    result = {"model": args.model, "set": args.dataset, "threshold_set": threshold_set, "rows": len(rows),
              "no_tools": args.no_tools, "escalated": len(escalated),
              "llm": os.environ.get("LLM_MODEL", "qwen3:1.7b"),
              "quota_wait_s": round(getattr(paced, "waited_s", 0.0), 1),
              "system2_latency_ms_p50": latency_p50,
              "fallback_rate": sum(r == "triage" for _, r, *_ in resolved) / max(len(resolved), 1),
              "escalated_only": score_escalated(rows, {i: laya[i].route for i in escalated},
                                                {i: r for i, (r, *_) in s2.items()}) if resolved else None,
              "end_to_end": {k: round(v, 4) for k, v in evaluate(rows, combined).items()},
              "system1_only": {k: round(v, 4) for k, v in evaluate(rows, laya).items()}}
    suffix = ("-notools" if args.no_tools else "") + (f"-{args.tag}" if args.tag else "")
    out = ROOT / "docs" / "results" / f"b5-{args.model}-{args.dataset}{suffix}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

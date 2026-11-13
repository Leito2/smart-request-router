"""Export decisions from the topics for eval/drift_monitor.py (stands in for the Postgres decision log export).

Usage: python scripts/export_decisions.py out.jsonl
Joins `routed` and `escalations` with `inbound-messages` (for the language) and writes one JSON line per
decision: {route, lang, confidence, kind: routed|escalation}.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke import read_topic  # noqa: E402


def top(distribution: dict[str, float]) -> tuple[str, float]:
    route = max(distribution, key=distribution.__getitem__)
    return route, distribution[route]


def build_decisions(inbound: list[dict], routed: list[dict], escalations: list[dict]) -> list[dict]:
    lang = {m["message_id"]: m.get("lang") or "unknown" for m in inbound}
    decisions = [{"route": r["route"], "lang": lang.get(r["message_id"], "unknown"),
                  "confidence": r["confidence"], "kind": "routed"} for r in routed]
    for e in escalations:
        route, confidence = top(e["laya"].get("route") or {"unknown": 0.0})
        decisions.append({"route": route, "lang": lang.get(e["message"]["message_id"], "unknown"),
                          "confidence": confidence, "kind": "escalation"})
    return decisions


def main() -> None:
    out = Path(sys.argv[1])
    decisions = build_decisions(read_topic("inbound-messages", 4), read_topic("routed", 4),
                                read_topic("escalations", 4))
    out.write_text("\n".join(json.dumps(d) for d in decisions) + "\n", encoding="utf-8")
    print(f"{len(decisions)} decisions -> {out}")


if __name__ == "__main__":
    main()

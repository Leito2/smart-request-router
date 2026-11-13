"""End-to-end smoke test of the System 1 stream (PLAN §10 M4): replayer → Redpanda → dataflow → dispatcher.

Needs `make up PROFILE=core`. Starts the mocks, dataflow and dispatcher as host processes, replays
messages, then reads the topics back and reports coverage and p50/p95 latency.
Writes docs/results/smoke.json.
Usage: python scripts/smoke.py [--messages 300] [--rate 20] [--model setfit|laya|rules]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BROKER = "localhost:9092"


def read_topic(topic: str, timeout_s: float = 8) -> list[dict]:
    from confluent_kafka import Consumer

    consumer = Consumer({"bootstrap.servers": BROKER, "group.id": f"smoke-{uuid.uuid4()}",
                         "auto.offset.reset": "earliest", "enable.auto.commit": False})
    consumer.subscribe([topic])
    items, idle_since = [], time.time()
    while time.time() - idle_since < timeout_s:
        message = consumer.poll(1.0)
        if message is None or message.error():
            continue
        items.append(json.loads(message.value()))
        idle_since = time.time()
    consumer.close()
    return items


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", type=int, default=300)
    parser.add_argument("--rate", type=float, default=20)
    parser.add_argument("--model", default="setfit")
    args = parser.parse_args()

    env = {**os.environ, "MODEL": args.model, "PYTHONPATH": str(ROOT), "USE_TF": "0"}
    procs = [subprocess.Popen([sys.executable, "-m", module], cwd=ROOT, env=env)
             for module in ("services.mocks.main", "services.dataflow.main", "services.dispatcher.main")]
    try:
        time.sleep(20 if args.model == "rules" else 40)           # model load + Quix partition assignment
        subprocess.run([sys.executable, "-m", "services.replayer.main", "--rate", str(args.rate),
                        "--duration", str(args.messages / args.rate), "--broker", BROKER],
                       cwd=ROOT, env=env, check=True)
        time.sleep(10)
        inbound = {m["message_id"]: m["ts_sent_ns"] for m in read_topic("inbound-messages")}
        routed, escalated = read_topic("routed"), read_topic("escalations")
        dlq = read_topic("dlq", 3)
    finally:
        for proc in procs:
            proc.terminate()

    latencies = [(r["ts_routed_ns"] - inbound[r["message_id"]]) / 1e6
                 for r in routed if r["message_id"] in inbound]
    reasons: dict[str, int] = {}
    for e in escalated:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    decided = len(routed) + len(escalated)
    result = {"model": args.model, "sent": len(inbound), "routed": len(routed), "escalated": len(escalated),
              "dlq": len(dlq), "coverage_s1": round(len(routed) / decided, 4) if decided else 0,
              "escalation_reasons": reasons,
              "latency_ms_p50": round(statistics.median(latencies), 1) if latencies else None,
              "latency_ms_p95": round(percentile(latencies, 0.95), 1) if latencies else None}
    out = ROOT / "docs" / "results" / "smoke.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Surge demo (PLAN §10 M6): steady traffic, then `payments` x5; prints how fast the detector alerts.

Needs `make up PROFILE=core`. Uses config/surge.demo.yaml (10 s windows). Writes docs/results/surge_demo.json.
Usage: python scripts/surge_demo.py [--model rules|laya]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SURGE_AT_S, SURGE_FOR_S, ROUTE = 90, 90, "payments"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rules")
    args = parser.parse_args()
    env = {**os.environ, "MODEL": args.model, "PYTHONPATH": str(ROOT), "USE_TF": "0",
           "SURGE_CONFIG": str(ROOT / "config" / "surge.demo.yaml")}
    procs = [subprocess.Popen([sys.executable, "-m", module], cwd=ROOT, env=env)
             for module in ("services.dataflow.main", "services.dataflow.surge_app")]
    alerts: list[dict] = []
    detected_after = 0.0
    try:
        time.sleep(12)
        from confluent_kafka import Consumer

        consumer = Consumer({"bootstrap.servers": "localhost:9092", "group.id": f"demo-{time.time_ns()}",
                             "auto.offset.reset": "latest"})
        consumer.subscribe(["alerts"])
        consumer.poll(2)
        started = time.time()
        command = [sys.executable, "-m", "services.replayer.main", "--rate", "10", "--duration", "200",
                   "--lang-mix", "en:1.0", "--surge", f"{ROUTE}:5:{SURGE_AT_S}:{SURGE_FOR_S}"]
        replay = subprocess.Popen(command, cwd=ROOT, env=env)
        while replay.poll() is None and not alerts:
            message = consumer.poll(1.0)
            if message is not None and not message.error():
                alerts.append(json.loads(message.value()))
                detected_after = time.time() - started - SURGE_AT_S
        replay.terminate()
        consumer.close()
    finally:
        for proc in procs:
            proc.terminate()

    result = {"surge_route": ROUTE, "surge_factor": 5, "detected": bool(alerts),
              "seconds_to_detect": round(detected_after, 1) if alerts else None, "alert": alerts[:1]}
    out = ROOT / "docs" / "results" / "surge_demo.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

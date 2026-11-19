"""Feeds the stack with traffic so the Grafana dashboards R1-R4 have something to show (PLAN §3.10).

Needs `docker compose --profile core --profile ops up -d` (Redpanda, Redis, Prometheus, Grafana). Starts the
dataflow and the surge detector with the chosen model, optionally the System 2 worker (needs the llm-gateway
reachable at GATEWAY_URL) and the P1/P3 mocks, replays messages, then holds so Prometheus keeps scraping.

    python scripts/dashboards_demo.py --rate 5 --duration 200 --surge payments:5:100:60 --hold 240     # R1-R3
    python scripts/dashboards_demo.py --system2 --rate 0.3 --duration 150 --hold 240                   # R4

Run it on fresh topics (`docker compose ... down -v` and `rm -rf .quix-state state`), see the README.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="setfit")
    parser.add_argument("--rate", type=float, default=5)
    parser.add_argument("--duration", type=float, default=200)
    parser.add_argument("--lang-mix", default="en:0.8,es:0.2")
    parser.add_argument("--surge", help="route:factor:at_s:duration_s, e.g. payments:5:100:60")
    parser.add_argument("--set", dest="dataset", default="test", choices=["test", "hard"])
    parser.add_argument("--system2", action="store_true", help="also run the System 2 worker (+ P1/P3 mocks)")
    parser.add_argument("--hold", type=float, default=240, help="seconds to stay up after the replay")
    args = parser.parse_args()

    env = {**os.environ, "MODEL": args.model, "PYTHONPATH": str(ROOT), "USE_TF": "0"}
    modules = ["services.dataflow.main", "services.dataflow.surge_app"]
    if args.system2:
        modules += ["services.mocks.main", "services.system2.main"]
    procs = [subprocess.Popen([sys.executable, "-m", module], cwd=ROOT, env=env) for module in modules]
    try:
        time.sleep(45)                                  # model load + Quix partition assignment
        command = [sys.executable, "-m", "services.replayer.main", "--rate", str(args.rate), "--duration",
                   str(args.duration), "--lang-mix", args.lang_mix, "--set", args.dataset]
        if args.surge:
            command += ["--surge", args.surge]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        print(f"replay done, holding {args.hold:.0f} s (Grafana: http://localhost:3000)", flush=True)
        time.sleep(args.hold)
    finally:
        for proc in procs:
            proc.terminate()


if __name__ == "__main__":
    main()

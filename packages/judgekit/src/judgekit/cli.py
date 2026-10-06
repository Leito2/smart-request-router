"""judgekit CLI. M0: `judgekit gate <rules.yaml> <metrics.json>`; run/report/calibrate arrive in M2/M7b."""
import argparse
import json
import sys
from pathlib import Path

from judgekit.gate import evaluate, load_rules


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="judgekit")
    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("gate", help="check metrics against thresholds")
    gate.add_argument("rules", type=Path)
    gate.add_argument("metrics", type=Path)
    args = parser.parse_args(argv)

    failures = evaluate(load_rules(args.rules), json.loads(args.metrics.read_text(encoding="utf-8")))
    for failure in failures:
        print(f"GATE FAIL {failure}")
    print("GATE PASS" if not failures else f"GATE FAILED ({len(failures)})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

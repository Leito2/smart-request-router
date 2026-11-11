"""Drift on prediction signals (PLAN §5.5): no labels needed. PSI of the route mix, language mix, System 1
confidence and routed/escalation split, current traffic vs the training distribution.

Usage: python eval/drift_monitor.py --current decisions.jsonl [--threshold 0.25] [--evidently]
`decisions.jsonl`: one JSON per decision {route, lang, confidence, kind: routed|escalation} (exported from the
Postgres decision log or the topics). Writes docs/results/drift.json; exit code 1 when something drifted.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from judgekit.drift import drift_report

ROOT = Path(__file__).resolve().parent.parent


def confidence_bucket(confidence: float) -> str:
    low = min(int(confidence * 10), 9)
    return "0.0-0.5" if low < 5 else f"0.{low}-{(low + 1) / 10:.1f}"


def feature_counts(decisions: list[dict]) -> dict[str, dict[str, int]]:
    return {
        "route": dict(Counter(d["route"] for d in decisions)),
        "lang": dict(Counter(d["lang"] for d in decisions)),
        "outcome": dict(Counter(d["kind"] for d in decisions)),
        "confidence": dict(Counter(confidence_bucket(d["confidence"]) for d in decisions
                                   if d.get("confidence") is not None)),
    }


def reference_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    train = [r for r in rows if r["split"] == "train"]
    return {"route": dict(Counter(r["route"] for r in train)),
            "lang": dict(Counter(r["lang"] for r in train))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--evidently", action="store_true", help="also write an Evidently HTML report")
    args = parser.parse_args()

    dataset = [json.loads(line) for line in
               (ROOT / "training/data/dataset.jsonl").read_text(encoding="utf-8").splitlines() if line]
    decisions = [json.loads(line) for line in args.current.read_text(encoding="utf-8").splitlines() if line]
    reference, current = reference_counts(dataset), feature_counts(decisions)
    report = drift_report(reference, current, threshold=args.threshold)
    out = ROOT / "docs" / "results" / "drift.json"
    out.write_text(json.dumps(report, indent=2) + "
", encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.evidently:
        import pandas as pd
        from evidently import Report
        from evidently.presets import DataDriftPreset

        ref_frame = pd.DataFrame([{"route": r["route"], "lang": r["lang"]} for r in dataset
                                  if r["split"] == "train"])
        cur_frame = pd.DataFrame([{"route": d["route"], "lang": d["lang"]} for d in decisions])
        snapshot = Report([DataDriftPreset()]).run(cur_frame, ref_frame)
        snapshot.save_html(str(ROOT / "docs" / "results" / "drift_evidently.html"))
    sys.exit(1 if any(v["drifted"] for v in report.values()) else 0)


if __name__ == "__main__":
    main()

"""Writes config/route_reference.json: the train split's route mix, the reference of the dashboards' PSI.

    uv run --no-sync python scripts/make_route_reference.py

Rerun it when the dataset changes. Only counts per route are stored, no text.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    from routercore.adapters.jsonl import read_jsonl

    rows = read_jsonl(ROOT / "training" / "data" / "dataset.jsonl")
    counts = Counter(r["route"] for r in rows if r["split"] == "train")
    out = ROOT / "config" / "route_reference.json"
    out.write_text(json.dumps(dict(sorted(counts.items())), indent=2) + "\n", encoding="utf-8")
    print(f"{sum(counts.values())} train rows, {len(counts)} routes -> {out}")


if __name__ == "__main__":
    main()

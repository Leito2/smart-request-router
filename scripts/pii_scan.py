"""PII scan of the datasets with Presidio, before any of them goes to Langfuse or the Hub (PLAN §12).

    uv run --no-sync python scripts/pii_scan.py              # scan, write docs/results/pii_scan.json
    uv run --no-sync python scripts/pii_scan.py --strict     # exit 1 if any strong finding (CI / pre-upload)

Prints and stores entity counts and row ids only, never the matched text. Strong = score >= 0.8 (card with
Luhn, IBAN, email, NIF all score 1.0); weak = a phone-like number, noisy by nature, for a human to look at.
Scope: identifiers by pattern. Names and addresses are not detected (no NER model), see
routercore.adapters.pii.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASETS = {"train+test": ROOT / "training/data/dataset.jsonl",
            "golden": ROOT / "training/golden_set/golden.jsonl",
            "hard": ROOT / "training/golden_set/hard_cases.jsonl"}


def scan(rows: list[dict], redactor) -> dict:
    from routercore.adapters.pii import STRONG

    strong, weak = Counter(), Counter()
    flagged: list[dict] = []
    for row in rows:
        found = redactor.findings(row["text"])
        for f in found:
            (strong if f.score >= STRONG else weak)[f.entity] += 1
        if found:
            flagged.append({"id": row["id"], "entities": sorted({f.entity for f in found}),
                            "strong": any(f.score >= STRONG for f in found)})
    return {"rows": len(rows), "rows_flagged": len(flagged), "strong": dict(strong), "weak": dict(weak),
            "flagged": flagged[:50]}


def main() -> None:
    from routercore.adapters.jsonl import read_jsonl
    from routercore.adapters.pii import PiiRedactor

    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="exit 1 on any strong finding")
    args = parser.parse_args()
    redactor = PiiRedactor()
    report = {name: scan(read_jsonl(path), redactor) for name, path in DATASETS.items()}
    out = ROOT / "docs" / "results" / "pii_scan.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, r in report.items():
        print(f"{name}: {r['rows']} rows, {r['rows_flagged']} flagged, "
              f"strong {r['strong'] or 0}, weak {r['weak'] or 0}")
    print(f"-> {out}")
    if args.strict and any(r["strong"] for r in report.values()):
        raise SystemExit("strong PII findings: do not upload these datasets")


if __name__ == "__main__":
    main()

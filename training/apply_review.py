"""Applies the frontier-model review (training/golden_set/reviews.jsonl) to golden.jsonl and prints a summary.

Rows judged `drop` are removed (the golden set can end slightly under 500; the summary says by how much).
Usage: python training/apply_review.py
"""
import json
import random
from collections import Counter
from pathlib import Path

GOLDEN = Path(__file__).parent / "golden_set"
FIXABLE = ("route", "urgency", "should_escalate")


def apply_reviews(golden: list[dict], reviews: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (updated golden without dropped rows, the review entries that changed or dropped something)."""
    by_id = {r["id"]: r for r in reviews}
    kept, changes = [], []
    for row in golden:
        review = by_id.get(row["id"])
        if review is None:
            kept.append(row)
            continue
        if review["verdict"] == "drop":
            changes.append({"id": row["id"], "text": row["text"], "verdict": "drop",
                            "note": review.get("note")})
            continue
        updated = {**row, "review": {"verdict": review["verdict"], "note": review.get("note")}}
        if review["verdict"] == "fix":
            before = {k: row.get(k) for k in FIXABLE}
            updated.update({k: review[k] for k in FIXABLE if review.get(k) is not None})
            changes.append({"id": row["id"], "text": row["text"], "verdict": "fix", "before": before,
                            "after": {k: updated.get(k) for k in FIXABLE}, "note": review.get("note")})
        kept.append(updated)
    return kept, changes


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    golden, reviews = load_jsonl(GOLDEN / "golden.jsonl"), load_jsonl(GOLDEN / "reviews.jsonl")
    kept, changes = apply_reviews(golden, reviews)
    (GOLDEN / "golden.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in kept) + "\n", encoding="utf-8")
    verdicts = Counter(r["verdict"] for r in reviews)
    print(f"reviewed {len(reviews)}/{len(golden)} rows: {dict(verdicts)}; golden now has {len(kept)} rows")
    print("5 examples of what the reviewer changed:")
    for change in random.Random(0).sample(changes, min(5, len(changes))):
        print(" -", json.dumps(change, ensure_ascii=False))


if __name__ == "__main__":
    main()

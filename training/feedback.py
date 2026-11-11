"""Improvement loop (PLAN §4.7): human corrections from the `feedback` table become training examples.

Usage: python training/feedback.py --feedback feedback.jsonl   (export of v_retrain_dataset + message text)
Writes training/data/dataset.jsonl with the corrections appended to the train split, then `make train` again.
Messages whose text is already in train are skipped (no duplicates).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _key(text: str) -> str:
    return " ".join(text.lower().split())


def merge_feedback(rows: list[dict], feedback: list[dict]) -> list[dict]:
    """Rows plus one train example per correction (urgency 0: it fixes the route, not the urgency)."""
    known = {_key(r["text"]) for r in rows if r["split"] == "train"}
    merged = list(rows)
    for item in feedback:
        if _key(item["text"]) in known:
            continue
        known.add(_key(item["text"]))
        merged.append({"id": f"fb-{item['message_id']}", "origin_id": f"fb-{item['message_id']}",
                       "text": item["text"], "lang": item.get("lang", "en"), "route": item["correct_route"],
                       "intent": "feedback", "urgency": 0, "needs_human": False, "needs_kb": False,
                       "split": "train"})
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", required=True, type=Path)
    args = parser.parse_args()
    path = ROOT / "training" / "data" / "dataset.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    feedback = [json.loads(line) for line in args.feedback.read_text(encoding="utf-8").splitlines() if line]
    merged = merge_feedback(rows, feedback)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in merged) + "\n",
                    encoding="utf-8")
    print(f"added {len(merged) - len(rows)} corrections ({len(feedback)} received)")


if __name__ == "__main__":
    main()

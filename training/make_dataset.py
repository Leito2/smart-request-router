"""Builds the router dataset, the hard cases and the golden set candidates (PLAN §4.2, M1).

Reads training/data/raw/*.jsonl (fetch_banking77.py) and training/data/translations.jsonl (translate_es.py).
Writes the full dataset to training/data/dataset.jsonl (git-ignored) and the small, reviewable artifacts to
training/golden_set/ (committed): golden.jsonl, hard_cases.jsonl and MANIFEST.json with content hashes.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from data_prep import build_dataset, general_records, select_golden
from general_train import general_examples
from hard_cases import hand_written, hard_records, multi_intent

from routercore.adapters.route_map_config import load_route_map

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "training" / "data"
GOLDEN = ROOT / "training" / "golden_set"
GOLDEN_SIZE, GOLDEN_HARD_SHARE, MULTI_INTENT_CASES = 500, 0.32, 250


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    text = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n"
    path.write_text(text, encoding="utf-8")
    return {"rows": len(rows), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def main() -> None:
    route_map = load_route_map(ROOT / "config" / "route_map.yaml")
    train, test = load_jsonl(DATA / "raw" / "train.jsonl"), load_jsonl(DATA / "raw" / "test.jsonl")
    translated = DATA / "translations.jsonl"
    translations = {r["origin_id"]: r["es"] for r in load_jsonl(translated)} if translated.exists() else {}

    records = build_dataset(train, test, translations, route_map, seed=0)
    records += general_records(general_examples(), seed=0)
    hard = hard_records(hand_written() + multi_intent(test, route_map, MULTI_INTENT_CASES, seed=1))
    regular_test = [r for r in records if r["split"] == "test"]
    picked = select_golden(regular_test, hard, size=GOLDEN_SIZE, hard_share=GOLDEN_HARD_SHARE, seed=2)
    golden = [{**g, "review": None} for g in picked]

    split_counts = Counter((r["split"], r["lang"]) for r in records)
    GOLDEN.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "PolyAI/banking77 (CC BY 4.0) + local Spanish translations + hand-written hard cases",
        "files": {
            "training/data/dataset.jsonl": write_jsonl(DATA / "dataset.jsonl", records),
            "training/golden_set/golden.jsonl": write_jsonl(GOLDEN / "golden.jsonl", golden),
            "training/golden_set/hard_cases.jsonl": write_jsonl(GOLDEN / "hard_cases.jsonl", hard),
        },
        "splits": {f"{s}/{lang}": n for (s, lang), n in sorted(split_counts.items())},
        "golden": {
            "by_route": dict(sorted(Counter(g["route"] for g in golden).items())),
            "by_lang": dict(sorted(Counter(g["lang"] for g in golden).items())),
            "urgent_rows": sum(g["urgency"] >= 2 for g in golden),
        },
    }
    (GOLDEN / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    print(json.dumps(manifest["splits"]), manifest["golden"], sep="\n")


if __name__ == "__main__":
    main()

"""Trains B1 (SetFit few-shot) on the train split and saves it to models/b1_setfit/ (git-ignored).

Usage: python training/baselines/train_b1.py [--per-class 16] [--iterations 10]   (needs `--group ml`)
       python training/baselines/train_b1.py --urgency-only     # keep the route head, refit the urgency head

The urgency head (`IntentUrgencyHead`) is also fitted out-of-fold: the P(urgent) of every urgent train row
goes to models/b1_setfit/urgency/oof_urgent_masses.json, which `choose_thresholds.py` uses for tau_urgent.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from routercore.adapters.jsonl import read_jsonl  # noqa: E402
from routercore.adapters.setfit_model import IntentUrgencyHead, SetFitRouter  # noqa: E402

MODEL_DIR = ROOT / "models" / "b1_setfit"
OOF_FILE = "oof_urgent_masses.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--urgency-only", action="store_true", help="refit only the urgency head")
    parser.add_argument("--folds", type=int, default=5, help="out-of-fold sample for tau_urgent (0 = skip)")
    args = parser.parse_args()

    records = read_jsonl(ROOT / "training/data/dataset.jsonl")
    started = time.time()
    if args.urgency_only:
        head, oof = IntentUrgencyHead.fit(records, folds=args.folds)
        router = SetFitRouter.load(MODEL_DIR).with_urgency_head(head, model_version="b1-setfit-u2")
    else:
        router = SetFitRouter.train(records, per_class=args.per_class, iterations=args.iterations)
        _, oof = IntentUrgencyHead.fit(records, folds=args.folds) if args.folds else (None, [])
    router.save(MODEL_DIR)
    if oof:
        oof_path = MODEL_DIR / "urgency" / OOF_FILE
        oof_path.write_text(json.dumps([round(x, 6) for x in oof]), encoding="utf-8")
    print(f"trained in {time.time() - started:.0f}s -> models/b1_setfit")


if __name__ == "__main__":
    main()

"""Trains B1 (SetFit few-shot) on the train split and saves it to models/b1_setfit/ (git-ignored).

Usage: python training/baselines/train_b1.py [--per-class 16] [--iterations 10]   (needs `--group ml`)
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from baselines.setfit_model import SetFitRouter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    lines = (ROOT / "training/data/dataset.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    started = time.time()
    router = SetFitRouter.train(records, per_class=args.per_class, iterations=args.iterations)
    router.save(ROOT / "models" / "b1_setfit")
    print(f"trained in {time.time() - started:.0f}s -> models/b1_setfit")


if __name__ == "__main__":
    main()

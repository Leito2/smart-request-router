"""Fine-tune Laya on our routes (PLAN §4.2). Local default: encoder frozen, 2-layer decision head trained.

Why frozen: the 322M encoder with AdamW states does not fit 4 GB of VRAM (laya.train has no LoRA). The full
fine-tune runs on Kaggle/Colab (training/notebooks/laya_full_ft.ipynb) and is imported with `--checkpoint`.

Usage: python training/laya_finetune.py [--epochs 3] [--full] [--device cuda|cpu]
Writes models/laya_ft (loads with laya.load) and docs/results/laya_finetune.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "routercore" / "src"))

from routercore.adapters.laya_model import CHECKPOINT, build_questions  # noqa: E402
from routercore.adapters.route_map_config import load_route_map  # noqa: E402


def to_laya_rows(records: list[dict], routes: dict[str, str], *, split: str) -> list[dict]:
    questions = build_questions(routes)
    return [{"state": r["text"], "questions": questions,
             "expected": {"route": r["route"], "urgency": str(r["urgency"])}}
            for r in records if r["split"] == split]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def main() -> None:
    from laya.train import TrainConfig, finetune

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--full", action="store_true", help="also train the encoder (needs > 4 GB VRAM)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    routes = load_route_map(ROOT / "config" / "route_map.yaml").routes
    records = [json.loads(line) for line in (ROOT / "training/data/dataset.jsonl").read_text(
        encoding="utf-8").splitlines() if line]
    data = ROOT / "training/data/laya"
    write_jsonl(to_laya_rows(records, routes, split="train"), data / "train.jsonl")
    write_jsonl(to_laya_rows(records, routes, split="val"), data / "val.jsonl")

    config = TrainConfig(epochs=args.epochs, micro_batch=args.micro_batch, grad_accum=args.grad_accum,
                         freeze_encoder=not args.full, shuffle_options=("choice",),
                         eval_data=str(data / "val.jsonl"))
    started = time.perf_counter()
    summary = finetune(str(data / "train.jsonl"), CHECKPOINT, str(ROOT / "models" / "laya_ft"), config,
                       device=args.device)
    summary["train_seconds"] = round(time.perf_counter() - started, 1)
    out = ROOT / "docs" / "results" / "laya_finetune.json"
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()

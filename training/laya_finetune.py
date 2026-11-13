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


def freeze_all_but_top(model, top: int) -> None:
    """Leave only the last `top` encoder layers (and the head) trainable: embeddings are ~60% of the 322M."""
    layers = len(model.encoder.layers)
    for name, param in model.encoder.named_parameters():
        keep = name.startswith("layers.") and int(name.split(".")[1]) >= layers - top
        param.requires_grad_(keep)


def main() -> None:
    import laya.train as laya_train
    from laya.train import TrainConfig, finetune

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--full", action="store_true", help="also train the encoder (needs > 4 GB VRAM)")
    parser.add_argument("--unfreeze-top", type=int, default=0,
                        help="train only the last N encoder layers + head (fits 4 GB with N~8)")
    parser.add_argument("--out", default="models/laya_ft", help="checkpoint dir, relative to the repo")
    parser.add_argument("--report", default="docs/results/laya_finetune.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=320, help="sequence budget: 12-option head + message")
    parser.add_argument("--limit", type=int, help="smoke test: train on the first N rows only")
    args = parser.parse_args()

    routes = load_route_map(ROOT / "config" / "route_map.yaml").routes
    records = [json.loads(line) for line in (ROOT / "training/data/dataset.jsonl").read_text(
        encoding="utf-8").splitlines() if line]
    data = ROOT / "training/data/laya"
    train_rows = to_laya_rows(records, routes, split="train")
    write_jsonl(train_rows[:args.limit] if args.limit else train_rows, data / "train.jsonl")
    write_jsonl(to_laya_rows(records, routes, split="val"), data / "val.jsonl")

    if args.unfreeze_top:
        # laya.train only filters the optimizer by requires_grad, so freezing after the load is enough
        load_checkpoint = laya_train.load_checkpoint

        def load_partially_frozen(model_dir):
            model, tok, cfg = load_checkpoint(model_dir)
            freeze_all_but_top(model, args.unfreeze_top)
            return model, tok, cfg

        laya_train.load_checkpoint = load_partially_frozen

    train_encoder = args.full or args.unfreeze_top > 0
    config = TrainConfig(epochs=args.epochs, micro_batch=args.micro_batch, grad_accum=args.grad_accum,
                         freeze_encoder=not train_encoder, shuffle_options=("choice",), max_len=args.max_len,
                         eval_data=str(data / "val.jsonl"))
    started = time.perf_counter()
    summary = finetune(str(data / "train.jsonl"), CHECKPOINT, str(ROOT / args.out), config,
                       device=args.device)
    summary["train_seconds"] = round(time.perf_counter() - started, 1)
    summary["unfreeze_top"] = args.unfreeze_top
    out = ROOT / args.report
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()

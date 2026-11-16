"""Temperature scaling of the decision model per question and per language (PLAN §4.4).

Usage: python training/calibrate.py [--checkpoint models/laya_ft] [--limit N]
Runs the model on the validation split (never on test), fits one temperature per question (and per language
with enough rows), reports ECE before and after, and writes models/calibration.json (read by LayaModel)
plus docs/results/calibration.json.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from judgekit.probabilistic import expected_calibration_error, fit_temperature  # noqa: E402
from routercore.adapters.jsonl import read_jsonl  # noqa: E402
from routercore.adapters.model_factory import build_model  # noqa: E402
from routercore.domain.calibration import apply_temperature  # noqa: E402

QUESTIONS = ("intent", "urgency")


def _argmax(distribution: dict[str, float]) -> str:
    return max(distribution, key=distribution.__getitem__)


def _ece(pairs: list[tuple[dict[str, float], str]]) -> float:
    return expected_calibration_error([max(p.values()) for p, _ in pairs],
                                      [_argmax(p) == truth for p, truth in pairs])


def fit_calibration(records: list[dict], *, min_per_language: int = 100) -> dict:
    """records: {"lang", "intent": (distribution, truth), "urgency": (...)} per validation row."""
    temperatures: dict[str, dict[str, float]] = {}
    before: dict[str, float] = {}
    after: dict[str, float] = {}
    for question in QUESTIONS:
        pairs = [r[question] for r in records]
        default = fit_temperature([p for p, _ in pairs], [t for _, t in pairs])
        temperatures[question] = {"default": round(default, 4)}
        for lang in sorted({r["lang"] for r in records}):
            subset = [r[question] for r in records if r["lang"] == lang]
            if len(subset) >= min_per_language:
                temperatures[question][lang] = round(fit_temperature([p for p, _ in subset],
                                                                     [t for _, t in subset]), 4)
        before[question] = round(_ece(pairs), 4)
        by_lang = temperatures[question]
        scaled = [(apply_temperature(p, by_lang.get(r["lang"], by_lang["default"])), t)
                  for r, (p, t) in zip(records, pairs, strict=True)]
        after[question] = round(_ece(scaled), 4)
    return {"temperatures": temperatures, "ece_before": before, "ece_after": after, "rows": len(records)}


def collect(model, rows: list[dict]) -> list[dict]:
    from routercore.application.contracts import InboundMessage

    records = []
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        # no lang: the model returns raw probabilities, without temperature
        messages = [InboundMessage(message_id=r["id"], customer_id="cal", channel="chat", text=r["text"],
                                   ts_sent_ns=0) for r in batch]
        for row, assessment in zip(batch, model.assess(messages), strict=True):
            records.append({"lang": row["lang"], "intent": (assessment.route_p, row["route"]),
                            "urgency": (assessment.urgency_p, str(row["urgency"]))})
    return records


def main() -> None:
    from routercore.adapters.laya_model import CHECKPOINT
    from routercore.adapters.route_map_config import load_route_map

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    route_map = load_route_map(ROOT / "config" / "route_map.yaml")
    rows = read_jsonl(ROOT / "training/data/dataset.jsonl")
    rows = [r for r in rows if r["split"] == "val"]
    if args.limit:
        rows = random.Random(0).sample(rows, min(args.limit, len(rows)))
    model = build_model("laya", route_map, ROOT / "models", device=args.device, checkpoint=args.checkpoint,
                        calibrated=False, model_version="laya-calibrating")
    report = fit_calibration(collect(model, rows))
    report["checkpoint"] = args.checkpoint
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    out = ROOT / "docs" / "results" / "calibration.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

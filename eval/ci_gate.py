"""Regression gate for PRs (PLAN §5.4): fixed 100-case golden subset vs the `production` metrics.

Usage: python eval/ci_gate.py --model rules|laya [--update-baseline]
Exit code 1 when a gate fails. The subset keeps every urgent row (there are few) and fills with an even
stride. CI uses `rules` until the Laya checkpoint is published (M9); `--model laya` is the real gate locally.
"""
import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

from harness import evaluate
from routers import ModelRouter

from judgekit.gate import Rule, load_rules
from judgekit.gate import evaluate as check_rules
from routercore.adapters.policy_config import load_policy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
BASELINE = ROOT / "eval" / "production_metrics.json"


def select_ci_subset(rows: list[dict], size: int) -> list[dict]:
    urgent = [r for r in rows if r["urgency"] >= 2]
    rest = [r for r in rows if r["urgency"] < 2]
    room = max(size - len(urgent), 0)
    stride = max(len(rest) // room, 1) if room else 1
    return (urgent + rest[::stride][:room])[:size]


def degrade_routes(routes: list[str], labels: list[str], *, share: float, seed: int) -> list[str]:
    """Inject a regression for demos: flip `share` of the predictions to a different route."""
    rng = random.Random(seed)
    out = list(routes)
    for i in rng.sample(range(len(out)), round(share * len(out))):
        out[i] = rng.choice([label for label in labels if label != out[i]])
    return out


def gate_metrics(current: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {"macro_f1_drop": baseline["macro_f1"] - current["macro_f1"],
            "urgent_recall": current["urgent_recall"],
            "ece_rise": current.get("ece", 0.0) - baseline.get("ece", 0.0)}


def gate_failures(rules: list[Rule], metrics: dict[str, float]) -> list[str]:
    return check_rules(rules, metrics)


def build_model(name: str):
    from routercore.adapters.route_map_config import load_route_map

    route_map = load_route_map(ROOT / "config" / "route_map.yaml")
    if name == "rules":
        from baselines.rules import RulesModel
        return RulesModel(route_map)
    from routercore.adapters.laya_model import CHECKPOINT, LayaModel, load_calibration
    fine_tuned, calibration = ROOT / "models" / "laya_ft", ROOT / "models" / "calibration.json"
    return LayaModel.load(route_map.routes, checkpoint=str(fine_tuned) if fine_tuned.exists() else CHECKPOINT,
                          model_version="laya-gate", supported_langs={"en", "es"},
                          calibration=load_calibration(calibration) if calibration.exists() else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rules")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--degrade", type=float, default=0.0, help="demo: flip this share of predictions")
    args = parser.parse_args()

    golden = [json.loads(line) for line in
              (ROOT / "training/golden_set/golden.jsonl").read_text(encoding="utf-8").splitlines() if line]
    subset = select_ci_subset(golden, args.size)
    policy, _ = load_policy(ROOT / "config" / "policy.yaml")   # judge the whole system, safety net included
    predictions = ModelRouter(build_model(args.model), policy).predict(subset)
    if args.degrade:
        labels = sorted({r["route"] for r in golden})
        flipped = degrade_routes([p.route for p in predictions], labels, share=args.degrade, seed=0)
        predictions = [replace(p, route=route) for p, route in zip(predictions, flipped, strict=True)]
    metrics = evaluate(subset, predictions)
    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))

    if args.update_baseline:
        BASELINE.write_text(json.dumps({"model": args.model, **metrics}, indent=2) + "\n", encoding="utf-8")
        print(f"baseline written to {BASELINE}")
        return
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    failures = gate_failures(load_rules(ROOT / "config" / "gates.yaml"), gate_metrics(metrics, baseline))
    for failure in failures:
        print(f"GATE FAILED: {failure}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

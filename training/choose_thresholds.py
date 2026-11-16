"""Pick the escalation thresholds on the risk-coverage curve (PLAN §4.5).

Usage: python training/choose_thresholds.py [--checkpoint models/laya_ft] [--device cuda]
       [--target-accuracy 0.9] [--min-urgent-recall 0.98] [--cache]
Runs the calibrated model on the validation split (never test), then searches tau / margin /
max_entropy for the highest System 1 coverage whose accepted messages reach the target accuracy.
tau_urgent is the largest value that still catches `--min-urgent-recall` of the truly urgent
messages (asymmetric safety net).
Writes config/policy.yaml (with a threshold_set id) and docs/results/thresholds.json.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from routercore.domain.calibration import apply_temperature  # noqa: E402
from routercore.domain.policy import Policy, decide  # noqa: E402

TAUS = [round(0.5 + 0.025 * i, 3) for i in range(0, 20)]
MARGINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
ENTROPIES = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]


def urgent_mass(urgency_p: dict[str, float]) -> float:
    return urgency_p.get("2", 0.0) + urgency_p.get("3", 0.0)


def pick_tau_urgent(records: list[dict], min_recall: float) -> float:
    """Largest tau_urgent (fewest escalations) that still flags `min_recall` of the urgent rows."""
    urgent = sorted(urgent_mass(r["urgency_p"]) for r in records if int(r["urgency"]) >= 2)
    if not urgent:
        return Policy().tau_urgent
    allowed_misses = int(len(urgent) * (1 - min_recall))
    # flagged = mass strictly above tau: sit just below the (allowed_misses+1)-th lowest mass
    cut = urgent[allowed_misses]
    return round(max(0.0, cut - 1e-4), 4)


def evaluate(records: list[dict], pol: Policy) -> tuple[float, float]:
    """(coverage, accuracy of the messages System 1 accepted)."""
    accepted = correct = 0
    for r in records:
        target, _ = decide(r["route_p"], r["urgency_p"], True, pol)
        if target != "system2":
            accepted += 1
            correct += target == r["route"]
    return accepted / len(records), (correct / accepted if accepted else 0.0)


def choose(records: list[dict], *, target_accuracy: float = 0.90, min_urgent_recall: float = 0.98) -> dict:
    tau_urgent = pick_tau_urgent(records, min_urgent_recall)
    best: dict | None = None
    curve = []
    for tau, margin, entropy in itertools.product(TAUS, MARGINS, ENTROPIES):
        pol = Policy(tau=tau, margin=margin, tau_urgent=tau_urgent, max_entropy=entropy)
        coverage, accuracy = evaluate(records, pol)
        curve.append({"tau": tau, "margin": margin, "max_entropy": entropy,
                      "coverage": round(coverage, 4), "accuracy": round(accuracy, 4)})
        if accuracy >= target_accuracy and (best is None or coverage > best["coverage"]):
            best = curve[-1]
    if best is None:  # target unreachable: report the most accurate point instead of pretending
        best = max(curve, key=lambda c: (c["accuracy"], c["coverage"]))
        best = {**best, "target_met": False}
    else:
        best = {**best, "target_met": True}
    return {"policy": {"tau": best["tau"], "margin": best["margin"], "tau_urgent": tau_urgent,
                       "max_entropy": best["max_entropy"]},
            "coverage": best["coverage"], "accuracy": best["accuracy"], "target_met": best["target_met"],
            "target_accuracy": target_accuracy, "min_urgent_recall": min_urgent_recall,
            "rows": len(records)}


def calibrated(raw: list[dict], calibration: dict) -> list[dict]:
    temps = calibration["temperatures"]
    out = []
    for r in raw:
        t_route = temps["intent"].get(r["lang"], temps["intent"]["default"])
        t_urg = temps["urgency"].get(r["lang"], temps["urgency"]["default"])
        out.append({**r, "route_p": apply_temperature(r["route_p"], t_route),
                    "urgency_p": apply_temperature(r["urgency_p"], t_urg)})
    return out


def write_policy(result: dict, path: Path, model: str = "laya") -> None:
    p = result["policy"]
    set_id = f"v1-t{p['tau']}-m{p['margin']}-u{p['tau_urgent']}-e{p['max_entropy']}"
    path.write_text(
        f"# Chosen by training/choose_thresholds.py ({model}) on the validation risk-coverage curve.\n"
        f"threshold_set: {set_id}\ntau: {p['tau']}\nmargin: {p['margin']}\ntau_urgent: {p['tau_urgent']}\n"
        f"max_entropy: {p['max_entropy']}\ntemperatures: {{}}        # per language and question, "
        "fitted on validation data (models/calibration.json)\n", encoding="utf-8")


def collect_raw(checkpoint: str, device: str, model_name: str = "laya") -> list[dict]:
    from calibrate import collect

    rows = [json.loads(line) for line in (ROOT / "training/data/dataset.jsonl").read_text(
        encoding="utf-8").splitlines() if line]
    rows = [r for r in rows if r["split"] == "val"]
    if model_name == "setfit":
        from routercore.adapters.setfit_model import SetFitRouter
        model = SetFitRouter.load(ROOT / "models" / "b1_setfit")
    else:
        from routercore.adapters.laya_model import LayaModel
        from routercore.adapters.route_map_config import load_route_map
        model = LayaModel.load(load_route_map(ROOT / "config" / "route_map.yaml").routes,
                               checkpoint=checkpoint, device=device, model_version="laya-thresholds")
    return [{"lang": rec["lang"], "route_p": rec["intent"][0], "route": rec["intent"][1],
             "urgency_p": rec["urgency"][0], "urgency": rec["urgency"][1]} for rec in collect(model, rows)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["laya", "setfit"], default="laya")
    parser.add_argument("--checkpoint", default=str(ROOT / "models" / "laya_ft"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    parser.add_argument("--min-urgent-recall", type=float, default=0.98)
    parser.add_argument("--cache", action="store_true", help="reuse the cached validation assessments")
    args = parser.parse_args()

    cache = ROOT / "models" / f"val_assessments.{args.model}.json"
    if args.cache and cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
    else:
        raw = collect_raw(args.checkpoint, args.device, args.model)
        cache.write_text(json.dumps(raw), encoding="utf-8")
    if args.model == "laya":
        calibration = json.loads((ROOT / "models" / "calibration.json").read_text(encoding="utf-8"))
        records = calibrated(raw, calibration)
    else:
        records = raw                       # SetFit's probabilities are used as they come (ECE ~0.03 on val)
    result = choose(records, target_accuracy=args.target_accuracy, min_urgent_recall=args.min_urgent_recall)
    result["model"] = args.model
    result["checkpoint"] = args.checkpoint if args.model == "laya" else "models/b1_setfit"
    suffix = "" if args.model == "laya" else ".setfit"
    report = ROOT / "docs" / "results" / f"thresholds{suffix}.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_policy(result, ROOT / "config" / f"policy{suffix}.yaml", args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

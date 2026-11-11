"""Evaluate one router on one dataset with the shared harness (PLAN §5).

Usage: python eval/run_eval.py --router b0|b1|b2|b3|b4 --set test|golden|hard [--policy] [--limit N]
[--mlflow]
Writes docs/results/<router>-<set>.json. B4 reads LLM_BASE_URL / LLM_MODEL / LLM_API_KEY from the environment
(default: local Ollama + qwen3:1.7b); LLM_EXTRA_BODY adds request fields (JSON).
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

from harness import evaluate
from routers import LlmOnlyRouter, ModelRouter

from routercore.adapters.policy_config import load_policy
from routercore.adapters.route_map_config import load_route_map

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))     # baselines/ lives next to the training scripts
NO_THINKING = '{"reasoning_effort": "none"}'      # qwen3 on Ollama otherwise spends the budget thinking
SETS = {"test": ROOT / "training/data/dataset.jsonl", "golden": ROOT / "training/golden_set/golden.jsonl",
        "hard": ROOT / "training/golden_set/hard_cases.jsonl"}


def load_rows(name: str, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in SETS[name].read_text(encoding="utf-8").splitlines() if line]
    if name == "test":
        rows = [r for r in rows if r["split"] == "test"]
    if not limit:
        return rows
    return random.Random(0).sample(rows, min(limit, len(rows)))    # fixed seed: same subset


def build_router(name: str, route_map, policy):
    if name == "b0":
        from baselines.rules import RulesModel
        return ModelRouter(RulesModel(route_map), policy)
    if name == "b1":
        from baselines.setfit_model import SetFitRouter
        return ModelRouter(SetFitRouter.load(ROOT / "models" / "b1_setfit"), policy)
    if name in ("b2", "b3"):
        from routercore.adapters.laya_model import CHECKPOINT, LayaModel, load_calibration
        calibration_path = ROOT / "models" / "calibration.json"
        checkpoint = str(ROOT / "models" / "laya_ft") if name == "b3" else CHECKPOINT
        use_calibration = name == "b3" and calibration_path.exists()
        model = LayaModel.load(
            route_map.routes, checkpoint=checkpoint, model_version=f"laya-{name}",
            calibration=load_calibration(calibration_path) if use_calibration else None,
            supported_langs={"en", "es"})
        return ModelRouter(model, policy)
    if name == "b4":
        base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434")
        model = os.environ.get("LLM_MODEL", "qwen3:1.7b")
        return LlmOnlyRouter(route_map.routes, base_url=base_url, model=model,
                             api_key=os.environ.get("LLM_API_KEY"),
                             extra_body=json.loads(os.environ.get("LLM_EXTRA_BODY", NO_THINKING)))
    raise SystemExit(f"router {name!r} not available yet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", required=True)
    parser.add_argument("--set", dest="dataset", choices=SETS, default="golden")
    parser.add_argument("--policy", action="store_true", help="apply the escalation policy (S1 + escalate)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    route_map = load_route_map(ROOT / "config" / "route_map.yaml")
    policy, threshold_set = load_policy(ROOT / "config" / "policy.yaml")
    rows = load_rows(args.dataset, args.limit)
    router = build_router(args.router, route_map, policy if args.policy else None)

    metrics = evaluate(rows, router.predict(rows))
    result = {"router": args.router, "set": args.dataset, "policy": args.policy,
              "threshold_set": threshold_set, "rows": len(rows),
              "metrics": {k: round(v, 4) for k, v in metrics.items()}}
    out = ROOT / "docs" / "results" / f"{args.router}-{args.dataset}{'-policy' if args.policy else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

    if args.mlflow:
        import mlflow
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
        mlflow.set_experiment("router-eval")
        with mlflow.start_run(run_name=f"{args.router}-{args.dataset}"):
            mlflow.log_params({"router": args.router, "set": args.dataset, "policy": args.policy,
                               "rows": len(rows), "threshold_set": threshold_set})
            mlflow.log_metrics(result["metrics"])


if __name__ == "__main__":
    main()

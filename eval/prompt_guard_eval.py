# ruff: noqa: E501
"""Llama Prompt Guard 2 (22M and 86M) on the router's traffic: does it catch attacks without blocking customers?

A prompt-attack check would sit in front of System 2 (the agent reads customer text and calls tools). It is only
worth adding if it almost never flags a real customer, so the benign side is the router's own data (test, golden,
hard; English and Spanish). The attack side is a hand-written banking set (`eval/prompt_guard/attacks.jsonl`, 80
attacks in English and Spanish plus 40 hard negatives that sound like attacks but are legitimate requests) and the
public `deepset/prompt-injections` set (English and German; its "injection" label is broader than what Prompt Guard
2 is trained for, so its recall is a lower bound).

    uv run --no-sync python eval/prompt_guard_eval.py            # both models -> docs/results/prompt_guard.json

Needs the gated weights in models/prompt_guard_22m and models/prompt_guard_86m (meta-llama/Llama-Prompt-Guard-2-*)
and the dataset in data/external/deepset_prompt_injections (both downloaded with huggingface_hub).
"""
import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "training")]

MODELS = {"22m": ROOT / "models" / "prompt_guard_22m", "86m": ROOT / "models" / "prompt_guard_86m"}
ATTACKS = ROOT / "eval" / "prompt_guard" / "attacks.jsonl"
DEEPSET = ROOT / "data" / "external" / "deepset_prompt_injections" / "data"
THRESHOLD = 0.5
FPR_BUDGET = 0.01          # at most 1 customer in 100 sent to triage by mistake


class Guard:
    def __init__(self, directory: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.positive = labels.get("malicious", labels.get("label_1", 1))

    def scores(self, texts: list[str], batch: int = 32) -> list[float]:
        out = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch):
                enc = self.tokenizer(texts[start:start + batch], return_tensors="pt", padding=True, truncation=True,
                                     max_length=512)
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1)[:, self.positive]
                out += probs.tolist()
        return out

    def latency_ms(self, texts: list[str]) -> dict:
        self.scores(texts[:5], batch=1)                                     # warm-up
        times = []
        for text in texts:
            start = time.perf_counter()
            self.scores([text], batch=1)
            times.append(1000 * (time.perf_counter() - start))
        times.sort()
        return {"n": len(times), "p50": round(statistics.median(times), 1),
                "p95": round(times[int(0.95 * (len(times) - 1))], 1)}


def load_sets() -> dict[str, list[dict]]:
    import pandas as pd
    from run_eval import load_rows

    from routercore.adapters.jsonl import read_jsonl
    sets = {f"router_{name}": [{"text": r["text"], "lang": r["lang"], "label": 0} for r in load_rows(name, None)]
            for name in ("test", "golden", "hard")}
    own = read_jsonl(ATTACKS)
    sets["own_attacks"] = [r for r in own if r["label"] == 1]
    sets["own_hard_negatives"] = [r for r in own if r["label"] == 0]
    frames = [pd.read_parquet(f) for f in sorted(DEEPSET.glob("*.parquet"))]
    sets["deepset"] = [{"text": t, "label": int(y), "lang": "?"} for f in frames for t, y in zip(f.text, f.label,
                                                                                                    strict=True)]
    return sets


def rate(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def evaluate(scores: dict[str, list[float]], sets: dict[str, list[dict]], threshold: float) -> dict:
    report = {}
    for name, rows in sets.items():
        flagged = [s >= threshold for s in scores[name]]
        by_lang, by_category = defaultdict(list), defaultdict(list)
        for row, flag in zip(rows, flagged, strict=True):
            by_lang[row.get("lang", "?")].append(flag)
            by_category[row.get("category", "-")].append(flag)
        positives = [f for r, f in zip(rows, flagged, strict=True) if r["label"] == 1]
        negatives = [f for r, f in zip(rows, flagged, strict=True) if r["label"] == 0]
        entry = {"n": len(rows), "recall": rate(positives), "false_positive_rate": rate(negatives),
                 "flagged_by_lang": {k: rate(v) for k, v in sorted(by_lang.items())}}
        if len(by_category) > 1:
            entry["flagged_by_category"] = {k: rate(v) for k, v in sorted(by_category.items())}
        report[name] = {k: v for k, v in entry.items() if v is not None}
    return report


def auc(scores: list[float], labels: list[int]) -> float:
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(labels, scores)), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="22m,86m")
    parser.add_argument("--latency-n", type=int, default=200)
    args = parser.parse_args()
    sets = load_sets()
    benign = ("router_test", "router_golden", "router_hard")
    results = {"threshold": THRESHOLD, "fpr_budget": FPR_BUDGET, "sizes": {k: len(v) for k, v in sets.items()}}
    for key in args.models.split(","):
        guard = Guard(MODELS[key])
        scores = {name: guard.scores([r["text"] for r in rows]) for name, rows in sets.items()}
        router_scores = sorted(s for name in benign for s in scores[name])
        tuned = router_scores[int((1 - FPR_BUDGET) * (len(router_scores) - 1))]   # 99th pct of customer scores
        own_scores = scores["own_attacks"] + scores["own_hard_negatives"]
        own_labels = [1] * len(scores["own_attacks"]) + [0] * len(scores["own_hard_negatives"])
        results[key] = {
            "at_0.5": evaluate(scores, sets, THRESHOLD),
            "tuned_threshold": round(tuned, 4),
            "at_tuned": evaluate(scores, sets, max(tuned, 1e-6) + 1e-9),
            "auc": {"own_attacks_vs_hard_negatives": auc(own_scores, own_labels),
                    "own_attacks_vs_router_customers": auc(scores["own_attacks"] + router_scores,
                                                           [1] * len(scores["own_attacks"]) + [0] * len(router_scores)),
                    "deepset": auc(scores["deepset"], [r["label"] for r in sets["deepset"]])},
            "latency_ms_cpu_batch1": guard.latency_ms([r["text"] for r in sets["router_hard"][:args.latency_n]]),
            "top_customer_false_positives": sorted(
                ({"set": name, "score": round(s, 3), "text": r["text"][:120]}
                 for name in benign for r, s in zip(sets[name], scores[name], strict=True) if s >= THRESHOLD),
                key=lambda x: -x["score"])[:10],
            "missed_own_attacks": [r["id"] for r, s in zip(sets["own_attacks"], scores["own_attacks"], strict=True)
                                   if s < THRESHOLD],
        }
        print(key, json.dumps({k: results[key][k] for k in ("tuned_threshold", "auc", "latency_ms_cpu_batch1")}),
              flush=True)
    out = ROOT / "docs" / "results" / "prompt_guard.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

"""Validate the local judge against a frontier-model reference (PLAN §5.3).

Cohen's kappa >= 0.6, or the judge is only a signal and never evidence.

  python eval/judge_validation.py build   100 cases (message + proposed route): B1 on the golden set
  python eval/judge_validation.py judge   the local judge (Gemma via Ollama/gateway) labels them
  python eval/judge_validation.py kappa   compares with docs/judge_validation/reference.jsonl

The reference is written by a frontier model following docs/judge_validation/REVIEW.md (a pause point: the
author switches model and gets a 5-example summary). Judge env: JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY.
"""
import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

from judgekit.calibration import cohen_kappa
from judgekit.judges import parse_json

ROOT = Path(__file__).resolve().parent.parent
DIRECTORY = ROOT / "docs" / "judge_validation"
KAPPA_MIN = 0.6
SYSTEM = ("You audit a customer-support router for a fintech. Given a customer message and the team it was "
          "routed to, answer whether that routing is acceptable (the right team, or one that would clearly "
          'handle it). Reply JSON only: {"acceptable": 1 or 0, "reason": "<one sentence>"}.')


def build_cases(rows: list[dict], predictions: list[str], *, size: int, seed: int) -> list[dict]:
    """Half the cases carry a correct proposal and half a wrong one, so kappa has both classes to agree on."""
    rng = random.Random(seed)
    right = [i for i, (r, p) in enumerate(zip(rows, predictions, strict=True)) if r["route"] == p]
    wrong = [i for i in range(len(rows)) if i not in set(right)]
    half = size // 2
    chosen = rng.sample(right, min(half, len(right))) + rng.sample(wrong, min(size - half, len(wrong)))
    rng.shuffle(chosen)
    return [{"id": rows[i]["id"], "text": rows[i]["text"], "proposed_route": predictions[i]} for i in chosen]


def parse_verdict(text: str) -> int | None:
    answer = parse_json(text).get("acceptable")
    if answer in (0, 1):
        return int(answer)
    stripped = text.strip()
    return int(stripped) if stripped in ("0", "1") else None


def agreement(judge: dict[str, int | None], reference: dict[str, int]) -> dict:
    ids = sorted(reference)
    unparsed = sum(judge.get(i) is None for i in ids)
    judged = [judge[i] if judge.get(i) is not None else 1 - reference[i] for i in ids]   # no answer = wrong
    kappa = cohen_kappa(judged, [reference[i] for i in ids])
    return {"n": len(ids), "kappa": round(kappa, 4), "unparsed": unparsed,
            "accuracy": round(sum(j == reference[i] for j, i in zip(judged, ids, strict=True)) / len(ids), 4),
            "usable": kappa >= KAPPA_MIN}


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def run_judge(cases: list[dict]) -> dict[str, int | None]:
    from judgekit.clients import ChatClient

    async with ChatClient(os.environ.get("JUDGE_BASE_URL", "http://localhost:11434"),
                          model=os.environ.get("JUDGE_MODEL", "gemma3:4b"),
                          api_key=os.environ.get("JUDGE_API_KEY")) as client:
        async def one(case: dict) -> tuple[str, int | None]:
            user = json.dumps({"message": case["text"], "routed_to": case["proposed_route"]},
                              ensure_ascii=False)
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            done = await client.complete(messages, max_tokens=80)
            return case["id"], parse_verdict(done.text)
        return dict(await asyncio.gather(*(one(c) for c in cases)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["build", "judge", "kappa"])
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    cases_path, judge_path = DIRECTORY / "cases.jsonl", DIRECTORY / "judge.json"

    if args.step == "build":
        sys.path.insert(0, str(ROOT / "eval"))
        sys.path.insert(0, str(ROOT / "training"))
        from routers import ModelRouter
        from run_eval import build_router, load_rows

        from routercore.adapters.route_map_config import load_route_map
        rows = load_rows("golden", None)
        router = build_router("b1", load_route_map(ROOT / "config" / "route_map.yaml"), None)
        predictions = [p.route for p in ModelRouter.predict(router, rows)]
        cases = build_cases(rows, predictions, size=100, seed=0)
        lines = "\n".join(json.dumps(c, ensure_ascii=False) for c in cases)
        cases_path.write_text(lines + "\n", encoding="utf-8")
        print(f"{len(cases)} cases -> {cases_path}")
    elif args.step == "judge":
        labels = asyncio.run(run_judge(_read(cases_path)))
        judge_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
        print(f"{sum(v is not None for v in labels.values())}/{len(labels)} parsed")
    else:
        reference = {r["id"]: int(r["acceptable"]) for r in _read(DIRECTORY / "reference.jsonl")}
        report = agreement(json.loads(judge_path.read_text(encoding="utf-8")), reference)
        (ROOT / "docs" / "results" / "judge_kappa.json").write_text(json.dumps(report, indent=2) + "\n",
                                                                    encoding="utf-8")
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["usable"] else 1)


if __name__ == "__main__":
    main()

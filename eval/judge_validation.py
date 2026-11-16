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
from routercore.adapters.jsonl import read_jsonl

ROOT = Path(__file__).resolve().parent.parent
DIRECTORY = ROOT / "docs" / "judge_validation"
KAPPA_MIN = 0.6
SYSTEM = ("You audit a customer-support router for a fintech. Given a customer message and the team it was "
          "routed to, answer whether that routing is acceptable (the right team, or one that would clearly "
          'handle it). Reply JSON only: {"acceptable": 1 or 0, "reason": "<one sentence>"}.')


STRICT = ("Be strict: accept only when the named team is the one primarily responsible for the customer's "
          "main request. A team that is merely related, or could forward the case, is not acceptable. "
          "If the message mixes topics, judge the topic the customer most needs solved. "
          "When in doubt, answer 0. ")


def system_prompt(rubric: int, routes: dict[str, str]) -> str:
    """Rubric 1 names the team only; 2 also lists what each team covers (what the reviewer had);
    3 adds a strictness instruction, because the judge at rubric 2 accepts too many wrong routings."""
    if rubric == 1:
        return SYSTEM
    listing = "\n".join(f"- {name}: {description}" for name, description in routes.items())
    covers = f"The teams and what each one covers:\n{listing}\nReply JSON only"
    if rubric == 3:
        covers = STRICT + covers
    return SYSTEM.replace("Reply JSON only", covers)


def seeded(name: str, seed: int) -> str:
    """Seed 0 keeps the original file names; other seeds are held-out sets with their own files."""
    stem, dot, ext = name.partition(".")
    return name if seed == 0 else f"{stem}_seed{seed}{dot}{ext}"


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
    return read_jsonl(path)


async def run_judge(cases: list[dict], system: str) -> dict[str, int | None]:
    from judgekit.clients import ChatClient

    async with ChatClient(os.environ.get("JUDGE_BASE_URL", "http://localhost:11434"),
                          model=os.environ.get("JUDGE_MODEL", "gemma3:4b"),
                          api_key=os.environ.get("JUDGE_API_KEY")) as client:
        # one local GPU: letting all 100 requests queue at once makes the slowest exceed the client timeout
        slots = asyncio.Semaphore(int(os.environ.get("JUDGE_CONCURRENCY", "4")))

        async def one(case: dict) -> tuple[str, int | None]:
            user = json.dumps({"message": case["text"], "routed_to": case["proposed_route"]},
                              ensure_ascii=False)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            async with slots:
                done = await client.complete(messages, max_tokens=80)
            return case["id"], parse_verdict(done.text)
        return dict(await asyncio.gather(*(one(c) for c in cases)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["build", "judge", "kappa"])
    parser.add_argument("--rubric", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--seed", type=int, default=0, help="non-zero = a fresh held-out case set")
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    cases_path = DIRECTORY / seeded("cases.jsonl", args.seed)
    judge_path = DIRECTORY / seeded("judge.json", args.seed)
    reference_path = DIRECTORY / seeded("reference.jsonl", args.seed)

    if args.step == "build":
        sys.path.insert(0, str(ROOT / "eval"))
        sys.path.insert(0, str(ROOT / "training"))
        from routers import ModelRouter
        from run_eval import build_router, load_rows

        from routercore.adapters.route_map_config import load_route_map
        rows = load_rows("golden", None)
        router = build_router("b1", load_route_map(ROOT / "config" / "route_map.yaml"), None)
        predictions = [p.route for p in ModelRouter.predict(router, rows)]
        cases = build_cases(rows, predictions, size=100, seed=args.seed)
        lines = "\n".join(json.dumps(c, ensure_ascii=False) for c in cases)
        cases_path.write_text(lines + "\n", encoding="utf-8")
        print(f"{len(cases)} cases -> {cases_path}")
    elif args.step == "judge":
        from routercore.adapters.route_map_config import load_route_map
        routes = load_route_map(ROOT / "config" / "route_map.yaml").routes
        labels = asyncio.run(run_judge(_read(cases_path), system_prompt(args.rubric, routes)))
        judge_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
        print(f"{sum(v is not None for v in labels.values())}/{len(labels)} parsed")
    else:
        reference = {r["id"]: int(r["acceptable"]) for r in _read(reference_path)}
        report = agreement(json.loads(judge_path.read_text(encoding="utf-8")), reference)
        (ROOT / "docs" / "results" / "judge_kappa.json").write_text(json.dumps(report, indent=2) + "\n",
                                                                    encoding="utf-8")
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["usable"] else 1)


if __name__ == "__main__":
    main()

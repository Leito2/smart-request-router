# Judge validation: frontier-model reference

Goal: measure how well the local judge (Gemma 3 4B) agrees with a strong reference (Cohen's kappa, rule: kappa >= 0.6).

## Protocol (a pause point)

1. `python eval/judge_validation.py build` writes `cases.jsonl`: 100 cases, each a customer message plus the
   route a router proposed (half right, half wrong, shuffled).
2. The author switches to a frontier model in Claude Code and asks it to follow this file.
3. For each case the reviewer decides, from the route list in `config/route_map.yaml`, whether the proposed
   route is **acceptable** (the right team, or one that would clearly handle the message): `1` or `0`.
   Judge the message only. Do not look at `judge.json`, at the golden label or at the gold route.
4. Write one line per case to `reference.jsonl`: `{"id": "...", "acceptable": 1}`.
5. Print a summary for the author: counts of 1 and 0, and 5 examples (message, proposed route, verdict, reason).
6. `python eval/judge_validation.py judge` then `kappa` produce `docs/results/judge_kappa.json`.

## If kappa < 0.6

Improve the rubric, then try the panel (local judge + `judge_golden`). If it still fails, the judge is used only as a
signal, never for conclusions, and the README says so.

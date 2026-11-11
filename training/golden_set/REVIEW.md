# Golden set review protocol

Run this with a strong model (switch the model in Claude Code first). It replaces hand labelling.

**Task.** Read `training/golden_set/golden.jsonl` (500 rows) and write one JSON line per row to
`training/golden_set/reviews.jsonl`:

```json
{"id": "...", "verdict": "ok | fix | drop", "route": "...", "urgency": 0, "should_escalate": null, "note": "..."}
```

**Rules**
- `route` must be one of the 12 routes in `config/route_map.yaml` (use its descriptions). For multi-intent
  rows the route is the **primary** intent (first message); `urgency` is the highest across intents.
- `urgency`: 3 stolen/lost/compromised, 2 unrecognised charge (possible fraud), 1 money stuck or a failed
  action, 0 informational.
- Spanish rows (`lang: es`) are machine translations: if the translation changes the meaning, use `drop`.
- Hard-case rows carry `should_escalate`: true when a fast model should hand the message to a slower agent
  (ambiguous, several intents, out of domain, sarcasm, too short). Confirm or correct it.
- `ok` = the row's labels are right; `fix` = give the corrected fields; `drop` = unusable.
- Keep `note` to one short sentence, only for `fix` and `drop`.

**Then** run `uv run python training/apply_review.py`: it applies the verdicts (dropped rows are removed, so the set can end a little under 500) and prints a
summary with 5 examples for the author to read.

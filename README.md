# 🧭 Smart Request Router — System 1 / System 2

> A calibrated decision model resolves most fintech support messages in milliseconds; only uncertain, urgent or
> out-of-distribution cases escalate to a LangGraph agent. Quix Streams (on Redpanda) adds per-customer context
> and real-time surge detection. Everything is measured, and the measurements include the ones that went badly.

**Headline (measured, 8 GB laptop, no cloud):**

- System 1 (SetFit, plan B) resolves **41–50 %** of messages on its own at **~86–92 % selective accuracy**, p95 **9–11 ms**
  (offline harness, test / golden sets); the streaming path (Redpanda → Quix → dispatcher) adds **p50 51 ms / p95 87 ms**.
- Because only the rest escalates, an LLM is called for **~50–60 %** of messages instead of **100 %** (LLM-only router).
- A volume surge on one route (×5) is detected in **13.9 s** (10 s windows).
- **The thesis is only half proven.** The calibrated decision model the project was built around (Laya) did **not** beat
  a SetFit baseline when fine-tuned on a 4 GB GPU, and a 1.7 B/4 B local LLM as System 2 did **not** significantly beat
  System 1 on the escalated cases. Details, numbers and what would change them are in [Part IV](#part-iv--proof) and
  [Part VI](#part-vi--reflection).

**Status:** v1.0 candidate. Design in [`PLAN.md`](PLAN.md) (Spanish); the course behind it: *Decision Models — System One AI*.

---

## TL;DR — Results at a Glance

All rows use the same harness (`eval/harness.py`), the same seeded subsets (`--limit` samples with seed 0) and the same
labels. "Policy" = System 1 answers when confident, otherwise it escalates. Files are in [`docs/results/`](docs/results).

| | Router | Set | Accuracy | Macro-F1 | S1 coverage | Selective acc. | Urgent recall | ECE | p95 latency |
|---|---|---|---|---|---|---|---|---|---|
| B0 | Keyword rules | test (3,080) | 0.591 | 0.543 | 1.00 | – | 0.325 | – | 0.06 ms |
| B1 | SetFit (2 heads), no escalation | test (3,080) | 0.778 | 0.704 | 1.00 | – | 0.879 | – | 79 ms |
| B1+π | SetFit + policy | test (600) | 0.765 | 0.689 | 0.50 | 0.920 | 0.956 | 0.026 | 9 ms |
| B1+π | SetFit + policy | golden (500) | 0.744 | 0.725 | 0.41 | 0.857 | 1.000 | 0.042 | 11 ms |
| B1+π | SetFit + policy | hard (394) | 0.594 | 0.488 | 0.35 | 0.676 | 0.973 | 0.104 | 18 ms |
| B3+π | Laya, head fine-tuned (frozen encoder) + policy | test (600) | 0.497 | 0.411 | 0.12 | – | 1.000 | 0.034 | 364 ms |
| B3+π | same | golden (500) | 0.416 | 0.405 | 0.09 | – | 1.000 | 0.059 | 404 ms |
| B4 | LLM only (qwen3:1.7b, local) | test (150) | 0.433 | 0.151 | 1.00 | – | 1.000 | – | 734 ms |

Escalated cases only (System 2 vs System 1 on the 76 of 80 hard cases that escalated; paired bootstrap, 95 % CI):

| System 2 | Accuracy S1 → S2 | Difference | 95 % CI | Significant? |
|---|---|---|---|---|
| qwen3:1.7b, with tools | 0.500 → 0.539 | +3.9 pp | [−6.6, +14.5] | no |
| qwen3:1.7b, no tools | 0.500 → 0.487 | −1.3 pp | [−13.2, +9.2] | no |

Infrastructure results: [`smoke.json`](docs/results/smoke.json), [`surge_demo.json`](docs/results/surge_demo.json),
[`onnx_parity.json`](docs/results/onnx_parity.json), [`drift.json`](docs/results/drift.json),
[`calibration.json`](docs/results/calibration.json), [`thresholds.setfit.json`](docs/results/thresholds.setfit.json).

---

## Part I — The Big Picture

### 1. The problem: routing support at fintech scale

A fintech receives thousands of support messages a day: *"my card was stolen"*, *"why was my transfer delayed?"*,
*"how do I top up?"*, *"why was payment pay_42 blocked?"*. Each must reach the right place fast: a human team
(security, cards, fees…), the help-center knowledge base (an answer), or the payments-decision service (an
explanation). Two things make this hard:

1. **Errors are asymmetric.** Sending a stolen-card report to the FAQ is far worse than sending an FAQ question to a human.
2. **Volume is bursty.** An outage on one route multiplies its traffic; someone should notice in seconds, not hours.

Routing every message with an LLM is accurate on hard cases but slow and expensive; routing everything with a small
classifier is fast and cheap but confidently wrong on the unusual ones.

### 2. Core concepts primer

- **Decision model vs classifier vs LLM.** A classifier gives one label. An LLM gives a (slow, costly) free-form
  judgement. A *decision model* answers several multiple-choice questions about a message in one cheap pass and returns a
  **probability distribution** for each (here: *which route?* and *how urgent, 0–3?*).
- **Calibration & ECE.** A model is *calibrated* when "80 % confident" means right 80 % of the time. Expected Calibration
  Error (ECE) measures the gap. Escalation decisions are only as good as the confidences, so we fit a temperature on a
  validation split (never on test).
- **Selective classification.** Answer only when confident, abstain (escalate) otherwise. The trade-off is the
  *risk–coverage curve*: more coverage → more errors. We choose thresholds on it.
- **System 1 / System 2 cascade.** Fast and cheap first; slow and powerful only when needed.
- **Agents with tools.** System 2 is a LangGraph agent that can look up the customer's recent messages, the payments
  decision, and the help center before deciding.
- **Stream processing in Python.** Quix Streams keeps per-customer state (dedupe, recent messages) and counts
  traffic per route in time windows.

### 3. The System 1 / System 2 thesis

> Most messages are easy. Spend the expensive thinking only where the cheap model is unsure, and make the cheap model
> *know* when it is unsure.

Four reasons send a message to System 2, checked **in this order** (`routercore.domain.policy.decide`):

| # | Reason | Rule | Why |
|---|---|---|---|
| 1 | `urgency_safety_net` | P(urgency ≥ high) > `tau_urgent` | Asymmetric cost: a low bar to escalate possible fraud, even if the route looks certain |
| 2 | `ood` | unsupported language, or normalized entropy > `max_entropy` | The model does not know what this is |
| 3 | `multi_intent` | top-2 route gap < `margin` | Two plausible destinations: someone must choose |
| 4 | `low_confidence` | top route < `tau` | Not sure enough |
| – | `system1` | otherwise | Route directly |

### 4. Architecture

```
replayer ─► inbound-messages ─► Quix dataflow ──► routed ───────────► dispatcher ─► Redis queues (human teams)
 (or API)    (Redpanda)        dedupe · context ·       │                 │       ├► P3 /ask  (help-center RAG, SSE)
                               System 1 decision ──► escalations          │       └► P1 /decisions/{payment_id}
                                       │                  │              Postgres (idempotent decision log)
                                       │                  ▼
                                       │        System 2 worker (LangGraph) ──► routed (decided_by=system2)
                                       ▼
                                 surge detector ──► alerts ──► Prometheus / Grafana
```

Code is organised as **Clean Architecture** ([ADR-0002](docs/adr/0002-clean-architecture-core.md)): `domain` ←
`application` ← `adapters` ← `services/*`, enforced by `import-linter`. The decision model sits behind a port
(`DecisionModel`), which is what made swapping Laya for SetFit a one-line composition change.

### 5. Design decisions

- **System 2 lives outside the dataflow.** An agent takes seconds; inside Quix it would block every other message
  (head-of-line blocking). The dataflow only *publishes* to `escalations`; a separate worker resolves them.
- **Context without waiting.** Each message is routed immediately with the customer's last 5 minutes of messages as state;
  sessions are never used to delay routing.
- **At-least-once + idempotent dispatcher.** Redpanda may redeliver; the dispatcher keys on `message_id` (unique in
  Postgres), so processing twice equals processing once.
- **Redpanda instead of Kafka.** Same API, no JVM, ~100 MB resident here.
- **Redis Streams for team queues**, one consumer group per team.
- **Mocks follow the real sibling contracts** (P1 decisions API, P3 SSE `/ask`), so the same adapters talk to the real services.

### 6. Journey of a message

1. `POST /messages` (or the replayer) puts an `InboundMessage` on `inbound-messages`.
2. The Quix dataflow drops duplicates (60 s), attaches the customer's recent messages (5 min), and calls the decision model.
3. `decide()` returns either a route → `routed`, or a reason → `escalations` (carrying the model's distributions and the context).
4. The System 2 worker runs the agent, validates its answer against the route map (≤ 2 retries, then falls back to `triage`),
   and publishes to `routed`. A safety-net escalation is never resolved below priority `high`.
5. The dispatcher records the decision (idempotent), then enqueues for a team, asks P3, or asks P1.

---

## Part II — Components

| Component | Where | What it does |
|---|---|---|
| Routing core | `packages/routercore` | Policy, contracts, calibration, surge detector, use cases, ports. Shared by API, dataflow and eval so they cannot drift |
| Quix dataflow | `services/dataflow` | Dedupe, per-customer state, System 1, outputs `routed`/`escalations`/`dlq`; a second app does surge windows |
| System 2 worker | `services/system2` | LangGraph agent (analyze ⇄ tools → decide → validate → emit), concurrency 2, 20 s timeout, fallback queue |
| Dispatcher | `services/dispatcher` | Idempotent delivery to Redis queues, P3 (SSE), P1 (HTTP); Postgres decision log |
| API | `services/api` | `POST /route`, `POST /messages`, `/healthz` |
| Replayer | `services/replayer` | Streams the dataset as events, with `--surge route:factor:at_s:duration_s` |
| Mocks | `services/mocks` | P1 and P3 with the real contracts |
| `judgekit` | `packages/judgekit` | Reusable async LLM-as-a-Judge: judges, panels, position-swap pairwise, Cohen's κ, bootstrap, ECE/Brier, PSI drift, YAML quality gate |
| Eval | `eval/` | Harness, routers B0–B5, CI gate, drift monitor, judge validation, versioned batch job |
| Training | `training/` | Dataset, SetFit baseline, Laya fine-tune, calibration, threshold selection, ONNX parity, promotion |

**Surge detection.** `SurgeDetector` is an EWMA of per-route volume with a Poisson standard-deviation floor; baselines are
not learned during a surge and one alert fires per surge (after `consecutive_windows` anomalous windows).

**System 2 tools.** `customer_history` (the escalation's context, Redis when available), `payment_decision` (P1),
`knowledge_lookup` (P3). The agent never publishes by itself; the worker owns delivery.

---

## Part III — The Model

**Data.** Banking77 (77 intents, English) mapped to **12 routes** and a rule-based urgency 0–3
([`config/route_map.yaml`](config/route_map.yaml)); Spanish by translating a subset with a local LLM; plus hand-made
"hard cases" (ambiguous, multi-intent, out-of-domain, urgent). 14,953 rows: train 10,487 / val 1,155 / test 3,311. A
500-row golden set (278 en / 222 es / 36 urgent) and 394 hard cases are held out. See [`docs/dataset_card.md`](docs/dataset_card.md).
Twelve routes are deliberate: Laya degrades with more than ~20 options (zero-shot on all 77 intents scores 0.425).

**Baselines.** B0 keyword rules; B1 SetFit with an intent head and an urgency head (16 shots/class).

**Fine-tuning Laya on 4 GB.** `laya.train` has no LoRA and the 322 M encoder plus optimizer state does not fit in
4 GB of VRAM, so the local run trains only the decision head (encoder frozen), 2 epochs over 20,574 items
(10,287 rows × 2 questions), 2 h 38 on a GTX 1650. Validation accuracy (route+urgency items) 0.443 → 0.585, ECE 0.222 → 0.025.
A full fine-tune on Kaggle/Colab is prepared in
[`training/notebooks/laya_full_ft.ipynb`](training/notebooks/laya_full_ft.ipynb) but has **not** been run.
A gotcha worth knowing: with `max_len=128` the 12-option question does not fit and `laya.train` silently skips every route
item, so only urgency was being trained; the default is now 320.

**Calibration.** Temperature scaling per question and per language, fitted on validation only
([`training/calibrate.py`](training/calibrate.py)). On `laya_ft`: ECE route 0.040 → 0.029, urgency 0.024 → 0.019.

**Thresholds.** [`training/choose_thresholds.py`](training/choose_thresholds.py) picks `tau`, `margin`, `max_entropy` for the
highest S1 coverage whose accepted messages reach 90 % accuracy on validation, and `tau_urgent` so ≥ 98 % of urgent
messages escalate. Thresholds belong to a model, so each has its own file (`config/policy.yaml` for Laya,
`config/policy.setfit.yaml` for SetFit).

| Model | Coverage @ 90 % acc (val) | tau / margin / tau_urgent / max_entropy |
|---|---|---|
| Laya (frozen encoder) | **3.4 %** | 0.775 / 0.0 / 0.017 / 0.6 |
| SetFit | **47.3 %** | 0.625 / 0.2 / 0.218 / 1.01 |

**ONNX.** The fp32 graph agrees 100 % with PyTorch (route and urgency); **int8 agrees only 75 %**, so fp32 is what would
ship (Laya warns that int8 hurts calibration). fp32 p95 on CPU: 0.89 s at batch 1, ≈ 1.1 s per message at batch 8–16.

**Improvement loop.** `training/feedback.py` merges human corrections into the training set; `training/promote.py` promotes a
candidate that passes the gate to `production` in an MLflow registry and keeps the previous one as `previous`
(one command to roll back). Verified directly on a local MLflow registry (reject without gate → promote → previous alias → rollback).

---

## Part IV — Proof

**Harness.** One function (`eval/harness.py`) computes macro-F1 (overall, en, es), selective accuracy, coverage, urgent
recall, ECE, LLM calls and tokens per 1k, and latency percentiles for any router, so B0–B5 are comparable.

**Quality gate.** [`eval/ci_gate.py`](eval/ci_gate.py) runs a 100-case golden subset and fails when macro-F1 drops > 0.01, ECE rises
> 0.02 or urgent recall < 0.98 ([`config/gates.yaml`](config/gates.yaml)). Verified both ways: it passes on the committed
baseline and fails with `--degrade 0.3` (macro-F1 −0.164, ECE +0.135). The baseline in `eval/production_metrics.json` is
the Laya checkpoint's; CI does not run the gate because the weights are not published (see Part VI).

**Judge validation.** [`eval/judge_validation.py`](eval/judge_validation.py) builds cases, collects a reference review, runs the
judge, and computes Cohen's κ (target ≥ 0.6). **Not completed:** the reference review needs a strong model/human pass
([`docs/judge_validation/REVIEW.md`](docs/judge_validation/REVIEW.md)).

**Drift.** PSI on prediction signals (route mix, language mix, confidence, routed/escalated split) against the training
distribution, no labels needed. On the surge-demo traffic (payments ×5) the route mix drifts (PSI 3.94, threshold 0.25) and
the language mix does not (0.21); an Evidently report is in `docs/results/drift_evidently.html`.

**Streaming.** `scripts/smoke.py` (Redpanda + Quix + dispatcher + mocks, 600 messages at 20/s, rules model so the number
isolates the plumbing): 78 % routed by System 1, 0 in the dead-letter topic, **p50 51 ms / p95 87 ms** from sent to routed.
Quix flushes output on every commit (default 5 s); `COMMIT_INTERVAL_S=0.5` is what keeps latency low.

**Surge.** `scripts/surge_demo.py`: 10 msg/s, `payments` ×5 after 90 s → alert after **13.9 s** (z-score 110).

**Batch evaluation.** `eval/batch_job.py` reads the golden set from a *versioned* S3 bucket and writes a report that cites the
dataset's `VersionId`, so a number traces back to the exact data (verified against a local S3-compatible server).

**Cost.** LLM calls per 1k messages: LLM-only 1,000; System 1 + System 2 = the escalated share (≈ 500–600 with SetFit's
thresholds; ≈ 950 with Laya's, whose coverage is 3–12 %).

---

## Part V — Run It Yourself

```bash
python scripts/doctor.py                     # prerequisites
uv sync --all-packages --dev                 # + --group services --group ml --group mlops as needed
uv run pytest -q                             # 150 tests
uv run ruff check . && uv run lint-imports   # lint + Clean Architecture contracts

make up PROFILE=core                         # Redpanda (pinned), Redis, Postgres, topics
uv run python scripts/smoke.py --model rules --messages 600 --rate 20
uv run python scripts/surge_demo.py --model rules
make up PROFILE=agent                        # + llm-gateway (needs ../llm-gateway)

uv run python eval/run_eval.py --router b1 --set golden --policy --limit 600
uv run python eval/run_system2.py --model b1 --set hard --limit 100    # B5 (set LLM_BASE_URL / LLM_MODEL)
uv run python eval/ci_gate.py --model laya                              # regression gate
```

Models are chosen with `MODEL=laya|setfit|rules` (services) or `--router b0…b4`. GPU runs of Laya use the CUDA torch
wheel; install it **outside the lock** and use `uv run --no-sync` afterwards.

**Troubleshooting.**

- *Latency looks huge in the smoke test:* consumers were still joining the group; the script warms up 20 s (rules) / 40 s (Laya).
- *`laya.train` skips all route items:* raise `--max-len` (320 works for 12 options).
- *KFP local runner fails on Windows* (JSON decode error on backslash paths): run `pipelines/run_local.py` on Linux/macOS/WSL.
- *MinIO images are no longer published:* the compose `minio` service is a moto S3 server, pinned by digest.
- *Docker + 8 GB RAM:* start only the profile you need and stop Ollama while training.

---

## Part VI — Reflection

**What worked.** The architecture: one routing core shared by every entry point; a policy that is explicit, testable and
asymmetric; a stream pipeline that is idempotent and fast; surge detection and drift without labels; a gate that actually
fails when quality drops; mocks that follow the real contracts. Calibration brought ECE under 0.05 for both models.

**What did not.**

1. **Laya, with only its head trained, loses to SetFit** (macro-F1 0.41 vs 0.69 on test, 30–60× slower), so the planned
   acceptance criterion *B3 ≥ B1* is **not met**. The cause is almost certainly the frozen encoder (4 GB VRAM); the full
   fine-tune notebook is the next step and was not run. Until then the working System 1 is SetFit, behind the same port (PLAN risk R1).
2. **int8 ONNX is not usable** (75 % agreement); fp32 only.
3. **System 2 did not significantly beat System 1** with the small local models available (qwen3:1.7b; gemma3:4b below).
   The harness, the paired-bootstrap gate and the agent work; a stronger model behind the gateway alias `smart_router` is the experiment to run.
4. **Judge validation (κ ≥ 0.6)** needs the reference review; not done.
5. **Langfuse tracing** is implemented (optional) but was not run against Langfuse Cloud (needs keys).
6. **P1/P3 real services** were not available; adapters follow their published contracts and are tested against mocks.
7. **KFP quality-loop** compiles; its local runner does not work on Windows (see troubleshooting).

**Deviations from PLAN.** Frozen encoder instead of LoRA; Quix handles messages one by one rather than micro-batches of 16
(ADR-3, the model call is not the bottleneck with SetFit); the gate baseline is the local Laya checkpoint rather than a
published one; MinIO replaced by a moto S3 server; the retrain step of the pipeline is a stub.

**Honest takeaways.** A decision model is only useful if its confidences can be trusted *and* its accuracy is competitive;
calibration fixes the first, not the second. Measuring the cheap model against a cheap baseline before building around
it would have saved weeks, and is exactly what the harness is for.

## License

MIT

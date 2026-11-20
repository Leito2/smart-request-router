# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A System 1 / System 2 support-message router for a fintech: a calibrated decision model resolves most messages; uncertain, urgent or out-of-distribution ones escalate to a LangGraph agent. Quix Streams (on Redpanda/Kafka) adds per-customer context and surge detection. **M1–M9 are implemented** (tagged `v1.2`; v1.2 adds a Llama Prompt Guard 2 86M screen in `monitor` mode in front of System 2, `services/system2/prompt_guard.py`, measured by `eval/prompt_guard_eval.py`): `services/`, `eval/`, `training/`, `scripts/`, `pipelines/`, `observability/` all exist. [`README.md`](README.md) holds the measured results, including the negative ones: Laya fine-tuned on the 4 GB GPU (last 8 encoder layers + head, 1 epoch, `training/laya_finetune.py --unfreeze-top 8`) roughly ties SetFit on the test sample but loses on golden/hard, calibration and latency, so **SetFit is the working System 1** (`MODEL=setfit`, PLAN risk R1; its urgency head is `IntentUrgencyHead` in `adapters/setfit_model.py`: a logistic regression over frozen embeddings that predicts the Banking77 intent and maps it to urgency, scoring multi-sentence messages sentence by sentence; retrain it alone with `training/baselines/train_b1.py --urgency-only`, which also writes the out-of-fold urgent masses that `choose_thresholds.py --model setfit` pools for `tau_urgent`); small local LLMs as System 2 do not significantly beat System 1. `models/laya_ft` is that 8-layer epoch-1 checkpoint (the head-only one is `models/laya_ft_frozen`). The full Laya fine-tune notebook (`training/notebooks/laya_full_ft.ipynb`) has not been run. The Hugging Face dataset and SetFit model are published (`scripts/publish_hf.py`); the Gradio Space is not (Hub needs PRO).

[`PLAN.md`](PLAN.md) (Spanish) is the source of truth for design: architecture/ADRs (§2), components (§3), model/calibration (§4), evaluation harness (§5), resource budget (§7), milestones with acceptance criteria (§10), risks (§11), `judgekit` (§12). Check §10 before implementing — each milestone defines what is in scope and which README sections it feeds. README is written progressively (concepts/macro first, technical detail last in each component).

## Commands

Python 3.12, managed with `uv` (workspace). `make` targets wrap these; on Windows without `make`, run the underlying commands.

```bash
uv sync --all-packages --dev       # make setup
uv run pytest -q                   # make test (all suites)
uv run pytest packages/routercore/tests/test_policy.py::test_name -q   # single test
uv run ruff check .                # make lint (CI runs lint, then lint-imports, then test)
uv run lint-imports                # Clean Architecture dependency rule (also in make lint)
uv run ruff format .               # make fmt
python scripts/doctor.py           # make doctor — prerequisite check (stdlib only)
make up PROFILE=core|agent|ops     # docker compose profiles; `make down` stops all
uv run judgekit gate <rules.yaml> <metrics.json>   # quality gate CLI (exit 1 on failure)
```

`uv.lock` is committed and CI installs with `uv sync --locked`, so any dependency change in a `pyproject.toml` must ship with the regenerated `uv.lock` (`uv lock`) or CI fails at install.

`make data|train|calibrate|eval|eval-full|smoke|surge-demo` are real targets. GPU runs of Laya must use `uv run --no-sync` (the venv holds CUDA torch installed outside the lock). Eval scripts need `PYTHONPATH=.:eval:training` (the Makefile exports it). Ruff: line length 110, rules `E,F,I,UP,B`. pytest `testpaths` are the two package test dirs plus top-level `tests/`.

## Architecture

uv workspace (`packages/*`) with two libraries, both depended on by the root project:

- **`routercore`** — shared by dataflow, API and eval so they cannot drift. Organized as **Clean Architecture** ([ADR-0002](docs/adr/0002-clean-architecture-core.md)); imports only point inward, `domain` ← `application` ← `adapters` ← `services/*`, enforced by `lint-imports`:
  - `domain/` (entities, pure Python — no pydantic/yaml): `decision.py` holds the routing vocabulary (`DecidedBy`, `EscalationReason`, `Priority`) and `Assessment` (the model's per-question distributions). `policy.py`: `decide(route_p, urgency_p, lang_supported, pol)` returns `(target, reason)`. **Check order is significant**: urgency safety net (`P(urgency≥high) > tau_urgent`, asymmetric by design) → OOD (unsupported language or normalized entropy > `max_entropy`) → multi-intent (top-2 margin < `margin`) → low confidence (`p1 < tau`) → else System 1 route.
  - `application/` (use cases): `contracts.py` (pydantic event contracts v1, `InboundMessage` → `Routed` | `Escalation`, which are the use cases' I/O), `ports.py` (`Protocol`s such as `DecisionModel`), and use cases like `RouteMessages`. Use cases **return** events instead of publishing them, because Quix/FastAPI/LangGraph own delivery. Add a port only together with the use case that needs it.
  - `adapters/`: adapters shared by several services, eval and training: `policy_config`, `route_map_config`, `jsonl.read_jsonl`, the models (`rules_model`, `setfit_model`, `laya_model`) behind `model_factory.build_model(name, route_map, models_dir)` — the one place that knows how each is built, never re-implement that switch in a script — and `http_clients` (P1/P3). Heavy deps are imported lazily and declared as optional extras of routercore (`http`, `setfit`, `laya`, `pii`). `pii` is the Presidio redactor (`adapters/pii.py`): `services/system2/tracing.py` masks every Langfuse trace with it and turns tracing off if it cannot load; `scripts/pii_scan.py` scans the datasets and `langfuse_datasets.py` refuses to upload on a hit.
  - Services (when built) are the frameworks & drivers ring: thin composition roots plus their own single-use adapters. Don't create per-service `domain/`/`application/` folders.
- **`judgekit`** — async LLM-as-a-Judge framework intended to be reused by other projects (P1–P4). Currently: `metrics.py`, `calibration.py` (Cohen's kappa for judge-vs-human agreement), `gate.py` (YAML metric thresholds → pass/fail; **missing metrics fail**), `cli.py`. Heavy deps are optional extras (`runner`, `report`, `hf`, `sinks`) added per milestone — keep the base install to `pyyaml`.

Runtime config lives in `config/`: `policy.yaml` (thresholds + `threshold_set` id, written by `training/choose_thresholds.py` on the validation risk–coverage curve — rerun it after any new checkpoint/calibration), `route_map.yaml` (~12 routes; kept small on purpose because Laya degrades with >20 options), `surge.yaml` (EWMA surge detector params). `Policy` dataclass defaults are independent test fixtures (no longer mirrored from `policy.yaml`).

Services (`services/`): `replayer` → Redpanda → `dataflow` (Quix; dedupe, per-customer context, System 1, `routed`/`escalations`/`dlq`; `surge_app` for windows) → `dispatcher` (idempotent by `message_id`; Redis queues, P3 SSE, P1 HTTP, Postgres log) and `system2` (LangGraph worker); `api` (`/route`, `/messages`); `mocks` (P1/P3 with the real contracts); `composition.py` picks the model with `MODEL=setfit|laya|rules` (default `setfit`). **Thresholds belong to a model:** `config/policy.yaml` is Laya's, `config/policy.setfit.yaml` SetFit's (`policy_config.policy_path`); regenerate with `training/choose_thresholds.py --model laya|setfit` after any retrain. Quix emits output on each commit, so `COMMIT_INTERVAL_S` (default 0.5) drives latency.

`docker-compose.yml` profiles: `core` (Redpanda, Redis, Postgres, topic init; every image pinned by digest), `agent` (`llm-gateway` from `../llm-gateway`), `ops` (S3 store — moto, since MinIO stopped publishing images —, Prometheus, Grafana), `batch` (versioned eval job). Memory limits are sized for an 8 GB machine — preserve them (Grafana is 300 MB: Grafana 13 is OOM-killed at 150 MB). KFP's local runner does not work on Windows (backslash paths); the pipeline compiles to `pipelines/quality_loop.yaml`.

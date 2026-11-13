# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A System 1 / System 2 support-message router for a fintech: a calibrated decision model resolves most messages; uncertain, urgent or out-of-distribution ones escalate to a LangGraph agent. Quix Streams (on Redpanda/Kafka) adds per-customer context and surge detection. **M1–M9 are implemented** (v1.0 candidate): `services/`, `eval/`, `training/`, `scripts/`, `pipelines/`, `observability/` all exist. [`README.md`](README.md) holds the measured results, including the negative ones: Laya with a frozen encoder (4 GB GPU) loses to the SetFit baseline, so **SetFit is the working System 1** (`MODEL=setfit`, PLAN risk R1); small local LLMs as System 2 do not significantly beat System 1. The full Laya fine-tune notebook (`training/notebooks/laya_full_ft.ipynb`) has not been run.

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
  - `adapters/`: only adapters shared by several services (`policy_config.load_policy` reads `config/policy.yaml`; the Laya ONNX model goes here in M3). Heavy deps for them belong in optional extras.
  - Services (when built) are the frameworks & drivers ring: thin composition roots plus their own single-use adapters. Don't create per-service `domain/`/`application/` folders.
- **`judgekit`** — async LLM-as-a-Judge framework intended to be reused by other projects (P1–P4). Currently: `metrics.py`, `calibration.py` (Cohen's kappa for judge-vs-human agreement), `gate.py` (YAML metric thresholds → pass/fail; **missing metrics fail**), `cli.py`. Heavy deps are optional extras (`runner`, `report`, `hf`, `sinks`) added per milestone — keep the base install to `pyyaml`.

Runtime config lives in `config/`: `policy.yaml` (thresholds + `threshold_set` id, written by `training/choose_thresholds.py` on the validation risk–coverage curve — rerun it after any new checkpoint/calibration), `route_map.yaml` (~12 routes; kept small on purpose because Laya degrades with >20 options), `surge.yaml` (EWMA surge detector params). `Policy` dataclass defaults are independent test fixtures (no longer mirrored from `policy.yaml`).

Services (`services/`): `replayer` → Redpanda → `dataflow` (Quix; dedupe, per-customer context, System 1, `routed`/`escalations`/`dlq`; `surge_app` for windows) → `dispatcher` (idempotent by `message_id`; Redis queues, P3 SSE, P1 HTTP, Postgres log) and `system2` (LangGraph worker); `api` (`/route`, `/messages`); `mocks` (P1/P3 with the real contracts); `composition.py` picks the model with `MODEL=setfit|laya|rules` (default `setfit`). **Thresholds belong to a model:** `config/policy.yaml` is Laya's, `config/policy.setfit.yaml` SetFit's (`policy_config.policy_path`); regenerate with `training/choose_thresholds.py --model laya|setfit` after any retrain. Quix emits output on each commit, so `COMMIT_INTERVAL_S` (default 0.5) drives latency.

`docker-compose.yml` profiles: `core` (Redpanda pinned, Redis, Postgres, topic init), `agent` (`llm-gateway` from `../llm-gateway`), `ops` (S3 store — moto, since MinIO stopped publishing images —, Prometheus, Grafana), `batch` (versioned eval job). Memory limits are sized for an 8 GB machine — preserve them. KFP's local runner does not work on Windows (backslash paths); the pipeline compiles to `pipelines/quality_loop.yaml`.

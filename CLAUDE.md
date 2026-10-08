# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A System 1 / System 2 support-message router for a fintech: a calibrated decision model ("Laya", CPU) resolves most messages; uncertain, urgent or out-of-distribution ones escalate to a LangGraph agent. Quix Streams (on Redpanda/Kafka) adds per-customer context and surge detection. The repo is at **M0 (bootstrap)** — only the shared packages and infra scaffolding exist; everything under `services/`, `eval/`, `training/`, `tests/` is an empty `.gitkeep` placeholder.

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

`make data|train|calibrate|eval|eval-full|smoke|surge-demo` are deliberate stubs that exit 1 until their milestone (PLAN §10) lands. Ruff: line length 110, rules `E,F,I,UP,B`. pytest `testpaths` are the two package test dirs plus top-level `tests/`.

## Architecture

uv workspace (`packages/*`) with two libraries, both depended on by the root project:

- **`routercore`** — shared by dataflow, API and eval so they cannot drift. Organized as **Clean Architecture** ([ADR-0002](docs/adr/0002-clean-architecture-core.md)); imports only point inward, `domain` ← `application` ← `adapters` ← `services/*`, enforced by `lint-imports`:
  - `domain/` (entities, pure Python — no pydantic/yaml): `decision.py` holds the routing vocabulary (`DecidedBy`, `EscalationReason`, `Priority`) and `Assessment` (the model's per-question distributions). `policy.py`: `decide(route_p, urgency_p, lang_supported, pol)` returns `(target, reason)`. **Check order is significant**: urgency safety net (`P(urgency≥high) > tau_urgent`, asymmetric by design) → OOD (unsupported language or normalized entropy > `max_entropy`) → multi-intent (top-2 margin < `margin`) → low confidence (`p1 < tau`) → else System 1 route.
  - `application/` (use cases): `contracts.py` (pydantic event contracts v1, `InboundMessage` → `Routed` | `Escalation`, which are the use cases' I/O), `ports.py` (`Protocol`s such as `DecisionModel`), and use cases like `RouteMessages`. Use cases **return** events instead of publishing them, because Quix/FastAPI/LangGraph own delivery. Add a port only together with the use case that needs it.
  - `adapters/`: only adapters shared by several services (`policy_config.load_policy` reads `config/policy.yaml`; the Laya ONNX model goes here in M3). Heavy deps for them belong in optional extras.
  - Services (when built) are the frameworks & drivers ring: thin composition roots plus their own single-use adapters. Don't create per-service `domain/`/`application/` folders.
- **`judgekit`** — async LLM-as-a-Judge framework intended to be reused by other projects (P1–P4). Currently: `metrics.py`, `calibration.py` (Cohen's kappa for judge-vs-human agreement), `gate.py` (YAML metric thresholds → pass/fail; **missing metrics fail**), `cli.py`. Heavy deps are optional extras (`runner`, `report`, `hf`, `sinks`) added per milestone — keep the base install to `pyyaml`.

Runtime config lives in `config/`: `policy.yaml` (thresholds + `threshold_set` id, currently a `v0-placeholder` until M3 calibration picks them on the risk–coverage curve), `route_map.yaml` (~12 routes; kept small on purpose because Laya degrades with >20 options), `surge.yaml` (EWMA surge detector params). `Policy` dataclass defaults mirror `policy.yaml` for tests; a test asserts they match, so update both together.

Planned (not yet built) services, per PLAN §3: replayer → Redpanda → Quix Streams dataflow (dedupe, per-customer grouping, Laya call, surge windows) → dispatcher / System 2 LangGraph worker → API. `docker-compose.yml` currently defines only the infra (Redpanda, Redis, Postgres in `core`; `llm-gateway` in `agent`/`ops`; Grafana in `ops`) with memory limits sized for an 8 GB machine — preserve those limits. `llm-gateway` builds from a sibling repo (`../llm-gateway`).

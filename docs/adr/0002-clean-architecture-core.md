# ADR-0002: Clean Architecture for the routing core

- **Status:** accepted
- **Date:** 2026-10-08

## Context
The same routing logic runs in the Quix Streams dataflow, the FastAPI `POST /route` and the evaluation
harness, and must not drift between them (PLAN §3.8). The decision model is a risky dependency that may be
swapped for Von or ModernBERT (PLAN §11 R1/R1b). The repo is already split by deployable unit
(`services/`, `packages/`, `training/`, `eval/`); that layout answers "what runs", not "what depends on what".

## Decision
`packages/routercore` follows Clean Architecture's dependency rule — source code only points inward:

| Ring | Where | Contains |
|---|---|---|
| Entities | `routercore.domain` | routing vocabulary, `Assessment`, `Policy`/`decide`. Pure Python. |
| Use cases | `routercore.application` | use cases (`RouteMessages`), their I/O models (`contracts`), ports (`DecisionModel`) |
| Interface adapters | `routercore.adapters` | adapters shared by several services (policy config; Laya ONNX in M3) |
| Frameworks & drivers | `services/*` | Quix, FastAPI, Redis, LangGraph wiring; single-service adapters; the composition root |

It is enforced by `import-linter` (`[tool.importlinter]` in `pyproject.toml`, run by `make lint` and CI).

## Considered options
- **Hexagonal (ports & adapters):** same dependency direction, but no explicit use-case layer; Clean was
  preferred so each use case (route, escalate, record feedback) is a named, testable unit.
- **Full four-ring layout inside every service:** rejected. Most services are thin composition roots; empty
  `domain/`/`application/` folders per service would be ceremony.

## Consequences
- Use cases **return** events rather than publishing them: Quix Streams and LangGraph own their main loop,
  so operators and graph nodes stay thin and call use cases, and the frameworks are not abstracted away.
- Ports are added with the use case that needs them, not up front.
- Contracts use pydantic and live in the application ring; the domain may not import pydantic or yaml.
- `training/`, `eval/` and `judgekit` are pipelines/libraries and are out of scope of this rule.
- Supersedes PLAN's `packages/routercore/laya.py`: the Laya model is an adapter behind `DecisionModel`.

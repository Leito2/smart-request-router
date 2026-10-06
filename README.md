# 🧭 Smart Request Router — System 1 / System 2

> A calibrated decision model (Laya, on CPU) resolves most fintech support messages in milliseconds; only
> uncertain, urgent, or out-of-distribution cases escalate to a LangGraph agent. Quix Streams adds per-customer
> context and real-time surge detection. **Headline (to be measured):** _{X}% resolved by System 1 at p95 {Y} ms,
> {C}% fewer LLM calls than LLM-only routing, urgent recall ≥ 98%._

**Status:** 🟡 M0 bootstrap. Full design in [`PLAN.md`](PLAN.md) (Spanish). Course behind it: Decision Models —
System One AI (Learning vault 06/34).

## TL;DR — Results at a Glance
## Part I — The Big Picture
### 1. The Problem: routing support at fintech scale
### 2. Core Concepts Primer
Decision models vs classifiers vs LLMs · calibration & ECE · selective classification · System 1/System 2 cascades ·
agents with tools · stream processing in Python
### 3. The System 1 / System 2 Thesis · 4. Architecture · 5. Design Decisions · 6. Journey of a Message
## Part II — Components (Concept → How it works here → Technical details)
## Part III — The Model (data, baselines, fine-tuning on 4 GB, calibration, thresholds, improvement loop)
## Part IV — Proof (evaluation harness, judge validation, results, cost analysis, observability)
## Part V — Run It Yourself

```bash
python scripts/doctor.py
uv sync --all-packages --dev
uv run pytest -q                 # M0: routing policy + contracts
make up PROFILE=core             # Redpanda, Redis, Postgres
```

## Part VI — Reflection

## License
MIT

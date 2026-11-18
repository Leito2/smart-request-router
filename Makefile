.DEFAULT_GOAL := help
PROFILE ?= core
export PYTHONPATH := .:eval:training
export PYTHONIOENCODING := utf-8
.PHONY: help doctor setup up down test lint fmt data train calibrate eval eval-full smoke surge-demo

help:      ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'
doctor:    ## Check prerequisites
	python scripts/doctor.py
setup:     ## Install Python dependencies (uv workspace)
	uv sync --all-packages --dev
up:        ## Start a profile: make up PROFILE=core|agent|ops
	docker compose --profile $(PROFILE) up -d
down:      ## Stop everything
	docker compose --profile core --profile agent --profile ops down
data:      ## Banking77 + Spanish translations (local Ollama) + hard cases + golden set candidates
	uv run --group data python training/fetch_banking77.py
	uv run python training/translate_es.py
	uv run python training/make_dataset.py
test:      ## Unit + contract tests
	uv run pytest -q
lint:      ## Lint + architecture dependency rule
	uv run ruff check .
	uv run lint-imports
fmt:       ## Format
	uv run ruff format .
smoke:     ## End-to-end System 1 stream (needs `make up`): coverage and p95
	uv run python scripts/smoke.py
demo:      ## 5-message demo, no Docker: System 1 + System 2 via the gateway (--no-s2: System 1 only)
	uv run --no-sync python scripts/demo.py

surge-demo: ## Inject `payments` x5 and show the surge alert (needs `make up`)
	uv run python scripts/surge_demo.py
train:     ## Fine-tune Laya (frozen encoder, GPU) -> calibrate -> choose thresholds
	uv run --no-sync python training/laya_finetune.py --epochs 2 --device cuda
	uv run --no-sync python training/calibrate.py --checkpoint models/laya_ft --device cuda
	uv run --no-sync python training/choose_thresholds.py --device cuda
calibrate: ## Temperature scaling + thresholds on the validation split only
	uv run --no-sync python training/calibrate.py --checkpoint models/laya_ft --device cuda
	uv run --no-sync python training/choose_thresholds.py --device cuda
eval:      ## Quick evaluation (seeded 600-row samples): B1 with policy on golden/hard/test
	for s in golden hard test; do uv run --no-sync python eval/run_eval.py --router b1 --set $$s --policy --limit 600; done
eval-full: ## Full sets for every baseline (slow; Laya needs LAYA_DEVICE=cuda)
	for r in b0 b1 b3; do for s in golden hard test; do uv run --no-sync python eval/run_eval.py --router $$r --set $$s --policy; done; done

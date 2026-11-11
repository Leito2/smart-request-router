.DEFAULT_GOAL := help
PROFILE ?= core
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
surge-demo: ## Inject `payments` x5 and show the surge alert (needs `make up`)
	uv run python scripts/surge_demo.py
train calibrate eval eval-full:   ## Implemented in later milestones (PLAN.md §10)
	@echo "'$@' arrives in a later milestone — see PLAN.md §10"; exit 1

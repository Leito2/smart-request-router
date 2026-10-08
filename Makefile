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
test:      ## Unit + contract tests
	uv run pytest -q
lint:      ## Lint + architecture dependency rule
	uv run ruff check .
	uv run lint-imports
fmt:       ## Format
	uv run ruff format .
data train calibrate eval eval-full smoke surge-demo:   ## Implemented in later milestones (PLAN.md §10)
	@echo "'$@' arrives in a later milestone — see PLAN.md §10"; exit 1

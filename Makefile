# Call Quality Analyzer task runner.
# Run `make` or `make help` to list available targets.

PORT  ?= 8000
HOST  ?= 0.0.0.0
IMAGE ?= call-quality-analyzer
TAG   ?= latest

.DEFAULT_GOAL := help

.PHONY: help install dev run health lint format clean docker-build docker-run docker-shell

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "; printf "Usage: make \033[36m<target>\033[0m\n\nTargets:\n"} \
		/^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Sync dependencies into .venv via uv.
	uv sync

dev: ## Run the app with hot reload (HOST/PORT overridable).
	uv run uvicorn main:app --reload --host $(HOST) --port $(PORT)

run: ## Run the app without reload (production-style).
	uv run uvicorn main:app --host $(HOST) --port $(PORT)

health: ## Hit /health on the running server.
	curl -fsS http://127.0.0.1:$(PORT)/health && echo

lint: ## Run ruff lint checks.
	uv run --group dev ruff check .

format: ## Format the codebase with ruff.
	uv run --group dev ruff format .

clean: ## Remove Python and tooling caches (leaves .venv intact).
	find . -type d \( -name __pycache__ -o -name .ruff_cache -o -name .pytest_cache -o -name .mypy_cache \) \
		-not -path './.venv/*' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -not -path './.venv/*' -delete

docker-build: ## Build the Docker image ($(IMAGE):$(TAG)).
	docker build -t $(IMAGE):$(TAG) .

docker-run: ## Run the Docker image, mapping $(PORT) -> 8000.
	docker run --rm -p $(PORT):8000 $(IMAGE):$(TAG)

docker-shell: ## Open a shell inside the Docker image.
	docker run --rm -it --entrypoint sh $(IMAGE):$(TAG)

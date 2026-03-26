.PHONY: help install lint format check test coverage docs-serve docs-build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install project with test dependencies
	pip install -e ".[test]"
	pre-commit install

docs-serve: ## Serve the documentation site locally
	uv run --extra docs mkdocs serve -f docs/mkdocs.local.yml

docs-build: ## Build the documentation site
	uv run --extra docs mkdocs build --strict -f docs/mkdocs.yml

lint: ## Run ruff linter (check only)
	ruff check .

format: ## Run ruff formatter + fix lint issues
	ruff format .
	ruff check --fix .

check: lint ## Run all checks (lint + type check)
	ty check

test: ## Run tests
	pytest

coverage: ## Run tests with coverage report
	coverage run -m pytest
	coverage report
	coverage html

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

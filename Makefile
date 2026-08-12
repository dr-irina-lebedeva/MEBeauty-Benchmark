.PHONY: help sync format lint types test test-slow coverage results figures check check-all

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

sync:  ## Install everything, locked
	uv sync --locked --all-extras --dev

format:  ## Fix formatting and safe lint violations
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

types:  ## Static type check
	uv run mypy src

test:  ## Fast tests
	uv run pytest

# The floor covers the fast suite only. It sits at 52% because the torch
# training loops are unreachable without downloading weights -- those are
# covered by `test-slow`, not by lowering the bar here.
coverage:  ## Fast tests with a coverage floor
	uv run pytest --cov=fbp_benchmark --cov-report=term-missing --cov-fail-under=52

# End-to-end: every method fits and predicts for one epoch. Downloads the
# dataset and pretrained weights, and is the only thing that catches a method
# that cannot run at all.
test-slow:  ## End-to-end method tests (slow, needs network)
	uv run pytest -m slow

results:  ## Regenerate the README results table from results/
	uv run fbp-benchmark report --update-readme

figures:  ## Regenerate the README figures from the dataset and results/
	uv run --with matplotlib python scripts/make_figures.py

check: lint types test  ## What CI gates on
	uv run fbp-benchmark report --check-readme

check-all: lint types coverage test-slow  ## Everything, including slow tests

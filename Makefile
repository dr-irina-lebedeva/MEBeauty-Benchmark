.PHONY: sync format format-check lint test test-slow check check-all

sync:
	uv sync --locked --all-extras --dev

format:
	uv run ruff format .
	uv run ruff check --fix .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

test:
	uv run pytest

# End-to-end: every method fits and predicts for one epoch. Slower, and the
# only thing that catches a method that cannot run at all.
test-slow:
	uv run pytest -m slow

check: lint format-check test

check-all: lint format-check test test-slow

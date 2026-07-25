.RECIPEPREFIX := >

.PHONY: sync format format-check lint test check

sync:
> uv sync --locked --all-extras --dev

format:
> uv run ruff format .
> uv run ruff check --fix .

format-check:
> uv run ruff format --check .

lint:
> uv run ruff check .

test:
> uv run pytest

check: lint format-check test

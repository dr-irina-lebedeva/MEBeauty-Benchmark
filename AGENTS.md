# MEBeauty Benchmark — Agent Instructions

## Purpose

Build a reproducible dataset and benchmark for general and personalized facial attractiveness prediction.

## Required workflow

- Work on a feature branch; never push directly to `main`.
- Read the relevant issue and inspect existing code before editing.
- Keep changes small and reviewable.
- Run `make check` before committing.
- Report changed files, tests run, and remaining uncertainties.

## Data and research rules

- Treat the legacy MEBeauty repository and raw data as read-only.
- Never modify raw files manually.
- Never commit images, model weights, private ratings, tokens, or credentials.
- Never upload data without explicit approval.
- Keep rater identifiers anonymous.
- Distinguish clearly between:
  - author-reported results;
  - reimplemented results;
  - reproduced results;
  - verified results.
- Do not claim scientific reproduction unless the protocol and metrics were actually run.
- Record dataset revision, Git SHA, configuration, seed, and environment for every experiment.

## Code standards

- Python 3.12.
- Use `uv` for environments and dependencies.
- Put reusable code under `src/mebeauty_benchmark/`.
- Put commands and conversion utilities under `scripts/`.
- Add tests for parsing, validation, preprocessing, and evaluation logic.
- Prefer deterministic scripts over manual notebook operations.
- Use Ruff for linting and formatting.
- Use type hints for public functions.
- Do not silently catch data-quality errors.

## Commands

- Install: `uv sync --locked --all-extras --dev`
- Format: `make format`
- Lint: `make lint`
- Test: `make test`
- Full check: `make check`

## Security

- Secrets belong in `.env` or platform secret storage.
- Never expose Hugging Face, GitHub, W&B, or RunPod credentials.
- Do not provide coding agents unrestricted access to private identifiable data.

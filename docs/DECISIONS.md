# Project Decisions

## Repository responsibilities

- GitHub: code, configurations, tests, and documentation
- Hugging Face: dataset and released model weights
- W&B: experiment tracking
- RunPod: temporary GPU compute
- MacBook: primary development environment

## Data policy

- Preserve original MEBeauty data unchanged
- Use deterministic scripts for cleaning and conversion
- Do not commit datasets, private ratings, or model weights to GitHub
- Verify redistribution rights before public release
- Keep raters anonymous

## Experiment policy

Every experiment must record:

- Git commit
- dataset revision
- configuration
- random seed
- environment
- metrics
- checkpoint reference

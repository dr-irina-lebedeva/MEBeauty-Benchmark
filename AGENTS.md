# Working in this repository

This is a **benchmark**, not a dataset pipeline. It contains method
implementations and the harness that runs them. It contains no code for
cleaning, building or publishing a dataset — that lives on the
`chore/legacy-inventory` branch and must not be merged back here.

## Rules

- **Never add dataset preparation code.** The dataset is consumed from the
  Hugging Face Hub as published. If it needs fixing, fix it in the dataset
  repository and republish.
- **Never invent a training schedule.** Every trainable method takes its
  hyperparameters from `setups.py`, quoted from its paper with `source`,
  `quote` and `deviation` recorded. If a published value cannot transfer, say
  so in `deviation` — do not silently substitute.
- **Never let a method define its own evaluation.** Metrics belong to
  `metrics.py`. A method returns predictions; the harness scores them.
- **Never remove the leak check.** `assert_no_test_leak` runs every method
  twice with shuffled test labels. It is the one guard against the mistake
  that would invalidate every number in the table.
- **Do not claim a result was reproduced without running it.** Results carry
  the commit and a dirty-tree flag for exactly this reason.

## Adding a method

Decorate the class with `@register(...)`, giving `era`, `reference`, and
`requires` if it needs landmarks or rating distributions. Add a `setups.py`
entry if it is `trainable=True`. A test in `tests/test_registry.py` fails if
you forget the second step.

## Before finishing

Run `make check`, and summarise what changed and what remains uncertain.

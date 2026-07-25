"""Basic package smoke tests."""

import mebeauty_benchmark


def test_package_imports() -> None:
    """The installed package should import successfully."""
    assert mebeauty_benchmark is not None

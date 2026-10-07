"""Basic smoke tests for the novomodelo-bridge package structure."""

import novomodelo_bridge


def test_version_string_is_set() -> None:
    """Package version should be a non-empty string."""
    assert isinstance(novomodelo_bridge.__version__, str)
    assert len(novomodelo_bridge.__version__) > 0

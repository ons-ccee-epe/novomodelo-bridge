"""Tests for the shared diversion-channel flow-limit rule."""

from __future__ import annotations

import pytest

from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.diversion import (
    channel_flow_limit,
    is_unlimited_flow,
    unlimited_channels_diagnostic,
)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (None, None),
        (100.0, 100.0),
        (0.0, 0.0),
        (9_999_998.0, 9_999_998.0),
        (9_999_999.0, None),
        (99_999_999.0, None),
    ],
)
def test_channel_flow_limit(given: float | None, expected: float | None) -> None:
    assert channel_flow_limit(given) == expected


@pytest.mark.parametrize(
    ("given", "expected"),
    [(None, False), (100.0, False), (9_999_999.0, True), (1e9, True)],
)
def test_is_unlimited_flow(given: float | None, expected: bool) -> None:
    assert is_unlimited_flow(given) is expected


def test_unlimited_channels_diagnostic_lists_each_channel() -> None:
    diagnostic = unlimited_channels_diagnostic(
        [(314, "PIMENTAL", 288, "BELO MONTE", 9_999_999.0)], source="AC DESVIO"
    )
    assert diagnostic.code == "diversion-channel-unlimited"
    assert diagnostic.severity is Severity.INFO
    assert "AC DESVIO" in diagnostic.summary
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [[314, "PIMENTAL", 288, "BELO MONTE", "9999999"]]

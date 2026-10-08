"""Tier-1 tests for the shared relative-tolerance idiom.

Pure Python, no deck. Covers ``relative_tolerance``'s value table (reference
below/at/well above the ``max(|reference|, 1.0)`` floor) and ``floats_differ``'s
truth table (equal, sub-epsilon, super-epsilon, near-zero reference).
"""

from __future__ import annotations

import pytest

from novomodelo_bridge.core.tolerances import (
    RELATIVE_TOLERANCE,
    floats_differ,
    relative_tolerance,
)


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        (0.5, RELATIVE_TOLERANCE * 1.0),
        (-0.5, RELATIVE_TOLERANCE * 1.0),
        (1.0, RELATIVE_TOLERANCE * 1.0),
        (1e6, RELATIVE_TOLERANCE * 1e6),
        (-1e6, RELATIVE_TOLERANCE * 1e6),
    ],
)
def test_relative_tolerance_value_table(reference: float, expected: float) -> None:
    assert relative_tolerance(reference) == expected


def test_floats_differ_equal_values_do_not_differ() -> None:
    assert floats_differ(1.0, 1.0) is False


def test_floats_differ_sub_epsilon_gap_does_not_differ() -> None:
    assert floats_differ(1.0, 1.0 + RELATIVE_TOLERANCE * 0.5) is False


def test_floats_differ_super_epsilon_gap_differs() -> None:
    assert floats_differ(1.0, 1.0 + 1e-6) is True


def test_floats_differ_near_zero_reference_uses_the_floor() -> None:
    # reference ~0 floors to max(|reference|, 1.0) == 1.0, so a 5e-10 gap
    # (which would dwarf a purely-reference-scaled 1e-9 * 1e-10 tolerance)
    # still falls inside the floored 1e-9 tolerance.
    assert floats_differ(6e-10, 1e-10) is False
    assert floats_differ(1.0, 1e-10) is True

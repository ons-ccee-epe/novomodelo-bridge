"""Tier-1 tests for the shared relative-tolerance idiom.

Pure Python, no deck. Covers ``relative_tolerance``'s value table (reference
below/at/well above the ``max(|reference|, 1.0)`` floor), ``floats_differ``'s
truth table (equal, sub-epsilon, super-epsilon, near-zero reference) and
``beyond_written_precision``'s.
"""

from __future__ import annotations

import pytest

from novomodelo_bridge.core.tolerances import (
    RELATIVE_TOLERANCE,
    beyond_written_precision,
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


@pytest.mark.parametrize(
    ("value", "reference", "expected"),
    [
        # BELO MONTE's RE ceiling over its 18 x 611.111 MW of units.
        (11000.0, 10999.998, False),
        (11000.0, 10999.0, True),
        (10999.0, 11000.0, True),
        # Near zero the gap floors at 1e-6 absolute.
        (5e-7, 0.0, False),
        (2e-6, 0.0, True),
    ],
)
def test_beyond_written_precision_truth_table(
    value: float, reference: float, expected: bool
) -> None:
    assert beyond_written_precision(value, reference) is expected

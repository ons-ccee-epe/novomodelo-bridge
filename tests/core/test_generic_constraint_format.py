"""Unit tests for the shared F3 sense<->interval mapping helper.

``generic_constraint_format`` is the single source of truth both the source
model and DECOMP generic-constraint writers use to map a pre-F3 ``(sense,
value)`` pair onto novomodelo's sense-free ``(bound_lower, bound_upper)``
interval. Model-agnostic: no source-model/DECOMP coupling in the tests
either.
"""

from __future__ import annotations

import pytest

from novomodelo_bridge.core.generic_constraint_format import (
    GENERIC_BOUNDS_COLUMNS,
    sense_to_interval,
)


class TestSenseToInterval:
    def test_ge_yields_lower_only(self) -> None:
        assert sense_to_interval(">=", 12.5) == (12.5, None)

    def test_le_yields_upper_only(self) -> None:
        assert sense_to_interval("<=", 12.5) == (None, 12.5)

    def test_eq_yields_both_endpoints_equal(self) -> None:
        assert sense_to_interval("==", 7.0) == (7.0, 7.0)

    def test_unknown_sense_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown constraint sense"):
            sense_to_interval("!=", 1.0)

    def test_empty_sense_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown constraint sense"):
            sense_to_interval("", 1.0)


class TestGenericBoundsColumns:
    def test_column_list_is_the_f3_shape(self) -> None:
        assert GENERIC_BOUNDS_COLUMNS == [
            "constraint_id",
            "stage_id",
            "block_id",
            "bound_lower",
            "bound_upper",
        ]

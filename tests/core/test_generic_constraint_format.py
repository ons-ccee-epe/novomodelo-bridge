"""Unit tests for the shared generic-constraint wire format.

``generic_constraint_format`` is the single source of truth both the source
model and DECOMP generic-constraint writers use to render a constraint's
``expression`` and to map a ``(sense, value)`` pair onto novomodelo's
sense-free ``(bound_lower, bound_upper)`` interval. Model-agnostic: no
source-model/DECOMP coupling in the tests either.
"""

from __future__ import annotations

import pytest

from novomodelo_bridge.core.generic_constraint_format import (
    GENERIC_BOUNDS_COLUMNS,
    render_expression,
    sense_to_interval,
)


class TestRenderExpression:
    def test_unit_and_fractional_terms_join_with_binary_signs(self) -> None:
        terms = [
            (1.0, "hydro_generation(0)"),
            (-1.0, "thermal_generation(2)"),
            (2.0, "line_direct(1)"),
        ]
        assert (
            render_expression(terms)
            == "hydro_generation(0) - thermal_generation(2) + 2.0 * line_direct(1)"
        )

    def test_unit_positive_is_the_bare_token(self) -> None:
        assert render_expression([(1.0, "hydro_outflow(0)")]) == "hydro_outflow(0)"

    def test_unit_negative_is_the_negated_token(self) -> None:
        assert render_expression([(-1.0, "hydro_outflow(0)")]) == "- hydro_outflow(0)"

    def test_fractional_positive_multiplies_the_token(self) -> None:
        assert (
            render_expression([(0.5, "hydro_outflow(0)")]) == "0.5 * hydro_outflow(0)"
        )

    def test_fractional_negative_first_term_leads_with_a_minus(self) -> None:
        assert (
            render_expression([(-0.5, "hydro_outflow(0)")])
            == "- 0.5 * hydro_outflow(0)"
        )

    def test_multi_term_join(self) -> None:
        terms = [
            (1.0, "hydro_outflow(0)"),
            (0.5, "hydro_outflow(1)"),
            (-1.0, "hydro_outflow(2)"),
        ]
        assert (
            render_expression(terms)
            == "hydro_outflow(0) + 0.5 * hydro_outflow(1) - hydro_outflow(2)"
        )

    def test_parameter_weight_names_the_parameter(self) -> None:
        assert (
            render_expression(
                [(-1.0, "hydro_outflow(0)"), ("coef_x", "hydro_outflow(1)")]
            )
            == "- hydro_outflow(0) + @coef_x * hydro_outflow(1)"
        )
        assert render_expression([("coef_x", "hydro_outflow(1)")]) == (
            "@coef_x * hydro_outflow(1)"
        )

    def test_empty_term_list_is_none(self) -> None:
        assert render_expression([]) is None


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

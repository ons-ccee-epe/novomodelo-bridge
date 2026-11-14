"""Unit tests for the compare agreement bands (``novomodelo_bridge.core.agreement``)."""

from __future__ import annotations

import pytest

from novomodelo_bridge.core.agreement import (
    FAR_FACTOR,
    Band,
    band,
    format_gap,
    gap,
    is_policy_row,
    variable_label,
)
from novomodelo_bridge.core.summary_counts import summary_tolerance


class TestGap:
    def test_summed_absolute_difference_over_summed_reference(self) -> None:
        # (|110 − 100| + |40 − 50|) / (100 + 50)
        assert gap([(100.0, 110.0), (50.0, 40.0)]) == pytest.approx(20 / 150)

    def test_a_near_zero_reference_point_cannot_inflate_it(self) -> None:
        # Per point, the second pair alone is a relative error of 10⁶.
        assert gap([(100.0, 100.0), (1e-6, 1.0)]) == pytest.approx(0.01, rel=1e-5)

    def test_negative_references_count_by_magnitude(self) -> None:
        assert gap([(-100.0, -90.0)]) == pytest.approx(0.1)

    def test_accepts_a_one_shot_iterable(self) -> None:
        assert gap(iter([(10.0, 11.0)])) == pytest.approx(0.1)

    def test_a_floor_stands_in_for_a_smaller_reference(self) -> None:
        assert gap([(1.0, 3.0)], floor=100.0) == pytest.approx(0.02)
        assert gap([(0.0, 3.0)], floor=100.0) == pytest.approx(0.03)
        # A reference larger than the floor is unaffected.
        assert gap([(200.0, 210.0)], floor=100.0) == pytest.approx(0.05)

    def test_zero_on_both_sides_is_no_gap(self) -> None:
        assert gap([(0.0, 0.0), (0.0, 1e-9)]) == 0.0

    def test_zero_reference_with_a_difference_is_undefined(self) -> None:
        assert gap([(0.0, 5.0), (0.0, 0.0)]) is None


class TestBand:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, Band.CLOSE),
            (0.05, Band.CLOSE),
            (0.050001, Band.MODERATE),
            (0.2, Band.MODERATE),
            (0.200001, Band.FAR),
            (None, Band.FAR),
        ],
    )
    def test_edges_at_the_tolerance_and_the_far_factor(
        self, value: float | None, expected: Band
    ) -> None:
        assert band(value, 0.05) is expected

    def test_far_band_scales_with_the_tolerance(self) -> None:
        assert FAR_FACTOR == 4.0
        assert band(0.08, 0.02) is Band.MODERATE
        assert band(0.081, 0.02) is Band.FAR


class TestFormatGap:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, "∞"),
            (0.0, "0%"),
            (0.0004, "<0.1%"),
            (0.006, "0.6%"),
            (0.05, "5%"),
            (0.0549, "5.5%"),
            (0.0996, "10%"),
            (0.46, "46%"),
            (9.99, "999%"),
            (9.996, ">999%"),
            (25.0, ">999%"),
        ],
    )
    def test_short_percentages(self, value: float | None, expected: str) -> None:
        assert format_gap(value) == expected


class TestPolicyRows:
    @pytest.mark.parametrize(
        ("entity_type", "variable"),
        [
            ("convergence", "lower_bound"),
            ("convergence", "upper_bound_mean"),
            ("hydro", "water_value_per_hm3"),
        ],
    )
    def test_bounds_and_water_values_are_policy(
        self, entity_type: str, variable: str
    ) -> None:
        assert is_policy_row(entity_type, variable)

    @pytest.mark.parametrize(
        ("entity_type", "variable"),
        [("hydro", "generation_mw"), ("bus", "spot_price"), ("ree", "ena_mwmes")],
    )
    def test_operation_rows_are_not(self, entity_type: str, variable: str) -> None:
        assert not is_policy_row(entity_type, variable)


class TestVariableLabel:
    def test_known_variable_has_a_label_with_its_unit(self) -> None:
        assert variable_label("storage_final_hm3") == "Storage (hm³)"

    def test_unknown_variable_keeps_its_name(self) -> None:
        assert variable_label("gen_within") == "gen_within"


class TestSummaryTolerance:
    def test_reads_the_tolerance_the_dataset_was_judged_with(self) -> None:
        assert summary_tolerance({"tolerance": 0.02}) == 0.02
        assert summary_tolerance({"tolerance": 1}) == 1.0

    @pytest.mark.parametrize("metadata", [{}, {"tolerance": "5%"}, {"tolerance": True}])
    def test_falls_back_to_the_default(self, metadata: dict[str, object]) -> None:
        from novomodelo_bridge.core.agreement import DEFAULT_TOLERANCE

        assert summary_tolerance(metadata) == DEFAULT_TOLERANCE

"""Tests for ``comparators.version.convergence``: the final-bounds comparison and
the paired per-iteration rows, over literal frames with the bounds reader's schema."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence

import polars as pl
import pytest

from novomodelo_bridge.comparators.version.convergence import (
    UB_COMPARABILITY,
    BoundsComparison,
    FinalBounds,
    classify_upper_bound,
    compare_bounds,
    final_bound_rows,
    final_bounds,
    paired_convergence_rows,
    risk_measures_differ,
)
from novomodelo_bridge.core.agreement import (
    ENTITY_GROUPS,
    VARIABLE_LABELS,
    counts_in_headline,
    is_policy_row,
)

_SCHEMA = {
    "iteration": pl.Int64,
    "lower_bound": pl.Float64,
    "upper_bound": pl.Float64,
    "upper_bound_std": pl.Float64,
    "upper_bound_kind": pl.Utf8,
}

_EMPTY = pl.DataFrame(schema=_SCHEMA)

_NEUTRAL: Mapping[int, float] = {}


def _frame(
    lower: Sequence[float | None],
    upper: Sequence[float | None],
    kind: str | None = "statistical",
) -> pl.DataFrame:
    count = len(lower)
    return pl.DataFrame(
        {
            "iteration": list(range(1, count + 1)),
            "lower_bound": list(lower),
            "upper_bound": list(upper),
            "upper_bound_std": [None] * count,
            "upper_bound_kind": [kind] * count,
        },
        schema=_SCHEMA,
    )


def _final(
    lower: float | None, upper: float | None, kind: str | None = "statistical"
) -> pl.DataFrame:
    return _frame([lower], [upper], kind)


def _compare(
    baseline: pl.DataFrame,
    candidate: pl.DataFrame,
    baseline_risk: Mapping[int, float] | None = _NEUTRAL,
    candidate_risk: Mapping[int, float] | None = _NEUTRAL,
) -> BoundsComparison:
    return compare_bounds(baseline, candidate, baseline_risk, candidate_risk)


def _statistical_pair() -> tuple[pl.DataFrame, pl.DataFrame]:
    return _final(1000.0, 1005.0), _final(1000.0, 1008.0)


class TestFinalBounds:
    def test_takes_the_highest_iteration(self) -> None:
        frame = _frame([10.0, 20.0, 30.0], [11.0, 21.0, 31.0], "exact")

        assert final_bounds(frame) == FinalBounds(3, 30.0, 31.0, "exact")

    def test_ignores_row_order(self) -> None:
        frame = _frame([10.0, 20.0, 30.0], [11.0, 21.0, 31.0]).reverse()

        result = final_bounds(frame)

        assert result is not None
        assert result.iteration == 3

    def test_a_null_lower_bound_in_the_tail_row_falls_back_to_the_previous_iteration(
        self,
    ) -> None:
        frame = _frame([10.0, 20.0, None], [11.0, 21.0, 31.0])

        result = final_bounds(frame)

        assert result == FinalBounds(2, 20.0, 21.0, "statistical")

    def test_a_non_finite_lower_bound_in_the_tail_row_is_skipped(self) -> None:
        frame = _frame([10.0, math.nan, math.inf], [11.0, 21.0, 31.0])

        result = final_bounds(frame)

        assert result is not None
        assert result.iteration == 1

    def test_a_null_upper_bound_is_kept_as_none(self) -> None:
        result = final_bounds(_final(10.0, None, None))

        assert result == FinalBounds(1, 10.0, None, None)

    def test_a_non_finite_upper_bound_is_kept_as_read(self) -> None:
        result = final_bounds(_final(10.0, math.nan))

        assert result is not None
        assert result.upper is not None
        assert math.isnan(result.upper)

    def test_an_empty_frame_gives_none(self) -> None:
        assert final_bounds(_EMPTY) is None

    def test_an_all_null_frame_gives_none(self) -> None:
        assert final_bounds(_frame([None, None], [None, None], None)) is None

    @pytest.mark.parametrize("column", list(_SCHEMA))
    def test_a_missing_column_raises_naming_it(self, column: str) -> None:
        with pytest.raises(ValueError, match=column):
            final_bounds(_final(1.0, 2.0).drop(column))


class TestUbComparabilityConstant:
    def test_the_seven_reasons_in_their_pinned_order(self) -> None:
        assert UB_COMPARABILITY == (
            "comparable",
            "bounds-missing",
            "kind-missing",
            "kind-differs",
            "risk-averse",
            "risk-unknown",
            "bound-nonpositive",
        )


class TestClassifyUpperBound:
    def test_matching_kinds_and_risk_neutral_sides_are_comparable(self) -> None:
        baseline, candidate = _statistical_pair()

        assert _compare(baseline, candidate).comparability == "comparable"

    @pytest.mark.parametrize("which", ["baseline", "candidate", "both"])
    def test_a_side_without_final_bounds_is_bounds_missing(self, which: str) -> None:
        full = _final(1000.0, 1005.0)
        baseline = _EMPTY if which in ("baseline", "both") else full
        candidate = _EMPTY if which in ("candidate", "both") else full

        assert _compare(baseline, candidate).comparability == "bounds-missing"

    def test_an_all_null_frame_is_bounds_missing_never_kind_missing(self) -> None:
        all_null = _frame([None], [None], None)

        result = _compare(all_null, _final(1000.0, 1005.0))

        assert result.comparability == "bounds-missing"

    def test_a_null_kind_on_either_side_is_kind_missing(self) -> None:
        with_kind = _final(1000.0, 1005.0)
        without_kind = _final(1000.0, 1005.0, None)

        assert _compare(without_kind, with_kind).comparability == "kind-missing"
        assert _compare(with_kind, without_kind).comparability == "kind-missing"

    def test_a_null_upper_bound_std_is_no_reason(self) -> None:
        exact = _final(1000.0, 1000.0, "exact")

        assert _compare(exact, exact).comparability == "comparable"

    def test_different_kinds_are_kind_differs(self) -> None:
        result = _compare(
            _final(1000.0, 1005.0, "statistical"), _final(1000.0, 1008.0, "exact")
        )

        assert result.comparability == "kind-differs"

    def test_a_positive_weight_on_either_side_is_risk_averse(self) -> None:
        baseline, candidate = _statistical_pair()

        assert (
            _compare(baseline, candidate, _NEUTRAL, {0: 0.35}).comparability
            == "risk-averse"
        )
        assert (
            _compare(baseline, candidate, {0: 0.35}, _NEUTRAL).comparability
            == "risk-averse"
        )

    def test_a_none_risk_map_is_risk_unknown_never_risk_neutral(self) -> None:
        baseline, candidate = _statistical_pair()

        assert _compare(baseline, candidate, None, _NEUTRAL).comparability == (
            "risk-unknown"
        )
        assert _compare(baseline, candidate, _NEUTRAL, None).comparability == (
            "risk-unknown"
        )
        assert _compare(baseline, candidate, None, None).comparability == (
            "risk-unknown"
        )

    def test_a_known_positive_weight_beats_an_unknown_measure_on_the_other_side(
        self,
    ) -> None:
        baseline, candidate = _statistical_pair()

        assert _compare(baseline, candidate, {0: 0.35}, None).comparability == (
            "risk-averse"
        )
        assert _compare(baseline, candidate, None, {0: 0.35}).comparability == (
            "risk-averse"
        )

    @pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf, -0.1, 1.5])
    def test_an_out_of_range_weight_is_risk_unknown(self, weight: float) -> None:
        baseline, candidate = _statistical_pair()

        assert _compare(baseline, candidate, {1: weight}, _NEUTRAL).comparability == (
            "risk-unknown"
        )
        assert _compare(baseline, candidate, _NEUTRAL, {0: 0.0, 1: weight}) == (
            _compare(baseline, candidate, _NEUTRAL, None)
        )

    @pytest.mark.parametrize("weight", [math.nan, -0.1, 1.5])
    def test_one_out_of_range_weight_makes_the_whole_map_unknown(
        self, weight: float
    ) -> None:
        baseline, candidate = _statistical_pair()

        result = _compare(baseline, candidate, {0: 0.35, 1: weight}, _NEUTRAL)

        assert result.comparability == "risk-unknown"

    @pytest.mark.parametrize("weight", [math.nan, 1.5])
    def test_an_out_of_range_weight_loses_to_a_known_positive_weight(
        self, weight: float
    ) -> None:
        baseline, candidate = _statistical_pair()

        result = _compare(baseline, candidate, {1: weight}, {0: 0.35})

        assert result.comparability == "risk-averse"

    @pytest.mark.parametrize(
        ("weight", "expected"), [(0.0, "comparable"), (1.0, "risk-averse")]
    )
    def test_the_range_limits_are_known_weights(
        self, weight: float, expected: str
    ) -> None:
        baseline, candidate = _statistical_pair()

        result = _compare(baseline, candidate, {0: weight}, _NEUTRAL)

        assert result.comparability == expected

    @pytest.mark.parametrize(
        ("lower", "upper"),
        [
            (0.0, 1005.0),
            (-5.0, 1005.0),
            (1000.0, None),
            (1000.0, math.nan),
            (1000.0, math.inf),
            (1000.0, 0.0),
            (1000.0, -3.0),
        ],
    )
    def test_a_nonpositive_or_unusable_bound_on_either_side_is_bound_nonpositive(
        self, lower: float, upper: float | None
    ) -> None:
        good = _final(1000.0, 1005.0)
        bad = _final(lower, upper)

        assert _compare(bad, good).comparability == "bound-nonpositive"
        assert _compare(good, bad).comparability == "bound-nonpositive"

    def test_bounds_missing_precedes_kind_missing(self) -> None:
        result = _compare(_EMPTY, _final(1000.0, 1005.0, None))

        assert result.comparability == "bounds-missing"

    def test_kind_missing_precedes_kind_differs_and_risk(self) -> None:
        result = _compare(
            _final(1000.0, 1005.0, None), _final(1000.0, 1008.0), {0: 0.35}, None
        )

        assert result.comparability == "kind-missing"

    def test_kind_differs_precedes_risk_averse(self) -> None:
        result = _compare(
            _final(1000.0, 1005.0, "statistical"),
            _final(1000.0, 1008.0, "exact"),
            {0: 0.35},
            {0: 0.35},
        )

        assert result.comparability == "kind-differs"

    def test_risk_averse_precedes_risk_unknown(self) -> None:
        baseline, candidate = _statistical_pair()

        result = _compare(baseline, candidate, {0: 0.35}, None)

        assert result.comparability == "risk-averse"

    def test_risk_unknown_precedes_bound_nonpositive(self) -> None:
        result = _compare(_final(0.0, 1005.0), _final(1000.0, 1008.0), None, _NEUTRAL)

        assert result.comparability == "risk-unknown"

    def test_classifier_takes_final_bounds_directly(self) -> None:
        final = FinalBounds(4, 1000.0, 1005.0, "statistical")

        assert classify_upper_bound(final, final, {}, {}) == "comparable"
        assert classify_upper_bound(None, final, {}, {}) == "bounds-missing"


class TestRiskMeasuresDiffer:
    @pytest.mark.parametrize(
        ("baseline", "candidate", "expected"),
        [
            ({}, {0: 0.0, 1: 0.0}, False),
            ({}, {1: 0.35}, True),
            ({1: 0.35}, {1: 0.35, 2: 0.0}, False),
            ({1: 0.35}, {1: 0.5}, True),
            ({1: 0.35}, None, False),
            (None, {1: 0.35}, False),
            (None, None, False),
            ({1: 0.35}, {2: 0.35}, True),
            ({1: 0.35}, {1: 0.35}, False),
        ],
    )
    def test_positive_weights_are_what_differs(
        self,
        baseline: Mapping[int, float] | None,
        candidate: Mapping[int, float] | None,
        expected: bool,
    ) -> None:
        assert risk_measures_differ(baseline, candidate) is expected

    @pytest.mark.parametrize("weight", [math.nan, math.inf, -0.1, 1.5])
    def test_an_out_of_range_weight_is_an_unknown_measure_not_known_to_differ(
        self, weight: float
    ) -> None:
        assert risk_measures_differ({1: weight}, {1: 0.35}) is False
        assert risk_measures_differ({1: 0.35}, {1: weight}) is False
        assert risk_measures_differ({}, {1: weight}) is False

    def test_it_is_true_under_kind_missing_when_the_weights_differ(self) -> None:
        result = _compare(
            _final(1000.0, 1005.0, None), _final(1000.0, 1008.0), {}, {1: 0.35}
        )

        assert result.comparability == "kind-missing"
        assert result.risk_measures_differ is True

    def test_it_is_true_under_kind_differs_when_the_weights_differ(self) -> None:
        result = _compare(
            _final(1000.0, 1005.0, "statistical"),
            _final(1000.0, 1008.0, "exact"),
            {},
            {1: 0.35},
        )

        assert result.comparability == "kind-differs"
        assert result.risk_measures_differ is True

    def test_it_is_false_under_risk_averse_when_both_sides_share_the_measure(
        self,
    ) -> None:
        baseline, candidate = _statistical_pair()

        result = _compare(baseline, candidate, {1: 0.35}, {1: 0.35})

        assert result.comparability == "risk-averse"
        assert result.risk_measures_differ is False

    def test_it_is_independent_of_missing_bounds(self) -> None:
        result = _compare(_EMPTY, _final(1000.0, 1008.0), {}, {1: 0.35})

        assert result.comparability == "bounds-missing"
        assert result.risk_measures_differ is True

    def test_it_is_false_when_a_risk_map_is_none_under_a_kind_reason(self) -> None:
        result = _compare(
            _final(1000.0, 1005.0, "statistical"),
            _final(1000.0, 1008.0, "exact"),
            {1: 0.35},
            None,
        )

        assert result.comparability == "kind-differs"
        assert result.risk_measures_differ is False


class TestComparableRows:
    def test_comparable_runs_give_both_rows_and_the_gaps(self) -> None:
        baseline, candidate = _statistical_pair()

        result = _compare(baseline, candidate)
        rows = final_bound_rows(result)

        assert result.comparability == "comparable"
        assert result.upper_bound_comparable is True
        assert result.baseline_gap_percent == pytest.approx(0.5)
        assert result.candidate_gap_percent == pytest.approx(0.8)
        assert result.gap_delta_pp == pytest.approx(0.3)
        assert [(r.entity_type, r.variable) for r in rows] == [
            ("final_bounds", "final_lower_bound"),
            ("final_bounds", "final_bound_ratio"),
        ]
        assert (rows[0].newave_value, rows[0].novomodelo_value) == (1000.0, 1000.0)
        assert rows[1].newave_value == pytest.approx(1.005)
        assert rows[1].novomodelo_value == pytest.approx(1.008)

    def test_row_identity_is_the_training_entity_at_id_and_stage_zero(self) -> None:
        baseline, candidate = _statistical_pair()

        for row in final_bound_rows(_compare(baseline, candidate)):
            assert row.entity_name == "Training"
            assert (row.newave_code, row.novomodelo_id, row.stage) == (0, 0, 0)

    def test_the_baseline_fills_the_newave_fields_and_diffs_follow_result_diff(
        self,
    ) -> None:
        result = _compare(_final(1000.0, 1005.0), _final(1100.0, 1108.0))

        lower, ratio = final_bound_rows(result)

        assert (lower.newave_value, lower.novomodelo_value) == (1000.0, 1100.0)
        assert lower.abs_diff == pytest.approx(100.0)
        assert lower.rel_diff == pytest.approx(0.1)
        assert ratio.abs_diff == pytest.approx(abs(1108.0 / 1100.0 - 1005.0 / 1000.0))

    def test_the_gap_delta_is_candidate_minus_baseline(self) -> None:
        result = _compare(_final(1000.0, 1008.0), _final(1000.0, 1005.0))

        assert result.gap_delta_pp == pytest.approx(-0.3)

    def test_each_side_uses_its_own_final_iteration_without_trimming(self) -> None:
        baseline = _frame(
            [900.0, 910.0, 920.0, 930.0, 1000.0],
            [1100.0, 1090.0, 1080.0, 1070.0, 1010.0],
        )
        candidate = _frame([900.0, 910.0, 995.0], [1100.0, 1090.0, 1005.0])

        result = _compare(baseline, candidate)

        assert result.baseline == FinalBounds(5, 1000.0, 1010.0, "statistical")
        assert result.candidate == FinalBounds(3, 995.0, 1005.0, "statistical")
        assert result.comparability == "comparable"
        assert final_bound_rows(result)[0].newave_value == 1000.0
        assert final_bound_rows(result)[0].novomodelo_value == 995.0

    def test_identical_exact_runs_have_zero_gaps_and_a_zero_ratio_diff(self) -> None:
        frame = _final(1000.0, 1000.0, "exact")

        result = _compare(frame, frame)
        ratio = final_bound_rows(result)[1]

        assert result.comparability == "comparable"
        assert result.baseline_gap_percent == 0.0
        assert result.candidate_gap_percent == 0.0
        assert result.gap_delta_pp == 0.0
        assert ratio.variable == "final_bound_ratio"
        assert ratio.abs_diff == 0.0

    def test_the_built_rows_count_in_the_headline_and_are_not_policy_rows(self) -> None:
        baseline, candidate = _statistical_pair()

        for row in final_bound_rows(_compare(baseline, candidate)):
            assert counts_in_headline(row.entity_type, row.variable)
            assert not is_policy_row(row.entity_type, row.variable)

    def test_the_names_are_the_ones_the_agreement_tables_define(self) -> None:
        baseline, candidate = _statistical_pair()

        for row in final_bound_rows(_compare(baseline, candidate)):
            assert row.entity_type in ENTITY_GROUPS
            assert row.variable in VARIABLE_LABELS


class TestRowsUnderEachReason:
    @pytest.mark.parametrize(
        ("baseline", "candidate", "baseline_risk", "candidate_risk", "reason"),
        [
            (
                _final(1000.0, 1005.0, None),
                _final(1000.0, 1008.0),
                {},
                {},
                "kind-missing",
            ),
            (
                _final(1000.0, 1005.0, "statistical"),
                _final(1000.0, 1008.0, "exact"),
                {},
                {},
                "kind-differs",
            ),
            (
                _final(1000.0, 1005.0),
                _final(1000.0, 1008.0),
                {},
                {0: 0.35},
                "risk-averse",
            ),
            (
                _final(1000.0, 1005.0),
                _final(1000.0, 1008.0),
                {},
                None,
                "risk-unknown",
            ),
            (
                _final(0.0, 1005.0),
                _final(1000.0, 1008.0),
                {},
                {},
                "bound-nonpositive",
            ),
        ],
    )
    def test_only_the_lower_bound_row_is_built(
        self,
        baseline: pl.DataFrame,
        candidate: pl.DataFrame,
        baseline_risk: Mapping[int, float] | None,
        candidate_risk: Mapping[int, float] | None,
        reason: str,
    ) -> None:
        result = _compare(baseline, candidate, baseline_risk, candidate_risk)

        rows = final_bound_rows(result)

        assert result.comparability == reason
        assert result.upper_bound_comparable is False
        assert [(r.entity_type, r.variable) for r in rows] == [
            ("final_bounds", "final_lower_bound")
        ]


class TestBoundsMissing:
    def test_an_empty_baseline_keeps_the_candidate_gap_and_builds_no_row(self) -> None:
        result = _compare(_EMPTY, _final(1000.0, 1008.0))

        assert result.comparability == "bounds-missing"
        assert result.baseline is None
        assert result.baseline_gap_percent is None
        assert result.candidate_gap_percent == pytest.approx(0.8)
        assert result.gap_delta_pp is None
        assert final_bound_rows(result) == []

    def test_an_empty_candidate_keeps_the_baseline_gap_and_builds_no_row(self) -> None:
        result = _compare(_final(1000.0, 1005.0), _EMPTY)

        assert result.comparability == "bounds-missing"
        assert result.baseline_gap_percent == pytest.approx(0.5)
        assert result.candidate_gap_percent is None
        assert result.gap_delta_pp is None
        assert final_bound_rows(result) == []

    def test_two_empty_frames_raise_nothing(self) -> None:
        result = _compare(_EMPTY, _EMPTY, None, None)

        assert result == BoundsComparison(
            None, None, "bounds-missing", None, None, None, False
        )
        assert final_bound_rows(result) == []

    def test_a_simulation_only_side_with_all_null_bounds_builds_no_row(self) -> None:
        result = _compare(_frame([None, None], [None, None]), _final(1.0, 2.0))

        assert result.comparability == "bounds-missing"
        assert final_bound_rows(result) == []


class TestGapFields:
    def test_the_gap_is_the_percent_excess_of_upper_over_lower(self) -> None:
        result = _compare(_final(200.0, 250.0), _final(400.0, 400.0))

        assert result.baseline_gap_percent == pytest.approx(25.0)
        assert result.candidate_gap_percent == pytest.approx(0.0)
        assert result.gap_delta_pp == pytest.approx(-25.0)

    @pytest.mark.parametrize(
        ("lower", "upper"),
        [
            (0.0, 1005.0),
            (-5.0, 1005.0),
            (1000.0, None),
            (1000.0, math.nan),
            (1000.0, math.inf),
            (1e-300, 1e300),
        ],
    )
    def test_an_undefined_side_gap_is_none_and_so_is_the_delta(
        self, lower: float, upper: float | None
    ) -> None:
        result = _compare(_final(lower, upper), _final(1000.0, 1008.0))

        assert result.baseline_gap_percent is None
        assert result.candidate_gap_percent == pytest.approx(0.8)
        assert result.gap_delta_pp is None

    def test_a_zero_upper_bound_still_has_a_gap(self) -> None:
        result = _compare(_final(1000.0, 0.0), _final(1000.0, 1008.0))

        assert result.baseline_gap_percent == pytest.approx(-100.0)

    @pytest.mark.parametrize(
        ("baseline", "candidate"),
        [
            (_final(1000.0, 1005.0), _final(1000.0, 1008.0)),
            (_EMPTY, _final(1000.0, 1008.0)),
            (_final(0.0, 1005.0), _final(1000.0, math.nan)),
            (_final(1000.0, math.inf), _final(1e-300, 1e300)),
            (_EMPTY, _EMPTY),
        ],
    )
    def test_every_gap_field_survives_strict_json(
        self, baseline: pl.DataFrame, candidate: pl.DataFrame
    ) -> None:
        result = _compare(baseline, candidate)

        fields = {
            "baseline": result.baseline_gap_percent,
            "candidate": result.candidate_gap_percent,
            "delta": result.gap_delta_pp,
        }

        json.dumps(fields, allow_nan=False)

    def test_an_overflowing_ratio_is_not_comparable(self) -> None:
        huge = _final(1e-300, 1e300)

        result = _compare(huge, _final(1000.0, 1008.0))
        rows = final_bound_rows(result)

        assert result.comparability == "bound-nonpositive"
        assert not result.upper_bound_comparable
        assert [r.variable for r in rows] == ["final_lower_bound"]


class TestPairedConvergenceRows:
    def test_only_the_common_iterations_are_paired(self) -> None:
        baseline = _frame(
            [10.0, 20.0, 30.0, 40.0, 50.0], [110.0, 120.0, 130.0, 140.0, 150.0]
        )
        candidate = _frame([11.0, 21.0, 31.0], [111.0, 121.0, 131.0])

        rows = paired_convergence_rows(baseline, candidate)

        assert final_bounds(baseline) == FinalBounds(5, 50.0, 150.0, "statistical")
        assert final_bounds(candidate) == FinalBounds(3, 31.0, 131.0, "statistical")
        assert len(rows) == 6
        assert {r.entity_type for r in rows} == {"convergence"}
        assert [r.entity_name for r in rows] == [
            "iteration_1",
            "iteration_1",
            "iteration_2",
            "iteration_2",
            "iteration_3",
            "iteration_3",
        ]

    def test_each_iteration_gives_lower_bound_then_upper_bound(self) -> None:
        baseline = _frame([10.0, 20.0], [110.0, 120.0])
        candidate = _frame([11.0, 22.0], [111.0, 118.0])

        rows = paired_convergence_rows(baseline, candidate)

        assert [
            (
                r.variable,
                r.stage,
                r.newave_code,
                r.novomodelo_id,
                r.newave_value,
                r.novomodelo_value,
            )
            for r in rows
        ] == [
            ("lower_bound", 1, 1, 1, 10.0, 11.0),
            ("upper_bound", 1, 1, 1, 110.0, 111.0),
            ("lower_bound", 2, 2, 2, 20.0, 22.0),
            ("upper_bound", 2, 2, 2, 120.0, 118.0),
        ]
        assert rows[0].abs_diff == pytest.approx(1.0)
        assert rows[0].rel_diff == pytest.approx(0.1)

    def test_an_iteration_with_a_null_bound_on_either_side_is_left_out(self) -> None:
        baseline = _frame([10.0, None, 30.0], [110.0, 120.0, None])
        candidate = _frame([11.0, 21.0, 31.0], [111.0, 121.0, 131.0])

        rows = paired_convergence_rows(baseline, candidate)

        assert {r.entity_name for r in rows} == {"iteration_1"}

    def test_a_non_finite_bound_builds_no_row(self) -> None:
        baseline = _frame([10.0, math.nan], [110.0, 120.0])
        candidate = _frame([11.0, 21.0], [math.inf, 121.0])

        assert paired_convergence_rows(baseline, candidate) == []

    def test_identical_frames_differ_by_exactly_zero(self) -> None:
        frame = _frame([10.0, 20.0], [110.0, 120.0])

        rows = paired_convergence_rows(frame, frame)

        assert len(rows) == 4
        assert {r.abs_diff for r in rows} == {0.0}

    def test_disjoint_or_empty_frames_give_no_rows(self) -> None:
        frame = _frame([10.0], [110.0])

        assert paired_convergence_rows(_EMPTY, frame) == []
        assert paired_convergence_rows(frame, _EMPTY) == []
        assert paired_convergence_rows(_EMPTY, _EMPTY) == []

    def test_the_rows_are_policy_rows_kept_out_of_the_headline(self) -> None:
        frame = _frame([10.0], [110.0])

        for row in paired_convergence_rows(frame, frame):
            assert is_policy_row(row.entity_type, row.variable)

    @pytest.mark.parametrize("column", list(_SCHEMA))
    def test_a_missing_column_raises_naming_it(self, column: str) -> None:
        frame = _frame([10.0], [110.0])

        with pytest.raises(ValueError, match=column):
            paired_convergence_rows(frame.drop(column), frame)
        with pytest.raises(ValueError, match=column):
            paired_convergence_rows(frame, frame.drop(column))


class TestMissingColumnInCompare:
    @pytest.mark.parametrize("column", list(_SCHEMA))
    def test_compare_bounds_raises_naming_the_column(self, column: str) -> None:
        frame = _final(1000.0, 1005.0)

        with pytest.raises(ValueError, match=column):
            _compare(frame.drop(column), frame)
        with pytest.raises(ValueError, match=column):
            _compare(frame, frame.drop(column))

"""Tests for ``comparators.version.rows``: the paired stage rows of two runs, over literal frames."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
import pytest

from novomodelo_bridge.comparators.model import ResultComparison
from novomodelo_bridge.comparators.version.alignment import AlignedPair
from novomodelo_bridge.comparators.version.rows import paired_stage_rows, result_row

_FURNAS = AlignedPair("Furnas", 3, 7)


def _frame(
    entity_ids: Sequence[int],
    stages: Sequence[int],
    **variables: Sequence[float | None],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": pl.Series(entity_ids, dtype=pl.Int64),
            "stage_id": pl.Series(stages, dtype=pl.Int64),
            **{
                name: pl.Series(values, dtype=pl.Float64)
                for name, values in variables.items()
            },
        }
    )


def _furnas_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    baseline = _frame([3, 3], [0, 1], generation_mw=[100.0, 110.0])
    candidate = _frame([7, 7], [0, 1], generation_mw=[90.0, 110.0])
    return baseline, candidate


class TestRowFields:
    def test_two_stages_give_two_rows_in_stage_order(self) -> None:
        baseline, candidate = _furnas_frames()

        rows = paired_stage_rows(
            "hydro", baseline, candidate, [_FURNAS], ["generation_mw"], {0, 1}
        )

        assert rows == [
            ResultComparison(
                entity_type="hydro",
                entity_name="Furnas",
                newave_code=3,
                novomodelo_id=7,
                stage=0,
                variable="generation_mw",
                newave_value=100.0,
                novomodelo_value=90.0,
                abs_diff=10.0,
                rel_diff=0.1,
            ),
            ResultComparison(
                entity_type="hydro",
                entity_name="Furnas",
                newave_code=3,
                novomodelo_id=7,
                stage=1,
                variable="generation_mw",
                newave_value=110.0,
                novomodelo_value=110.0,
                abs_diff=0.0,
                rel_diff=0.0,
            ),
        ]

    def test_the_baseline_value_takes_the_newave_slot(self) -> None:
        baseline = _frame([3], [0], generation_mw=[100.0])
        candidate = _frame([7], [0], generation_mw=[60.0])

        (row,) = paired_stage_rows(
            "thermal", baseline, candidate, [_FURNAS], ["generation_mw"], [0]
        )

        assert (row.newave_value, row.novomodelo_value) == (100.0, 60.0)
        assert (row.newave_code, row.novomodelo_id) == (3, 7)
        assert row.entity_type == "thermal"
        assert row.rel_diff == pytest.approx(0.4)

    def test_a_result_row_fills_each_slot_from_its_own_argument(self) -> None:
        row = result_row("hydro", "Furnas", 3, 7, 2, "generation_mw", 100.0, 60.0)

        assert row == ResultComparison(
            entity_type="hydro",
            entity_name="Furnas",
            newave_code=3,
            novomodelo_id=7,
            stage=2,
            variable="generation_mw",
            newave_value=100.0,
            novomodelo_value=60.0,
            abs_diff=40.0,
            rel_diff=0.4,
        )

    def test_the_stage_filter_selects_the_stages_asked_for(self) -> None:
        baseline, candidate = _furnas_frames()

        rows = paired_stage_rows(
            "hydro", baseline, candidate, [_FURNAS], ["generation_mw"], {1}
        )

        assert [(row.stage, row.abs_diff) for row in rows] == [(1, 0.0)]

    def test_a_stage_neither_frame_holds_builds_no_row(self) -> None:
        baseline, candidate = _furnas_frames()

        rows = paired_stage_rows(
            "hydro", baseline, candidate, [_FURNAS], ["generation_mw"], {5}
        )

        assert rows == []

    def test_a_repeated_stage_builds_one_row(self) -> None:
        baseline, candidate = _furnas_frames()

        rows = paired_stage_rows(
            "hydro", baseline, candidate, [_FURNAS], ["generation_mw"], [1, 1]
        )

        assert [row.stage for row in rows] == [1]


class TestMissingValues:
    def test_a_null_and_a_nan_build_no_row(self) -> None:
        baseline = _frame([3, 3], [0, 1], generation_mw=[100.0, float("nan")])
        candidate = _frame([7, 7], [0, 1], generation_mw=[None, 110.0])

        rows = paired_stage_rows(
            "hydro", baseline, candidate, [_FURNAS], ["generation_mw"], {0, 1}
        )

        assert rows == []

    @pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("on_baseline", [True, False])
    def test_a_non_finite_value_on_either_side_builds_no_row(
        self, value: float | None, on_baseline: bool
    ) -> None:
        bad = _frame([3], [0], generation_mw=[value])
        good = _frame([7], [0], generation_mw=[50.0])
        baseline, candidate = (bad, good) if on_baseline else (good, bad)
        pair = (
            AlignedPair("Furnas", 3, 7) if on_baseline else AlignedPair("Furnas", 7, 3)
        )

        rows = paired_stage_rows(
            "hydro", baseline, candidate, [pair], ["generation_mw"], [0]
        )

        assert rows == []

    def test_a_missing_value_is_never_read_as_a_zero(self) -> None:
        baseline = _frame([3], [0], generation_mw=[None])
        candidate = _frame([7], [0], generation_mw=[0.0])

        rows = paired_stage_rows(
            "hydro", baseline, candidate, [_FURNAS], ["generation_mw"], [0]
        )

        assert rows == []

    def test_a_true_zero_on_both_sides_builds_a_row(self) -> None:
        baseline = _frame([3], [0], generation_mw=[0.0])
        candidate = _frame([7], [0], generation_mw=[0.0])

        (row,) = paired_stage_rows(
            "hydro", baseline, candidate, [_FURNAS], ["generation_mw"], [0]
        )

        assert (row.abs_diff, row.rel_diff) == (0.0, None)

    def test_one_missing_variable_leaves_the_others_of_the_point(self) -> None:
        baseline = _frame([3], [0], generation_mw=[100.0], spillage_m3s=[None])
        candidate = _frame([7], [0], generation_mw=[80.0], spillage_m3s=[5.0])

        rows = paired_stage_rows(
            "hydro",
            baseline,
            candidate,
            [_FURNAS],
            ["spillage_m3s", "generation_mw"],
            [0],
        )

        assert [row.variable for row in rows] == ["generation_mw"]

    def test_a_pair_missing_from_one_frame_builds_no_row(self) -> None:
        baseline, candidate = _furnas_frames()
        absent_from_candidate = AlignedPair("Itaipu", 3, 99)
        absent_from_baseline = AlignedPair("Tucurui", 99, 7)

        rows = paired_stage_rows(
            "hydro",
            baseline,
            candidate,
            [absent_from_candidate, absent_from_baseline],
            ["generation_mw"],
            {0, 1},
        )

        assert rows == []

    def test_a_stage_held_by_one_side_only_builds_no_row_there(self) -> None:
        baseline = _frame([3, 3], [0, 1], generation_mw=[100.0, 110.0])
        candidate = _frame([7], [1], generation_mw=[100.0])

        rows = paired_stage_rows(
            "hydro", baseline, candidate, [_FURNAS], ["generation_mw"], {0, 1}
        )

        assert [row.stage for row in rows] == [1]


class TestOrderAndPairing:
    def test_rows_follow_pairs_then_stages_then_variables(self) -> None:
        baseline = _frame(
            [1, 1, 2, 2],
            [0, 1, 0, 1],
            generation_mw=[10.0, 11.0, 20.0, 21.0],
            spillage_m3s=[1.0, 2.0, 3.0, 4.0],
        )
        candidate = _frame(
            [8, 8, 9, 9],
            [0, 1, 0, 1],
            generation_mw=[10.0, 11.0, 20.0, 21.0],
            spillage_m3s=[1.0, 2.0, 3.0, 4.0],
        )
        pairs = [AlignedPair("Zeta", 2, 9), AlignedPair("Alpha", 1, 8)]

        rows = paired_stage_rows(
            "hydro",
            baseline,
            candidate,
            pairs,
            ["spillage_m3s", "generation_mw"],
            {1, 0},
        )

        assert [(row.entity_name, row.stage, row.variable) for row in rows] == [
            ("Zeta", 0, "spillage_m3s"),
            ("Zeta", 0, "generation_mw"),
            ("Zeta", 1, "spillage_m3s"),
            ("Zeta", 1, "generation_mw"),
            ("Alpha", 0, "spillage_m3s"),
            ("Alpha", 0, "generation_mw"),
            ("Alpha", 1, "spillage_m3s"),
            ("Alpha", 1, "generation_mw"),
        ]

    def test_frames_are_never_joined_on_entity_id(self) -> None:
        baseline = _frame([1, 2], [0, 0], generation_mw=[100.0, 200.0])
        candidate = _frame([1, 2], [0, 0], generation_mw=[200.0, 100.0])
        pairs = [AlignedPair("A", 1, 2), AlignedPair("B", 2, 1)]

        rows = paired_stage_rows(
            "thermal", baseline, candidate, pairs, ["generation_mw"], [0]
        )

        assert [(row.entity_name, row.abs_diff) for row in rows] == [
            ("A", 0.0),
            ("B", 0.0),
        ]

    def test_only_the_requested_variables_are_read(self) -> None:
        baseline = _frame([3], [0], generation_mw=[100.0], spillage_m3s=[1.0])
        candidate = _frame([7], [0], generation_mw=[90.0], spillage_m3s=[2.0])

        rows = paired_stage_rows(
            "hydro", baseline, candidate, [_FURNAS], ["spillage_m3s"], [0]
        )

        assert [(row.variable, row.newave_value) for row in rows] == [
            ("spillage_m3s", 1.0)
        ]


class TestDifferences:
    def test_identical_frames_give_zero_differences_across_distinct_ids(self) -> None:
        values = [0.1 + 0.2, 1e-9, 123456.789, 3.0e12]
        baseline = _frame([3] * 4, [0, 1, 2, 3], generation_mw=values)
        candidate = _frame([7] * 4, [0, 1, 2, 3], generation_mw=values)

        rows = paired_stage_rows(
            "hydro", baseline, candidate, [_FURNAS], ["generation_mw"], range(4)
        )

        assert len(rows) == 4
        assert {row.abs_diff for row in rows} == {0.0}
        assert {row.rel_diff for row in rows} == {0.0}

    def test_a_zero_baseline_has_no_relative_difference(self) -> None:
        baseline = _frame([3, 3], [0, 1], generation_mw=[0.0, 0.0])
        candidate = _frame([7, 7], [0, 1], generation_mw=[0.0, 4.0])

        rows = paired_stage_rows(
            "hydro", baseline, candidate, [_FURNAS], ["generation_mw"], {0, 1}
        )

        assert [(row.abs_diff, row.rel_diff) for row in rows] == [
            (0.0, None),
            (4.0, None),
        ]


class TestInvalidCalls:
    @pytest.mark.parametrize("missing_from", ["both", "baseline", "candidate"])
    def test_a_variable_missing_from_a_frame_raises_naming_it_and_the_type(
        self, missing_from: str
    ) -> None:
        baseline, candidate = _furnas_frames()
        if missing_from in ("both", "baseline"):
            baseline = baseline.rename({"generation_mw": "other_mw"})
        if missing_from in ("both", "candidate"):
            candidate = candidate.rename({"generation_mw": "other_mw"})

        with pytest.raises(ValueError, match="generation_mw") as excinfo:
            paired_stage_rows(
                "hydro", baseline, candidate, [_FURNAS], ["generation_mw"], {0, 1}
            )

        assert "hydro" in str(excinfo.value)

    @pytest.mark.parametrize("side", ["baseline", "candidate"])
    def test_two_rows_for_one_entity_and_stage_raise_naming_the_type_and_the_side(
        self, side: str
    ) -> None:
        baseline, candidate = _furnas_frames()
        repeated = _frame([3, 3], [0, 0], generation_mw=[100.0, 105.0])
        if side == "baseline":
            baseline = repeated
        else:
            candidate = _frame([7, 7], [0, 0], generation_mw=[90.0, 95.0])

        with pytest.raises(ValueError, match="thermal") as excinfo:
            paired_stage_rows(
                "thermal", baseline, candidate, [_FURNAS], ["generation_mw"], {0, 1}
            )

        assert f"the {side} frame" in str(excinfo.value)

    def test_two_rows_for_one_entity_and_stage_raise_for_an_entity_no_pair_names(
        self,
    ) -> None:
        baseline = _frame([3, 9, 9], [0, 0, 0], generation_mw=[100.0, 1.0, 2.0])
        candidate = _frame([7], [0], generation_mw=[90.0])

        with pytest.raises(ValueError, match="baseline"):
            paired_stage_rows(
                "thermal", baseline, candidate, [_FURNAS], ["generation_mw"], {0}
            )

    def test_one_entity_in_two_stages_and_two_entities_in_one_stage_are_not_duplicates(
        self,
    ) -> None:
        baseline = _frame([3, 3, 4], [0, 1, 0], generation_mw=[1.0, 2.0, 3.0])
        candidate = _frame([7, 7, 8], [0, 1, 0], generation_mw=[1.0, 2.0, 3.0])

        rows = paired_stage_rows(
            "thermal", baseline, candidate, [_FURNAS], ["generation_mw"], {0, 1}
        )

        assert [row.stage for row in rows] == [0, 1]

    def test_a_column_the_frames_lack_raises_even_with_no_pairs(self) -> None:
        baseline, candidate = _furnas_frames()

        with pytest.raises(ValueError, match="turbined_m3s") as excinfo:
            paired_stage_rows("hydro", baseline, candidate, [], ["turbined_m3s"], {0})

        assert "hydro" in str(excinfo.value)


class TestEmptyInputs:
    def test_empty_frames_give_no_rows(self) -> None:
        empty = _frame([], [], generation_mw=[])

        assert (
            paired_stage_rows("hydro", empty, empty, [_FURNAS], ["generation_mw"], {0})
            == []
        )

    def test_no_pairs_give_no_rows(self) -> None:
        baseline, candidate = _furnas_frames()

        assert (
            paired_stage_rows("hydro", baseline, candidate, [], ["generation_mw"], {0})
            == []
        )

    def test_no_variables_give_no_rows(self) -> None:
        baseline, candidate = _furnas_frames()

        assert paired_stage_rows("hydro", baseline, candidate, [_FURNAS], [], {0}) == []

    def test_no_stages_give_no_rows(self) -> None:
        baseline, candidate = _furnas_frames()

        assert (
            paired_stage_rows(
                "hydro", baseline, candidate, [_FURNAS], ["generation_mw"], set()
            )
            == []
        )

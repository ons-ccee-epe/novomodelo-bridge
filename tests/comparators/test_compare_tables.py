"""Unit tests for the compare report's per-plant gap tables, folded tables and
training costs."""

from __future__ import annotations

import re

import polars as pl
import pytest

from novomodelo_bridge.comparators.charts._shared import _plant_gap_table
from novomodelo_bridge.comparators.charts.costs import PairedCost, training_costs
from novomodelo_bridge.comparators.model import ResultComparison
from novomodelo_bridge.ui.html.document import folded_table

_VARIABLES = [
    ("generation_mw", "Generation (MW)"),
    ("storage_final_hm3", "Storage (hm³)"),
    ("water_value_per_hm3", "Water value (R$/hm³)"),
]


def _result(
    name: str, code: int, variable: str, ref: float, novomodelo: float, stage: int = 0
) -> ResultComparison:
    return ResultComparison(
        entity_type="hydro",
        entity_name=name,
        newave_code=code,
        novomodelo_id=code,
        stage=stage,
        variable=variable,
        newave_value=ref,
        novomodelo_value=novomodelo,
        abs_diff=abs(novomodelo - ref),
        rel_diff=abs(novomodelo - ref) / ref if ref else None,
    )


def _plant_rows(html: str) -> list[str]:
    return re.findall(r'<tr class="gap-row" data-plant="([^"]*)"', html)


def _cells(html: str, plant: str) -> list[str]:
    row = re.search(rf'data-plant="{re.escape(plant)}".*?</tr>', html)
    assert row is not None, plant
    return re.findall(r'<td class="cb-num([^"]*)"[^>]*>([^<]*)</td>', row.group(0))


class TestPlantGapTable:
    def test_cells_are_tinted_by_band_and_plants_ranked_worst_first(self) -> None:
        results = [
            _result("ALPHA", 1, "generation_mw", 100.0, 101.0),
            _result("ALPHA", 1, "storage_final_hm3", 500.0, 500.0),
            _result("BETA", 2, "generation_mw", 100.0, 150.0),
        ]

        html = _plant_gap_table(results, "hydro", _VARIABLES, "hydro", 0.05)

        assert _plant_rows(html) == ["2_BETA", "1_ALPHA"]
        assert _cells(html, "1_ALPHA") == [(" gap-close", "1%"), (" gap-close", "0%")]
        # BETA has no storage row: a dash that sorts below every gap.
        assert _cells(html, "2_BETA") == [(" gap-far", "50%"), ("", "—")]
        assert 'data-gap="-1"' in html
        # The water value column is left out when no plant has one.
        assert "Water value" not in html

    def test_all_plants_row_pools_every_plant(self) -> None:
        results = [
            _result("ALPHA", 1, "generation_mw", 100.0, 101.0),
            _result("BETA", 2, "generation_mw", 100.0, 120.0),
        ]

        html = _plant_gap_table(results, "hydro", _VARIABLES[:1], "hydro", 0.05)

        total = re.search(r'<tbody class="gap-total">(.*?)</tbody>', html)
        assert total is not None
        assert "All plants" in total.group(1)
        # (1 + 20) / 200
        assert '<td class="cb-num gap-moderate" data-gap="0.105"' in total.group(1)

    def test_a_far_water_value_does_not_rank_a_plant_first(self) -> None:
        results = [
            _result("ALPHA", 1, "generation_mw", 100.0, 100.0),
            _result("ALPHA", 1, "water_value_per_hm3", 100.0, 900.0),
            _result("BETA", 2, "generation_mw", 100.0, 110.0),
            _result("BETA", 2, "water_value_per_hm3", 100.0, 100.0),
        ]

        html = _plant_gap_table(results, "hydro", _VARIABLES, "hydro", 0.05)

        assert _plant_rows(html) == ["2_BETA", "1_ALPHA"]
        assert "Water value" in html

    def test_a_zero_reference_plant_is_judged_against_a_typical_plant(self) -> None:
        results = [
            _result("ALPHA", 1, "generation_mw", 0.0, 0.5),
            _result("BETA", 2, "generation_mw", 100.0, 100.0),
        ]

        html = _plant_gap_table(results, "hydro", _VARIABLES, "hydro", 0.05)

        # 0.5 over the mean plant's 50 instead of an undefined gap.
        assert _cells(html, "1_ALPHA") == [(" gap-close", "1%")]
        assert "∞" not in html

    def test_a_variable_zero_for_every_plant_stays_undefined(self) -> None:
        results = [
            _result("ALPHA", 1, "generation_mw", 0.0, 0.5),
            _result("BETA", 2, "generation_mw", 0.0, 0.0),
        ]

        html = _plant_gap_table(results, "hydro", _VARIABLES, "hydro", 0.05)

        assert _plant_rows(html) == ["1_ALPHA", "2_BETA"]
        assert _cells(html, "1_ALPHA") == [(" gap-far", "∞")]
        assert _cells(html, "2_BETA") == [(" gap-close", "0%")]

    def test_rows_open_the_plant_charts_of_the_prefix(self) -> None:
        html = _plant_gap_table(
            [_result("A<B", 7, "generation_mw", 1.0, 1.0)],
            "hydro",
            _VARIABLES,
            "hydro",
            0.05,
            reference_label="DECOMP",
        )

        assert 'data-prefix="hydro"' in html
        assert _plant_rows(html) == ["7_A&lt;B"]
        assert 'onclick="gapTableOpen(this)"' in html
        assert "function gapTableOpen" in html
        assert (
            "Σ|Novomodelo − DECOMP| / Σ|DECOMP|, Σ|DECOMP| no smaller than the "
            "mean plant's"
        ) in html
        assert "close up to 5%, far beyond 20%" in html

    def test_bands_follow_the_tolerance(self) -> None:
        html = _plant_gap_table(
            [_result("ALPHA", 1, "generation_mw", 100.0, 103.0)],
            "hydro",
            _VARIABLES,
            "hydro",
            0.01,
        )

        assert _cells(html, "1_ALPHA") == [(" gap-moderate", "3%")]
        assert "close up to 1%, far beyond 4%" in html

    def test_no_plant_no_table(self) -> None:
        assert _plant_gap_table([], "hydro", _VARIABLES, "hydro", 0.05) == ""


class TestFoldedTable:
    def test_summary_then_shown_then_the_folded_full_table(self) -> None:
        html = folded_table(
            "2 of 3 plants match <REF>",
            "<table>one</table>",
            "<table>all</table>",
            "All 3",
        )

        assert html == (
            '<p class="fold-summary">2 of 3 plants match &lt;REF&gt;</p>'
            "<table>one</table>"
            '<details class="fold"><summary>All 3</summary><table>all</table></details>'
        )


class TestTrainingCosts:
    def test_final_lower_bound_and_simulated_mean_cost(self) -> None:
        conv = pl.DataFrame(
            {"iteration": [2, 1, 3], "lower_bound": [20.0, 10.0, None]},
            schema={"iteration": pl.Int64, "lower_bound": pl.Float64},
        )

        rows = training_costs(
            conv, None, {"deficit": 1.0, "thermal": 3.0}, {"deficit": 2.0}
        )

        assert rows == (
            PairedCost("Final lower bound", 20.0, None),
            PairedCost("Simulated mean cost", 4.0, 2.0),
        )

    def test_a_row_neither_model_reports_is_left_out(self) -> None:
        assert training_costs(None, None, {"thermal": 1.0}, {}) == (
            PairedCost("Simulated mean cost", 1.0, None),
        )
        assert training_costs(None, None, None, None) == ()

    def test_paired_cost_difference(self) -> None:
        cost = PairedCost("x", 200.0, 230.0)

        assert cost.diff == pytest.approx(30.0)
        assert cost.pct == pytest.approx(15.0)

    def test_paired_cost_percent_undefined_near_a_zero_reference(self) -> None:
        assert PairedCost("x", 0.0, 5.0).pct is None
        assert PairedCost("x", None, 5.0).diff is None

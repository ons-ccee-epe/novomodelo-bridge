"""Unit tests for the operative-data additions to ``compare results``.

Covers per-plant extras + bounds, system spillage in MWmes, and line interchange
comparison — without spinning up a real Novomodelo or the source model
case.
"""

from __future__ import annotations

import polars as pl
import pytest

from novomodelo_bridge.comparators.model import ResultComparison
from novomodelo_bridge.comparators.newave.alignment import EntityAlignment, LineEntity
from novomodelo_bridge.comparators.newave.results import (
    _build_gen_max_overlay,
    _compare_lines,
    _compare_system_spillage,
)

# ---------------------------------------------------------------------------
# system spillage comparison
# ---------------------------------------------------------------------------


class TestCompareSystemSpillage:
    def test_emits_three_variables_per_stage(self) -> None:
        nw_sin = pl.DataFrame(
            {
                "newave_code": [0, 0, 0],
                "stage": [9, 9, 9],
                "variable": ["VERTOT", "VERTcont", "VERTfio"],
                "value": [1000.0, 800.0, 200.0],
            }
        )
        novomodelo_spill = pl.DataFrame(
            {
                "stage_id": [0],
                "total_mw": [950.0],
                "reservoir_mw": [800.0],
                "rorov_mw": [150.0],
            }
        )
        out = _compare_system_spillage(nw_sin, novomodelo_spill)
        variables = {r.variable for r in out}
        assert variables == {
            "spill_energy_total_mw",
            "spill_energy_reservoir_mw",
            "spill_energy_rorov_mw",
        }
        assert all(r.entity_type == "system_spillage" for r in out)
        assert all(r.stage == 0 for r in out)
        total = next(r for r in out if r.variable == "spill_energy_total_mw")
        assert total.newave_value == 1000.0
        assert total.novomodelo_value == 950.0
        assert total.abs_diff == pytest.approx(50.0)

    def test_empty_inputs_return_no_rows(self) -> None:
        empty_sin = pl.DataFrame(
            schema={
                "newave_code": pl.Int64,
                "stage": pl.Int64,
                "variable": pl.Utf8,
                "value": pl.Float64,
            }
        )
        empty_novomodelo = pl.DataFrame(
            schema={
                "stage_id": pl.Int64,
                "total_mw": pl.Float64,
                "reservoir_mw": pl.Float64,
                "rorov_mw": pl.Float64,
            }
        )
        assert _compare_system_spillage(empty_sin, empty_novomodelo) == []


# ---------------------------------------------------------------------------
# GHMAX_FPHC overlay builder
# ---------------------------------------------------------------------------


class TestGenMaxOverlay:
    def test_joins_source_model_ghmax_with_novomodelo_lp_max(self) -> None:
        nw_hydro = pl.DataFrame(
            {
                "newave_code": [1, 1, 2],
                "stage": [9, 10, 9],
                "variable": ["GHMAX_FPHC", "GHMAX_FPHC", "GHMAX_FPHC"],
                "value": [500.0, 480.0, 200.0],
            }
        )
        novomodelo_lp = pl.DataFrame(
            {
                "entity_id": [0, 0, 1],
                "stage_id": [0, 1, 0],
                "novomodelo_lp_gen_max_mw": [510.0, 490.0, 205.0],
            }
        )
        nw_names = {1: "PLANT_A", 2: "PLANT_B"}
        novomodelo_meta = {0: {"name": "Plant_A"}, 1: {"name": "Plant_B"}}
        out = _build_gen_max_overlay(
            nw_hydro, novomodelo_lp, nw_names, novomodelo_meta, 9
        )
        assert {
            "entity_id",
            "stage_id",
            "nw_ghmax_fphc_mw",
            "novomodelo_lp_gen_max_mw",
        } <= set(out.columns)
        # Verify Plant_A stage 0 row carries both sides.
        row = out.filter((pl.col("entity_id") == 0) & (pl.col("stage_id") == 0)).row(
            0, named=True
        )
        assert row["nw_ghmax_fphc_mw"] == 500.0
        assert row["novomodelo_lp_gen_max_mw"] == 510.0


# ---------------------------------------------------------------------------
# line comparison engine
# ---------------------------------------------------------------------------


class TestCompareLines:
    def test_directional_pair_match_emits_rows(self) -> None:
        nw_intercambio = pl.DataFrame(
            {
                "from_submarket_code": [1, 1, 2],
                "to_submarket_code": [2, 2, 1],
                "from_name": ["SE", "SE", "S"],
                "to_name": ["S", "S", "SE"],
                "stage": [9, 10, 9],
                "variable": ["INTERC", "INTERC", "INTERC"],
                "value": [100.0, -50.0, 25.0],
            }
        )
        novomodelo_line = pl.DataFrame(
            {
                "entity_id": [0, 0],
                "stage_id": [0, 1],
                "net_flow_mw": [99.5, -48.0],
            }
        )
        alignment = EntityAlignment(
            lines=[
                LineEntity(
                    novomodelo_line_id=0,
                    name="SE-S",
                    source_bus_id=0,
                    target_bus_id=1,
                    newave_de=1,
                    newave_para=2,
                ),
            ],
        )
        out = _compare_lines(nw_intercambio, novomodelo_line, alignment, nw_offset=9)
        assert len(out) == 2
        assert all(r.entity_type == "line" for r in out)
        assert all(r.variable == "net_flow_mw" for r in out)
        stages = sorted(r.stage for r in out)
        assert stages == [0, 1]
        # The (2 → 1) the source model row should NOT match since the alignment only
        # contains (1 → 2) — directional match per the design. Confirm by checking
        # newave values come from the (1 → 2) rows only.
        nw_vals = {r.stage: r.newave_value for r in out}
        assert nw_vals[0] == 100.0
        assert nw_vals[1] == -50.0

    def test_pre_study_stages_filtered_via_offset(self) -> None:
        # int*.out emits Jan-Aug pre-study with zero values; offset = 9
        # must shift them to stage_0based < 0 so they are dropped.
        nw_intercambio = pl.DataFrame(
            {
                "from_submarket_code": [1, 1, 1],
                "to_submarket_code": [2, 2, 2],
                "from_name": ["SE", "SE", "SE"],
                "to_name": ["S", "S", "S"],
                "stage": [1, 8, 9],
                "variable": ["INTERC", "INTERC", "INTERC"],
                "value": [0.0, 0.0, 100.0],
            }
        )
        novomodelo_line = pl.DataFrame(
            {
                "entity_id": [0],
                "stage_id": [0],
                "net_flow_mw": [99.0],
            }
        )
        alignment = EntityAlignment(
            lines=[
                LineEntity(
                    novomodelo_line_id=0,
                    name="SE-S",
                    source_bus_id=0,
                    target_bus_id=1,
                    newave_de=1,
                    newave_para=2,
                ),
            ],
        )
        out = _compare_lines(nw_intercambio, novomodelo_line, alignment, nw_offset=9)
        assert len(out) == 1
        assert out[0].stage == 0
        assert out[0].newave_value == 100.0


# ---------------------------------------------------------------------------
# End-to-end smoke: HTML report contains new sections and tabs
# ---------------------------------------------------------------------------


class TestHtmlReportNewSections:
    def test_renders_new_tab_and_sections(self) -> None:
        from novomodelo_bridge.comparators.analyze import build_results_dataset
        from novomodelo_bridge.comparators.model import PercentileData
        from novomodelo_bridge.comparators.report_builder import (
            build_comparison_report,
        )

        results = [
            ResultComparison(
                entity_type="hydro",
                entity_name="PLANT_A",
                newave_code=1,
                novomodelo_id=0,
                stage=0,
                variable="evaporation_m3s",
                newave_value=5.0,
                novomodelo_value=4.8,
                abs_diff=0.2,
                rel_diff=0.04,
            ),
            ResultComparison(
                entity_type="hydro",
                entity_name="PLANT_A",
                newave_code=1,
                novomodelo_id=0,
                stage=0,
                variable="total_inflow_m3s",
                newave_value=200.0,
                novomodelo_value=195.0,
                abs_diff=5.0,
                rel_diff=0.025,
            ),
            ResultComparison(
                entity_type="hydro",
                entity_name="PLANT_A",
                newave_code=1,
                novomodelo_id=0,
                stage=0,
                variable="outflow_m3s",
                newave_value=180.0,
                novomodelo_value=178.0,
                abs_diff=2.0,
                rel_diff=0.011,
            ),
            ResultComparison(
                entity_type="hydro",
                entity_name="PLANT_A",
                newave_code=1,
                novomodelo_id=0,
                stage=0,
                variable="withdrawal_m3s",
                newave_value=10.0,
                novomodelo_value=10.0,
                abs_diff=0.0,
                rel_diff=0.0,
            ),
            ResultComparison(
                entity_type="line",
                entity_name="SE-S",
                newave_code=1,
                novomodelo_id=0,
                stage=0,
                variable="net_flow_mw",
                newave_value=100.0,
                novomodelo_value=99.0,
                abs_diff=1.0,
                rel_diff=0.01,
            ),
            ResultComparison(
                entity_type="system_spillage",
                entity_name="SIN",
                newave_code=0,
                novomodelo_id=0,
                stage=0,
                variable="spill_energy_total_mw",
                newave_value=1000.0,
                novomodelo_value=950.0,
                abs_diff=50.0,
                rel_diff=0.05,
            ),
        ]
        pctiles = PercentileData(
            novomodelo_spillage_energy=pl.DataFrame(
                {
                    "stage_id": [0],
                    "total_mw": [950.0],
                    "reservoir_mw": [800.0],
                    "rorov_mw": [150.0],
                }
            ),
            line_meta=[
                {
                    "id": 0,
                    "name": "SE-S",
                    "source_bus_id": 0,
                    "target_bus_id": 1,
                    "capacity": {"direct_mw": 1000.0, "reverse_mw": 800.0},
                }
            ],
            line_bounds=pl.DataFrame(),
        )
        dataset = build_results_dataset(results, pctiles, 0.05)
        html = build_comparison_report(dataset)
        # New tab registered
        assert "tab-network" in html
        # Network tab content
        assert "Line Net Flow" in html
        # Hydro Operation now uses per-bus facets.
        assert "Storage by Bus" in html
        # Hydro Details new variables.
        assert "Evaporation" in html
        assert "Total Inflow / QAFLUH" in html
        assert "Water Withdrawal" in html
        assert "Total Outflow" in html

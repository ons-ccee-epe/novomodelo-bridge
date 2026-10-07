"""Unit tests for novomodelo_bridge.dashboard.tabs.plants.

Covers module constants, can_render, _compute_hydro_percentiles,
_build_hydro_json, build_hydro_explorer, _compute_thermal_percentiles,
_build_thermal_json, build_thermal_explorer, and render() sub-tab structure.
Uses real polars/pandas objects; never calls render() with Plotly generation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import polars as pl

from novomodelo_bridge.dashboard.tabs.plants import (
    TAB_ID,
    TAB_LABEL,
    TAB_ORDER,
    _block_hours_dict,
    _build_hydro_json,
    _build_thermal_json,
    _compute_hydro_percentiles,
    _compute_thermal_percentiles,
    _weighted_lp_generation_bounds,
    build_hydro_explorer,
    build_thermal_explorer,
    can_render,
    render,
)

# ---------------------------------------------------------------------------
# Data factories
# ---------------------------------------------------------------------------


def _make_hydros_lf(
    n_scenarios: int = 2,
    n_stages: int = 2,
    n_blocks: int = 1,
    hydro_ids: list[int] | None = None,
) -> pl.LazyFrame:
    """Return a minimal hydros LazyFrame with all expected columns."""
    if hydro_ids is None:
        hydro_ids = [0, 1]
    rows: list[dict] = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            for block_id in range(n_blocks):
                for hydro_id in hydro_ids:
                    rows.append(
                        {
                            "scenario_id": scenario_id,
                            "stage_id": stage_id,
                            "block_id": block_id,
                            "hydro_id": hydro_id,
                            "storage_initial_hm3": 100.0 + hydro_id,
                            "storage_final_hm3": 90.0 + hydro_id,
                            "inflow_m3s": 50.0 + hydro_id,
                            "evaporation_m3s": 1.0,
                            "turbined_m3s": 30.0 + hydro_id,
                            "spillage_m3s": 5.0,
                            "generation_mw": 200.0 + hydro_id * 10,
                            "water_value_per_hm3": 150.0,
                        }
                    )
    return pl.DataFrame(rows).lazy()


def _make_bh_df(
    n_stages: int = 2,
    n_blocks: int = 1,
    hours_per_block: float = 720.0,
) -> pl.DataFrame:
    """Return a block-hours DataFrame matching _make_hydros_lf structure."""
    rows: list[dict] = []
    for stage_id in range(n_stages):
        for block_id in range(n_blocks):
            rows.append(
                {
                    "stage_id": stage_id,
                    "block_id": block_id,
                    "_bh": hours_per_block,
                }
            )
    return pl.DataFrame(rows)


def _make_block_hours(
    n_stages: int = 2,
    n_blocks: int = 1,
    hours_per_block: float = 720.0,
) -> dict[tuple[int, int], float]:
    """Return a ``(stage_id, block_id) -> hours`` map matching _make_bh_df."""
    return {
        (stage_id, block_id): hours_per_block
        for stage_id in range(n_stages)
        for block_id in range(n_blocks)
    }


def _make_hydro_meta(hydro_ids: list[int] | None = None) -> dict[int, dict]:
    """Return a minimal hydro_meta dict."""
    if hydro_ids is None:
        hydro_ids = [0, 1]
    names = {0: "Agua Vermelha", 1: "Barra Grande"}
    return {
        hid: {
            "bus_id": hid,
            "name": names.get(hid, f"Hydro {hid}"),
            "vol_max": 1000.0 + hid * 100,
            "vol_min": 10.0,
            "max_gen_mw": 300.0 + hid * 50,
            "max_gen_physical": 280.0,
            "max_turbined": 40.0,
            "productivity": 0.9,
            "downstream_id": None,
        }
        for hid in hydro_ids
    }


def _make_hydro_bounds(
    hydro_ids: list[int] | None = None, n_stages: int = 2
) -> pd.DataFrame:
    """Return a minimal hydro_bounds DataFrame."""
    if hydro_ids is None:
        hydro_ids = [0]
    rows: list[dict] = []
    for hid in hydro_ids:
        for stage_id in range(n_stages):
            rows.append(
                {
                    "hydro_id": hid,
                    "stage_id": stage_id,
                    "min_storage_hm3": 10.0,
                    "max_storage_hm3": 950.0,
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 45.0,
                    "min_spillage_m3s": 0.0,
                    "max_spillage_m3s": 500.0,
                    "min_outflow_m3s": 5.0,
                    "max_outflow_m3s": 545.0,
                }
            )
    return pd.DataFrame(rows)


def _make_lp_bounds(
    hydro_ids: list[int] | None = None, n_stages: int = 2
) -> pd.DataFrame:
    """Return a minimal lp_bounds DataFrame for hydro generation bounds."""
    if hydro_ids is None:
        hydro_ids = [0]
    rows: list[dict] = []
    for hid in hydro_ids:
        for stage_id in range(n_stages):
            rows.append(
                {
                    "entity_type_code": 0,  # hydro
                    "entity_id": hid,
                    "stage_id": stage_id,
                    "bound_type_code": 6,  # gen_min
                    "bound_value": 0.0,
                }
            )
            rows.append(
                {
                    "entity_type_code": 0,
                    "entity_id": hid,
                    "stage_id": stage_id,
                    "bound_type_code": 7,  # gen_max
                    "bound_value": 300.0,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Module constants
# ---------------------------------------------------------------------------


def test_module_constants() -> None:
    assert TAB_ID == "tab-plants"
    assert TAB_LABEL == "Plant Explorer"
    assert TAB_ORDER == 50


# ---------------------------------------------------------------------------
# 2. can_render
# ---------------------------------------------------------------------------


def test_can_render_true_when_simulation_available() -> None:
    data = MagicMock()
    data.simulation_available = True
    assert can_render(data) is True


def test_can_render_false_when_simulation_missing() -> None:
    data = MagicMock()
    data.simulation_available = False
    assert can_render(data) is False


# ---------------------------------------------------------------------------
# 3. _compute_hydro_percentiles
# ---------------------------------------------------------------------------


def test_compute_hydro_percentiles_returns_correct_columns() -> None:
    hydros_lf = _make_hydros_lf(n_scenarios=2, n_stages=2, n_blocks=1, hydro_ids=[0, 1])
    bh_df = _make_bh_df(n_stages=2, n_blocks=1)

    result = _compute_hydro_percentiles(hydros_lf, bh_df)

    assert isinstance(result, pl.DataFrame)
    # Should have hydro_id and stage_id columns
    assert "hydro_id" in result.columns
    assert "stage_id" in result.columns
    # Verify flow variable percentile columns are present
    for var in ["generation_mw", "turbined_m3s", "spillage_m3s", "evaporation_m3s"]:
        for sfx in ["p10", "p50", "p90"]:
            assert f"{var}_{sfx}" in result.columns, f"Missing column: {var}_{sfx}"
    # Verify stage-level variable percentile columns are present
    for var in ["storage_final_hm3", "inflow_m3s", "water_value_per_hm3"]:
        for sfx in ["p10", "p50", "p90"]:
            assert f"{var}_{sfx}" in result.columns, f"Missing column: {var}_{sfx}"


def test_compute_hydro_percentiles_correct_row_count() -> None:
    """2 hydros x 2 stages = 4 rows."""
    hydros_lf = _make_hydros_lf(n_scenarios=3, n_stages=2, n_blocks=1, hydro_ids=[0, 1])
    bh_df = _make_bh_df(n_stages=2, n_blocks=1)

    result = _compute_hydro_percentiles(hydros_lf, bh_df)

    assert len(result) == 4  # 2 hydros x 2 stages


def test_compute_hydro_percentiles_p50_between_p10_p90() -> None:
    """P50 must be between P10 and P90 for each row and variable."""
    hydros_lf = _make_hydros_lf(n_scenarios=5, n_stages=2, n_blocks=1, hydro_ids=[0])
    bh_df = _make_bh_df(n_stages=2, n_blocks=1)

    result = _compute_hydro_percentiles(hydros_lf, bh_df)

    for var in ["generation_mw", "storage_final_hm3"]:
        p10_col = f"{var}_p10"
        p50_col = f"{var}_p50"
        p90_col = f"{var}_p90"
        if p10_col in result.columns:
            for row in result.iter_rows(named=True):
                p10 = row[p10_col] or 0.0
                p50 = row[p50_col] or 0.0
                p90 = row[p90_col] or 0.0
                assert p10 <= p50 <= p90, (
                    f"{var}: p10={p10} <= p50={p50} <= p90={p90} failed"
                )


# ---------------------------------------------------------------------------
# 4. _build_hydro_json
# ---------------------------------------------------------------------------


def test_build_hydro_json_with_bounds_has_stor_keys() -> None:
    """Plant with hydro_bounds should have non-empty stor_min / stor_max."""
    hydro_meta = _make_hydro_meta([0, 1])
    hydros_lf = _make_hydros_lf(n_scenarios=2, n_stages=2, hydro_ids=[0, 1])
    bh_df = _make_bh_df(n_stages=2)
    percentiles = _compute_hydro_percentiles(hydros_lf, bh_df)
    stages = sorted(percentiles["stage_id"].unique().to_list())

    # Only plant 0 has bounds
    hydro_bounds = _make_hydro_bounds(hydro_ids=[0], n_stages=2)
    lp_bounds = pd.DataFrame()

    hydro_data, xlabels = _build_hydro_json(
        hydro_meta=hydro_meta,
        percentiles=percentiles,
        stages=stages,
        stage_dates={0: "2024-01-01", 1: "2024-02-01"},
        bus_names={0: "Bus-A", 1: "Bus-B"},
        hydro_bounds=hydro_bounds,
        lp_bounds=lp_bounds,
        block_hours=_make_block_hours(n_stages=2),
    )

    # Plant 0 (has bounds) — must have non-empty arrays
    entry_0 = hydro_data["0"]
    assert "stor_min" in entry_0
    assert "stor_max" in entry_0
    assert len(entry_0["stor_min"]) == len(stages)
    assert len(entry_0["stor_max"]) == len(stages)
    # Values should be non-zero (we set min=10.0, max=950.0)
    assert any(v > 0 for v in entry_0["stor_min"])
    assert any(v > 0 for v in entry_0["stor_max"])

    # Plant 1 (no bounds) — must have empty arrays
    entry_1 = hydro_data["1"]
    assert entry_1["stor_min"] == []
    assert entry_1["stor_max"] == []


def test_build_hydro_json_with_lp_bounds_has_gen_keys() -> None:
    """Plant with lp_bounds should have gen_min / gen_max arrays."""
    hydro_meta = _make_hydro_meta([0])
    hydros_lf = _make_hydros_lf(n_scenarios=2, n_stages=2, hydro_ids=[0])
    bh_df = _make_bh_df(n_stages=2)
    percentiles = _compute_hydro_percentiles(hydros_lf, bh_df)
    stages = sorted(percentiles["stage_id"].unique().to_list())

    lp_bounds = _make_lp_bounds(hydro_ids=[0], n_stages=2)

    hydro_data, _ = _build_hydro_json(
        hydro_meta=hydro_meta,
        percentiles=percentiles,
        stages=stages,
        stage_dates={},
        bus_names={0: "Bus-A"},
        hydro_bounds=pd.DataFrame(),
        lp_bounds=lp_bounds,
        block_hours=_make_block_hours(n_stages=2),
    )

    entry = hydro_data["0"]
    assert len(entry["gen_min"]) == len(stages)
    assert len(entry["gen_max"]) == len(stages)
    # gen_max should be 300.0 as set in _make_lp_bounds
    assert all(v == 300.0 for v in entry["gen_max"])


def test_build_hydro_json_sorted_alphabetically() -> None:
    """Options dict keys are sorted by plant name."""
    hydro_meta = _make_hydro_meta([0, 1])
    # Plant 0 = "Agua Vermelha", Plant 1 = "Barra Grande"
    hydros_lf = _make_hydros_lf(n_scenarios=2, n_stages=2, hydro_ids=[0, 1])
    bh_df = _make_bh_df(n_stages=2)
    percentiles = _compute_hydro_percentiles(hydros_lf, bh_df)
    stages = sorted(percentiles["stage_id"].unique().to_list())

    hydro_data, _ = _build_hydro_json(
        hydro_meta=hydro_meta,
        percentiles=percentiles,
        stages=stages,
        stage_dates={},
        bus_names={},
        hydro_bounds=pd.DataFrame(),
        lp_bounds=pd.DataFrame(),
        block_hours=_make_block_hours(n_stages=2),
    )

    assert hydro_data["0"]["name"] == "Agua Vermelha"
    assert hydro_data["1"]["name"] == "Barra Grande"


def test_build_hydro_json_outflow_is_turbined_plus_spillage() -> None:
    """Derived outflow_p50 should equal turb_p50 + spill_p50 at each stage."""
    hydro_meta = _make_hydro_meta([0])
    hydros_lf = _make_hydros_lf(n_scenarios=2, n_stages=3, hydro_ids=[0])
    bh_df = _make_bh_df(n_stages=3)
    percentiles = _compute_hydro_percentiles(hydros_lf, bh_df)
    stages = sorted(percentiles["stage_id"].unique().to_list())

    hydro_data, _ = _build_hydro_json(
        hydro_meta=hydro_meta,
        percentiles=percentiles,
        stages=stages,
        stage_dates={},
        bus_names={0: "Bus-A"},
        hydro_bounds=pd.DataFrame(),
        lp_bounds=pd.DataFrame(),
        block_hours=_make_block_hours(n_stages=3),
    )

    entry = hydro_data["0"]
    for t, sp, of in zip(entry["turb_p50"], entry["spill_p50"], entry["outflow_p50"]):
        assert abs(of - (t + sp)) < 1e-6, f"outflow {of} != turb {t} + spill {sp}"


# ---------------------------------------------------------------------------
# 5. build_hydro_explorer — empty hydro_meta
# ---------------------------------------------------------------------------


def test_build_hydro_explorer_empty_meta_returns_no_data_paragraph() -> None:
    result = build_hydro_explorer(
        hydros_lf=pl.LazyFrame(),
        hydro_meta={},
        bus_names={},
        stage_dates={},
        bh_df=pl.DataFrame({"stage_id": [], "block_id": [], "_bh": []}),
        hydro_bounds=pd.DataFrame(),
        lp_bounds=pd.DataFrame(),
    )
    assert result == "<p>No hydro plant data available.</p>"


# ---------------------------------------------------------------------------
# 6. build_hydro_explorer — normal path
# ---------------------------------------------------------------------------


def test_build_hydro_explorer_normal_contains_expected_ids() -> None:
    """HTML output must contain key IDs and init calls for JS interactivity."""
    hydros_lf = _make_hydros_lf(n_scenarios=2, n_stages=3, n_blocks=1, hydro_ids=[0, 1])
    bh_df = _make_bh_df(n_stages=3, n_blocks=1)
    hydro_meta = _make_hydro_meta([0, 1])
    hydro_bounds = _make_hydro_bounds(hydro_ids=[0], n_stages=3)
    lp_bounds = _make_lp_bounds(hydro_ids=[0], n_stages=3)

    html = build_hydro_explorer(
        hydros_lf=hydros_lf,
        hydro_meta=hydro_meta,
        bus_names={0: "Bus-A", 1: "Bus-B"},
        stage_dates={0: "2024-01-01", 1: "2024-02-01", 2: "2024-03-01"},
        bh_df=bh_df,
        hydro_bounds=hydro_bounds,
        lp_bounds=lp_bounds,
    )

    # Table structure
    assert 'id="hp-tbody"' in html
    assert 'id="hp-search"' in html

    # Chart div IDs — water balance
    assert 'id="hp-stor"' in html
    assert 'id="hp-inflow"' in html
    assert 'id="hp-spill"' in html
    assert 'id="hp-evap"' in html

    # Chart div IDs — generation
    assert 'id="hp-gen"' in html
    assert 'id="hp-turb"' in html
    assert 'id="hp-outflow"' in html
    assert 'id="hp-wv"' in html

    # JS init calls
    assert "initPlantExplorer" in html
    assert "initComparisonMode" in html
    assert "syncHover" in html


def test_build_hydro_explorer_has_two_table_rows() -> None:
    """With 2 hydro plants, tbody must contain 2 <tr> elements."""
    hydros_lf = _make_hydros_lf(n_scenarios=2, n_stages=2, hydro_ids=[0, 1])
    bh_df = _make_bh_df(n_stages=2)
    hydro_meta = _make_hydro_meta([0, 1])

    html = build_hydro_explorer(
        hydros_lf=hydros_lf,
        hydro_meta=hydro_meta,
        bus_names={0: "Bus-A", 1: "Bus-B"},
        stage_dates={},
        bh_df=bh_df,
        hydro_bounds=pd.DataFrame(),
        lp_bounds=pd.DataFrame(),
    )

    # Count <tr elements after hp-tbody opening (simple string count)
    tbody_start = html.find('id="hp-tbody"')
    tbody_end = html.find("</tbody>", tbody_start)
    tbody_html = html[tbody_start:tbody_end]
    tr_count = tbody_html.count("<tr ")
    assert tr_count == 2, f"Expected 2 rows, got {tr_count}"


def test_build_hydro_explorer_rows_sorted_alphabetically() -> None:
    """Table rows must be in alphabetical order by plant name."""
    hydros_lf = _make_hydros_lf(n_scenarios=2, n_stages=2, hydro_ids=[0, 1])
    bh_df = _make_bh_df(n_stages=2)
    hydro_meta = _make_hydro_meta([0, 1])
    # Plant 0 = "Agua Vermelha", Plant 1 = "Barra Grande"

    html = build_hydro_explorer(
        hydros_lf=hydros_lf,
        hydro_meta=hydro_meta,
        bus_names={},
        stage_dates={},
        bh_df=bh_df,
        hydro_bounds=pd.DataFrame(),
        lp_bounds=pd.DataFrame(),
    )

    pos_agua = html.find("Agua Vermelha")
    pos_barra = html.find("Barra Grande")
    assert pos_agua != -1
    assert pos_barra != -1
    assert pos_agua < pos_barra, "Agua Vermelha should appear before Barra Grande"


def test_build_hydro_explorer_bounds_json_embedded() -> None:
    """When hydro_bounds is provided, the JSON blob must contain stor_min/stor_max."""
    hydros_lf = _make_hydros_lf(n_scenarios=2, n_stages=2, hydro_ids=[0])
    bh_df = _make_bh_df(n_stages=2)
    hydro_meta = _make_hydro_meta([0])
    hydro_bounds = _make_hydro_bounds(hydro_ids=[0], n_stages=2)

    html = build_hydro_explorer(
        hydros_lf=hydros_lf,
        hydro_meta=hydro_meta,
        bus_names={0: "Bus-A"},
        stage_dates={0: "2024-01-01", 1: "2024-02-01"},
        bh_df=bh_df,
        hydro_bounds=hydro_bounds,
        lp_bounds=pd.DataFrame(),
    )

    # The JSON blob "HP = {...}" must contain the bounds keys
    assert '"stor_min"' in html
    assert '"stor_max"' in html


def test_build_hydro_explorer_empty_bounds_skips_bound_keys() -> None:
    """When hydro_bounds is empty, bound arrays for plants must be empty ([])."""
    hydros_lf = _make_hydros_lf(n_scenarios=2, n_stages=2, hydro_ids=[0])
    bh_df = _make_bh_df(n_stages=2)
    hydro_meta = _make_hydro_meta([0])

    html = build_hydro_explorer(
        hydros_lf=hydros_lf,
        hydro_meta=hydro_meta,
        bus_names={0: "Bus-A"},
        stage_dates={},
        bh_df=bh_df,
        hydro_bounds=pd.DataFrame(),
        lp_bounds=pd.DataFrame(),
    )

    # The JSON blob must still contain the key but with empty array
    assert '"stor_min":[]' in html
    assert '"stor_max":[]' in html


# ---------------------------------------------------------------------------
# Thermal data factories
# ---------------------------------------------------------------------------


def _make_thermals_lf(
    n_scenarios: int = 2,
    n_stages: int = 2,
    n_blocks: int = 1,
    thermal_ids: list[int] | None = None,
) -> pl.LazyFrame:
    """Return a minimal thermals LazyFrame with all expected columns."""
    if thermal_ids is None:
        thermal_ids = [0, 1]
    rows: list[dict] = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            for block_id in range(n_blocks):
                for thermal_id in thermal_ids:
                    rows.append(
                        {
                            "scenario_id": scenario_id,
                            "stage_id": stage_id,
                            "block_id": block_id,
                            "thermal_id": thermal_id,
                            "generation_mw": 100.0 + thermal_id * 20,
                            "generation_cost": 5000.0 + thermal_id * 500,
                            "generation_mwh": 72000.0 + thermal_id * 5000,
                        }
                    )
    return pl.DataFrame(rows).lazy()


def _make_thermal_meta(thermal_ids: list[int] | None = None) -> dict[int, dict]:
    """Return a minimal thermal_meta dict."""
    if thermal_ids is None:
        thermal_ids = [0, 1]
    names = {0: "Angra 1", 1: "Termeletrica Sul"}
    return {
        tid: {
            "bus_id": tid,
            "name": names.get(tid, f"Thermal {tid}"),
            "max_mw": 500.0 + tid * 100,
            "cost_per_mwh": 150.0 + tid * 50,
        }
        for tid in thermal_ids
    }


def _make_thermal_lp_bounds(
    thermal_ids: list[int] | None = None,
    n_stages: int = 2,
    bound_type_code: int = 7,
    bound_value: float = 500.0,
) -> pd.DataFrame:
    """Return lp_bounds rows for thermals (entity_type_code=1)."""
    if thermal_ids is None:
        thermal_ids = [0]
    rows: list[dict] = []
    for tid in thermal_ids:
        for stage_id in range(n_stages):
            rows.append(
                {
                    "entity_type_code": 1,  # thermal
                    "entity_id": tid,
                    "stage_id": stage_id,
                    "bound_type_code": bound_type_code,
                    "bound_value": bound_value,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. _compute_thermal_percentiles
# ---------------------------------------------------------------------------


def test_compute_thermal_percentiles_returns_correct_columns() -> None:
    """Output must have thermal_id, stage_id, and p10/p50/p90 for all 3 metrics."""
    thermals_lf = _make_thermals_lf(
        n_scenarios=2, n_stages=2, n_blocks=1, thermal_ids=[0, 1]
    )
    bh_df = _make_bh_df(n_stages=2, n_blocks=1)

    result = _compute_thermal_percentiles(thermals_lf, bh_df)

    assert isinstance(result, pl.DataFrame)
    assert "thermal_id" in result.columns
    assert "stage_id" in result.columns
    for var in ["generation_mw", "generation_cost", "generation_mwh"]:
        for sfx in ["p10", "p50", "p90"]:
            assert f"{var}_{sfx}" in result.columns, f"Missing column: {var}_{sfx}"


def test_compute_thermal_percentiles_correct_row_count() -> None:
    """2 thermals x 2 stages = 4 rows."""
    thermals_lf = _make_thermals_lf(
        n_scenarios=3, n_stages=2, n_blocks=1, thermal_ids=[0, 1]
    )
    bh_df = _make_bh_df(n_stages=2, n_blocks=1)

    result = _compute_thermal_percentiles(thermals_lf, bh_df)

    assert len(result) == 4  # 2 thermals x 2 stages


def test_compute_thermal_percentiles_p50_between_p10_p90() -> None:
    """P50 must be between P10 and P90 for each row."""
    thermals_lf = _make_thermals_lf(
        n_scenarios=5, n_stages=2, n_blocks=1, thermal_ids=[0]
    )
    bh_df = _make_bh_df(n_stages=2, n_blocks=1)

    result = _compute_thermal_percentiles(thermals_lf, bh_df)

    for var in ["generation_mw", "generation_cost", "generation_mwh"]:
        for row in result.iter_rows(named=True):
            p10 = row[f"{var}_p10"] or 0.0
            p50 = row[f"{var}_p50"] or 0.0
            p90 = row[f"{var}_p90"] or 0.0
            assert p10 <= p50 <= p90, (
                f"{var}: p10={p10} <= p50={p50} <= p90={p90} failed"
            )


# ---------------------------------------------------------------------------
# 8. _build_thermal_json
# ---------------------------------------------------------------------------


def test_build_thermal_json_with_lp_bounds_has_gen_max() -> None:
    """Plant 0 with gen_max LP bound; plant 1 without bound must have empty list."""
    thermal_meta = _make_thermal_meta([0, 1])
    thermals_lf = _make_thermals_lf(n_scenarios=2, n_stages=2, thermal_ids=[0, 1])
    bh_df = _make_bh_df(n_stages=2)
    percentiles = _compute_thermal_percentiles(thermals_lf, bh_df)
    stages = sorted(percentiles["stage_id"].unique().to_list())

    # Only plant 0 has LP gen_max bound
    lp_bounds = _make_thermal_lp_bounds(
        thermal_ids=[0], n_stages=2, bound_type_code=7, bound_value=450.0
    )

    thermal_data, _ = _build_thermal_json(
        thermal_meta=thermal_meta,
        percentiles=percentiles,
        stages=stages,
        stage_dates={0: "2024-01-01", 1: "2024-02-01"},
        bus_names={0: "Bus-A", 1: "Bus-B"},
        lp_bounds=lp_bounds,
        block_hours=_make_block_hours(n_stages=2),
    )

    # Plant 0 — gen_max must be populated with 450.0 values
    entry_0 = thermal_data["0"]
    assert "gen_max" in entry_0
    assert len(entry_0["gen_max"]) == len(stages)
    assert all(v == 450.0 for v in entry_0["gen_max"])

    # Plant 1 — no LP bounds, gen_max must be empty list
    entry_1 = thermal_data["1"]
    assert entry_1["gen_max"] == []


def test_build_thermal_json_sorted_alphabetically() -> None:
    """Thermal data dict keys must correspond to plants sorted by name."""
    thermal_meta = _make_thermal_meta([0, 1])
    # Plant 0 = "Angra 1", Plant 1 = "Termeletrica Sul"
    thermals_lf = _make_thermals_lf(n_scenarios=2, n_stages=2, thermal_ids=[0, 1])
    bh_df = _make_bh_df(n_stages=2)
    percentiles = _compute_thermal_percentiles(thermals_lf, bh_df)
    stages = sorted(percentiles["stage_id"].unique().to_list())

    thermal_data, _ = _build_thermal_json(
        thermal_meta=thermal_meta,
        percentiles=percentiles,
        stages=stages,
        stage_dates={},
        bus_names={},
        lp_bounds=pd.DataFrame(),
        block_hours=_make_block_hours(n_stages=2),
    )

    assert thermal_data["0"]["name"] == "Angra 1"
    assert thermal_data["1"]["name"] == "Termeletrica Sul"


def test_build_thermal_json_empty_lp_bounds() -> None:
    """With empty lp_bounds, gen_min and gen_max must be empty lists."""
    thermal_meta = _make_thermal_meta([0])
    thermals_lf = _make_thermals_lf(n_scenarios=2, n_stages=2, thermal_ids=[0])
    bh_df = _make_bh_df(n_stages=2)
    percentiles = _compute_thermal_percentiles(thermals_lf, bh_df)
    stages = sorted(percentiles["stage_id"].unique().to_list())

    thermal_data, _ = _build_thermal_json(
        thermal_meta=thermal_meta,
        percentiles=percentiles,
        stages=stages,
        stage_dates={},
        bus_names={0: "Bus-A"},
        lp_bounds=pd.DataFrame(),
        block_hours=_make_block_hours(n_stages=2),
    )

    entry = thermal_data["0"]
    assert entry["gen_min"] == []
    assert entry["gen_max"] == []


# ---------------------------------------------------------------------------
# 8b. _weighted_lp_generation_bounds — block-hours weighting
# ---------------------------------------------------------------------------


def _angra_like_lp_bounds() -> pd.DataFrame:
    """LP-bounds dump for a must-run thermal fixed per block at 568/544/516 MW.

    Mirrors novomodelo's ``bounds.parquet``: a stage-level fallback row (block_id
    NULL, from the plant's thermals.json default) plus one override row per
    block, for both gen_min (6) and gen_max (7). All blocks are overridden, so
    the stage-level 460 is never in force.
    """
    rows: list[dict] = [
        {
            "entity_type_code": 1,
            "entity_id": 0,
            "stage_id": 2,
            "block_id": None,
            "bound_type_code": 6,
            "bound_value": 460.0,
        },
        {
            "entity_type_code": 1,
            "entity_id": 0,
            "stage_id": 2,
            "block_id": None,
            "bound_type_code": 7,
            "bound_value": 460.0,
        },
    ]
    for block_id, mw in [(0, 568.0), (1, 544.0), (2, 516.0)]:
        rows.append(
            {
                "entity_type_code": 1,
                "entity_id": 0,
                "stage_id": 2,
                "block_id": block_id,
                "bound_type_code": 6,
                "bound_value": mw,
            }
        )
        rows.append(
            {
                "entity_type_code": 1,
                "entity_id": 0,
                "stage_id": 2,
                "block_id": block_id,
                "bound_type_code": 7,
                "bound_value": mw,
            }
        )
    return pd.DataFrame(rows)


def test_weighted_lp_generation_bounds_per_block_is_hours_weighted() -> None:
    """Per-block bounds collapse to the block-hours weighted mean, not last-block.

    ANGRA-1 regression: 568/544/516 MW over 24/67/77 h -> 534.5952, NOT the 516
    the old block_id-ignoring collapse produced.
    """
    lp_bounds = _angra_like_lp_bounds()
    block_hours = {(2, 0): 24.0, (2, 1): 67.0, (2, 2): 77.0}

    result = _weighted_lp_generation_bounds(lp_bounds, 1, block_hours)

    expected = (568.0 * 24 + 544.0 * 67 + 516.0 * 77) / 168.0
    assert abs(result[0][2][7] - expected) < 1e-9
    assert abs(result[0][2][6] - expected) < 1e-9
    # Must not be the naive last-block value.
    assert abs(result[0][2][7] - 516.0) > 1.0


def test_weighted_lp_generation_bounds_stage_level_passthrough() -> None:
    """A stage-level-only row (block_id NULL) is returned unchanged."""
    lp_bounds = pd.DataFrame(
        [
            {
                "entity_type_code": 1,
                "entity_id": 0,
                "stage_id": 0,
                "block_id": None,
                "bound_type_code": 7,
                "bound_value": 300.0,
            },
        ]
    )
    result = _weighted_lp_generation_bounds(lp_bounds, 1, {(0, 0): 720.0})
    assert result[0][0][7] == 300.0


def test_weighted_lp_generation_bounds_no_block_column_is_stage_level() -> None:
    """A frame with no block_id column is treated as entirely stage-level."""
    lp_bounds = _make_thermal_lp_bounds(
        thermal_ids=[0], n_stages=2, bound_type_code=7, bound_value=450.0
    )
    assert "block_id" not in lp_bounds.columns
    result = _weighted_lp_generation_bounds(lp_bounds, 1, _make_block_hours(n_stages=2))
    assert result[0][0][7] == 450.0
    assert result[0][1][7] == 450.0


def test_weighted_lp_generation_bounds_empty_frame() -> None:
    """An empty lp_bounds frame yields an empty mapping (no crash)."""
    assert _weighted_lp_generation_bounds(pd.DataFrame(), 1, {}) == {}


def test_build_thermal_json_per_block_bound_matches_weighted_generation() -> None:
    """The gen_max line must equal the block-hours weighted generation.

    End-to-end guard for the ANGRA-1 "534.5952 > 516" false violation: a
    must-run plant fixed at 568/544/516 MW per block has a 534.5952 MW
    hours-weighted dispatch, and its plotted bound must sit at the same value —
    not at the last block's 516 MW cap.
    """
    thermal_meta = _make_thermal_meta([0])
    thermals_lf = pl.DataFrame(
        [
            {
                "scenario_id": 0,
                "stage_id": 2,
                "block_id": b,
                "thermal_id": 0,
                "generation_mw": mw,
                "generation_cost": 0.0,
                "generation_mwh": 0.0,
            }
            for b, mw in [(0, 568.0), (1, 544.0), (2, 516.0)]
        ]
    ).lazy()
    bh_df = pl.DataFrame(
        {"stage_id": [2, 2, 2], "block_id": [0, 1, 2], "_bh": [24.0, 67.0, 77.0]}
    )
    percentiles = _compute_thermal_percentiles(thermals_lf, bh_df)

    thermal_data, _ = _build_thermal_json(
        thermal_meta=thermal_meta,
        percentiles=percentiles,
        stages=[2],
        stage_dates={2: "2024-01-01"},
        bus_names={0: "Bus-A"},
        lp_bounds=_angra_like_lp_bounds(),
        block_hours=_block_hours_dict(bh_df),
    )

    entry = thermal_data["0"]
    expected = (568.0 * 24 + 544.0 * 67 + 516.0 * 77) / 168.0
    assert abs(entry["gen_max"][0] - expected) < 1e-3
    assert abs(entry["gen_min"][0] - expected) < 1e-3
    # The block-hours weighted generation must not exceed its (weighted) bound.
    assert entry["gen_p50"][0] <= entry["gen_max"][0] + 1e-6
    assert abs(entry["gen_max"][0] - 516.0) > 1.0


# ---------------------------------------------------------------------------
# 9. build_thermal_explorer — empty thermal_meta
# ---------------------------------------------------------------------------


def test_build_thermal_explorer_empty_meta_returns_no_data_paragraph() -> None:
    result = build_thermal_explorer(
        thermals_lf=pl.LazyFrame(),
        thermal_meta={},
        bus_names={},
        stage_dates={},
        lp_bounds=pd.DataFrame(),
        bh_df=pl.DataFrame({"stage_id": [], "block_id": [], "_bh": []}),
    )
    assert result == "<p>No thermal plant data available.</p>"


# ---------------------------------------------------------------------------
# 10. build_thermal_explorer — normal path
# ---------------------------------------------------------------------------


def test_build_thermal_explorer_normal_contains_expected_ids() -> None:
    """HTML must contain key IDs and JS init calls for thermal interactivity."""
    thermals_lf = _make_thermals_lf(
        n_scenarios=2, n_stages=3, n_blocks=1, thermal_ids=[0, 1]
    )
    bh_df = _make_bh_df(n_stages=3, n_blocks=1)
    thermal_meta = _make_thermal_meta([0, 1])
    lp_bounds = _make_thermal_lp_bounds(
        thermal_ids=[0], n_stages=3, bound_type_code=7, bound_value=500.0
    )

    html = build_thermal_explorer(
        thermals_lf=thermals_lf,
        thermal_meta=thermal_meta,
        bus_names={0: "Bus-A", 1: "Bus-B"},
        stage_dates={0: "2024-01-01", 1: "2024-02-01", 2: "2024-03-01"},
        lp_bounds=lp_bounds,
        bh_df=bh_df,
    )

    # Table structure
    assert 'id="tt-tbody"' in html
    assert 'id="tt-search"' in html

    # Chart div IDs
    assert 'id="tt-gen"' in html
    assert 'id="tt-cost"' in html
    assert 'id="tt-energy"' in html

    # JS init calls
    assert "initPlantExplorer" in html
    assert "initComparisonMode" in html
    assert "syncHover" in html


def test_build_thermal_explorer_has_two_table_rows() -> None:
    """With 2 thermal plants, tbody must contain 2 <tr> elements."""
    thermals_lf = _make_thermals_lf(n_scenarios=2, n_stages=2, thermal_ids=[0, 1])
    bh_df = _make_bh_df(n_stages=2)
    thermal_meta = _make_thermal_meta([0, 1])

    html = build_thermal_explorer(
        thermals_lf=thermals_lf,
        thermal_meta=thermal_meta,
        bus_names={0: "Bus-A", 1: "Bus-B"},
        stage_dates={},
        lp_bounds=pd.DataFrame(),
        bh_df=bh_df,
    )

    tbody_start = html.find('id="tt-tbody"')
    tbody_end = html.find("</tbody>", tbody_start)
    tbody_html = html[tbody_start:tbody_end]
    tr_count = tbody_html.count("<tr ")
    assert tr_count == 2, f"Expected 2 rows, got {tr_count}"


def test_build_thermal_explorer_rows_sorted_alphabetically() -> None:
    """Table rows must be in alphabetical order by plant name."""
    thermals_lf = _make_thermals_lf(n_scenarios=2, n_stages=2, thermal_ids=[0, 1])
    bh_df = _make_bh_df(n_stages=2)
    thermal_meta = _make_thermal_meta([0, 1])
    # Plant 0 = "Angra 1", Plant 1 = "Termeletrica Sul"

    html = build_thermal_explorer(
        thermals_lf=thermals_lf,
        thermal_meta=thermal_meta,
        bus_names={},
        stage_dates={},
        lp_bounds=pd.DataFrame(),
        bh_df=bh_df,
    )

    pos_angra = html.find("Angra 1")
    pos_termo = html.find("Termeletrica Sul")
    assert pos_angra != -1
    assert pos_termo != -1
    assert pos_angra < pos_termo, "Angra 1 should appear before Termeletrica Sul"


# ---------------------------------------------------------------------------
# 11. render() — sub-tab structure
# ---------------------------------------------------------------------------


def test_render_contains_subtab_structure() -> None:
    """render() output must contain sub-tab group, panel IDs, and switchSubTab."""
    # Build real polars LazyFrames for both hydro and thermal
    hydros_lf = _make_hydros_lf(n_scenarios=2, n_stages=2, n_blocks=1, hydro_ids=[0])
    thermals_lf = _make_thermals_lf(
        n_scenarios=2, n_stages=2, n_blocks=1, thermal_ids=[0]
    )
    bh_df = _make_bh_df(n_stages=2, n_blocks=1)
    hydro_meta = _make_hydro_meta([0])
    thermal_meta = _make_thermal_meta([0])

    data = MagicMock()
    data.hydros_lf = hydros_lf
    data.thermals_lf = thermals_lf
    data.bh_df = bh_df
    data.hydro_meta = hydro_meta
    data.thermal_meta = thermal_meta
    data.bus_names = {0: "Bus-A"}
    data.stage_dates = {0: "2024-01-01", 1: "2024-02-01"}
    data.hydro_bounds = pd.DataFrame()
    data.lp_bounds = pd.DataFrame()

    html = render(data)

    # Sub-tab group wrapper
    assert 'data-subtab-group="plants-explorer"' in html

    # Both panel IDs
    assert 'id="plants-hydro"' in html
    assert 'id="plants-thermal"' in html

    # Hydro panel visible by default, thermal hidden
    assert 'id="plants-hydro"' in html
    assert 'style="display:block;"' in html
    assert 'style="display:none;"' in html

    # switchSubTab and initPlantExplorer are declared via REQUIRED_JS and
    # emitted once by the document shell — not inlined here.
    assert "function switchSubTab" not in html
    assert "function initPlantExplorer" not in html

    # Button onclick calls
    assert "switchSubTab('plants-hydro', 'plants-explorer')" in html
    assert "switchSubTab('plants-thermal', 'plants-explorer')" in html


# ---------------------------------------------------------------------------
# 12. Band toggle
# ---------------------------------------------------------------------------


def test_render_band_toggle_checkbox() -> None:
    """render() HTML must contain the pe-band-toggle checkbox and its label."""
    hydros_lf = _make_hydros_lf(n_scenarios=2, n_stages=2, n_blocks=1, hydro_ids=[0])
    bh_df = _make_bh_df(n_stages=2, n_blocks=1)
    hydro_meta = _make_hydro_meta([0])

    data = MagicMock()
    data.hydros_lf = hydros_lf
    data.thermals_lf = pl.LazyFrame()
    data.bh_df = bh_df
    data.hydro_meta = hydro_meta
    data.thermal_meta = {}
    data.bus_names = {0: "Bus-A"}
    data.stage_dates = {0: "2024-01-01", 1: "2024-02-01"}
    data.hydro_bounds = pd.DataFrame()
    data.lp_bounds = pd.DataFrame()

    html = render(data)

    assert 'id="pe-band-toggle"' in html
    assert "Show p10-p90" in html


def test_render_band_toggle_js_variable() -> None:
    """render() HTML must contain the _peBandVisible JS variable declaration."""
    hydros_lf = _make_hydros_lf(n_scenarios=2, n_stages=2, n_blocks=1, hydro_ids=[0])
    bh_df = _make_bh_df(n_stages=2, n_blocks=1)
    hydro_meta = _make_hydro_meta([0])

    data = MagicMock()
    data.hydros_lf = hydros_lf
    data.thermals_lf = pl.LazyFrame()
    data.bh_df = bh_df
    data.hydro_meta = hydro_meta
    data.thermal_meta = {}
    data.bus_names = {0: "Bus-A"}
    data.stage_dates = {0: "2024-01-01", 1: "2024-02-01"}
    data.hydro_bounds = pd.DataFrame()
    data.lp_bounds = pd.DataFrame()

    html = render(data)

    assert "var _peBandVisible = true" in html


def test_render_hydro_detail_respects_band_var() -> None:
    """renderHydroDetail JS body must reference _peBandVisible for band traces."""
    hydros_lf = _make_hydros_lf(n_scenarios=2, n_stages=2, n_blocks=1, hydro_ids=[0])
    bh_df = _make_bh_df(n_stages=2, n_blocks=1)
    hydro_meta = _make_hydro_meta([0])

    html = build_hydro_explorer(
        hydros_lf=hydros_lf,
        hydro_meta=hydro_meta,
        bus_names={0: "Bus-A"},
        stage_dates={0: "2024-01-01", 1: "2024-02-01"},
        bh_df=bh_df,
        hydro_bounds=pd.DataFrame(),
        lp_bounds=pd.DataFrame(),
    )

    # The renderHydroDetail function must contain _peBandVisible references
    render_start = html.find("function renderHydroDetail")
    render_end = html.find("function renderHydroComparison", render_start)
    render_body = html[render_start:render_end]
    assert "_peBandVisible" in render_body


def test_render_thermal_detail_respects_band_var() -> None:
    """renderThermalDetail JS body must reference _peBandVisible for band traces."""
    thermals_lf = _make_thermals_lf(
        n_scenarios=2, n_stages=2, n_blocks=1, thermal_ids=[0]
    )
    bh_df = _make_bh_df(n_stages=2, n_blocks=1)
    thermal_meta = _make_thermal_meta([0])

    html = build_thermal_explorer(
        thermals_lf=thermals_lf,
        thermal_meta=thermal_meta,
        bus_names={0: "Bus-A"},
        stage_dates={0: "2024-01-01", 1: "2024-02-01"},
        lp_bounds=pd.DataFrame(),
        bh_df=bh_df,
    )

    render_start = html.find("function renderThermalDetail")
    render_end = html.find("function renderThermalComparison", render_start)
    render_body = html[render_start:render_end]
    assert "_peBandVisible" in render_body

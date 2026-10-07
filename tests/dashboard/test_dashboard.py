"""Tests for the dashboard data layer and tab registry.

Covers:
- Data loader helpers in novomodelo_bridge.dashboard.data
- Tab registry (get_renderable_tabs, TAB_MODULES)
- can_render contracts for constraints and stochastic tabs
- TabModule protocol compliance for every registered module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.errors import NovomodeloOutputError
from novomodelo_bridge.dashboard.data import (
    _aggregate_timing_by_iteration,
    _correct_wall_times_from_convergence,
    _normalize_output_columns,
    entity_name,
    load_entity_metadata,
    load_hydro_bus_map,
    load_hydro_metadata,
    load_names,
    load_ncs_bus_map,
    load_stage_labels,
    load_thermal_metadata,
    resolve_hydro_bus_id,
    scan_entity,
)
from novomodelo_bridge.dashboard.tabs import (
    TAB_MODULES,
    collect_required_js,
    get_renderable_tabs,
)
from novomodelo_bridge.ui.html.document import build_html
from tests.conftest import hydro_with_group

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# load_names
# ---------------------------------------------------------------------------


def test_load_names_returns_dict_with_hydros(tmp_path: Path) -> None:
    hydros_json = {
        "hydros": [{"id": 0, "name": "Itaipu"}, {"id": 1, "name": "Tucurui"}]
    }
    _write_json(tmp_path / "system" / "hydros.json", hydros_json)

    result = load_names(tmp_path)

    assert result[("hydros", 0)] == "Itaipu"
    assert result[("hydros", 1)] == "Tucurui"


def test_load_names_missing_system_dir_returns_empty(tmp_path: Path) -> None:
    # No files written — system/ directory does not exist
    result = load_names(tmp_path)

    assert result == {}


def test_load_names_uses_id_as_fallback_when_name_absent(tmp_path: Path) -> None:
    # Arrange — item has no "name" key
    hydros_json = {"hydros": [{"id": 5}]}
    _write_json(tmp_path / "system" / "hydros.json", hydros_json)

    result = load_names(tmp_path)

    assert result[("hydros", 5)] == "5"


def test_load_names_multiple_entity_types(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "system" / "hydros.json",
        {"hydros": [{"id": 0, "name": "H0"}]},
    )
    _write_json(
        tmp_path / "system" / "buses.json",
        {"buses": [{"id": 0, "name": "B0"}]},
    )

    result = load_names(tmp_path)

    assert result[("hydros", 0)] == "H0"
    assert result[("buses", 0)] == "B0"


# ---------------------------------------------------------------------------
# load_stage_labels
# ---------------------------------------------------------------------------


def test_load_stage_labels_returns_formatted_labels(tmp_path: Path) -> None:
    stages_json = {
        "stages": [
            {"id": 0, "start_date": "2024-01-01"},
            {"id": 1, "start_date": "2024-02-01"},
        ]
    }
    _write_json(tmp_path / "stages.json", stages_json)

    result = load_stage_labels(tmp_path)

    assert result[0] == "Jan 2024"
    assert result[1] == "Feb 2024"


def test_load_stage_labels_weekly_stages_use_day_resolution(tmp_path: Path) -> None:
    """Stages sharing a calendar month (weekly week/month decks) must not
    collapse onto one month-only x-axis point — each gets a day-resolution
    label from its own start date.
    """
    stages_json = {
        "stages": [
            {"id": 0, "start_date": "2026-03-14"},
            {"id": 1, "start_date": "2026-03-21"},
            {"id": 2, "start_date": "2026-03-28"},
            {"id": 3, "start_date": "2026-04-04"},
        ]
    }
    _write_json(tmp_path / "stages.json", stages_json)

    result = load_stage_labels(tmp_path)

    # Three stages fall in March; month-only labels would all read "Mar 2026".
    assert result == {
        0: "14 Mar 2026",
        1: "21 Mar 2026",
        2: "28 Mar 2026",
        3: "04 Apr 2026",
    }
    assert len(set(result.values())) == 4


def test_load_stage_labels_missing_file_returns_empty(tmp_path: Path) -> None:
    result = load_stage_labels(tmp_path)

    assert result == {}


def test_load_stage_labels_falls_back_to_stage_id_on_invalid_date(
    tmp_path: Path,
) -> None:
    stages_json = {
        "stages": [
            {"id": 3, "start_date": "not-a-date"},
        ]
    }
    _write_json(tmp_path / "stages.json", stages_json)

    result = load_stage_labels(tmp_path)

    assert result[3] == "3"


def test_load_stage_labels_falls_back_to_stage_id_when_start_date_absent(
    tmp_path: Path,
) -> None:
    stages_json = {"stages": [{"id": 7}]}
    _write_json(tmp_path / "stages.json", stages_json)

    result = load_stage_labels(tmp_path)

    assert result[7] == "7"


# ---------------------------------------------------------------------------
# load_hydro_bus_map
# ---------------------------------------------------------------------------


def test_load_hydro_bus_map_returns_mapping(tmp_path: Path) -> None:
    """0.13-shaped hydros (unit_groups[].bus_id, no top-level bus_id)
    resolve to the same bus ids the pre-0.13 top-level ``bus_id`` field
    produced, proving the relocation is value-preserving."""
    hydros_json = {
        "hydros": [
            hydro_with_group(0, 10),
            hydro_with_group(1, 20),
        ]
    }
    _write_json(tmp_path / "system" / "hydros.json", hydros_json)

    result = load_hydro_bus_map(tmp_path)

    assert result == {0: 10, 1: 20}


def test_load_hydro_bus_map_missing_file_returns_empty(tmp_path: Path) -> None:
    result = load_hydro_bus_map(tmp_path)

    assert result == {}


def test_load_hydro_bus_map_omits_multi_bus_plant(tmp_path: Path) -> None:
    """A plant whose unit_groups disagree on bus is dropped from the
    map entirely -- neither collapsed onto one of its buses nor kept at a
    guessed value -- and a Diagnostic is raised naming it."""
    ambiguous = hydro_with_group(1, 10)
    ambiguous["unit_groups"].append({**ambiguous["unit_groups"][0], "bus_id": 20})
    hydros_json = {"hydros": [hydro_with_group(0, 5), ambiguous]}
    _write_json(tmp_path / "system" / "hydros.json", hydros_json)

    with dx.collect() as collected:
        result = load_hydro_bus_map(tmp_path)

    assert result == {0: 5}

    ambiguous_diags = [d for d in collected if d.code == "hydro-unit-groups-multi-bus"]
    assert len(ambiguous_diags) == 1
    assert ambiguous_diags[0].severity is dx.Severity.WARNING
    assert "1" in ambiguous_diags[0].summary
    assert "10" in ambiguous_diags[0].summary
    assert "20" in ambiguous_diags[0].summary


def test_load_hydro_bus_map_missing_unit_groups_raises_named_error(
    tmp_path: Path,
) -> None:
    """A hand-edited/pre-0.13 hydro with no unit_groups raises a typed
    error naming the plant, rather than a bare KeyError."""
    hydros_json = {"hydros": [{"id": 3, "name": "ORPHAN"}]}
    _write_json(tmp_path / "system" / "hydros.json", hydros_json)

    with pytest.raises(NovomodeloOutputError, match=r"hydro 3 \(ORPHAN\)"):
        load_hydro_bus_map(tmp_path)


def test_load_hydro_bus_map_empty_unit_groups_raises_named_error(
    tmp_path: Path,
) -> None:
    """Empty-list variant of the missing-unit_groups guard."""
    hydros_json = {"hydros": [{"id": 4, "name": "EMPTY_GROUPS", "unit_groups": []}]}
    _write_json(tmp_path / "system" / "hydros.json", hydros_json)

    with pytest.raises(NovomodeloOutputError, match=r"hydro 4 \(EMPTY_GROUPS\)"):
        load_hydro_bus_map(tmp_path)


# ---------------------------------------------------------------------------
# load_thermal_metadata
# ---------------------------------------------------------------------------


def test_load_thermal_metadata_extracts_scalar_cost(tmp_path: Path) -> None:
    thermals_json = {
        "thermals": [
            {
                "id": 0,
                "bus_id": 5,
                "name": "Gas Plant",
                "cost_per_mwh": 150.0,
                "generation": {"min_mw": 0.0, "max_mw": 100.0},
            }
        ]
    }
    _write_json(tmp_path / "system" / "thermals.json", thermals_json)

    result = load_thermal_metadata(tmp_path)

    assert result[0]["bus_id"] == 5
    assert result[0]["name"] == "Gas Plant"
    assert result[0]["max_mw"] == pytest.approx(100.0)
    assert result[0]["cost_per_mwh"] == pytest.approx(150.0)


def test_load_thermal_metadata_legacy_cost_segments(tmp_path: Path) -> None:
    thermals_json = {
        "thermals": [
            {
                "id": 2,
                "bus_id": 3,
                "name": "Coal",
                "cost_segments": [{"capacity_mw": 300.0, "cost_per_mwh": 80.0}],
                "generation": {"max_mw": 300.0},
            }
        ]
    }
    _write_json(tmp_path / "system" / "thermals.json", thermals_json)

    result = load_thermal_metadata(tmp_path)

    assert result[2]["max_mw"] == pytest.approx(300.0)
    assert result[2]["cost_per_mwh"] == pytest.approx(80.0)


def test_load_thermal_metadata_missing_file_returns_empty(tmp_path: Path) -> None:
    result = load_thermal_metadata(tmp_path)

    assert result == {}


# ---------------------------------------------------------------------------
# load_ncs_bus_map
# ---------------------------------------------------------------------------


def test_load_ncs_bus_map_returns_mapping(tmp_path: Path) -> None:
    ncs_json = {
        "non_controllable_sources": [
            {"id": 0, "bus_id": 7},
            {"id": 1, "bus_id": 8},
        ]
    }
    _write_json(tmp_path / "system" / "non_controllable_sources.json", ncs_json)

    result = load_ncs_bus_map(tmp_path)

    assert result == {0: 7, 1: 8}


def test_load_ncs_bus_map_missing_file_returns_empty(tmp_path: Path) -> None:
    result = load_ncs_bus_map(tmp_path)

    assert result == {}


# ---------------------------------------------------------------------------
# load_hydro_metadata
# ---------------------------------------------------------------------------


def test_load_hydro_metadata_extracts_fields(tmp_path: Path) -> None:
    """Bus id relocated into unit_groups[0].bus_id still resolves to the
    same value (1) the pre-0.13 top-level ``bus_id`` field produced."""
    hydros_json = {
        "hydros": [
            hydro_with_group(
                0,
                1,
                name="Belo Monte",
                reservoir={"max_storage_hm3": 5000.0, "min_storage_hm3": 100.0},
                generation={
                    "max_generation_mw": 11000.0,
                    "productivity_mw_per_m3s": 0.08,
                    "max_turbined_m3s": 150000.0,
                },
            )
        ]
    }
    _write_json(tmp_path / "system" / "hydros.json", hydros_json)

    result = load_hydro_metadata(tmp_path)

    assert result[0]["name"] == "Belo Monte"
    assert result[0]["bus_id"] == 1
    assert result[0]["vol_max"] == pytest.approx(5000.0)
    assert result[0]["vol_min"] == pytest.approx(100.0)
    assert result[0]["max_gen_mw"] == pytest.approx(11000.0)
    assert result[0]["max_turbined"] == pytest.approx(150000.0)
    assert result[0]["productivity"] == pytest.approx(0.08)


def test_load_hydro_metadata_missing_file_returns_empty(tmp_path: Path) -> None:
    result = load_hydro_metadata(tmp_path)

    assert result == {}


def test_load_hydro_metadata_omits_bus_id_for_multi_bus_plant(
    tmp_path: Path,
) -> None:
    """A plant whose unit_groups disagree on bus keeps its non-bus
    metadata (name, volumes, productivity, ...) but has no "bus_id" key --
    it is not collapsed onto one of its buses -- and a Diagnostic names it."""
    ambiguous = hydro_with_group(1, 10, name="AMBIGUOUS")
    ambiguous["unit_groups"].append({**ambiguous["unit_groups"][0], "bus_id": 20})
    hydros_json = {"hydros": [ambiguous]}
    _write_json(tmp_path / "system" / "hydros.json", hydros_json)

    with dx.collect() as collected:
        result = load_hydro_metadata(tmp_path)

    assert result[1]["name"] == "AMBIGUOUS"
    assert "bus_id" not in result[1]

    ambiguous_diags = [d for d in collected if d.code == "hydro-unit-groups-multi-bus"]
    assert len(ambiguous_diags) == 1
    assert "1" in ambiguous_diags[0].summary
    assert "AMBIGUOUS" in ambiguous_diags[0].summary


def test_load_hydro_metadata_missing_unit_groups_raises_named_error(
    tmp_path: Path,
) -> None:
    """Absent unit_groups raises the same typed error naming the plant,
    from the metadata call site too."""
    hydros_json = {
        "hydros": [
            {
                "id": 2,
                "name": "ORPHAN",
                "generation": {},
                "reservoir": {},
            }
        ]
    }
    _write_json(tmp_path / "system" / "hydros.json", hydros_json)

    with pytest.raises(NovomodeloOutputError, match=r"hydro 2 \(ORPHAN\)"):
        load_hydro_metadata(tmp_path)


def test_resolve_hydro_bus_id_is_the_single_shared_implementation(
    tmp_path: Path,
) -> None:
    """Both load_hydro_bus_map and load_hydro_metadata derive a plant's
    bus by calling the one shared helper, rather than each inlining its own
    copy of the unit_groups scan."""
    hydros_json = {"hydros": [hydro_with_group(0, 7)]}
    _write_json(tmp_path / "system" / "hydros.json", hydros_json)

    with patch(
        "novomodelo_bridge.dashboard.data.resolve_hydro_bus_id",
        wraps=resolve_hydro_bus_id,
    ) as mock_resolve:
        bus_map = load_hydro_bus_map(tmp_path)
        metadata = load_hydro_metadata(tmp_path)

    assert mock_resolve.call_count == 2
    assert bus_map[0] == metadata[0]["bus_id"] == 7


def test_load_entity_metadata_emits_multi_bus_diagnostic_once(
    tmp_path: Path,
) -> None:
    """Regression: load_entity_metadata calls both
    load_hydro_bus_map and load_hydro_metadata over the same hydros.json.
    The existing ambiguous-plant tests above call one loader at a time, so
    the double emission through the real load_entity_metadata path was
    previously untested. An ambiguous plant's
    hydro-unit-groups-multi-bus warning must fire once per case load, not
    once per loader."""
    ambiguous = hydro_with_group(1, 10, name="AMBIGUOUS")
    ambiguous["unit_groups"].append({**ambiguous["unit_groups"][0], "bus_id": 20})
    hydros_json = {"hydros": [hydro_with_group(0, 5), ambiguous]}
    _write_json(tmp_path / "system" / "hydros.json", hydros_json)

    with dx.collect() as collected:
        metadata = load_entity_metadata(tmp_path)

    assert metadata.hydro_bus_map == {0: 5}
    assert "bus_id" not in metadata.hydro_meta[1]

    ambiguous_diags = [d for d in collected if d.code == "hydro-unit-groups-multi-bus"]
    assert len(ambiguous_diags) == 1
    assert "AMBIGUOUS" in ambiguous_diags[0].summary


# ---------------------------------------------------------------------------
# entity_name
# ---------------------------------------------------------------------------


def test_entity_name_returns_name_when_key_present() -> None:
    names: dict[tuple[str, int], str] = {("hydros", 0): "Itaipu", ("buses", 5): "SE"}

    assert entity_name(names, "hydros", 0) == "Itaipu"
    assert entity_name(names, "buses", 5) == "SE"


def test_entity_name_returns_str_id_when_key_absent() -> None:
    names: dict[tuple[str, int], str] = {}

    assert entity_name(names, "hydros", 42) == "42"


# ---------------------------------------------------------------------------
# scan_entity
# ---------------------------------------------------------------------------


def test_scan_entity_calls_scan_parquet_with_correct_path(tmp_path: Path) -> None:
    entity_dir = tmp_path / "output" / "simulation" / "hydros"
    entity_dir.mkdir(parents=True)
    (entity_dir / "scenario_id=0000").mkdir()
    (entity_dir / "scenario_id=0000" / "data.parquet").write_bytes(b"")

    expected_path = str(entity_dir / "**" / "*.parquet")
    mock_lf = MagicMock(spec=pl.LazyFrame)

    with patch(
        "novomodelo_bridge.dashboard.data.pl.scan_parquet", return_value=mock_lf
    ) as mock_scan:
        result = scan_entity(tmp_path, "hydros")

    mock_scan.assert_called_once_with(expected_path, hive_partitioning=True)
    assert result is mock_lf


def test_scan_entity_returns_empty_lazyframe_when_dir_missing(tmp_path: Path) -> None:
    """scan_entity must return an empty LazyFrame when the simulation dir is absent."""
    result = scan_entity(tmp_path, "hydros")
    assert isinstance(result, pl.LazyFrame)
    assert result.collect().is_empty()


# ---------------------------------------------------------------------------
# get_renderable_tabs — ordering
# ---------------------------------------------------------------------------


def test_get_renderable_tabs_returns_tabs_sorted_by_tab_order() -> None:
    # Arrange: two mock modules with reversed ORDER values
    mock_high = MagicMock()
    mock_high.TAB_ID = "tab-high"
    mock_high.TAB_LABEL = "High"
    mock_high.TAB_ORDER = 200
    mock_high.can_render.return_value = True
    mock_high.render.return_value = "<div>high</div>"

    mock_low = MagicMock()
    mock_low.TAB_ID = "tab-low"
    mock_low.TAB_LABEL = "Low"
    mock_low.TAB_ORDER = 5
    mock_low.can_render.return_value = True
    mock_low.render.return_value = "<div>low</div>"

    fake_data = MagicMock()

    with patch("novomodelo_bridge.dashboard.tabs.TAB_MODULES", [mock_high, mock_low]):
        result = get_renderable_tabs(fake_data)

    ids = [tab_id for tab_id, _label, _html in result]
    assert ids == ["tab-low", "tab-high"]


# ---------------------------------------------------------------------------
# get_renderable_tabs — filtering
# ---------------------------------------------------------------------------


def test_get_renderable_tabs_excludes_modules_where_can_render_is_false() -> None:
    # Arrange: one renderable, one not
    mock_yes = MagicMock()
    mock_yes.TAB_ID = "tab-yes"
    mock_yes.TAB_LABEL = "Yes"
    mock_yes.TAB_ORDER = 10
    mock_yes.can_render.return_value = True
    mock_yes.render.return_value = "<div>yes</div>"

    mock_no = MagicMock()
    mock_no.TAB_ID = "tab-no"
    mock_no.TAB_LABEL = "No"
    mock_no.TAB_ORDER = 20
    mock_no.can_render.return_value = False

    fake_data = MagicMock()

    with patch("novomodelo_bridge.dashboard.tabs.TAB_MODULES", [mock_yes, mock_no]):
        result = get_renderable_tabs(fake_data)

    ids = [tab_id for tab_id, _label, _html in result]
    assert "tab-yes" in ids
    assert "tab-no" not in ids


# ---------------------------------------------------------------------------
# get_renderable_tabs — error handling
# ---------------------------------------------------------------------------


def test_get_renderable_tabs_shows_placeholder_when_render_raises() -> None:
    # Arrange: one tab raises, one succeeds. The failing tab must stay visible
    # with an error placeholder rather than vanishing.
    mock_bad = MagicMock()
    mock_bad.TAB_ID = "tab-bad"
    mock_bad.TAB_LABEL = "Bad"
    mock_bad.TAB_ORDER = 10
    mock_bad.can_render.return_value = True
    mock_bad.render.side_effect = RuntimeError("rendering <failed>")

    mock_good = MagicMock()
    mock_good.TAB_ID = "tab-good"
    mock_good.TAB_LABEL = "Good"
    mock_good.TAB_ORDER = 20
    mock_good.can_render.return_value = True
    mock_good.render.return_value = "<div>good</div>"

    fake_data = MagicMock()

    with patch("novomodelo_bridge.dashboard.tabs.TAB_MODULES", [mock_bad, mock_good]):
        result = get_renderable_tabs(fake_data)

    by_id = {tab_id: html for tab_id, _label, html in result}
    assert "tab-bad" in by_id  # not dropped
    assert "tab-good" in by_id
    assert "failed to render" in by_id["tab-bad"]
    # The exception text is HTML-escaped in the placeholder.
    assert "rendering &lt;failed&gt;" in by_id["tab-bad"]
    assert "<failed>" not in by_id["tab-bad"]


def test_get_renderable_tabs_returns_correct_tuple_structure() -> None:
    mock_mod = MagicMock()
    mock_mod.TAB_ID = "tab-x"
    mock_mod.TAB_LABEL = "X Tab"
    mock_mod.TAB_ORDER = 0
    mock_mod.can_render.return_value = True
    mock_mod.render.return_value = "<section>content</section>"

    fake_data = MagicMock()

    with patch("novomodelo_bridge.dashboard.tabs.TAB_MODULES", [mock_mod]):
        result = get_renderable_tabs(fake_data)

    assert len(result) == 1
    tab_id, label, html = result[0]
    assert tab_id == "tab-x"
    assert label == "X Tab"
    assert html == "<section>content</section>"


# ---------------------------------------------------------------------------
# collect_required_js — shared plant-explorer JS union
# ---------------------------------------------------------------------------


def test_collect_required_js_includes_stochastic_js_on_training_only_case() -> None:
    """A training-only case (Plants does not render) must still get
    PLANT_EXPLORER_JS from the Stochastic tab's own REQUIRED_JS declaration."""
    mock_stochastic = MagicMock()
    mock_stochastic.TAB_ORDER = 10
    mock_stochastic.can_render.return_value = True
    mock_stochastic.REQUIRED_JS = ["function initPlantExplorer() {}"]

    mock_plants = MagicMock()
    mock_plants.TAB_ORDER = 50
    mock_plants.can_render.return_value = False
    mock_plants.REQUIRED_JS = [
        "function switchSubTab() {}",
        "function initPlantExplorer() {}",
    ]

    fake_data = MagicMock()
    fake_data.simulation_available = False
    fake_data.stochastic_available = True

    with patch(
        "novomodelo_bridge.dashboard.tabs.TAB_MODULES", [mock_stochastic, mock_plants]
    ):
        result = collect_required_js(fake_data)

    assert "function initPlantExplorer" in result


def test_collect_required_js_dedupes_shared_block_across_tabs() -> None:
    """When both Stochastic and Plants render, the shared block is emitted once."""
    mock_stochastic = MagicMock()
    mock_stochastic.TAB_ORDER = 10
    mock_stochastic.can_render.return_value = True
    mock_stochastic.REQUIRED_JS = ["function initPlantExplorer() {}"]

    mock_plants = MagicMock()
    mock_plants.TAB_ORDER = 50
    mock_plants.can_render.return_value = True
    mock_plants.REQUIRED_JS = [
        "function switchSubTab() {}",
        "function initPlantExplorer() {}",
    ]

    fake_data = MagicMock()

    with patch(
        "novomodelo_bridge.dashboard.tabs.TAB_MODULES", [mock_stochastic, mock_plants]
    ):
        result = collect_required_js(fake_data)

    assert result.count("function initPlantExplorer") == 1
    assert result.count("function switchSubTab") == 1


def test_collect_required_js_tolerates_module_with_no_required_js_attr() -> None:
    """A module that declares no REQUIRED_JS contributes nothing (getattr default)."""
    mock_bare = MagicMock(spec=["TAB_ORDER", "can_render"])
    mock_bare.TAB_ORDER = 5
    mock_bare.can_render.return_value = True

    fake_data = MagicMock()

    with patch("novomodelo_bridge.dashboard.tabs.TAB_MODULES", [mock_bare]):
        result = collect_required_js(fake_data)

    assert result == ""


def test_collect_required_js_excludes_modules_where_can_render_is_false() -> None:
    """A non-renderable module's REQUIRED_JS is not included, even alone."""
    mock_no = MagicMock()
    mock_no.TAB_ORDER = 10
    mock_no.can_render.return_value = False
    mock_no.REQUIRED_JS = ["function shouldNotAppear() {}"]

    fake_data = MagicMock()

    with patch("novomodelo_bridge.dashboard.tabs.TAB_MODULES", [mock_no]):
        result = collect_required_js(fake_data)

    assert result == ""


# ---------------------------------------------------------------------------
# build_html — required_js shell parameter
# ---------------------------------------------------------------------------


def test_build_html_required_js_emitted_once_in_head() -> None:
    """A non-empty required_js is emitted exactly once, inside a <script> in <head>."""
    result = build_html(
        title="Test",
        tab_defs=[("tab-a", "A")],
        tab_contents={"tab-a": "<p>A</p>"},
        css="",
        js="",
        required_js="/*MARK*/",
    )

    assert result.count("/*MARK*/") == 1
    head = result.split("<head>", 1)[1].split("</head>", 1)[0]
    assert "/*MARK*/" in head
    assert "<script>" in head


def test_build_html_empty_required_js_adds_no_extra_script() -> None:
    """Empty required_js (the comparison-report path) adds no <script> to <head>."""
    result = build_html(
        title="Test",
        tab_defs=[("tab-a", "A")],
        tab_contents={"tab-a": "<p>A</p>"},
        css="",
        js="",
        required_js="",
    )

    head = result.split("<head>", 1)[1].split("</head>", 1)[0]
    # Only the plotly CDN <script> tag remains — no extra shared-JS block.
    assert head.count("<script") == 1


# ---------------------------------------------------------------------------
# TabModule protocol compliance — parametrized over all registered modules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", TAB_MODULES)
def test_tab_module_has_tab_id(module: Any) -> None:
    assert hasattr(module, "TAB_ID"), f"{module} missing TAB_ID"
    assert isinstance(module.TAB_ID, str), f"{module}.TAB_ID must be str"
    assert module.TAB_ID, f"{module}.TAB_ID must not be empty"


@pytest.mark.parametrize("module", TAB_MODULES)
def test_tab_module_has_tab_label(module: Any) -> None:
    assert hasattr(module, "TAB_LABEL"), f"{module} missing TAB_LABEL"
    assert isinstance(module.TAB_LABEL, str), f"{module}.TAB_LABEL must be str"
    assert module.TAB_LABEL, f"{module}.TAB_LABEL must not be empty"


@pytest.mark.parametrize("module", TAB_MODULES)
def test_tab_module_has_tab_order(module: Any) -> None:
    assert hasattr(module, "TAB_ORDER"), f"{module} missing TAB_ORDER"
    assert isinstance(module.TAB_ORDER, int), f"{module}.TAB_ORDER must be int"


@pytest.mark.parametrize("module", TAB_MODULES)
def test_tab_module_has_can_render_callable(module: Any) -> None:
    assert hasattr(module, "can_render"), f"{module} missing can_render"
    assert callable(module.can_render), f"{module}.can_render must be callable"


@pytest.mark.parametrize("module", TAB_MODULES)
def test_tab_module_has_render_callable(module: Any) -> None:
    assert hasattr(module, "render"), f"{module} missing render"
    assert callable(module.render), f"{module}.render must be callable"


# ---------------------------------------------------------------------------
# Integration test — full build_dashboard() pipeline
# ---------------------------------------------------------------------------


class TestDashboardIntegration:
    """Integration tests for build_dashboard() with a minimal mock case directory.

    Exercises the full pipeline: DashboardData.load(), get_renderable_tabs(),
    and build_html() assembling a complete self-contained HTML file.
    """

    @pytest.fixture()
    def case_dir(self, tmp_path: Path) -> Path:
        """Build a minimal case directory for DashboardData.load()."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        case = tmp_path / "test_case"
        case.mkdir()

        # ---- stages.json ----
        stages_data = {
            "stages": [
                {
                    "id": 0,
                    "start_date": "2026-01-01",
                    "blocks": [
                        {"id": 0, "hours": 120.0},
                        {"id": 1, "hours": 300.0},
                    ],
                },
                {
                    "id": 1,
                    "start_date": "2026-02-01",
                    "blocks": [
                        {"id": 0, "hours": 112.0},
                        {"id": 1, "hours": 280.0},
                    ],
                },
            ]
        }
        _write_json(case / "stages.json", stages_data)

        # ---- config.json ----
        _write_json(case / "config.json", {"num_scenarios": 2, "num_stages": 2})

        # ---- system/ JSON files ----
        # 0.13-shaped hydros.json (unit_groups[].bus_id, no top-level
        # bus_id) — exercises the full DashboardData.load() pipeline,
        # including load_hydro_bus_map/load_hydro_metadata, against this
        # shape end-to-end.
        _write_json(
            case / "system" / "hydros.json",
            {
                "hydros": [
                    hydro_with_group(
                        0,
                        0,
                        name="HYDRO_A",
                        reservoir={
                            "max_storage_hm3": 5000.0,
                            "min_storage_hm3": 100.0,
                        },
                        generation={
                            "max_generation_mw": 1000.0,
                            "productivity_mw_per_m3s": 0.08,
                            "max_turbined_m3s": 12000.0,
                        },
                    )
                ]
            },
        )
        _write_json(
            case / "system" / "buses.json",
            {"buses": [{"id": 0, "name": "SE"}]},
        )
        _write_json(
            case / "system" / "thermals.json",
            {
                "thermals": [
                    {
                        "id": 0,
                        "name": "GAS_A",
                        "bus_id": 0,
                        "cost_per_mwh": 200.0,
                        "generation": {"min_mw": 0.0, "max_mw": 300.0},
                    }
                ]
            },
        )
        _write_json(
            case / "system" / "lines.json",
            {"lines": []},
        )
        _write_json(
            case / "system" / "non_controllable_sources.json",
            {"non_controllable_sources": []},
        )

        # ---- scenarios/ ----
        (case / "scenarios").mkdir(parents=True, exist_ok=True)
        load_stats_table = pa.table(
            {
                "bus_id": pa.array([0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "mean_mw": pa.array([50000.0, 48000.0], type=pa.float64()),
                "std_mw": pa.array([0.0, 0.0], type=pa.float64()),
            }
        )
        pq.write_table(
            load_stats_table, case / "scenarios" / "load_seasonal_stats.parquet"
        )
        _write_json(
            case / "scenarios" / "load_factors.json",
            {
                "load_factors": [
                    {
                        "bus_id": 0,
                        "stage_id": 0,
                        "block_factors": [
                            {"block_id": 0, "factor": 1.05},
                            {"block_id": 1, "factor": 0.95},
                        ],
                    },
                    {
                        "bus_id": 0,
                        "stage_id": 1,
                        "block_factors": [
                            {"block_id": 0, "factor": 1.03},
                            {"block_id": 1, "factor": 0.97},
                        ],
                    },
                ]
            },
        )

        # ---- output/training/convergence.parquet ----
        conv_dir = case / "output" / "training"
        conv_dir.mkdir(parents=True)
        conv_table = pa.table(
            {
                "iteration": pa.array([1, 2], type=pa.int32()),
                "lower_bound": pa.array([1.0e9, 1.1e9], type=pa.float64()),
                "upper_bound_mean": pa.array([1.5e9, 1.4e9], type=pa.float64()),
                "upper_bound_std": pa.array([1.0e7, 9.0e6], type=pa.float64()),
                "gap_percent": pa.array([33.3, 21.4], type=pa.float64()),
                "cuts_added": pa.array([10, 8], type=pa.int32()),
                "cuts_removed": pa.array([0, 0], type=pa.int32()),
                "cuts_active": pa.array([10, 18], type=pa.int64()),
                "time_forward_ms": pa.array([100, 90], type=pa.int64()),
                "time_backward_ms": pa.array([200, 180], type=pa.int64()),
                "time_total_ms": pa.array([300, 270], type=pa.int64()),
                "forward_passes": pa.array([5, 5], type=pa.int32()),
                "lp_solves": pa.array([100, 90], type=pa.int64()),
            }
        )
        pq.write_table(conv_table, conv_dir / "convergence.parquet")

        # ---- Simulation entity directories (hive-partitioned) ----
        # Columns for each entity (scenario_id is inferred from directory name
        # by polars hive_partitioning=True; it must NOT be in the file columns).

        sim_base = case / "output" / "simulation"

        def _write_sim_parquet(entity: str, scenario_id: int, table: pa.Table) -> None:
            d = sim_base / entity / f"scenario_id={scenario_id:04d}"
            d.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, d / "data.parquet")

        # hydros
        hydro_table = pa.table(
            {
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "block_id": pa.array([0, 0], type=pa.int32()),
                "hydro_id": pa.array([0, 0], type=pa.int32()),
                "generation_mw": pa.array([800.0, 750.0], type=pa.float64()),
                "generation_mwh": pa.array([96000.0, 84000.0], type=pa.float64()),
                "spillage_m3s": pa.array([0.0, 0.0], type=pa.float64()),
                "turbined_m3s": pa.array([10000.0, 9500.0], type=pa.float64()),
                "storage_final_hm3": pa.array([4500.0, 4600.0], type=pa.float64()),
                "storage_initial_hm3": pa.array([4400.0, 4500.0], type=pa.float64()),
                "inflow_m3s": pa.array([500.0, 480.0], type=pa.float64()),
                "outflow_m3s": pa.array([10000.0, 9500.0], type=pa.float64()),
                "incremental_inflow_m3s": pa.array([500.0, 480.0], type=pa.float64()),
                "water_value_per_hm3": pa.array([1.5, 1.4], type=pa.float64()),
                "spillage_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "evaporation_m3s": pa.array([0.0, 0.0], type=pa.float64()),
                "productivity_mw_per_m3s": pa.array([0.08, 0.08], type=pa.float64()),
                "storage_binding_code": pa.array([0, 0], type=pa.int8()),
                "operative_state_code": pa.array([1, 1], type=pa.int8()),
                "turbined_slack_m3s": pa.array([0.0, 0.0], type=pa.float64()),
                "outflow_slack_below_m3s": pa.array([0.0, 0.0], type=pa.float64()),
                "outflow_slack_above_m3s": pa.array([0.0, 0.0], type=pa.float64()),
                "generation_slack_mw": pa.array([0.0, 0.0], type=pa.float64()),
                "storage_violation_below_hm3": pa.array([0.0, 0.0], type=pa.float64()),
                "filling_target_violation_hm3": pa.array([0.0, 0.0], type=pa.float64()),
                "diverted_inflow_m3s": pa.array([0.0, 0.0], type=pa.float64()),
                "diverted_outflow_m3s": pa.array([0.0, 0.0], type=pa.float64()),
                "evaporation_violation_pos_m3s": pa.array(
                    [0.0, 0.0], type=pa.float64()
                ),
                "evaporation_violation_neg_m3s": pa.array(
                    [0.0, 0.0], type=pa.float64()
                ),
                "inflow_nonnegativity_slack_m3s": pa.array(
                    [0.0, 0.0], type=pa.float64()
                ),
                "water_withdrawal_violation_pos_m3s": pa.array(
                    [0.0, 0.0], type=pa.float64()
                ),
                "water_withdrawal_violation_neg_m3s": pa.array(
                    [0.0, 0.0], type=pa.float64()
                ),
            }
        )
        for sid in (0, 1):
            _write_sim_parquet("hydros", sid, hydro_table)

        # thermals
        thermal_table = pa.table(
            {
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "block_id": pa.array([0, 0], type=pa.int32()),
                "thermal_id": pa.array([0, 0], type=pa.int32()),
                "generation_mw": pa.array([200.0, 210.0], type=pa.float64()),
                "generation_mwh": pa.array([24000.0, 23520.0], type=pa.float64()),
                "generation_cost": pa.array([4.8e6, 4.7e6], type=pa.float64()),
                "is_gnl": pa.array([False, False]),
                "gnl_committed_mw": pa.array([0.0, 0.0], type=pa.float64()),
                "gnl_decision_mw": pa.array([0.0, 0.0], type=pa.float64()),
                "operative_state_code": pa.array([1, 1], type=pa.int8()),
            }
        )
        for sid in (0, 1):
            _write_sim_parquet("thermals", sid, thermal_table)

        # non_controllables
        ncs_table = pa.table(
            {
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "block_id": pa.array([0, 0], type=pa.int32()),
                "non_controllable_id": pa.array([0, 0], type=pa.int32()),
                "generation_mw": pa.array([100.0, 90.0], type=pa.float64()),
                "generation_mwh": pa.array([12000.0, 10080.0], type=pa.float64()),
                "available_mw": pa.array([110.0, 100.0], type=pa.float64()),
                "curtailment_mw": pa.array([10.0, 10.0], type=pa.float64()),
                "curtailment_mwh": pa.array([1200.0, 1120.0], type=pa.float64()),
                "curtailment_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "operative_state_code": pa.array([1, 1], type=pa.int8()),
            }
        )
        for sid in (0, 1):
            _write_sim_parquet("non_controllables", sid, ncs_table)

        # buses
        bus_table = pa.table(
            {
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "block_id": pa.array([0, 0], type=pa.int32()),
                "bus_id": pa.array([0, 0], type=pa.int32()),
                "load_mw": pa.array([1000.0, 980.0], type=pa.float64()),
                "load_mwh": pa.array([120000.0, 109760.0], type=pa.float64()),
                "deficit_mw": pa.array([0.0, 0.0], type=pa.float64()),
                "deficit_mwh": pa.array([0.0, 0.0], type=pa.float64()),
                "excess_mw": pa.array([0.0, 0.0], type=pa.float64()),
                "excess_mwh": pa.array([0.0, 0.0], type=pa.float64()),
                "spot_price": pa.array([150.0, 145.0], type=pa.float64()),
            }
        )
        for sid in (0, 1):
            _write_sim_parquet("buses", sid, bus_table)

        # exchanges (empty — no lines defined)
        exchange_table = pa.table(
            {
                "stage_id": pa.array([], type=pa.int32()),
                "block_id": pa.array([], type=pa.int32()),
                "line_id": pa.array([], type=pa.int32()),
                "direct_flow_mw": pa.array([], type=pa.float64()),
                "reverse_flow_mw": pa.array([], type=pa.float64()),
                "net_flow_mw": pa.array([], type=pa.float64()),
                "net_flow_mwh": pa.array([], type=pa.float64()),
                "losses_mw": pa.array([], type=pa.float64()),
                "losses_mwh": pa.array([], type=pa.float64()),
                "exchange_cost": pa.array([], type=pa.float64()),
            }
        )
        for sid in (0, 1):
            _write_sim_parquet("exchanges", sid, exchange_table)

        # costs
        cost_table = pa.table(
            {
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "block_id": pa.array([None, None], type=pa.int32()),
                "total_cost": pa.array([5.0e9, 4.8e9], type=pa.float64()),
                "immediate_cost": pa.array([5.0e8, 4.8e8], type=pa.float64()),
                "future_cost": pa.array([4.5e9, 4.32e9], type=pa.float64()),
                "discount_factor": pa.array([1.0, 0.99], type=pa.float64()),
                "thermal_cost": pa.array([4.8e6, 4.7e6], type=pa.float64()),
                "contract_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "deficit_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "excess_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "storage_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "filling_target_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "hydro_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "outflow_violation_below_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "outflow_violation_above_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "turbined_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "generation_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "evaporation_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "withdrawal_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "inflow_penalty_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "generic_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "spillage_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "turbined_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "curtailment_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "exchange_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "pumping_cost": pa.array([0.0, 0.0], type=pa.float64()),
            }
        )
        for sid in (0, 1):
            _write_sim_parquet("costs", sid, cost_table)

        return case

    def test_build_dashboard_integration(self, case_dir: Path, tmp_path: Path) -> None:
        """build_dashboard() writes a valid HTML file with at least 3 tab sections."""
        from novomodelo_bridge.dashboard import build_dashboard

        output_path = tmp_path / "dashboard.html"

        build_dashboard(case_dir, output_path)

        assert output_path.exists(), "Dashboard HTML file was not written"
        html = output_path.read_text(encoding="utf-8")

        assert "<!DOCTYPE html>" in html

        section_count = html.count('<section id="tab-')
        assert section_count >= 3, (
            f"Expected at least 3 tab sections, found {section_count}"
        )

        assert case_dir.resolve().name in html

    def test_build_dashboard_head_includes_plotly_title_shim(
        self, case_dir: Path, tmp_path: Path
    ) -> None:
        """The dashboard <head> must carry the client-side title-normalizer shim.

        plotly.js 3.x drops a bare-string ``title`` at render; the shim wraps
        ``Plotly.newPlot``/``Plotly.react`` before any tab's chart script runs
        (see ``PLOTLY_TITLE_SHIM_JS``), so it must land in ``<head>``, not the
        end-of-body ``<script>``, which executes too late.
        """
        from novomodelo_bridge.dashboard import build_dashboard

        output_path = tmp_path / "dashboard.html"
        build_dashboard(case_dir, output_path)
        html = output_path.read_text(encoding="utf-8")

        head = html.split("<head>", 1)[1].split("</head>", 1)[0]
        assert "normalizePlotlyTitles" in head

    def test_build_dashboard_with_per_block_line_bounds(
        self, case_dir: Path, tmp_path: Path
    ) -> None:
        """build_dashboard() renders end-to-end on a 0.13 case whose
        constraints/line_bounds.parquet carries per-block (block_id
        non-null) override rows alongside the stage-level base row.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        from novomodelo_bridge.dashboard import build_dashboard
        from novomodelo_bridge.dashboard.data import DashboardData

        constraints_dir = case_dir / "constraints"
        constraints_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "line_id": pa.array([0, 0, 0], type=pa.int32()),
                    "stage_id": pa.array([0, 0, 0], type=pa.int32()),
                    "direct_mw": pa.array([100.0, 80.0, 120.0], type=pa.float64()),
                    "reverse_mw": pa.array([100.0, 80.0, 120.0], type=pa.float64()),
                    "block_id": pa.array([None, 0, 1], type=pa.int32()),
                }
            ),
            constraints_dir / "line_bounds.parquet",
        )

        data = DashboardData.load(case_dir)
        assert len(data.line_block_bounds) == 2
        assert data.line_block_bounds["block_id"].notna().all()

        output_path = tmp_path / "dashboard.html"
        build_dashboard(case_dir, output_path)

        assert output_path.exists(), "Dashboard HTML file was not written"
        html = output_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html


# ---------------------------------------------------------------------------
# DashboardData.load() — new v2 fields
#
# These tests reuse the full minimal case directory from
# TestDashboardIntegration via the module-level ``_v2_case`` fixture.
# Each test writes one optional file, calls DashboardData.load(), and
# asserts the corresponding new field.
# ---------------------------------------------------------------------------


@pytest.fixture()
def _v2_case(tmp_path: Path) -> Path:
    """Build the same minimal Novomodelo case used by TestDashboardIntegration.

    Returns the ``case`` Path so individual tests can write optional files
    before calling ``DashboardData.load()``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    case = tmp_path / "v2_case"
    case.mkdir()

    _write_json(
        case / "stages.json",
        {
            "stages": [
                {
                    "id": 0,
                    "start_date": "2026-01-01",
                    "blocks": [
                        {"id": 0, "hours": 120.0},
                        {"id": 1, "hours": 300.0},
                    ],
                },
                {
                    "id": 1,
                    "start_date": "2026-02-01",
                    "blocks": [
                        {"id": 0, "hours": 112.0},
                        {"id": 1, "hours": 280.0},
                    ],
                },
            ]
        },
    )
    _write_json(case / "system" / "hydros.json", {"hydros": []})
    _write_json(case / "system" / "buses.json", {"buses": []})
    _write_json(case / "system" / "thermals.json", {"thermals": []})
    _write_json(case / "system" / "lines.json", {"lines": []})
    _write_json(
        case / "system" / "non_controllable_sources.json",
        {"non_controllable_sources": []},
    )

    (case / "scenarios").mkdir(parents=True, exist_ok=True)
    load_stats_table = pa.table(
        {
            "bus_id": pa.array([], type=pa.int32()),
            "stage_id": pa.array([], type=pa.int32()),
            "mean_mw": pa.array([], type=pa.float64()),
            "std_mw": pa.array([], type=pa.float64()),
        }
    )
    pq.write_table(load_stats_table, case / "scenarios" / "load_seasonal_stats.parquet")
    _write_json(case / "scenarios" / "load_factors.json", {"load_factors": []})

    conv_dir = case / "output" / "training"
    conv_dir.mkdir(parents=True)
    conv_table = pa.table(
        {
            "iteration": pa.array([1, 2], type=pa.int32()),
            "lower_bound": pa.array([1.0e9, 1.1e9], type=pa.float64()),
            "upper_bound_mean": pa.array([1.5e9, 1.4e9], type=pa.float64()),
            "upper_bound_std": pa.array([1.0e7, 9.0e6], type=pa.float64()),
            "gap_percent": pa.array([33.3, 21.4], type=pa.float64()),
            "cuts_added": pa.array([10, 8], type=pa.int32()),
            "cuts_removed": pa.array([0, 0], type=pa.int32()),
            "cuts_active": pa.array([10, 18], type=pa.int64()),
            "time_forward_ms": pa.array([100, 90], type=pa.int64()),
            "time_backward_ms": pa.array([200, 180], type=pa.int64()),
            "time_total_ms": pa.array([300, 270], type=pa.int64()),
            "forward_passes": pa.array([5, 5], type=pa.int32()),
            "lp_solves": pa.array([100, 90], type=pa.int64()),
        }
    )
    pq.write_table(conv_table, conv_dir / "convergence.parquet")

    sim_base = case / "output" / "simulation"

    def _write_sim_parquet(entity: str, scenario_id: int, table: pa.Table) -> None:
        d = sim_base / entity / f"scenario_id={scenario_id:04d}"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, d / "data.parquet")

    empty_entity = pa.table(
        {
            "stage_id": pa.array([], type=pa.int32()),
            "block_id": pa.array([], type=pa.int32()),
        }
    )
    for entity in ("hydros", "thermals", "non_controllables", "buses", "exchanges"):
        for sid in (0, 1):
            _write_sim_parquet(entity, sid, empty_entity)

    costs_table = pa.table(
        {
            "stage_id": pa.array([0, 1], type=pa.int32()),
            "block_id": pa.array([None, None], type=pa.int32()),
            "total_cost": pa.array([5.0e9, 4.8e9], type=pa.float64()),
            "immediate_cost": pa.array([5.0e8, 4.8e8], type=pa.float64()),
            "future_cost": pa.array([4.5e9, 4.32e9], type=pa.float64()),
            "discount_factor": pa.array([1.0, 0.99], type=pa.float64()),
            "thermal_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "contract_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "deficit_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "excess_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "storage_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "filling_target_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "hydro_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "outflow_violation_below_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "outflow_violation_above_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "turbined_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "generation_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "evaporation_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "withdrawal_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "inflow_penalty_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "generic_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "spillage_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "turbined_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "curtailment_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "exchange_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "pumping_cost": pa.array([0.0, 0.0], type=pa.float64()),
        }
    )
    for sid in (0, 1):
        _write_sim_parquet("costs", sid, costs_table)

    return case


def test_load_config_present(_v2_case: Path) -> None:
    """config and discount_rate are populated from config.json when it exists."""
    from novomodelo_bridge.dashboard.data import DashboardData

    _write_json(
        _v2_case / "config.json",
        {"discount_rate": 0.12, "iterations": 500},
    )

    data = DashboardData.load(_v2_case)

    assert data.config["iterations"] == 500
    assert data.discount_rate == pytest.approx(0.12)


def test_load_config_absent(_v2_case: Path) -> None:
    """config defaults to {} and discount_rate to 0.0 when config.json is absent."""
    from novomodelo_bridge.dashboard.data import DashboardData

    # Ensure no config.json is present
    config_path = _v2_case / "config.json"
    config_path.unlink(missing_ok=True)

    data = DashboardData.load(_v2_case)

    assert data.config == {}
    assert data.discount_rate == pytest.approx(0.0)


def test_load_training_metadata_present(_v2_case: Path) -> None:
    """training_metadata is populated from output/training/metadata.json."""
    from novomodelo_bridge.dashboard.data import DashboardData

    _write_json(
        _v2_case / "output" / "training" / "metadata.json",
        {
            "cobre_version": "0.3.2",
            "hostname": "test",
            "solver": "highs",
            "started_at": "2026-04-06T04:27:06Z",
            "completed_at": "2026-04-06T04:29:22Z",
            "duration_seconds": 136.135,
            "status": "complete",
            "convergence": {"termination_reason": "iteration_limit"},
        },
    )

    data = DashboardData.load(_v2_case)

    assert data.training_metadata["cobre_version"] == "0.3.2"
    assert data.training_metadata["duration_seconds"] == 136.135
    assert (
        data.training_metadata["convergence"]["termination_reason"] == "iteration_limit"
    )


def test_load_stages_data_preserved(_v2_case: Path) -> None:
    """stages_data contains the raw stages list from stages.json."""
    from novomodelo_bridge.dashboard.data import DashboardData

    data = DashboardData.load(_v2_case)

    assert "stages" in data.stages_data
    assert isinstance(data.stages_data["stages"], list)
    assert len(data.stages_data["stages"]) > 0


def test_simulation_metadata_field(_v2_case: Path) -> None:
    """simulation_metadata is populated from output/simulation/metadata.json."""
    from novomodelo_bridge.dashboard.data import DashboardData

    _write_json(
        _v2_case / "output" / "simulation" / "metadata.json",
        {
            "cobre_version": "0.3.2",
            "duration_seconds": 48.837,
            "status": "complete",
            "scenarios": {"total": 100, "completed": 100, "failed": 0},
        },
    )

    data = DashboardData.load(_v2_case)

    assert data.simulation_metadata["status"] == "complete"
    assert data.simulation_metadata["scenarios"]["completed"] == 100


# ---------------------------------------------------------------------------
# DashboardData.load() — constraint bounds and scenario stats
# ---------------------------------------------------------------------------


def test_load_hydro_bounds_present(_v2_case: Path) -> None:
    """hydro_bounds is a non-empty DataFrame with expected columns when file exists."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from novomodelo_bridge.dashboard.data import DashboardData

    bounds_dir = _v2_case / "constraints"
    bounds_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "hydro_id": pa.array([0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "min_storage_hm3": pa.array([100.0, 100.0], type=pa.float64()),
                "max_storage_hm3": pa.array([5000.0, 5000.0], type=pa.float64()),
            }
        ),
        bounds_dir / "hydro_bounds.parquet",
    )

    data = DashboardData.load(_v2_case)

    assert not data.hydro_bounds.empty
    assert list(data.hydro_bounds.columns) == [
        "hydro_id",
        "stage_id",
        "min_storage_hm3",
        "max_storage_hm3",
    ]


def test_load_hydro_bounds_absent(_v2_case: Path) -> None:
    """hydro_bounds is an empty DataFrame when hydro_bounds.parquet is absent."""
    from novomodelo_bridge.dashboard.data import DashboardData

    # Ensure the file is not present
    hb_path = _v2_case / "constraints" / "hydro_bounds.parquet"
    hb_path.unlink(missing_ok=True)

    data = DashboardData.load(_v2_case)

    assert data.hydro_bounds.empty


def test_load_thermal_bounds_present(_v2_case: Path) -> None:
    """thermal_bounds is a non-empty DataFrame with expected columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from novomodelo_bridge.dashboard.data import DashboardData

    bounds_dir = _v2_case / "constraints"
    bounds_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "thermal_id": pa.array([0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "min_generation_mw": pa.array([0.0, 0.0], type=pa.float64()),
                "max_generation_mw": pa.array([300.0, 300.0], type=pa.float64()),
            }
        ),
        bounds_dir / "thermal_bounds.parquet",
    )

    data = DashboardData.load(_v2_case)

    assert not data.thermal_bounds.empty
    assert list(data.thermal_bounds.columns) == [
        "thermal_id",
        "stage_id",
        "min_generation_mw",
        "max_generation_mw",
    ]


def test_load_ncs_stats_present(_v2_case: Path) -> None:
    """ncs_stats is a non-empty DataFrame when non_controllable_stats.parquet exists."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from novomodelo_bridge.dashboard.data import DashboardData

    scenarios_dir = _v2_case / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "non_controllable_id": pa.array([0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "mean_mw": pa.array([80.0, 75.0], type=pa.float64()),
                "std_mw": pa.array([5.0, 4.0], type=pa.float64()),
            }
        ),
        scenarios_dir / "non_controllable_stats.parquet",
    )

    data = DashboardData.load(_v2_case)

    assert not data.ncs_stats.empty
    assert "non_controllable_id" in data.ncs_stats.columns
    assert "mean_mw" in data.ncs_stats.columns


def test_load_line_block_bounds_present(_v2_case: Path) -> None:
    """line_block_bounds holds only the per-block (block_id non-null) rows.

    Novomodelo 0.13 deleted the standalone per-block exchange-factor JSON document
    and folded it into absolute-MW override rows inside line_bounds.parquet;
    the stage-level base row (block_id is null) must not leak
    into this field.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from novomodelo_bridge.dashboard.data import DashboardData

    constraints_dir = _v2_case / "constraints"
    constraints_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "line_id": pa.array([0, 0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 0, 0], type=pa.int32()),
                "direct_mw": pa.array([100.0, 80.0, 120.0], type=pa.float64()),
                "reverse_mw": pa.array([100.0, 80.0, 120.0], type=pa.float64()),
                "block_id": pa.array([None, 0, 1], type=pa.int32()),
            }
        ),
        constraints_dir / "line_bounds.parquet",
    )

    data = DashboardData.load(_v2_case)

    assert not data.line_block_bounds.empty
    assert len(data.line_block_bounds) == 2
    assert data.line_block_bounds["block_id"].notna().all()
    assert sorted(data.line_block_bounds["direct_mw"].tolist()) == [80.0, 120.0]


def test_load_line_block_bounds_absent_is_empty_not_raising(_v2_case: Path) -> None:
    """A case whose lines are uniform across blocks has no override rows.

    Only the stage-level base row (block_id is null) is present; this is a
    legitimate steady state, not a version problem, so it must degrade
    to an empty frame rather than raise.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from novomodelo_bridge.dashboard.data import DashboardData

    constraints_dir = _v2_case / "constraints"
    constraints_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "line_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([0], type=pa.int32()),
                "direct_mw": pa.array([100.0], type=pa.float64()),
                "reverse_mw": pa.array([100.0], type=pa.float64()),
                "block_id": pa.array([None], type=pa.int32()),
            }
        ),
        constraints_dir / "line_bounds.parquet",
    )

    data = DashboardData.load(_v2_case)

    assert data.line_block_bounds.empty


def test_load_line_block_bounds_no_file_is_empty_not_raising(_v2_case: Path) -> None:
    """No line_bounds.parquet at all still yields an empty frame, not a raise."""
    from novomodelo_bridge.dashboard.data import DashboardData

    data = DashboardData.load(_v2_case)

    assert data.line_block_bounds.empty


# ---------------------------------------------------------------------------
# stochastic data fields
# ---------------------------------------------------------------------------


def test_load_inflow_history_present(_v2_case: Path) -> None:
    """The converter's windowed inflow_history loads, and the stochastic tab
    reads it without falling back to its error placeholder."""
    from datetime import date

    import pyarrow as pa
    import pyarrow.parquet as pq

    from novomodelo_bridge.dashboard.data import DashboardData
    from novomodelo_bridge.dashboard.tabs import stochastic
    from novomodelo_bridge.newave.converters.inflow_windows import (
        INFLOW_HISTORY_WINDOW_SCHEMA,
    )

    scenarios_dir = _v2_case / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "hydro_id": [0, 0, 1, 1],
                "start_date": [date(2000, 1, 1), date(2001, 1, 1)] * 2,
                "end_date": [date(2000, 2, 1), date(2001, 2, 1)] * 2,
                "value_m3s": [100.0, 110.0, 50.0, 55.0],
            },
            schema=INFLOW_HISTORY_WINDOW_SCHEMA,
        ),
        scenarios_dir / "inflow_history.parquet",
    )

    data = DashboardData.load(_v2_case)

    assert not data.inflow_history.empty
    assert not stochastic._compute_historical_stats(data).empty


def test_load_inflow_history_absent(_v2_case: Path) -> None:
    """inflow_history is an empty DataFrame when the file is missing."""
    from novomodelo_bridge.dashboard.data import DashboardData

    ih_path = _v2_case / "scenarios" / "inflow_history.parquet"
    assert not ih_path.exists()

    data = DashboardData.load(_v2_case)

    assert data.inflow_history.empty


def test_load_correlation_present(_v2_case: Path) -> None:
    """correlation contains the expected key when correlation.json exists."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from novomodelo_bridge.dashboard.data import DashboardData

    stochastic_dir = _v2_case / "output" / "stochastic"
    stochastic_dir.mkdir(parents=True, exist_ok=True)

    # Write the required stochastic parquet files so stochastic_available is True
    for filename in (
        "inflow_seasonal_stats.parquet",
        "inflow_ar_coefficients.parquet",
        "noise_openings.parquet",
    ):
        pq.write_table(
            pa.table({"dummy": pa.array([], type=pa.int32())}),
            stochastic_dir / filename,
        )
    _write_json(stochastic_dir / "fitting_report.json", {})
    _write_json(
        stochastic_dir / "correlation.json",
        {"correlations": [[1.0, 0.3], [0.3, 1.0]]},
    )

    data = DashboardData.load(_v2_case)

    assert "correlations" in data.correlation
    assert data.correlation["correlations"] == [[1.0, 0.3], [0.3, 1.0]]


def test_load_correlation_absent_no_stochastic(_v2_case: Path) -> None:
    """correlation is an empty dict when the stochastic output directory is missing."""
    from novomodelo_bridge.dashboard.data import DashboardData

    stochastic_dir = _v2_case / "output" / "stochastic"
    assert not stochastic_dir.exists()

    data = DashboardData.load(_v2_case)

    assert data.correlation == {}


def test_load_inflow_lags_lf_present(_v2_case: Path) -> None:
    """inflow_lags_lf is a LazyFrame that collects when the directory exists."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from novomodelo_bridge.dashboard.data import DashboardData

    lags_dir = _v2_case / "output" / "simulation" / "inflow_lags" / "scenario_id=0"
    lags_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "hydro_id": pa.array([0, 1], type=pa.int32()),
                "stage_id": pa.array([0, 0], type=pa.int32()),
                "lag_value": pa.array([1.5, 2.3], type=pa.float64()),
            }
        ),
        lags_dir / "data.parquet",
    )

    data = DashboardData.load(_v2_case)

    collected = data.inflow_lags_lf.collect()
    assert len(collected) > 0


# ---------------------------------------------------------------------------
# compute_non_fictitious_bus_ids
# ---------------------------------------------------------------------------


def test_compute_non_fictitious_bus_ids_filters_zero_load() -> None:
    """Bus with zero mean_mw in all stages is excluded; nonzero bus is included."""
    from novomodelo_bridge.dashboard.data import compute_non_fictitious_bus_ids

    load_stats = pd.DataFrame(
        {
            "bus_id": [0, 0, 1, 1],
            "stage_id": [0, 1, 0, 1],
            "mean_mw": [100.0, 80.0, 0.0, 0.0],
        }
    )

    result = compute_non_fictitious_bus_ids(load_stats)

    assert result == [0]


def test_compute_non_fictitious_bus_ids_all_nonzero() -> None:
    """All buses with nonzero load in at least one stage are returned sorted."""
    from novomodelo_bridge.dashboard.data import compute_non_fictitious_bus_ids

    load_stats = pd.DataFrame(
        {
            "bus_id": [0, 1, 0, 1],
            "stage_id": [0, 0, 1, 1],
            "mean_mw": [50.0, 10.0, 48.0, 9.0],
        }
    )

    result = compute_non_fictitious_bus_ids(load_stats)

    assert result == [0, 1]


def test_compute_non_fictitious_bus_ids_empty_df() -> None:
    """Empty DataFrame returns an empty list."""
    from novomodelo_bridge.dashboard.data import compute_non_fictitious_bus_ids

    result = compute_non_fictitious_bus_ids(pd.DataFrame())

    assert result == []


def test_compute_non_fictitious_bus_ids_missing_column() -> None:
    """DataFrame missing mean_mw column returns an empty list (defensive)."""
    from novomodelo_bridge.dashboard.data import compute_non_fictitious_bus_ids

    load_stats = pd.DataFrame({"bus_id": [0, 1], "stage_id": [0, 0]})

    result = compute_non_fictitious_bus_ids(load_stats)

    assert result == []


def test_non_fictitious_bus_ids_field_on_data(_v2_case: Path) -> None:
    """non_fictitious_bus_ids is populated as a sorted list of int on DashboardData."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from novomodelo_bridge.dashboard.data import DashboardData

    # Overwrite the load_stats with two buses: 0 has load, 1 is fictitious
    load_stats_table = pa.table(
        {
            "bus_id": pa.array([0, 0, 1, 1], type=pa.int32()),
            "stage_id": pa.array([0, 1, 0, 1], type=pa.int32()),
            "mean_mw": pa.array([50000.0, 48000.0, 0.0, 0.0], type=pa.float64()),
            "std_mw": pa.array([0.0, 0.0, 0.0, 0.0], type=pa.float64()),
        }
    )
    pq.write_table(
        load_stats_table,
        _v2_case / "scenarios" / "load_seasonal_stats.parquet",
    )

    data = DashboardData.load(_v2_case)

    assert isinstance(data.non_fictitious_bus_ids, list)
    assert data.non_fictitious_bus_ids == [0]
    assert all(isinstance(bid, int) for bid in data.non_fictitious_bus_ids)


# ---------------------------------------------------------------------------
# Wall-time correction — _correct_wall_times_from_convergence
# ---------------------------------------------------------------------------


def _make_timing_raw_multiworker(
    n_iters: int = 2, n_workers: int = 3, fwd_per_worker: int = 100
) -> pd.DataFrame:
    """Simulate the multi-worker timing shape: 1 rank row + N per-worker rows/iter."""
    rows: list[dict[str, Any]] = []
    for it in range(1, n_iters + 1):
        rows.append(
            {
                "iteration": it,
                "rank": 0,
                "worker_id": None,
                "forward_wall_ms": 0,
                "backward_wall_ms": 0,
                "lower_bound_ms": 7,
                "overhead_ms": 3,
                "fwd_setup_ms": 0,
                "bwd_setup_ms": 0,
            }
        )
        for w in range(n_workers):
            rows.append(
                {
                    "iteration": it,
                    "rank": 0,
                    "worker_id": w,
                    "forward_wall_ms": fwd_per_worker,
                    "backward_wall_ms": fwd_per_worker * 4,
                    "lower_bound_ms": 0,
                    "overhead_ms": 0,
                    "fwd_setup_ms": 5,
                    "bwd_setup_ms": 10,
                }
            )
    return pd.DataFrame(rows)


def test_correct_wall_times_overrides_forward_wall_ms() -> None:
    """Correction replaces inflated sum with convergence's per-iter wall value."""
    timing_raw = _make_timing_raw_multiworker(
        n_iters=2, n_workers=4, fwd_per_worker=100
    )
    agg = _aggregate_timing_by_iteration(timing_raw)
    # Aggregated forward_wall_ms is inflated: 4 workers × 100 = 400
    assert agg.loc[agg["iteration"] == 1, "forward_wall_ms"].iloc[0] == 400

    conv = pd.DataFrame(
        {
            "iteration": [1, 2],
            "time_forward_ms": [110, 120],
            "time_backward_ms": [440, 480],
        }
    )
    corrected = _correct_wall_times_from_convergence(agg, conv)
    assert corrected.loc[corrected["iteration"] == 1, "forward_wall_ms"].iloc[0] == 110
    assert corrected.loc[corrected["iteration"] == 2, "forward_wall_ms"].iloc[0] == 120
    assert corrected.loc[corrected["iteration"] == 1, "backward_wall_ms"].iloc[0] == 440
    assert corrected.loc[corrected["iteration"] == 2, "backward_wall_ms"].iloc[0] == 480


def test_correct_wall_times_preserves_other_columns() -> None:
    """Only forward/backward wall cols are overridden; others unchanged."""
    timing_raw = _make_timing_raw_multiworker(n_iters=2, n_workers=3, fwd_per_worker=50)
    agg = _aggregate_timing_by_iteration(timing_raw)
    # fwd_setup_ms (aggregate CPU) should stay summed: 3 workers × 5 = 15
    assert agg.loc[agg["iteration"] == 1, "fwd_setup_ms"].iloc[0] == 15
    # rank-only lower_bound_ms should equal the rank row's value (7)
    assert agg.loc[agg["iteration"] == 1, "lower_bound_ms"].iloc[0] == 7

    conv = pd.DataFrame(
        {
            "iteration": [1, 2],
            "time_forward_ms": [55, 60],
            "time_backward_ms": [220, 240],
        }
    )
    corrected = _correct_wall_times_from_convergence(agg, conv)
    # Unchanged columns:
    assert corrected.loc[corrected["iteration"] == 1, "fwd_setup_ms"].iloc[0] == 15
    assert corrected.loc[corrected["iteration"] == 1, "lower_bound_ms"].iloc[0] == 7
    assert corrected.loc[corrected["iteration"] == 1, "bwd_setup_ms"].iloc[0] == 30


def test_correct_wall_times_noop_when_conv_empty() -> None:
    """Empty conv frame leaves timing unchanged."""
    timing_raw = _make_timing_raw_multiworker(n_iters=1, n_workers=2)
    agg = _aggregate_timing_by_iteration(timing_raw)
    result = _correct_wall_times_from_convergence(agg, pd.DataFrame())
    pd.testing.assert_frame_equal(result, agg)


def test_correct_wall_times_noop_when_conv_missing_columns() -> None:
    """Correction leaves timing untouched when required conv columns are absent."""
    timing_raw = _make_timing_raw_multiworker(n_iters=1, n_workers=2)
    agg = _aggregate_timing_by_iteration(timing_raw)
    bad_conv = pd.DataFrame({"iteration": [1], "upper_bound": [10.0]})
    result = _correct_wall_times_from_convergence(agg, bad_conv)
    pd.testing.assert_frame_equal(result, agg)


def test_correct_wall_times_noop_on_empty_timing() -> None:
    """Empty timing frame passes through unchanged."""
    conv = pd.DataFrame(
        {"iteration": [1], "time_forward_ms": [50], "time_backward_ms": [200]}
    )
    result = _correct_wall_times_from_convergence(pd.DataFrame(), conv)
    assert result.empty


def test_correct_wall_times_partial_iteration_mapping() -> None:
    """When conv covers only some iterations, unmatched rows keep their original
    value."""
    timing_raw = _make_timing_raw_multiworker(n_iters=3, n_workers=2, fwd_per_worker=10)
    agg = _aggregate_timing_by_iteration(timing_raw)
    # Only provide conv for iteration 2.
    conv = pd.DataFrame(
        {"iteration": [2], "time_forward_ms": [17], "time_backward_ms": [67]}
    )
    corrected = _correct_wall_times_from_convergence(agg, conv)
    # iter 1 and 3 keep their (inflated) sum (2 workers × 10 = 20); iter 2 overridden.
    assert corrected.loc[corrected["iteration"] == 1, "forward_wall_ms"].iloc[0] == 20
    assert corrected.loc[corrected["iteration"] == 2, "forward_wall_ms"].iloc[0] == 17
    assert corrected.loc[corrected["iteration"] == 3, "forward_wall_ms"].iloc[0] == 20


# ---------------------------------------------------------------------------
# Tab registry smoke test — regression guard
# ---------------------------------------------------------------------------


def test_tab_registry_contains_all_modules() -> None:
    assert len(TAB_MODULES) == 9
    for module in TAB_MODULES:
        assert module.TAB_ID.startswith("tab-"), (
            f"{module}.TAB_ID = {module.TAB_ID!r} does not start with 'tab-'"
        )


# ---------------------------------------------------------------------------
# Section loaders — DashboardData.load is composed from cohesive, independently
# callable loaders rather than one monolithic block.
# ---------------------------------------------------------------------------


class TestSectionLoaders:
    def test_loaders_are_independently_callable(self, _v2_case: Path) -> None:
        from novomodelo_bridge.dashboard.data import (
            load_entity_metadata,
            load_scenario_inputs,
            load_temporal_context,
        )

        temporal = load_temporal_context(_v2_case)
        assert temporal.stage_hours[0] == 420.0  # 120 + 300
        assert temporal.discount_rate == 0.0

        entities = load_entity_metadata(_v2_case)
        assert isinstance(entities.names, dict)
        assert isinstance(entities.hydro_meta, dict)

        scenario = load_scenario_inputs(_v2_case)
        assert isinstance(scenario.load_factors_list, list)

    def test_loaders_compose_to_full_aggregate(self, _v2_case: Path) -> None:
        from novomodelo_bridge.dashboard.data import (
            DashboardData,
            load_temporal_context,
        )

        data = DashboardData.load(_v2_case)
        temporal = load_temporal_context(_v2_case)
        # Aggregate temporal fields come straight from the section loader.
        assert data.stage_hours == temporal.stage_hours
        assert data.block_hours == temporal.block_hours
        assert data.discount_rate == temporal.discount_rate
        assert data.line_meta == temporal.line_meta


# ---------------------------------------------------------------------------
# _normalize_output_columns — novomodelo 0.14 diagnostic-output axis renames
# ---------------------------------------------------------------------------


def test_normalize_output_columns_maps_014_solver_axes() -> None:
    """0.14 ``stage_id``/``opening_index`` map to the dashboard's legacy names."""
    df = pd.DataFrame(
        {
            "stage_id": [0, 1, None],
            "opening_index": [0, None, 2],
            "lp_solves": [3, 4, 5],
        }
    )
    out = _normalize_output_columns(df)
    assert "stage" in out.columns and "stage_id" not in out.columns
    assert "opening" in out.columns and "opening_index" not in out.columns
    # A NULL stage/opening survives as NaN (not refilled to a -1 sentinel), so
    # the chart layer's ``>= 0`` filters and ``notna()`` checks keep working.
    assert out["stage"].isna().sum() == 1
    assert out["opening"].isna().sum() == 1


def test_normalize_output_columns_maps_014_convergence_upper_bound() -> None:
    """0.14 ``upper_bound`` maps to ``upper_bound_mean``; ``upper_bound_std`` kept."""
    df = pd.DataFrame(
        {
            "iteration": [0, 1],
            "upper_bound": [1.5e9, 1.4e9],
            "upper_bound_std": [1.0e7, None],
            "upper_bound_kind": ["statistical", "exact"],
        }
    )
    out = _normalize_output_columns(df)
    assert "upper_bound_mean" in out.columns and "upper_bound" not in out.columns
    # The prefix-sharing std/kind columns must not be swept up by the rename.
    assert "upper_bound_std" in out.columns
    assert "upper_bound_kind" in out.columns


def test_normalize_output_columns_leaves_pre_014_frame_unchanged() -> None:
    """A 0.13 frame already carries the legacy names — pass through untouched."""
    df = pd.DataFrame(
        {"stage": [0, 1], "upper_bound_mean": [1.0, 2.0], "upper_bound_std": [0.1, 0.1]}
    )
    out = _normalize_output_columns(df)
    assert list(out.columns) == ["stage", "upper_bound_mean", "upper_bound_std"]


def test_normalize_output_columns_does_not_clobber_existing_legacy_name() -> None:
    """If both spellings are present the legacy column wins (no overwrite)."""
    df = pd.DataFrame({"stage": [7], "stage_id": [0]})
    out = _normalize_output_columns(df)
    # ``stage`` already exists, so ``stage_id`` is not renamed onto it.
    assert out["stage"].tolist() == [7]
    assert "stage_id" in out.columns


def test_normalize_output_columns_empty_frame() -> None:
    """An empty (absent-file) frame is returned as-is."""
    empty = pd.DataFrame()
    assert _normalize_output_columns(empty).empty

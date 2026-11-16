"""Tests for ``comparators.plant_buses``: the buses that share a hydro plant,
judged as one group, and the rows moved off the headline for them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from novomodelo_bridge.comparators.analyze import build_results_dataset
from novomodelo_bridge.comparators.model import PercentileData, ResultComparison
from novomodelo_bridge.comparators.plant_buses import (
    SharedPlantGroup,
    find_shared_plant_groups,
    merge_shared_plant_buses,
)
from novomodelo_bridge.comparators.verdict import build_compare_verdict
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.agreement import (
    SHARED_PLANT_NET_FLOW,
    SHARED_PLANT_NET_IMPORT,
    TIED_NET_FLOW,
)
from novomodelo_bridge.core.diagnostics import Severity

SE, S, NE, IV = 0, 1, 2, 5
_BUS_NAMES = {SE: "SE", S: "S", NE: "NE", IV: "IV"}


def _hydro(hydro_id: int, name: str, *buses: int) -> dict[str, Any]:
    return {
        "id": hydro_id,
        "name": name,
        "unit_groups": [{"id": i, "bus_id": bus} for i, bus in enumerate(buses)],
    }


def _line(line_id: int, source: int, target: int) -> dict[str, Any]:
    return {
        "id": line_id,
        "name": f"{_BUS_NAMES[source]}-{_BUS_NAMES[target]}",
        "source_bus_id": source,
        "target_bus_id": target,
    }


_ITAIPU = _hydro(47, "ITAIPU", SE, IV)
_LINES = [_line(0, SE, NE), _line(2, SE, IV), _line(6, IV, S)]


def _write_case(case_dir: Path, hydros: list[dict[str, Any]]) -> Path:
    system = case_dir / "system"
    system.mkdir(parents=True)
    (system / "hydros.json").write_text(
        json.dumps({"hydros": hydros}), encoding="utf-8"
    )
    (system / "lines.json").write_text(json.dumps({"lines": _LINES}), encoding="utf-8")
    (system / "buses.json").write_text(
        json.dumps({"buses": [{"id": i, "name": n} for i, n in _BUS_NAMES.items()]}),
        encoding="utf-8",
    )
    output = case_dir / "output"
    output.mkdir()
    return output


def _row(
    entity_type: str,
    entity_id: int,
    variable: str,
    ref: float,
    novomodelo: float,
    *,
    stage: int = 0,
) -> ResultComparison:
    diff = abs(novomodelo - ref)
    return ResultComparison(
        entity_type=entity_type,
        entity_name=_BUS_NAMES[entity_id]
        if entity_type == "bus"
        else f"line {entity_id}",
        newave_code=entity_id + 1,
        novomodelo_id=entity_id,
        stage=stage,
        variable=variable,
        newave_value=ref,
        novomodelo_value=novomodelo,
        abs_diff=diff,
        rel_diff=diff / abs(ref) if ref else None,
    )


def _two_placements() -> list[ResultComparison]:
    """Itaipu puts 800 MW at IV in the source model and 1400 MW in
    Novomodelo. IV sends 500 MW to S in both and the rest to SE, which sends
    100 MW to NE: SE's own net import and the SE-IV flow differ, and the
    net import of SE and IV together, -600 MW, does not."""
    return [
        _row("line", 0, "net_flow_mw", 100.0, 100.0),
        _row("line", 2, "net_flow_mw", -300.0, -900.0),
        _row("line", 6, "net_flow_mw", 500.0, 500.0),
        _row("bus", SE, "net_import_mw", 200.0, 800.0),
        _row("bus", S, "net_import_mw", 500.0, 500.0),
        _row("bus", NE, "net_import_mw", 100.0, 100.0),
    ]


class TestFindSharedPlantGroups:
    def test_a_plant_on_two_buses_joins_them(self) -> None:
        assert find_shared_plant_groups([_ITAIPU, _hydro(1, "FURNAS", SE)]) == (
            SharedPlantGroup(bus_ids=(SE, IV), plants=("ITAIPU",)),
        )

    def test_plants_on_one_bus_each_join_nothing(self) -> None:
        hydros = [_hydro(1, "FURNAS", SE), _hydro(2, "MACHADINHO", S, S)]

        assert find_shared_plant_groups(hydros) == ()

    def test_two_plants_sharing_a_bus_form_one_group(self) -> None:
        hydros = [_hydro(1, "A", SE, IV), _hydro(2, "B", IV, S)]

        assert find_shared_plant_groups(hydros) == (
            SharedPlantGroup(bus_ids=(SE, S, IV), plants=("A", "B")),
        )


class TestMergeSharedPlantBuses:
    def test_judges_the_group_and_moves_its_own_rows(self, tmp_path: Path) -> None:
        output = _write_case(tmp_path, [_ITAIPU])

        with dx.collect() as found:
            merged = merge_shared_plant_buses(_two_placements(), output)

        assert [(r.entity_type, r.entity_name, r.variable) for r in merged] == [
            ("line", "line 0", "net_flow_mw"),
            ("line", "line 2", SHARED_PLANT_NET_FLOW),
            ("line", "line 6", "net_flow_mw"),
            ("bus", "SE", SHARED_PLANT_NET_IMPORT),
            ("bus", "S", "net_import_mw"),
            ("bus", "NE", "net_import_mw"),
            ("bus", "SE+IV", "net_import_mw"),
        ]
        group = merged[-1]
        assert (group.novomodelo_id, group.newave_code, group.stage) == (SE, 1, 0)
        assert (group.newave_value, group.novomodelo_value) == (-600.0, -600.0)
        (warning,) = found
        assert (warning.code, warning.severity) == (
            "buses-share-plant",
            Severity.WARNING,
        )
        assert warning.table is not None
        assert warning.table.rows == [["0, 5", "SE+IV", "ITAIPU", "SE-IV"]]

    def test_the_headline_judges_the_group_not_its_buses(self, tmp_path: Path) -> None:
        output = _write_case(tmp_path, [_ITAIPU])
        with dx.collect():
            merged = merge_shared_plant_buses(_two_placements(), output)

        dataset = build_results_dataset(merged, PercentileData(), 0.05)
        verdict = build_compare_verdict(dataset)

        gaps = {row["variable"]: row["gap"] for row in dataset.summary.to_dicts()}
        assert gaps["net_import_mw"] == 0.0
        assert gaps[SHARED_PLANT_NET_IMPORT] > 0.0
        assert gaps[SHARED_PLANT_NET_FLOW] > 0.0
        assert (verdict.total, verdict.all_close) == (2, True)

    def test_a_tied_line_across_the_boundary_counts_in_the_group(
        self, tmp_path: Path
    ) -> None:
        output = _write_case(tmp_path, [_ITAIPU])
        rows = [
            _row("line", 0, TIED_NET_FLOW, 100.0, 100.0)
            if r.novomodelo_id == 0 and r.entity_type == "line"
            else r
            for r in _two_placements()
        ]

        with dx.collect():
            merged = merge_shared_plant_buses(rows, output)

        group = merged[-1]
        assert group.entity_name == "SE+IV"
        assert (group.newave_value, group.novomodelo_value) == (-600.0, -600.0)
        assert merged[0].variable == TIED_NET_FLOW

    def test_a_stage_missing_a_boundary_line_has_no_group_row(
        self, tmp_path: Path
    ) -> None:
        output = _write_case(tmp_path, [_ITAIPU])
        rows = [
            *_two_placements(),
            _row("line", 0, "net_flow_mw", 50.0, 50.0, stage=1),
            _row("line", 2, "net_flow_mw", -10.0, -20.0, stage=1),
        ]

        with dx.collect():
            merged = merge_shared_plant_buses(rows, output)

        assert [r.stage for r in merged if r.entity_name == "SE+IV"] == [0]

    def test_without_shared_plants_changes_nothing(self, tmp_path: Path) -> None:
        output = _write_case(tmp_path, [_hydro(47, "ITAIPU", SE)])
        rows = _two_placements()

        with dx.collect() as found:
            merged = merge_shared_plant_buses(rows, output)

        assert merged == rows
        assert found == []

    def test_without_bus_or_line_rows_reads_nothing(self, tmp_path: Path) -> None:
        rows = [_row("bus", SE, "spot_price", 100.0, 100.0)]

        with dx.collect() as found:
            merged = merge_shared_plant_buses(rows, tmp_path / "missing" / "output")

        assert merged == rows
        assert found == []

"""Tests for ``comparators.line_ties``: the loops of lines whose flows the two
models can route differently at the same cost, and the rows moved off the
headline for them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from novomodelo_bridge.comparators.analyze import build_results_dataset
from novomodelo_bridge.comparators.line_ties import find_tied_loops, mark_tied_lines
from novomodelo_bridge.comparators.model import PercentileData, ResultComparison
from novomodelo_bridge.comparators.verdict import build_compare_verdict
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.agreement import TIED_NET_FLOW
from novomodelo_bridge.core.diagnostics import Severity

SE, S, NE, FC = 0, 1, 2, 3
_BUS_NAMES = {SE: "SE", S: "S", NE: "NE", FC: "FC"}


def _line(line_id: int, source: int, target: int, cost: float | None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": line_id,
        "name": f"{_BUS_NAMES[source]}-{_BUS_NAMES[target]}",
        "source_bus_id": source,
        "target_bus_id": target,
    }
    if cost is not None:
        entry["exchange_cost"] = cost
    return entry


def _triangle_through_fc() -> list[dict[str, Any]]:
    """SE-NE direct at full cost, and SE-FC-NE at half cost per leg, as the
    conversion writes a route through a fictitious subsystem; plus S-SE,
    on no loop."""
    return [
        _line(0, SE, NE, 2e-4),
        _line(1, NE, FC, 1e-4),
        _line(2, SE, FC, 1e-4),
        _line(3, S, SE, 2e-4),
    ]


class TestFindTiedLoops:
    def test_a_direct_line_and_its_halved_route_through_a_fictitious_bus_tie(
        self,
    ) -> None:
        (loop,) = find_tied_loops(_triangle_through_fc(), None)

        assert loop.buses == (SE, NE, FC)
        assert loop.line_ids == {0, 1, 2}
        assert loop.halves == (("SE-NE",), ("NE-FC", "SE-FC"))
        assert loop.cost == 2e-4

    def test_a_loop_whose_costs_split_unevenly_does_not_tie(self) -> None:
        lines = [_line(0, SE, NE, 1.0), _line(1, NE, FC, 2.0), _line(2, SE, FC, 4.0)]

        assert find_tied_loops(lines, None) == ()

    def test_two_parallel_lines_of_equal_cost_tie(self) -> None:
        lines = [_line(0, SE, S, 1.0), _line(1, SE, S, 1.0)]

        (loop,) = find_tied_loops(lines, None)

        assert loop.line_ids == {0, 1}
        assert loop.halves == (("SE-S",), ("SE-S",))

    def test_two_parallel_lines_of_different_cost_do_not_tie(self) -> None:
        lines = [_line(0, SE, S, 1.0), _line(1, SE, S, 2.0)]

        assert find_tied_loops(lines, None) == ()

    def test_a_ring_of_four_ties_on_two_lines_against_two(self) -> None:
        lines = [
            _line(0, SE, S, 1.0),
            _line(1, S, NE, 2.0),
            _line(2, NE, FC, 1.0),
            _line(3, SE, FC, 2.0),
        ]

        (loop,) = find_tied_loops(lines, None)

        assert loop.line_ids == {0, 1, 2, 3}
        assert loop.halves == (("SE-S", "S-NE"), ("NE-FC", "SE-FC"))
        assert loop.cost == 3.0

    def test_the_outline_of_two_tied_triangles_is_left_out(self) -> None:
        # SE-NE-FC and SE-S-FC tie and share SE-FC; the ring SE-NE-FC-S
        # around them ties as well but holds no line of its own.
        lines = [
            _line(0, SE, NE, 2e-4),
            _line(1, NE, FC, 1e-4),
            _line(2, SE, FC, 1e-4),
            _line(3, S, SE, 2e-4),
            _line(4, S, FC, 1e-4),
        ]

        loops = find_tied_loops(lines, None)

        assert [loop.line_ids for loop in loops] == [{0, 1, 2}, {2, 3, 4}]

    def test_a_line_without_cost_takes_the_default(self) -> None:
        lines = [_line(0, SE, NE, 2e-4), _line(1, NE, FC, None), _line(2, SE, FC, None)]

        (loop,) = find_tied_loops(lines, 1e-4)

        assert loop.line_ids == {0, 1, 2}

    def test_a_loop_through_a_line_of_unknown_cost_is_not_judged(self) -> None:
        lines = [_line(0, SE, NE, 2e-4), _line(1, NE, FC, 1e-4), _line(2, SE, FC, None)]

        assert find_tied_loops(lines, None) == ()


def _write_case(case_dir: Path, lines: list[dict[str, Any]]) -> Path:
    system = case_dir / "system"
    system.mkdir(parents=True)
    (system / "lines.json").write_text(json.dumps({"lines": lines}), encoding="utf-8")
    (system / "buses.json").write_text(
        json.dumps({"buses": [{"id": i, "name": n} for i, n in _BUS_NAMES.items()]}),
        encoding="utf-8",
    )
    output = case_dir / "output"
    output.mkdir()
    return output


def _row(
    entity_type: str, entity_id: int, variable: str, ref: float, novomodelo: float
) -> ResultComparison:
    diff = abs(novomodelo - ref)
    return ResultComparison(
        entity_type=entity_type,
        entity_name=f"{entity_type} {entity_id}",
        newave_code=entity_id,
        novomodelo_id=entity_id,
        stage=0,
        variable=variable,
        newave_value=ref,
        novomodelo_value=novomodelo,
        abs_diff=diff,
        rel_diff=diff / abs(ref) if ref else None,
    )


def _two_routings() -> list[ResultComparison]:
    """SE sends 100 MW to NE: the source model on the direct line, Novomodelo
    through FC. The bus net imports agree and the triangle's flows do not;
    S-SE carries the same 50 MW in both."""
    return [
        _row("line", 0, "net_flow_mw", 100.0, 0.0),
        _row("line", 1, "net_flow_mw", 0.0, -100.0),
        _row("line", 2, "net_flow_mw", 0.0, 100.0),
        _row("line", 3, "net_flow_mw", 50.0, 50.0),
        _row("bus", SE, "net_import_mw", -50.0, -50.0),
        _row("bus", S, "net_import_mw", -50.0, -50.0),
        _row("bus", NE, "net_import_mw", 100.0, 100.0),
    ]


class TestMarkTiedLines:
    def test_moves_the_loop_lines_and_names_the_loop_once(self, tmp_path: Path) -> None:
        output = _write_case(tmp_path, _triangle_through_fc())

        with dx.collect() as found:
            marked = mark_tied_lines(_two_routings(), output)

        assert [(r.entity_type, r.novomodelo_id, r.variable) for r in marked] == [
            ("line", 0, TIED_NET_FLOW),
            ("line", 1, TIED_NET_FLOW),
            ("line", 2, TIED_NET_FLOW),
            ("line", 3, "net_flow_mw"),
            ("bus", SE, "net_import_mw"),
            ("bus", S, "net_import_mw"),
            ("bus", NE, "net_import_mw"),
        ]
        (warning,) = found
        assert (warning.code, warning.severity) == (
            "line-routes-tied",
            Severity.WARNING,
        )
        assert warning.table is not None
        assert warning.table.rows == [
            ["SE – NE – FC", "SE-NE = NE-FC + SE-FC", "0.0002"]
        ]

    def test_the_headline_judges_the_buses_not_the_tied_routes(
        self, tmp_path: Path
    ) -> None:
        output = _write_case(tmp_path, _triangle_through_fc())
        with dx.collect():
            marked = mark_tied_lines(_two_routings(), output)

        dataset = build_results_dataset(marked, PercentileData(), 0.05)
        verdict = build_compare_verdict(dataset)

        gaps = {row["variable"]: row["gap"] for row in dataset.summary.to_dicts()}
        assert gaps["net_import_mw"] == 0.0
        assert gaps[TIED_NET_FLOW] > 0.0
        assert (verdict.total, verdict.all_close) == (2, True)

    def test_a_line_without_cost_reads_the_default_from_penalties(
        self, tmp_path: Path
    ) -> None:
        lines = _triangle_through_fc()
        for line in lines[1:3]:
            del line["exchange_cost"]
        output = _write_case(tmp_path, lines)
        (tmp_path / "penalties.json").write_text(
            json.dumps({"line": {"exchange_cost": 1e-4}}), encoding="utf-8"
        )

        with dx.collect():
            marked = mark_tied_lines(_two_routings(), output)

        assert {r.novomodelo_id for r in marked if r.variable == TIED_NET_FLOW} == {
            0,
            1,
            2,
        }

    def test_without_line_rows_reads_nothing_and_warns_nothing(
        self, tmp_path: Path
    ) -> None:
        rows = [r for r in _two_routings() if r.entity_type == "bus"]

        with dx.collect() as found:
            marked = mark_tied_lines(rows, tmp_path / "missing" / "output")

        assert marked == rows
        assert found == []

    def test_a_tied_loop_with_no_compared_line_warns_nothing(
        self, tmp_path: Path
    ) -> None:
        output = _write_case(tmp_path, _triangle_through_fc())
        rows = [r for r in _two_routings() if r.novomodelo_id == 3]

        with dx.collect() as found:
            marked = mark_tied_lines(rows, output)

        assert marked == rows
        assert found == []

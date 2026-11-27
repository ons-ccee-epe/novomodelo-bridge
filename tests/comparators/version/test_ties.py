"""Tests of ``comparators.version.ties`` over pairs of ``build_solved_case`` trees.

``system/lines.json`` and ``system/hydros.json`` are rewritten with ``json.dumps``
to set the exchange costs and the plants that share buses. The baseline is built
with ``id_offset`` 100 and the candidate with 0, so a mark read with the wrong
side's ids misses. The mirrored view and the length invariant are read off the
public shared functions, patched pass-through or faulty in the module under test.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from novomodelo_bridge.comparators.model import ResultComparison
from novomodelo_bridge.comparators.version import ties
from novomodelo_bridge.comparators.version.alignment import name_key
from novomodelo_bridge.comparators.version.rows import result_row
from novomodelo_bridge.comparators.version.sides import SideInputs, read_side
from novomodelo_bridge.comparators.version.ties import apply_tie_marks
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.agreement import (
    SHARED_PLANT_NET_FLOW,
    SHARED_PLANT_NET_IMPORT,
    TIED_NET_FLOW,
)
from novomodelo_bridge.novomodelo.readers import NovomodeloReadError
from tests.novomodelo_output_builders import SolvedCaseSpec, build_solved_case

_STAGES = (0, 1)
_BUSES = ("SUDESTE", "SUL", "NORTE")
_PLANT = ("SUDESTE", "SUL")
_Lines = tuple[tuple[str, str, str], ...]
_TRIANGLE: _Lines = (
    ("SE-S", "SUDESTE", "SUL"),
    ("S-N", "SUL", "NORTE"),
    ("N-SE", "NORTE", "SUDESTE"),
)
_CHAIN = _TRIANGLE[:2]
_STUB: _Lines = (*_TRIANGLE, ("N-NE", "NORTE", "NORDESTE"))
_FOUR_BUSES = (*_BUSES, "NORDESTE")
_UNTIED = (1.0, 2.0, 4.0, 8.0)
_TIED = (1.0, 1.0, 2.0)
_ENTITY_TYPES = {
    "buses": "bus",
    "lines": "line",
    "hydros": "hydro",
    "thermals": "thermal",
}
_BASELINE_OFFSET = 100

_Garble = Callable[[list[ResultComparison]], list[ResultComparison]]
_Calls = list[tuple[list[ResultComparison], Path]]


def _case(
    tmp_path: Path,
    side: str,
    *,
    offset: int = 0,
    buses: Sequence[str] = _BUSES,
    lines: _Lines = _TRIANGLE,
    costs: Sequence[float] | None = None,
    plant: Sequence[str] = (),
) -> Path:
    """A solved case whose lines carry *costs* (not tying, by default) and whose
    first plant has unit groups on the buses of *plant*."""
    output = build_solved_case(
        tmp_path / side,
        SolvedCaseSpec(buses=tuple(buses), lines=lines, id_offset=offset),
    )
    path = output.parent / "system" / "lines.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    line_costs = _UNTIED[: len(lines)] if costs is None else costs
    for entry, cost in zip(document["lines"], line_costs, strict=True):
        entry["exchange_cost"] = cost
    path.write_text(json.dumps(document), encoding="utf-8")
    if plant:
        _share_plant(output, plant)
    return output


def _baseline(tmp_path: Path, **spec: Any) -> Path:
    return _case(tmp_path, "baseline", offset=_BASELINE_OFFSET, **spec)


def _candidate(tmp_path: Path, **spec: Any) -> Path:
    return _case(tmp_path, "candidate", **spec)


def _ids(output: Path, kind: str) -> dict[str, int]:
    path = output.parent / "system" / f"{kind}.json"
    entries = json.loads(path.read_text(encoding="utf-8"))[kind]
    return {name_key(entry["name"]): entry["id"] for entry in entries}


def _share_plant(output: Path, bus_names: Sequence[str]) -> None:
    path = output.parent / "system" / "hydros.json"
    ids = _ids(output, "buses")
    document = json.loads(path.read_text(encoding="utf-8"))
    groups = document["hydros"][0]["unit_groups"]
    document["hydros"][0]["unit_groups"] = [
        {**groups[0], "bus_id": ids[name_key(name)]} for name in bus_names
    ]
    path.write_text(json.dumps(document), encoding="utf-8")


def _rows(
    baseline: Path,
    candidate: Path,
    kind: str,
    names: Sequence[str],
    variable: str,
) -> list[ResultComparison]:
    """One row per name and stage; the value follows the name's position in *names*
    and the stage, and the candidate's is half a unit above the baseline's."""
    ours, theirs = _ids(baseline, kind), _ids(candidate, kind)
    return [
        result_row(
            _ENTITY_TYPES[kind],
            name,
            ours[name_key(name)],
            theirs[name_key(name)],
            stage,
            variable,
            10.0 * (position + 1) + stage,
            10.0 * (position + 1) + stage + 0.5,
        )
        for position, name in enumerate(names)
        for stage in _STAGES
    ]


def _line_rows(
    baseline: Path, candidate: Path, lines: _Lines
) -> list[ResultComparison]:
    names = [name for name, _, _ in lines]
    return _rows(baseline, candidate, "lines", names, "net_flow_mw")


def _bus_rows(
    baseline: Path, candidate: Path, names: Sequence[str] = _BUSES
) -> list[ResultComparison]:
    return _rows(baseline, candidate, "buses", names, "net_import_mw")


def _sides(baseline: Path, candidate: Path) -> tuple[SideInputs, SideInputs]:
    with dx.collect():
        return read_side(baseline, "baseline"), read_side(candidate, "candidate")


def _apply(
    baseline: Path, candidate: Path, rows: Sequence[ResultComparison]
) -> tuple[list[ResultComparison], list[dx.Diagnostic]]:
    baseline_side, candidate_side = _sides(baseline, candidate)
    with dx.collect() as sink:
        result = apply_tie_marks(rows, baseline_side, candidate_side)
    return result, sink


def _codes(sink: Sequence[dx.Diagnostic]) -> list[str]:
    return [diagnostic.code for diagnostic in sink]


def _prefixes(sink: Sequence[dx.Diagnostic]) -> list[str]:
    return sorted(diagnostic.summary.split(": ")[0] for diagnostic in sink)


def _tied(rows: Sequence[ResultComparison]) -> list[ResultComparison]:
    return [replace(row, variable=TIED_NET_FLOW) for row in rows]


def _in_shared_plant(rows: Sequence[ResultComparison]) -> list[ResultComparison]:
    """*rows* of the plant on ``SUDESTE`` and ``SUL`` as the marks leave them: the
    net imports of the two buses and the flow between them move."""
    return [
        replace(row, variable=SHARED_PLANT_NET_IMPORT)
        if row.entity_type == "bus" and row.variable == "net_import_mw"
        else replace(row, variable=SHARED_PLANT_NET_FLOW)
        if row.entity_name == "SE-S" and row.variable == "net_flow_mw"
        else row
        for row in rows
    ]


def _record(monkeypatch: pytest.MonkeyPatch, name: str) -> _Calls:
    real = getattr(ties, name)
    calls: _Calls = []

    def spy(rows: Sequence[ResultComparison], directory: Path) -> object:
        calls.append((list(rows), directory))
        return real(rows, directory)

    monkeypatch.setattr(ties, name, spy)
    return calls


def _break(
    monkeypatch: pytest.MonkeyPatch, name: str, directory: Path, garble: _Garble
) -> None:
    real = getattr(ties, name)

    def faulty(rows: Sequence[ResultComparison], where: Path) -> object:
        result = real(rows, where)
        return garble(result) if where == directory else result

    monkeypatch.setattr(ties, name, faulty)


class TestNoMarks:
    def test_no_tie_and_no_shared_plant_leaves_the_rows_and_the_sink_alone(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _baseline(tmp_path), _candidate(tmp_path)
        rows = _line_rows(baseline, candidate, _TRIANGLE) + _bus_rows(
            baseline, candidate
        )

        result, sink = _apply(baseline, candidate, rows)

        assert result == rows
        assert sink == []

    def test_rows_without_a_line_or_a_net_import_read_no_line_list(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _baseline(tmp_path), _candidate(tmp_path)
        (candidate.parent / "system" / "lines.json").write_text(
            "not json", encoding="utf-8"
        )
        rows = _rows(
            baseline, candidate, "hydros", ["FURNAS"], "generation_mw"
        ) + _rows(baseline, candidate, "thermals", ["GAS_A"], "generation_mw")

        result, sink = _apply(baseline, candidate, rows)

        assert result == rows
        assert sink == []

    def test_an_empty_input_gives_an_empty_result(self, tmp_path: Path) -> None:
        baseline, candidate = _baseline(tmp_path), _candidate(tmp_path)

        result, sink = _apply(baseline, candidate, [])

        assert result == []
        assert sink == []


class TestTiedLines:
    def test_a_loop_tied_in_the_candidate_only_moves_its_rows(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _baseline(tmp_path), _candidate(tmp_path, costs=_TIED)
        rows = _line_rows(baseline, candidate, _TRIANGLE)

        result, sink = _apply(baseline, candidate, rows)

        assert result == _tied(rows)
        assert _codes(sink) == ["line-routes-tied"]
        assert sink[0].summary.startswith("Candidate output: ")

    def test_a_loop_tied_in_the_baseline_only_moves_its_rows_by_the_baseline_ids(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _baseline(tmp_path, costs=_TIED), _candidate(tmp_path)
        rows = _line_rows(baseline, candidate, _TRIANGLE)

        result, sink = _apply(baseline, candidate, rows)

        assert result == _tied(rows)
        assert {row.novomodelo_id for row in result} == {0, 1, 2}
        assert {row.newave_code for row in result} == {100, 101, 102}
        assert _codes(sink) == ["line-routes-tied"]
        assert sink[0].summary.startswith("Baseline output: ")

    def test_a_loop_tied_in_both_cases_warns_once_per_case(
        self, tmp_path: Path
    ) -> None:
        baseline = _baseline(tmp_path, costs=_TIED)
        candidate = _candidate(tmp_path, costs=_TIED)
        rows = _line_rows(baseline, candidate, _TRIANGLE)

        result, sink = _apply(baseline, candidate, rows)

        assert result == _tied(rows)
        assert _codes(sink) == ["line-routes-tied", "line-routes-tied"]
        assert _prefixes(sink) == ["Baseline output", "Candidate output"]

    def test_a_line_off_the_tied_loop_keeps_its_variable(self, tmp_path: Path) -> None:
        baseline = _baseline(tmp_path, buses=_FOUR_BUSES, lines=_STUB)
        candidate = _candidate(
            tmp_path, buses=_FOUR_BUSES, lines=_STUB, costs=(*_TIED, 8.0)
        )
        rows = _line_rows(baseline, candidate, _STUB)

        result, _ = _apply(baseline, candidate, rows)

        assert {r.entity_name: r.variable for r in result} == {
            "SE-S": TIED_NET_FLOW,
            "S-N": TIED_NET_FLOW,
            "N-SE": TIED_NET_FLOW,
            "N-NE": "net_flow_mw",
        }


class TestSharedPlantBuses:
    def test_a_plant_shared_in_both_cases_moves_the_rows_and_adds_a_group_row_per_stage(
        self, tmp_path: Path
    ) -> None:
        baseline = _baseline(tmp_path, lines=_CHAIN, plant=_PLANT)
        candidate = _candidate(tmp_path, lines=_CHAIN, plant=_PLANT)
        rows = _bus_rows(baseline, candidate, _PLANT) + _line_rows(
            baseline, candidate, _CHAIN
        )

        result, sink = _apply(baseline, candidate, rows)

        outflow = {r.stage: r for r in rows if r.entity_name == "S-N"}
        group = [
            result_row(
                "bus",
                "SUDESTE+SUL",
                _BASELINE_OFFSET,
                0,
                stage,
                "net_import_mw",
                -outflow[stage].newave_value,
                -outflow[stage].novomodelo_value,
            )
            for stage in _STAGES
        ]
        assert result == [*_in_shared_plant(rows), *group]
        assert _codes(sink) == ["buses-share-plant", "buses-share-plant"]
        assert _prefixes(sink) == ["Baseline output", "Candidate output"]

    def test_the_group_row_has_the_candidate_ids_under_the_baseline_name(
        self, tmp_path: Path
    ) -> None:
        baseline = _baseline(
            tmp_path, buses=("SUL", "SUDESTE", "NORTE"), lines=_CHAIN, plant=_PLANT
        )
        candidate = _candidate(tmp_path, lines=_CHAIN, plant=_PLANT)
        rows = _bus_rows(baseline, candidate, _PLANT) + _line_rows(
            baseline, candidate, _CHAIN
        )

        result, _ = _apply(baseline, candidate, rows)

        group = result[len(rows) :]
        assert [(r.entity_name, r.newave_code, r.novomodelo_id) for r in group] == [
            ("SUL+SUDESTE", 101, 0)
        ] * len(_STAGES)

    def test_bus_names_are_compared_through_the_alignment_key(
        self, tmp_path: Path
    ) -> None:
        spelled = ("sudeste", " Sul", "NORTE")
        baseline = _baseline(
            tmp_path,
            buses=spelled,
            lines=(("SE-S", spelled[0], spelled[1]), ("S-N", spelled[1], spelled[2])),
            plant=_PLANT,
        )
        candidate = _candidate(tmp_path, lines=_CHAIN, plant=_PLANT)
        rows = _bus_rows(baseline, candidate, _PLANT) + _line_rows(
            baseline, candidate, _CHAIN
        )

        result, _ = _apply(baseline, candidate, rows)

        assert [r.entity_name for r in result[len(rows) :]] == ["sudeste+ Sul"] * len(
            _STAGES
        )

    @pytest.mark.parametrize(
        ("shared_in", "prefix"),
        [("baseline", "Baseline output: "), ("candidate", "Candidate output: ")],
    )
    def test_a_plant_shared_in_one_case_moves_its_rows_and_adds_no_group_row(
        self, tmp_path: Path, shared_in: str, prefix: str
    ) -> None:
        baseline = _baseline(
            tmp_path, lines=_CHAIN, plant=_PLANT if shared_in == "baseline" else ()
        )
        candidate = _candidate(
            tmp_path, lines=_CHAIN, plant=_PLANT if shared_in == "candidate" else ()
        )
        rows = _bus_rows(baseline, candidate, _PLANT) + _line_rows(
            baseline, candidate, _CHAIN
        )

        result, sink = _apply(baseline, candidate, rows)

        assert result == _in_shared_plant(rows)
        assert _codes(sink) == ["buses-share-plant"]
        assert sink[0].summary.startswith(prefix)

    def test_groups_with_different_members_move_every_member_and_add_no_group_row(
        self, tmp_path: Path
    ) -> None:
        baseline = _baseline(tmp_path, plant=_PLANT)
        candidate = _candidate(tmp_path, plant=("SUDESTE", "NORTE"))
        rows = _bus_rows(baseline, candidate) + _line_rows(
            baseline, candidate, _TRIANGLE
        )

        result, sink = _apply(baseline, candidate, rows)

        assert len(result) == len(rows)
        assert {r.variable for r in result if r.entity_type == "bus"} == {
            SHARED_PLANT_NET_IMPORT
        }
        assert {
            r.entity_name: r.variable for r in result if r.entity_type == "line"
        } == {
            "SE-S": SHARED_PLANT_NET_FLOW,
            "S-N": "net_flow_mw",
            "N-SE": SHARED_PLANT_NET_FLOW,
        }
        assert _codes(sink) == ["buses-share-plant", "buses-share-plant"]

    @pytest.mark.parametrize("extra_in", ["baseline", "candidate"])
    def test_a_boundary_line_of_one_case_only_adds_no_group_row(
        self, tmp_path: Path, extra_in: str
    ) -> None:
        baseline = _baseline(
            tmp_path,
            lines=_TRIANGLE if extra_in == "baseline" else _CHAIN,
            plant=_PLANT,
        )
        candidate = _candidate(
            tmp_path,
            lines=_TRIANGLE if extra_in == "candidate" else _CHAIN,
            plant=_PLANT,
        )
        rows = _bus_rows(baseline, candidate, _PLANT) + _line_rows(
            baseline, candidate, _CHAIN
        )

        result, sink = _apply(baseline, candidate, rows)

        assert result == _in_shared_plant(rows)
        assert _codes(sink) == ["buses-share-plant", "buses-share-plant"]

    def test_a_group_row_is_kept_only_at_the_stages_both_cases_build_it(
        self, tmp_path: Path
    ) -> None:
        # "X" crosses the group's boundary in the baseline and lies inside it in the
        # candidate, and only has a row at stage 0: the baseline builds the group
        # row at stage 0 only, the candidate at both stages.
        out = ("S-N", "SUL", "NORTE")
        baseline = _baseline(tmp_path, lines=(out, ("X", "SUL", "NORTE")), plant=_PLANT)
        candidate = _candidate(
            tmp_path, lines=(out, ("X", "SUDESTE", "SUL")), plant=_PLANT
        )
        flows = _rows(baseline, candidate, "lines", ["S-N", "X"], "net_flow_mw")
        rows = [
            *_bus_rows(baseline, candidate, _PLANT),
            *(r for r in flows if (r.entity_name, r.stage) != ("X", 1)),
        ]

        result, _ = _apply(baseline, candidate, rows)

        outflow = next(r for r in rows if (r.entity_name, r.stage) == ("S-N", 0))
        assert result[len(rows) :] == [
            result_row(
                "bus",
                "SUDESTE+SUL",
                _BASELINE_OFFSET,
                0,
                0,
                "net_import_mw",
                -outflow.newave_value,
                -outflow.novomodelo_value,
            )
        ]

    def test_a_tied_line_inside_a_group_keeps_the_tied_variable(
        self, tmp_path: Path
    ) -> None:
        baseline = _baseline(tmp_path, plant=_PLANT)
        candidate = _candidate(tmp_path, plant=_PLANT, costs=_TIED)
        rows = _bus_rows(baseline, candidate, _PLANT) + _line_rows(
            baseline, candidate, _TRIANGLE
        )

        result, sink = _apply(baseline, candidate, rows)

        marked = result[: len(rows)]
        assert {r.variable for r in marked if r.entity_type == "line"} == {
            TIED_NET_FLOW
        }
        assert {r.variable for r in marked if r.entity_type == "bus"} == {
            SHARED_PLANT_NET_IMPORT
        }
        assert [(r.entity_name, r.variable, r.stage) for r in result[len(rows) :]] == [
            ("SUDESTE+SUL", "net_import_mw", stage) for stage in _STAGES
        ]
        assert sorted(_codes(sink)) == [
            "buses-share-plant",
            "buses-share-plant",
            "line-routes-tied",
        ]


class TestPassThrough:
    def test_rows_of_other_types_and_variables_keep_their_place_and_value(
        self, tmp_path: Path
    ) -> None:
        baseline = _baseline(tmp_path, lines=_CHAIN, plant=_PLANT)
        candidate = _candidate(tmp_path, lines=_CHAIN, plant=_PLANT)
        hydro = _rows(baseline, candidate, "hydros", ["FURNAS"], "generation_mw")
        thermal = _rows(baseline, candidate, "thermals", ["GAS_A"], "generation_mw")
        price = _rows(baseline, candidate, "buses", _PLANT, "spot_price")
        imports = _bus_rows(baseline, candidate, _PLANT)
        flows = _line_rows(baseline, candidate, _CHAIN)
        rows = [*hydro, *flows[:2], *price, *thermal, *imports, *flows[2:]]

        result, _ = _apply(baseline, candidate, rows)

        assert result[: len(rows)] == _in_shared_plant(rows)
        assert len(result) == len(rows) + len(_STAGES)
        assert result[0 : len(hydro)] == hydro
        assert [r for r in result if r.variable == "spot_price"] == price


class TestMirroredView:
    @pytest.mark.parametrize("name", ["mark_tied_lines", "merge_shared_plant_buses"])
    def test_each_function_reads_the_rows_for_the_candidate_then_their_mirror_for_the_baseline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        baseline = _baseline(tmp_path, lines=_CHAIN, plant=_PLANT)
        candidate = _candidate(tmp_path, lines=_CHAIN, plant=_PLANT)
        rows = _bus_rows(baseline, candidate, _PLANT) + _line_rows(
            baseline, candidate, _CHAIN
        )
        calls = _record(monkeypatch, name)

        _apply(baseline, candidate, rows)

        (given, given_dir), (mirrored, mirrored_dir) = calls
        assert (given_dir, mirrored_dir) == (candidate, baseline)
        assert given == rows
        for row, mirror in zip(rows, mirrored, strict=True):
            assert (mirror.newave_code, mirror.novomodelo_id) == (
                row.novomodelo_id,
                row.newave_code,
            )
            assert (mirror.newave_value, mirror.novomodelo_value) == (
                row.novomodelo_value,
                row.newave_value,
            )
            assert (
                mirror.entity_type,
                mirror.entity_name,
                mirror.variable,
                mirror.stage,
                mirror.abs_diff,
                mirror.rel_diff,
            ) == (
                row.entity_type,
                row.entity_name,
                row.variable,
                row.stage,
                row.abs_diff,
                row.rel_diff,
            )

    def test_the_shared_plant_marking_reads_the_rows_the_tie_marking_returned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        baseline, candidate = _baseline(tmp_path, costs=_TIED), _candidate(tmp_path)
        rows = _line_rows(baseline, candidate, _TRIANGLE)
        calls = _record(monkeypatch, "merge_shared_plant_buses")

        _apply(baseline, candidate, rows)

        (given, _), (mirrored, _) = calls
        assert given == _tied(rows)
        assert {row.variable for row in mirrored} == {TIED_NET_FLOW}


class TestInvariants:
    @pytest.mark.parametrize(
        ("name", "side", "garble"),
        [
            ("mark_tied_lines", "candidate", lambda rows: rows[:-1]),
            ("mark_tied_lines", "baseline", lambda rows: rows[:-1]),
            ("mark_tied_lines", "candidate", lambda rows: [*rows, rows[0]]),
            ("mark_tied_lines", "baseline", lambda rows: [*rows, rows[0]]),
            ("merge_shared_plant_buses", "candidate", lambda rows: rows[:-1]),
            ("merge_shared_plant_buses", "baseline", lambda rows: rows[:-1]),
        ],
        ids=[
            "tied-candidate-short",
            "tied-baseline-short",
            "tied-candidate-long",
            "tied-baseline-long",
            "shared-candidate-short",
            "shared-baseline-short",
        ],
    )
    def test_a_pass_with_the_wrong_row_count_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        side: str,
        garble: _Garble,
    ) -> None:
        baseline, candidate = _baseline(tmp_path), _candidate(tmp_path)
        rows = _line_rows(baseline, candidate, _TRIANGLE)
        baseline_side, candidate_side = _sides(baseline, candidate)
        _break(
            monkeypatch, name, candidate if side == "candidate" else baseline, garble
        )

        with pytest.raises(ValueError, match="tie-marking pass"), dx.collect():
            apply_tie_marks(rows, baseline_side, candidate_side)

    @pytest.mark.parametrize("name", ["mark_tied_lines", "merge_shared_plant_buses"])
    def test_a_reader_failure_propagates_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        baseline, candidate = _baseline(tmp_path), _candidate(tmp_path)
        rows = _line_rows(baseline, candidate, _TRIANGLE)
        failure = NovomodeloReadError("system/lines.json is unreadable")

        def fail(rows: Sequence[ResultComparison], directory: Path) -> object:
            raise failure

        monkeypatch.setattr(ties, name, fail)

        with pytest.raises(NovomodeloReadError) as raised:
            _apply(baseline, candidate, rows)

        assert raised.value is failure

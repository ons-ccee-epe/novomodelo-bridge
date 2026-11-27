"""Tests of ``comparators.version.operation``: the cleared variables, the per-side
frames and the operation rows.

End-to-end cases read ``build_solved_case`` trees damaged with the builder's
helpers. The realized flows, the line direction and the empty cases use literal
frames; the hydro fold patches the public reader entry points of the module
under test.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NoReturn

import polars as pl
import pytest

from novomodelo_bridge.comparators.model import ResultComparison
from novomodelo_bridge.comparators.version import operation
from novomodelo_bridge.comparators.version.alignment import (
    AlignedPair,
    NameAlignment,
    align_by_name,
)
from novomodelo_bridge.comparators.version.availability import (
    SectionAvailability,
    Unavailable,
    with_entity_lists,
)
from novomodelo_bridge.comparators.version.operation import (
    ROW_GROUPS,
    operation_rows,
    operation_variables,
    read_operation_frames,
)
from novomodelo_bridge.comparators.version.requirements import (
    BUS,
    BUS_NET_IMPORT,
    HYDRO,
    HYDRO_BOUNDS,
    LINE,
    REQUIREMENTS,
    SYSTEM_SPILLAGE,
    THERMAL,
    decide_sections,
)
from novomodelo_bridge.comparators.version.sides import SideInputs, read_side
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.agreement import ENTITY_GROUPS, VARIABLE_LABELS
from novomodelo_bridge.novomodelo.readers import NovomodeloReadError
from tests.novomodelo_output_builders import (
    SolvedCaseSpec,
    build_solved_case,
    corrupt_file,
    drop_column,
)

_STAGES = (0, 1)
_FULL = SolvedCaseSpec(load_stats=True)
_HYDRO_VARIABLES = (
    "storage_final_hm3",
    "generation_mw",
    "turbined_m3s",
    "spillage_m3s",
    "inflow_m3s",
    "water_value_per_hm3",
    "evaporation_m3s",
    "outflow_m3s",
    "total_inflow_m3s",
    "productivity_mw_per_m3s",
)
_SPILLAGE_VARIABLES = (
    "spill_energy_total_mw",
    "spill_energy_reservoir_mw",
    "spill_energy_rorov_mw",
)
_ALL_VARIABLES = {
    "hydro": (*_HYDRO_VARIABLES, "withdrawal_m3s"),
    "thermal": ("generation_mw",),
    "bus": ("spot_price", "deficit_mw", "net_import_mw"),
    "line": ("net_flow_mw",),
    "system_spillage": _SPILLAGE_VARIABLES,
}
_SLACKS = (
    "evaporation_violation_pos_m3s",
    "evaporation_violation_neg_m3s",
    "water_withdrawal_violation_pos_m3s",
    "water_withdrawal_violation_neg_m3s",
)
_BUS_IDS = {"SUDESTE": 1, "SUL": 2, "NORTE": 3, "ABSENT": 9}
_Ends = tuple[str | None, str | None]


def _available(section: str, *variables: str) -> SectionAvailability:
    return SectionAvailability(section, True, variables, ())


def _unavailable(section: str) -> SectionAvailability:
    return SectionAvailability(
        section, False, (), (Unavailable(section, None, "both"),)
    )


def _sections(*replacements: SectionAvailability) -> dict[str, SectionAvailability]:
    sections = {
        name: _available(name, *requirement.variables)
        for name, requirement in REQUIREMENTS.items()
    }
    return {**sections, **{r.section: r for r in replacements}}


def _frame(
    ids: Sequence[int], stages: Sequence[int], **variables: Sequence[float | None]
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": pl.Series(ids, dtype=pl.Int64),
            "stage_id": pl.Series(stages, dtype=pl.Int64),
            **{
                name: pl.Series(values, dtype=pl.Float64)
                for name, values in variables.items()
            },
        }
    )


def _side(
    side: str,
    *,
    offset: int = 0,
    lines: Sequence[_Ends] = (),
    bus_spellings: Sequence[str] = ("SUDESTE", "SUL", "NORTE"),
) -> SideInputs:
    """A side with the buses of *bus_spellings* and one line per entry of *lines*,
    named ``L<position>``; an end that is ``None`` or ``ABSENT`` has no bus."""
    entries: list[dict[str, object]] = []
    for position, ends in enumerate(lines):
        line: dict[str, object] = {"id": offset + position, "name": f"L{position}"}
        for key, end in zip(("source_bus_id", "target_bus_id"), ends, strict=True):
            if end is not None:
                line[key] = offset + _BUS_IDS[end]
        entries.append(line)
    return SideInputs(
        side=side,
        output_dir=Path("unused"),
        version=None,
        hydros=[],
        buses=[
            {"id": offset + _BUS_IDS[name.strip().upper()], "name": name}
            for name in bus_spellings
        ],
        thermals=[],
        lines=entries,
        risk_aversion={},
        partitions={},
    )


def _alignment(entity_type: str, *pairs: AlignedPair) -> NameAlignment:
    return NameAlignment(entity_type, pairs, (), (), ())


def _loud(sink: Sequence[dx.Diagnostic]) -> list[dx.Diagnostic]:
    return [d for d in sink if d.severity is not dx.Severity.INFO]


def _types_in_order(rows: Sequence[ResultComparison]) -> list[str]:
    return [key for key, _ in itertools.groupby(row.entity_type for row in rows)]


class TestOperationVariables:
    def test_every_section_available_clears_every_variable_in_row_group_order(
        self,
    ) -> None:
        cleared = operation_variables(_sections())

        assert list(cleared) == list(ROW_GROUPS)
        assert cleared == _ALL_VARIABLES

    def test_the_hydro_variables_follow_the_requirement_then_the_withdrawal(
        self,
    ) -> None:
        cleared = operation_variables(_sections())

        assert cleared["hydro"] == (*REQUIREMENTS[HYDRO].variables, "withdrawal_m3s")

    def test_the_names_are_the_ones_the_agreement_labels(self) -> None:
        cleared = operation_variables(_sections())

        assert set(ROW_GROUPS) <= set(ENTITY_GROUPS)
        assert set(itertools.chain.from_iterable(cleared.values())) <= set(
            VARIABLE_LABELS
        )

    def test_unavailable_bounds_leave_the_withdrawal_out(self) -> None:
        cleared = operation_variables(_sections(_unavailable(HYDRO_BOUNDS)))

        assert cleared == {**_ALL_VARIABLES, "hydro": _HYDRO_VARIABLES}

    def test_unavailable_net_import_leaves_the_bus_with_two_variables(self) -> None:
        cleared = operation_variables(_sections(_unavailable(BUS_NET_IMPORT)))

        assert cleared["bus"] == ("spot_price", "deficit_mw")

    @pytest.mark.parametrize(
        ("section", "entity_type"),
        [
            (HYDRO, "hydro"),
            (THERMAL, "thermal"),
            (BUS, "bus"),
            (LINE, "line"),
            (SYSTEM_SPILLAGE, "system_spillage"),
        ],
    )
    def test_an_unavailable_section_leaves_its_entity_type_out(
        self, section: str, entity_type: str
    ) -> None:
        sections = _sections(
            _unavailable(section),
            _unavailable(HYDRO_BOUNDS),
            _unavailable(BUS_NET_IMPORT),
        )

        cleared = operation_variables(sections)

        assert entity_type not in cleared
        assert list(cleared) == [t for t in ROW_GROUPS if t != entity_type]

    @pytest.mark.parametrize(
        ("section", "entity_type"),
        [
            (HYDRO, "hydro"),
            (THERMAL, "thermal"),
            (BUS, "bus"),
            (LINE, "line"),
            (SYSTEM_SPILLAGE, "system_spillage"),
        ],
    )
    def test_an_available_section_without_a_cleared_variable_leaves_its_type_out(
        self, section: str, entity_type: str
    ) -> None:
        sections = _sections(
            _available(section),
            _available(HYDRO_BOUNDS),
            _available(BUS_NET_IMPORT),
        )

        cleared = operation_variables(sections)

        assert entity_type not in cleared

    def test_a_bus_clearing_only_the_net_import_keeps_that_variable(self) -> None:
        cleared = operation_variables(_sections(_available(BUS)))

        assert cleared["bus"] == ("net_import_mw",)

    def test_every_section_unavailable_clears_nothing(self) -> None:
        sections = {name: _unavailable(name) for name in REQUIREMENTS}

        assert operation_variables(sections) == {}


@dataclass(frozen=True)
class _Run:
    variables: dict[str, tuple[str, ...]]
    baseline_frames: dict[str, pl.DataFrame]
    candidate_frames: dict[str, pl.DataFrame]
    rows: list[ResultComparison]
    diagnostics: list[dx.Diagnostic]


def _alignments(
    baseline: SideInputs, candidate: SideInputs
) -> dict[str, NameAlignment]:
    lists = {
        "hydro": (baseline.hydros, candidate.hydros),
        "bus": (baseline.buses, candidate.buses),
        "thermal": (baseline.thermals, candidate.thermals),
        "line": (baseline.lines, candidate.lines),
    }
    return {
        entity_type: align_by_name(entity_type, ours, theirs)
        for entity_type, (ours, theirs) in lists.items()
        if ours is not None and theirs is not None
    }


def _availability(side: SideInputs) -> dict[str, frozenset[str] | None]:
    return with_entity_lists(side.partitions, side.usable_entity_lists)


def _run(
    baseline_output: Path,
    candidate_output: Path,
    stages: Sequence[int] = _STAGES,
    *,
    extra: Mapping[str, tuple[str, ...]] | None = None,
) -> _Run:
    with dx.collect() as sink:
        baseline = read_side(baseline_output, "baseline")
        candidate = read_side(candidate_output, "candidate")
        sections = decide_sections(_availability(baseline), _availability(candidate))
        variables = {**operation_variables(sections), **(extra or {})}
        baseline_frames = read_operation_frames(baseline, variables)
        candidate_frames = read_operation_frames(candidate, variables)
        rows = operation_rows(
            baseline,
            candidate,
            baseline_frames,
            candidate_frames,
            _alignments(baseline, candidate),
            variables,
            stages,
        )
    return _Run(variables, baseline_frames, candidate_frames, rows, sink)


def _pair(
    tmp_path: Path, baseline: SolvedCaseSpec, candidate: SolvedCaseSpec
) -> tuple[Path, Path]:
    return (
        build_solved_case(tmp_path / "baseline", baseline),
        build_solved_case(tmp_path / "candidate", candidate),
    )


def _variables_of(rows: Sequence[ResultComparison]) -> set[tuple[str, str]]:
    return {(row.entity_type, row.variable) for row in rows}


class TestSolvedCases:
    def test_identical_runs_with_distinct_ids_differ_by_exactly_zero(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _pair(tmp_path, _FULL, replace(_FULL, id_offset=100))

        run = _run(baseline, candidate)

        assert run.rows
        assert {row.abs_diff for row in run.rows} == {0.0}
        assert {
            row.novomodelo_id - row.newave_code
            for row in run.rows
            if row.entity_type != "system_spillage"
        } == {100}
        assert {
            (row.newave_code, row.novomodelo_id)
            for row in run.rows
            if row.entity_type == "system_spillage"
        } == {(0, 0)}
        assert _variables_of(run.rows) == {
            (entity_type, variable)
            for entity_type, variables in _ALL_VARIABLES.items()
            for variable in variables
            if variable != "withdrawal_m3s"
        }
        assert _loud(run.diagnostics) == []

    def test_rows_follow_the_row_groups_then_the_pairs(self, tmp_path: Path) -> None:
        baseline, candidate = _pair(tmp_path, _FULL, _FULL)

        run = _run(baseline, candidate)

        assert _types_in_order(run.rows) == list(ROW_GROUPS)
        hydro_names = [
            row.entity_name
            for row in run.rows
            if (row.entity_type, row.variable) == ("hydro", "generation_mw")
        ]
        assert hydro_names == ["FURNAS", "FURNAS", "ITAIPU", "ITAIPU"]

    def test_a_scaled_candidate_moves_the_flows_and_not_the_productivity(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _pair(
            tmp_path, _FULL, SolvedCaseSpec(load_stats=True, scale=1.1)
        )

        rows = _run(baseline, candidate).rows

        generation = [
            row
            for row in rows
            if (row.entity_type, row.variable) == ("hydro", "generation_mw")
        ]
        productivity = [
            row
            for row in rows
            if (row.entity_type, row.variable) == ("hydro", "productivity_mw_per_m3s")
        ]
        assert generation
        assert [row.rel_diff for row in generation] == pytest.approx([0.1] * 4)
        assert productivity
        assert [row.abs_diff for row in productivity] == pytest.approx([0.0] * 4)

    def test_a_line_the_candidate_runs_the_other_way_round_has_its_baseline_negated(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _pair(
            tmp_path,
            _FULL,
            SolvedCaseSpec(load_stats=True, lines=(("SE-S", "SUL", "SUDESTE"),)),
        )

        rows = [r for r in _run(baseline, candidate).rows if r.entity_type == "line"]

        assert [(row.entity_name, row.stage) for row in rows] == [
            ("SE-S", 0),
            ("SE-S", 1),
        ]
        assert [row.newave_value for row in rows] == pytest.approx(
            [-row.novomodelo_value for row in rows]
        )
        assert all(row.newave_value < 0 < row.novomodelo_value for row in rows)

    @pytest.mark.parametrize(
        ("field", "entity_type"), [("thermals", "thermal"), ("lines", "line")]
    )
    def test_a_case_without_an_entity_type_builds_no_frame_and_no_row_for_it(
        self, tmp_path: Path, field: str, entity_type: str
    ) -> None:
        spec = replace(_FULL, **{field: ()})
        baseline, candidate = _pair(tmp_path, spec, spec)

        run = _run(baseline, candidate)

        assert entity_type not in run.variables
        assert entity_type not in run.baseline_frames
        assert entity_type not in run.candidate_frames
        assert entity_type not in {row.entity_type for row in run.rows}
        assert {"hydro", "bus", "system_spillage"} <= {
            row.entity_type for row in run.rows
        }
        assert run.diagnostics == []

    def test_a_case_without_lines_has_no_net_import_row(self, tmp_path: Path) -> None:
        spec = SolvedCaseSpec(load_stats=True, lines=())
        baseline, candidate = _pair(tmp_path, spec, spec)

        rows = _run(baseline, candidate).rows

        assert ("bus", "spot_price") in _variables_of(rows)
        assert ("bus", "net_import_mw") not in _variables_of(rows)

    def test_a_thermal_list_one_side_cannot_read_leaves_the_thermals_out(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _pair(tmp_path, _FULL, _FULL)
        (candidate.parent / "system" / "thermals.json").write_text(
            json.dumps({"thermals": [{"id": 1}]}), encoding="utf-8"
        )

        run = _run(baseline, candidate)

        assert "thermal" not in run.variables
        assert "thermal" not in run.baseline_frames
        assert "thermal" not in run.candidate_frames
        assert "thermal" not in {row.entity_type for row in run.rows}
        assert {"hydro", "bus"} <= {row.entity_type for row in run.rows}
        assert [d.code for d in run.diagnostics] == ["report-file-unreadable"]

    @pytest.mark.parametrize("damaged", ["baseline", "candidate"])
    def test_a_dropped_thermal_generation_column_builds_no_thermal_row_and_no_warning(
        self, tmp_path: Path, damaged: str
    ) -> None:
        baseline, candidate = _pair(tmp_path, _FULL, _FULL)
        drop_column(
            {"baseline": baseline, "candidate": candidate}[damaged],
            "simulation/thermals",
            "generation_mw",
        )

        run = _run(baseline, candidate)

        assert "thermal" not in run.variables
        assert "thermal" not in run.baseline_frames
        assert "thermal" not in run.candidate_frames
        assert "thermal" not in {row.entity_type for row in run.rows}
        assert run.diagnostics == []

    def test_a_variable_one_side_cannot_supply_leaves_both_sides_rows_out(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _pair(tmp_path, _FULL, _FULL)
        drop_column(candidate, "simulation/hydros", "incremental_inflow_m3s")

        run = _run(baseline, candidate)

        assert "total_inflow_m3s" not in run.variables["hydro"]
        assert ("hydro", "total_inflow_m3s") not in _variables_of(run.rows)
        assert ("hydro", "generation_mw") in _variables_of(run.rows)
        assert run.diagnostics == []

    def test_a_withdrawal_no_target_supplies_is_zero_and_builds_rows(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _pair(tmp_path, _FULL, _FULL)

        run = _run(
            baseline, candidate, extra={"hydro": (*_HYDRO_VARIABLES, "withdrawal_m3s")}
        )

        assert run.baseline_frames["hydro"]["withdrawal_m3s"].to_list() == [0.0] * 4
        withdrawal = [r for r in run.rows if r.variable == "withdrawal_m3s"]
        assert len(withdrawal) == 4
        assert {(r.newave_value, r.novomodelo_value) for r in withdrawal} == {
            (0.0, 0.0)
        }

    def test_only_the_stages_passed_are_built(self, tmp_path: Path) -> None:
        baseline, candidate = _pair(tmp_path, _FULL, _FULL)

        rows = _run(baseline, candidate, stages=(1,)).rows

        assert {row.stage for row in rows} == {1}


class TestReadOperationFrames:
    def _read(self, tmp_path: Path) -> tuple[SideInputs, Path]:
        output = build_solved_case(tmp_path / "case", _FULL)
        with dx.collect():
            return read_side(output, "candidate"), output

    def test_only_the_entity_types_of_the_variables_are_read(
        self, tmp_path: Path
    ) -> None:
        side, _ = self._read(tmp_path)

        assert list(read_operation_frames(side, {"bus": ("spot_price",)})) == ["bus"]
        assert read_operation_frames(side, {}) == {}

    def test_each_frame_holds_the_keys_and_the_variables(self, tmp_path: Path) -> None:
        side, _ = self._read(tmp_path)
        variables = {**_ALL_VARIABLES, "hydro": _HYDRO_VARIABLES}

        frames = read_operation_frames(side, variables)

        assert list(frames) == list(ROW_GROUPS)
        for entity_type, columns in variables.items():
            assert {"entity_id", "stage_id", *columns} <= set(
                frames[entity_type].columns
            )
        assert frames["system_spillage"]["entity_id"].to_list() == [0, 0]

    def test_a_reader_warning_reaches_the_sink_tagged_with_the_side(
        self, tmp_path: Path
    ) -> None:
        side, output = self._read(tmp_path)
        drop_column(output, "simulation/buses", "spot_price")
        drop_column(output, "simulation/buses", "deficit_mw")

        with dx.collect() as sink:
            read_operation_frames(side, {"bus": ("spot_price",)})

        assert [d.code for d in sink] == ["report-file-unreadable"]
        assert sink[0].summary.startswith("Candidate output: ")

    def test_a_present_file_a_reader_cannot_read_raises(self, tmp_path: Path) -> None:
        side, output = self._read(tmp_path)
        corrupt_file(output, "simulation/thermals")

        with pytest.raises(NovomodeloReadError):
            read_operation_frames(side, {"thermal": ("generation_mw",)})


def _means(**columns: Sequence[float | None]) -> pl.DataFrame:
    count = len(next(iter(columns.values())))
    nulls = [None] * count
    names = (*_SLACKS, "evaporation_m3s", "turbined_m3s", "generation_mw")
    return _frame(
        [0] * count, range(count), **{**dict.fromkeys(names, nulls), **columns}
    )


def _compose_hydro(
    monkeypatch: pytest.MonkeyPatch,
    means: pl.DataFrame,
    variables: Sequence[str],
    *,
    withdrawal: pl.DataFrame | None = None,
) -> pl.DataFrame:
    monkeypatch.setattr(operation, "read_novomodelo_hydro_means", lambda _dir: means)
    if withdrawal is not None:
        monkeypatch.setattr(
            operation, "read_novomodelo_hydro_withdrawal", lambda _dir: withdrawal
        )
    side = _side("baseline")
    return read_operation_frames(side, {"hydro": variables})["hydro"]


def _must_not_run(*_args: object) -> NoReturn:
    raise AssertionError("the reader must not run")


class TestHydroFrame:
    def test_a_positive_slack_adds_and_a_negative_one_subtracts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        means = _means(
            evaporation_m3s=[5.0, 5.0, 5.0],
            evaporation_violation_pos_m3s=[2.0, None, 2.0],
            evaporation_violation_neg_m3s=[None, 1.5, 1.0],
        )

        frame = _compose_hydro(monkeypatch, means, ["evaporation_m3s"])

        assert frame["evaporation_m3s"].to_list() == [7.0, 3.5, 6.0]

    def test_a_missing_evaporation_reads_as_zero_before_its_slacks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        means = _means(
            evaporation_m3s=[None, None],
            evaporation_violation_pos_m3s=[2.0, None],
            evaporation_violation_neg_m3s=[1.0, None],
        )

        frame = _compose_hydro(monkeypatch, means, ["evaporation_m3s"])

        assert frame["evaporation_m3s"].to_list() == [1.0, 0.0]

    def test_the_withdrawal_is_the_target_less_its_slacks_and_a_missing_target_is_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        means = _means(
            water_withdrawal_violation_pos_m3s=[None, 3.0, 3.0],
            water_withdrawal_violation_neg_m3s=[4.0, 1.0, None],
            evaporation_m3s=[0.0] * 3,
        )
        withdrawal = _frame([0, 0], [0, 1], withdrawal_m3s=[10.0, 10.0])

        frame = _compose_hydro(
            monkeypatch, means, ["withdrawal_m3s"], withdrawal=withdrawal
        )

        assert frame["withdrawal_m3s"].to_list() == [6.0, 12.0, 3.0]

    def test_a_withdrawal_frame_without_rows_leaves_the_slacks_over_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        means = _means(water_withdrawal_violation_pos_m3s=[3.0, 3.0])
        withdrawal = pl.DataFrame(
            schema={
                "entity_id": pl.Int64,
                "stage_id": pl.Int64,
                "withdrawal_m3s": pl.Float64,
            }
        )

        frame = _compose_hydro(
            monkeypatch, means, ["withdrawal_m3s"], withdrawal=withdrawal
        )

        assert frame["withdrawal_m3s"].to_list() == [3.0, 3.0]

    @pytest.mark.parametrize("dtype", [pl.Int16, pl.UInt32, pl.Float64])
    def test_keys_of_another_dtype_join_as_int64(
        self, monkeypatch: pytest.MonkeyPatch, dtype: pl.DataType
    ) -> None:
        means = _means(evaporation_m3s=[1.0, 1.0]).cast(
            {"entity_id": pl.Int32, "stage_id": pl.Int32}
        )
        withdrawal = _frame([0, 0], [0, 1], withdrawal_m3s=[10.0, 20.0]).cast(
            {"entity_id": dtype, "stage_id": dtype}
        )

        frame = _compose_hydro(
            monkeypatch, means, ["withdrawal_m3s"], withdrawal=withdrawal
        )

        assert frame["withdrawal_m3s"].to_list() == [10.0, 20.0]
        assert frame.schema["entity_id"] == pl.Int64
        assert frame.schema["stage_id"] == pl.Int64

    def test_the_total_flows_and_the_withdrawal_are_read_only_when_cleared(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            operation, "read_novomodelo_hydro_total_flows", _must_not_run
        )
        monkeypatch.setattr(
            operation, "read_novomodelo_hydro_withdrawal", _must_not_run
        )

        frame = _compose_hydro(
            monkeypatch, _means(evaporation_m3s=[1.0]), ["evaporation_m3s"]
        )

        assert "total_inflow_m3s" not in frame.columns
        assert "withdrawal_m3s" not in frame.columns

    @pytest.mark.parametrize(
        ("turbined", "expected"),
        [
            (4.0, 2.0),
            (2.0e-6, 4.0e6),
            (1.0e-6, None),
            (0.0, None),
            (None, None),
        ],
    )
    def test_the_productivity_is_generation_over_a_turbined_flow_above_the_threshold(
        self,
        monkeypatch: pytest.MonkeyPatch,
        turbined: float | None,
        expected: float | None,
    ) -> None:
        means = _means(turbined_m3s=[turbined], generation_mw=[8.0])

        frame = _compose_hydro(monkeypatch, means, ["productivity_mw_per_m3s"])

        (value,) = frame["productivity_mw_per_m3s"].to_list()
        assert value == (None if expected is None else pytest.approx(expected))

    def test_the_spec_example_folds_the_slack_and_drops_the_zero_turbining(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        means = _means(
            evaporation_m3s=[5.0, 5.0],
            evaporation_violation_pos_m3s=[2.0, 2.0],
            turbined_m3s=[4.0, 0.0],
            generation_mw=[8.0, 0.0],
        )

        frame = _compose_hydro(
            monkeypatch, means, ["evaporation_m3s", "productivity_mw_per_m3s"]
        )

        assert frame["evaporation_m3s"].to_list() == [7.0, 7.0]
        assert frame["productivity_mw_per_m3s"].to_list() == [2.0, None]

    def test_the_spillage_frame_renames_its_columns_and_sits_on_entity_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spillage = pl.DataFrame(
            {
                "stage_id": [0, 1],
                "total_mw": [3.0, 4.0],
                "reservoir_mw": [1.0, 1.5],
                "rorov_mw": [2.0, 2.5],
            }
        )
        monkeypatch.setattr(
            operation, "read_novomodelo_spillage_energy", lambda _dir: spillage
        )

        frame = read_operation_frames(
            _side("baseline"), {"system_spillage": _SPILLAGE_VARIABLES}
        )["system_spillage"]

        assert frame.select("entity_id", "stage_id", *_SPILLAGE_VARIABLES).rows() == [
            (0, 0, 3.0, 1.0, 2.0),
            (0, 1, 4.0, 1.5, 2.5),
        ]
        assert frame.schema["entity_id"] == pl.Int64


def _flows(offset: int, **variables: Sequence[float | None]) -> pl.DataFrame:
    count = len(next(iter(variables.values())))
    return _frame([offset] * count, range(count), **variables)


class TestOperationRows:
    def _rows(
        self,
        baseline_frames: Mapping[str, pl.DataFrame],
        candidate_frames: Mapping[str, pl.DataFrame],
        alignments: Mapping[str, NameAlignment],
        variables: Mapping[str, Sequence[str]],
        stages: Sequence[int] = _STAGES,
    ) -> list[ResultComparison]:
        return operation_rows(
            _side("baseline"),
            _side("candidate", offset=10),
            baseline_frames,
            candidate_frames,
            alignments,
            variables,
            stages,
        )

    def test_a_withdrawal_target_in_one_run_only_differs_from_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        means = _means(water_withdrawal_violation_pos_m3s=[0.0, 0.0])
        target = _frame([0, 0], [0, 1], withdrawal_m3s=[10.0, 8.0])
        baseline = _compose_hydro(
            monkeypatch, means, ["withdrawal_m3s"], withdrawal=target
        )
        candidate = _compose_hydro(
            monkeypatch, means, ["withdrawal_m3s"], withdrawal=target.clear()
        )

        rows = self._rows(
            {"hydro": baseline},
            {"hydro": candidate},
            {"hydro": _alignment("hydro", AlignedPair("A", 0, 0))},
            {"hydro": ("withdrawal_m3s",)},
        )

        assert [
            (r.stage, r.newave_value, r.novomodelo_value, r.abs_diff, r.rel_diff)
            for r in rows
        ] == [(0, 10.0, 0.0, 10.0, 1.0), (1, 8.0, 0.0, 8.0, 1.0)]

    def test_a_bus_without_load_builds_no_net_import_row(self) -> None:
        baseline = {
            "bus": _flows(1, spot_price=[100.0, 110.0], net_import_mw=[None, 20.0])
        }
        candidate = {
            "bus": _flows(11, spot_price=[100.0, 110.0], net_import_mw=[5.0, 20.0])
        }

        rows = self._rows(
            baseline,
            candidate,
            {"bus": _alignment("bus", AlignedPair("SUDESTE", 1, 11))},
            {"bus": ("spot_price", "net_import_mw")},
        )

        assert [(row.stage, row.variable) for row in rows] == [
            (0, "spot_price"),
            (1, "spot_price"),
            (1, "net_import_mw"),
        ]

    def test_system_spillage_pairs_the_single_sin_entity_without_an_alignment(
        self,
    ) -> None:
        frame = _flows(0, spill_energy_total_mw=[3.0, 4.0])

        rows = self._rows(
            {"system_spillage": frame},
            {"system_spillage": frame},
            {},
            {"system_spillage": ("spill_energy_total_mw",)},
        )

        assert [
            (r.entity_type, r.entity_name, r.newave_code, r.novomodelo_id, r.stage)
            for r in rows
        ] == [
            ("system_spillage", "SIN", 0, 0, 0),
            ("system_spillage", "SIN", 0, 0, 1),
        ]

    @pytest.mark.parametrize("missing", ["variables", "baseline", "candidate"])
    def test_an_entity_type_missing_from_one_input_builds_no_row(
        self, missing: str
    ) -> None:
        thermal = _flows(1, generation_mw=[10.0, 11.0])
        inputs: dict[str, dict] = {
            "variables": {"thermal": ("generation_mw",)},
            "baseline": {"thermal": thermal},
            "candidate": {"thermal": thermal},
        }
        inputs[missing] = {}

        rows = self._rows(
            inputs["baseline"],
            inputs["candidate"],
            {"thermal": _alignment("thermal", AlignedPair("GAS_A", 1, 1))},
            inputs["variables"],
        )

        assert rows == []

    def test_a_type_without_an_alignment_builds_no_row_and_leaves_the_others(
        self,
    ) -> None:
        thermal = _flows(1, generation_mw=[10.0, 11.0])
        spillage = _flows(0, spill_energy_total_mw=[3.0, 4.0])
        frames = {"thermal": thermal, "system_spillage": spillage}

        rows = self._rows(
            frames,
            frames,
            {},
            {
                "thermal": ("generation_mw",),
                "system_spillage": ("spill_energy_total_mw",),
            },
        )

        assert _types_in_order(rows) == ["system_spillage"]

    def test_rows_follow_the_row_groups_whatever_the_order_of_the_inputs(self) -> None:
        frames = {
            "system_spillage": _flows(0, spill_energy_total_mw=[1.0]),
            "line": _flows(0, net_flow_mw=[1.0]),
            "bus": _flows(0, spot_price=[1.0]),
            "thermal": _flows(0, generation_mw=[1.0]),
            "hydro": _flows(0, generation_mw=[1.0]),
        }
        variables = {
            "system_spillage": ("spill_energy_total_mw",),
            "line": ("net_flow_mw",),
            "bus": ("spot_price",),
            "thermal": ("generation_mw",),
            "hydro": ("generation_mw",),
        }
        alignments = {
            entity_type: _alignment(entity_type, AlignedPair("A", 0, 0))
            for entity_type in ("line", "bus", "thermal", "hydro")
        }
        sides = (
            _side("baseline", lines=[("SUDESTE", "SUL")]),
            _side("candidate", lines=[("SUDESTE", "SUL")]),
        )

        rows = operation_rows(*sides, frames, frames, alignments, variables, (0,))

        assert _types_in_order(rows) == list(ROW_GROUPS)

    def test_a_pair_reads_each_side_by_its_own_id(self) -> None:
        baseline = {"thermal": _frame([1, 2], [0, 0], generation_mw=[100.0, 200.0])}
        candidate = {"thermal": _frame([11, 12], [0, 0], generation_mw=[200.0, 100.0])}
        pairs = (AlignedPair("A", 1, 12), AlignedPair("B", 2, 11))

        rows = self._rows(
            baseline,
            candidate,
            {"thermal": _alignment("thermal", *pairs)},
            {"thermal": ("generation_mw",)},
            (0,),
        )

        assert [(row.entity_name, row.abs_diff) for row in rows] == [
            ("A", 0.0),
            ("B", 0.0),
        ]

    def test_a_repeated_row_for_one_entity_and_stage_raises(self) -> None:
        repeated = _frame([1, 1], [0, 0], generation_mw=[100.0, 101.0])

        with pytest.raises(ValueError, match="thermal"):
            self._rows(
                {"thermal": repeated},
                {"thermal": _frame([11], [0], generation_mw=[100.0])},
                {"thermal": _alignment("thermal", AlignedPair("A", 1, 11))},
                {"thermal": ("generation_mw",)},
                (0,),
            )


_SE_S: _Ends = ("SUDESTE", "SUL")
_S_SE: _Ends = ("SUL", "SUDESTE")


def _line_row(
    baseline: SideInputs,
    candidate: SideInputs,
    baseline_flows: Sequence[float | None],
    candidate_flows: Sequence[float | None],
    pairs: Sequence[AlignedPair],
) -> list[ResultComparison]:
    count = len(baseline_flows)
    return operation_rows(
        baseline,
        candidate,
        {"line": _frame([0] * count, range(count), net_flow_mw=baseline_flows)},
        {"line": _frame([10] * count, range(count), net_flow_mw=candidate_flows)},
        {"line": _alignment("line", *pairs)},
        {"line": ("net_flow_mw",)},
        range(count),
    )


class TestLineDirection:
    @pytest.mark.parametrize(
        ("baseline_ends", "candidate_ends", "baseline_value"),
        [
            pytest.param(_SE_S, _S_SE, -300.0, id="exchanged"),
            pytest.param(_SE_S, _SE_S, 300.0, id="same-direction"),
            pytest.param(_SE_S, ("SUDESTE", "NORTE"), 300.0, id="other-target"),
            pytest.param(_SE_S, ("NORTE", "SUDESTE"), 300.0, id="other-source"),
            pytest.param(("SUL", "SUL"), ("SUL", "SUL"), 300.0, id="same-bus"),
            pytest.param(
                (None, "SUL"), ("SUL", "SUDESTE"), 300.0, id="no-baseline-source"
            ),
            pytest.param(
                ("SUDESTE", "SUL"), ("SUL", None), 300.0, id="no-candidate-target"
            ),
            pytest.param((None, "SUL"), ("SUL", None), 300.0, id="missing-ends-mirror"),
            pytest.param(
                ("SUDESTE", "ABSENT"), ("ABSENT", "SUDESTE"), 300.0, id="unknown-bus"
            ),
        ],
    )
    def test_only_a_line_whose_known_ends_are_exchanged_has_its_baseline_negated(
        self, baseline_ends: _Ends, candidate_ends: _Ends, baseline_value: float
    ) -> None:
        baseline = _side("baseline", lines=[baseline_ends])
        candidate = _side("candidate", offset=10, lines=[candidate_ends])

        (row,) = _line_row(
            baseline, candidate, [300.0], [-300.0], [AlignedPair("L0", 0, 10)]
        )

        assert (row.newave_value, row.novomodelo_value) == (baseline_value, -300.0)

    def test_an_exchanged_line_agrees_with_the_candidate(self) -> None:
        baseline = _side("baseline", lines=[_SE_S])
        candidate = _side("candidate", offset=10, lines=[_S_SE])

        (row,) = _line_row(
            baseline, candidate, [300.0], [-300.0], [AlignedPair("L0", 0, 10)]
        )

        assert (row.newave_value, row.novomodelo_value, row.abs_diff) == (
            -300.0,
            -300.0,
            0.0,
        )

    def test_the_bus_names_are_compared_by_their_name_key(self) -> None:
        baseline = _side(
            "baseline", lines=[_SE_S], bus_spellings=("sudeste ", "Sul", "NORTE")
        )
        candidate = _side("candidate", offset=10, lines=[_S_SE])

        (row,) = _line_row(
            baseline, candidate, [300.0], [-300.0], [AlignedPair("L0", 0, 10)]
        )

        assert row.newave_value == -300.0

    def test_the_negation_follows_the_pair_and_leaves_the_other_lines_and_the_candidate(
        self,
    ) -> None:
        baseline = _side("baseline", lines=[_SE_S, _SE_S])
        candidate = _side("candidate", offset=10, lines=[_SE_S, _S_SE])
        pairs = [AlignedPair("L0", 0, 10), AlignedPair("L1", 1, 11)]
        baseline_frame = _frame([0, 1], [0, 0], net_flow_mw=[300.0, 50.0])
        candidate_frame = _frame([10, 11], [0, 0], net_flow_mw=[300.0, -50.0])

        rows = operation_rows(
            baseline,
            candidate,
            {"line": baseline_frame},
            {"line": candidate_frame},
            {"line": _alignment("line", *pairs)},
            {"line": ("net_flow_mw",)},
            (0,),
        )

        assert [(r.entity_name, r.newave_value, r.novomodelo_value) for r in rows] == [
            ("L0", 300.0, 300.0),
            ("L1", -50.0, -50.0),
        ]
        assert baseline_frame["net_flow_mw"].to_list() == [300.0, 50.0]

    def test_the_negation_covers_every_stage_and_keeps_a_missing_flow_missing(
        self,
    ) -> None:
        baseline = _side("baseline", lines=[_SE_S])
        candidate = _side("candidate", offset=10, lines=[_S_SE])

        rows = _line_row(
            baseline,
            candidate,
            [300.0, None, 0.0],
            [-300.0, -5.0, 0.0],
            [AlignedPair("L0", 0, 10)],
        )

        assert [(row.stage, row.newave_value) for row in rows] == [
            (0, -300.0),
            (2, 0.0),
        ]

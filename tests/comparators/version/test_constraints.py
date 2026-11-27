"""Tests of ``comparators.version.constraints``.

The list check and the comparison run over ``build_solved_case`` trees whose
constraint file and resolved echo each test writes, never over mocks. The
difference rule runs over literal frames built by ``resolve_generic_constraints``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from novomodelo_bridge.comparators.constraints import stage_bounds_novomodelo
from novomodelo_bridge.comparators.version import findings
from novomodelo_bridge.comparators.version.alignment import (
    AlignedPair,
    NameAlignment,
    align_by_name,
)
from novomodelo_bridge.comparators.version.availability import (
    Unavailable,
    with_entity_lists,
)
from novomodelo_bridge.comparators.version.constraints import (
    ConstraintComparison,
    DifferingConstraint,
    compare_generic_constraints,
    constraint_list_usable,
    definition_differences,
)
from novomodelo_bridge.comparators.version.requirements import (
    GENERIC_CONSTRAINTS,
    decide_sections,
)
from novomodelo_bridge.comparators.version.sides import SideInputs, read_side
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.novomodelo.case_io import case_dir_for
from novomodelo_bridge.novomodelo.constraint_expr import (
    ResolvedGenericConstraints,
    resolve_generic_constraints,
)
from novomodelo_bridge.novomodelo.readers import (
    NovomodeloReadError,
    read_novomodelo_resolved_generic_constraints,
)
from tests.conftest import write_generic_constraint_echo
from tests.novomodelo_output_builders import (
    SolvedCaseSpec,
    build_solved_case,
    corrupt_file,
)

_STAGES = (0, 1)
_LHS_SCHEMA = pl.Schema(
    {"constraint_id": pl.Int32, "stage_id": pl.Int32, "lhs_value": pl.Float64}
)
_FILE_WARNING = "constraints/generic_constraints.json could not be read"
_ECHO = "generic_constraints/resolved_echo.parquet"
_BASELINE_OFFSET, _CANDIDATE_OFFSET = 0, 100


@dataclass(frozen=True)
class _Constraint:
    """One constraint of a built case: a unit-coefficient ``hydro_storage_final``
    term on the case's first hydro, with a lower and an upper bound per stage."""

    id: int
    name: str
    coefficient: float = 1.0
    lower: tuple[float | None, float | None] = (100.0, 100.0)
    upper: tuple[float | None, float | None] = (None, None)


_BASELINE = (_Constraint(0, "RE_1"), _Constraint(1, "VMINOP_1"))
_CANDIDATE = (_Constraint(5, "RE_1"), _Constraint(6, "VMINOP_1"))


def _constraints_path(output: Path) -> Path:
    return case_dir_for(output) / "constraints" / "generic_constraints.json"


def _write_constraints(
    output: Path, constraints: Sequence[_Constraint], *, hydro: int
) -> None:
    path = _constraints_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "constraints": [
                    {
                        "id": c.id,
                        "name": c.name,
                        "description": f"{c.name} of the case",
                        "expression": f"hydro_storage_final(id={hydro})",
                        "slack": {"enabled": True},
                    }
                    for c in constraints
                ]
            }
        ),
        encoding="utf-8",
    )
    if constraints:
        write_generic_constraint_echo(
            output,
            [
                (
                    c.id,
                    c.name,
                    stage,
                    None,
                    f"hydro_storage_final(id={hydro}, block=all)",
                    c.coefficient,
                    c.lower[stage],
                    c.upper[stage],
                )
                for c in constraints
                for stage in _STAGES
            ],
        )


def _write_text(output: Path, text: str) -> Path:
    path = _constraints_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _side(
    root: Path, side: str, constraints: Sequence[_Constraint], *, offset: int
) -> SideInputs:
    output = build_solved_case(root / side, SolvedCaseSpec(id_offset=offset))
    _write_constraints(output, constraints, hydro=offset)
    with dx.collect():
        return read_side(output, side)


def _sides(
    root: Path,
    baseline: Sequence[_Constraint] = _BASELINE,
    candidate: Sequence[_Constraint] = _CANDIDATE,
) -> tuple[SideInputs, SideInputs]:
    return (
        _side(root, "baseline", baseline, offset=_BASELINE_OFFSET),
        _side(root, "candidate", candidate, offset=_CANDIDATE_OFFSET),
    )


def _alignments(
    baseline: SideInputs, candidate: SideInputs
) -> dict[str, NameAlignment]:
    return {
        "hydro": align_by_name("hydro", baseline.hydros, candidate.hydros),
        "bus": align_by_name("bus", baseline.buses, candidate.buses),
    }


def _compare(
    baseline: SideInputs, candidate: SideInputs, stages: Sequence[int] = _STAGES
) -> tuple[ConstraintComparison, list[dx.Diagnostic]]:
    with dx.collect() as sink:
        result = compare_generic_constraints(
            baseline, candidate, _alignments(baseline, candidate), stages
        )
    return result, sink


def _usable(side: SideInputs) -> tuple[bool, list[dx.Diagnostic]]:
    with dx.collect() as sink:
        usable = constraint_list_usable(side)
    return usable, sink


def _ids(frame: pl.DataFrame) -> list[int]:
    return sorted(frame["constraint_id"].unique().to_list())


class TestConstraintListUsable:
    def test_a_case_without_the_file_is_usable_and_silent(self, tmp_path: Path) -> None:
        output = build_solved_case(tmp_path / "case")
        side = read_side(output, "candidate")

        usable, sink = _usable(side)

        assert usable is True
        assert sink == []

    def test_a_list_of_named_constraints_is_usable_and_silent(
        self, tmp_path: Path
    ) -> None:
        baseline, _ = _sides(tmp_path)

        usable, sink = _usable(baseline)

        assert usable is True
        assert sink == []

    def test_an_empty_list_is_usable_and_silent(self, tmp_path: Path) -> None:
        side = _side(tmp_path, "baseline", (), offset=0)

        usable, sink = _usable(side)

        assert usable is True
        assert sink == []

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("not json", id="unparseable"),
            pytest.param('{"constraints": {}}', id="no-list"),
            pytest.param('{"other": []}', id="no-key"),
            pytest.param("[]", id="top-level-list"),
            pytest.param('{"constraints": [{"name": "RE_1"}]}', id="missing-id"),
            pytest.param(
                '{"constraints": [{"id": true, "name": "A"}]}', id="boolean-id"
            ),
            pytest.param(
                '{"constraints": [{"id": 0, "name": "RE_1"}, {"id": 1, "name": " "}]}',
                id="blank-name",
            ),
            pytest.param('{"constraints": [{"id": 0}]}', id="missing-name"),
            pytest.param('{"constraints": [{"id": 0, "name": 7}]}', id="numeric-name"),
            pytest.param('{"constraints": ["RE_1"]}', id="entry-not-an-object"),
        ],
    )
    def test_an_unreadable_list_is_unusable_with_one_tagged_warning(
        self, tmp_path: Path, text: str
    ) -> None:
        output = build_solved_case(tmp_path / "case")
        path = _write_text(output, text)
        side = read_side(output, "candidate")

        usable, sink = _usable(side)

        assert usable is False
        [warning] = sink
        assert warning.code == "report-file-unreadable"
        assert warning.severity is dx.Severity.WARNING
        assert warning.summary == (
            f"Candidate output: {_FILE_WARNING}, so the comparison leaves out "
            "the generic constraints of this output."
        )
        assert warning.notes[0] == f"Path: {path}"
        assert warning.notes[1].startswith("Reason: ")

    def test_the_baseline_warning_is_tagged_with_the_baseline(
        self, tmp_path: Path
    ) -> None:
        output = build_solved_case(tmp_path / "case")
        _write_text(output, "not json")
        side = read_side(output, "baseline")

        usable, sink = _usable(side)

        assert usable is False
        assert [d.summary.split(",")[0] for d in sink] == [
            f"Baseline output: {_FILE_WARNING}"
        ]

    def test_the_verdict_never_comes_from_a_diagnostic_the_lenient_reader_emits(
        self, tmp_path: Path
    ) -> None:
        output = build_solved_case(tmp_path / "case")
        _write_text(output, '{"constraints": [{"id": 0, "name": " "}]}')
        side = read_side(output, "candidate")

        usable, sink = _usable(side)

        assert usable is False
        assert [d.code for d in sink] == ["report-file-unreadable"]

    def test_the_check_raises_nothing_for_a_path_that_is_a_directory(
        self, tmp_path: Path
    ) -> None:
        output = build_solved_case(tmp_path / "case")
        _constraints_path(output).mkdir(parents=True)
        side = read_side(output, "candidate")

        usable, sink = _usable(side)

        assert usable is False
        assert [d.code for d in sink] == ["report-file-unreadable"]


def _availability(
    side: SideInputs, *, usable: bool
) -> dict[str, frozenset[str] | None]:
    return with_entity_lists(
        side.partitions, {**side.usable_entity_lists, "generic_constraint": usable}
    )


def _sections(baseline: SideInputs, candidate: SideInputs):
    return decide_sections(
        _availability(baseline, usable=constraint_list_usable(baseline)),
        _availability(candidate, usable=constraint_list_usable(candidate)),
    )


class TestUnusableListThroughAvailability:
    def test_an_unparseable_list_on_one_side_makes_the_section_unavailable_there(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _sides(tmp_path)
        _write_text(candidate.output_dir, "not json")

        with dx.collect():
            sections = _sections(baseline, candidate)

        verdict = sections[GENERIC_CONSTRAINTS]
        assert verdict.available is False
        assert verdict.unavailable == (
            Unavailable(GENERIC_CONSTRAINTS, None, "candidate"),
        )
        assert findings.one_sided_entities([]) is None
        assert findings.differing_constraints(()) is None

    def test_a_usable_list_on_both_sides_leaves_the_section_available(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _sides(tmp_path)

        with dx.collect():
            sections = _sections(baseline, candidate)

        assert sections[GENERIC_CONSTRAINTS].available is True
        assert sections[GENERIC_CONSTRAINTS].unavailable == ()

    def test_an_unparseable_list_on_both_sides_gives_one_row_for_both(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _sides(tmp_path)
        _write_text(baseline.output_dir, "not json")
        _write_text(candidate.output_dir, '{"constraints": [{"id": 0, "name": ""}]}')

        with dx.collect() as sink:
            sections = _sections(baseline, candidate)

        assert sections[GENERIC_CONSTRAINTS].unavailable == (
            Unavailable(GENERIC_CONSTRAINTS, None, "both"),
        )
        assert [d.summary.split(" ")[0] for d in sink] == ["Baseline", "Candidate"]

    def test_an_absent_file_is_an_empty_list_and_keeps_the_section_available(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _sides(tmp_path)
        _constraints_path(candidate.output_dir).unlink()

        with dx.collect():
            sections = _sections(baseline, candidate)

        assert Unavailable(GENERIC_CONSTRAINTS, None, "candidate") not in (
            sections[GENERIC_CONSTRAINTS].unavailable
        )


class TestCompareGenericConstraints:
    def test_equal_constraints_with_different_ids_pair_by_name_and_do_not_differ(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _sides(tmp_path)

        result, sink = _compare(baseline, candidate)

        assert [
            (p.name, p.baseline_id, p.candidate_id) for p in result.alignment.pairs
        ] == [
            ("RE_1", 0, 5),
            ("VMINOP_1", 1, 6),
        ]
        assert result.alignment.entity_type == "generic_constraint"
        assert [c["id"] for c in result.gc_constraints] == [5, 6]
        assert [c["name"] for c in result.gc_constraints] == ["RE_1", "VMINOP_1"]
        assert _ids(result.gc_lhs_newave) == [5, 6]
        assert result.gc_lhs_novomodelo.height == 4
        assert_frame_equal(result.gc_lhs_newave, result.gc_lhs_novomodelo)
        assert result.gc_lhs_newave.schema == _LHS_SCHEMA
        assert result.differing == ()
        assert sink == []

    def test_the_bounds_are_the_candidates_stage_bounds(self, tmp_path: Path) -> None:
        baseline, candidate = _sides(tmp_path)
        resolved = read_novomodelo_resolved_generic_constraints(
            case_dir_for(candidate.output_dir), candidate.output_dir
        )

        result, _ = _compare(baseline, candidate)

        assert_frame_equal(
            result.gc_bounds, stage_bounds_novomodelo(resolved, candidate.output_dir)
        )
        assert _ids(result.gc_bounds) == [5, 6]
        assert result.gc_bounds["bound_lower"].to_list() == [100.0] * 4

    def test_the_baseline_lhs_follows_the_candidates_ids_whatever_their_order(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _sides(
            tmp_path, candidate=(_Constraint(7, "VMINOP_1"), _Constraint(3, "RE_1"))
        )

        result, _ = _compare(baseline, candidate)

        assert _ids(result.gc_lhs_newave) == [3, 7]
        by_name = {p.name: p.candidate_id for p in result.alignment.pairs}
        assert by_name == {"RE_1": 3, "VMINOP_1": 7}
        assert result.differing == ()

    def test_a_changed_bound_and_a_changed_coefficient_are_listed_in_pair_order(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _sides(
            tmp_path,
            candidate=(
                replace(_CANDIDATE[0], lower=(100.0, 120.0)),
                replace(_CANDIDATE[1], coefficient=2.0),
            ),
        )

        result, _ = _compare(baseline, candidate)

        assert result.differing == (
            DifferingConstraint("RE_1", terms=False, bounds=True),
            DifferingConstraint("VMINOP_1", terms=True, bounds=False),
        )
        assert [p.name for p in result.alignment.pairs] == ["RE_1", "VMINOP_1"]

    def test_a_changed_bound_outside_the_common_stages_is_not_a_difference(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _sides(
            tmp_path,
            baseline=(*_BASELINE, _Constraint(2, "ONLY_HERE")),
            candidate=(replace(_CANDIDATE[0], lower=(100.0, 120.0)), _CANDIDATE[1]),
        )

        result, _ = _compare(baseline, candidate, stages=(0,))

        assert result.differing == ()
        assert result.alignment.only_in_baseline == ("ONLY_HERE",)
        assert result.alignment.only_in_candidate == ()
        assert _ids(result.gc_lhs_newave) == [5, 6]

    def test_the_lhs_frames_are_not_restricted_to_the_common_stages(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _sides(tmp_path)

        result, _ = _compare(baseline, candidate, stages=(0,))

        assert sorted(result.gc_lhs_newave["stage_id"].unique().to_list()) == [0, 1]
        assert sorted(result.gc_lhs_novomodelo["stage_id"].unique().to_list()) == [0, 1]

    def test_an_unpaired_baseline_constraint_leaves_the_rekeyed_frame(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _sides(
            tmp_path, baseline=(*_BASELINE, _Constraint(2, "ONLY_HERE"))
        )
        baseline_resolved = read_novomodelo_resolved_generic_constraints(
            case_dir_for(baseline.output_dir), baseline.output_dir
        )

        result, _ = _compare(baseline, candidate)

        assert sorted(baseline_resolved.terms["constraint_id"].unique().to_list()) == [
            0,
            1,
            2,
        ]
        assert _ids(result.gc_lhs_newave) == [5, 6]
        assert result.gc_lhs_newave.height == 4
        assert result.alignment.only_in_baseline == ("ONLY_HERE",)

    def test_a_candidate_only_constraint_is_listed_and_keeps_its_frames(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _sides(
            tmp_path, candidate=(*_CANDIDATE, _Constraint(9, "NEW_ONE"))
        )

        result, _ = _compare(baseline, candidate)

        assert result.alignment.only_in_candidate == ("NEW_ONE",)
        assert _ids(result.gc_lhs_novomodelo) == [5, 6, 9]
        assert _ids(result.gc_lhs_newave) == [5, 6]
        assert [c["name"] for c in result.gc_constraints] == [
            "RE_1",
            "VMINOP_1",
            "NEW_ONE",
        ]
        assert result.differing == ()

    def test_a_name_used_twice_on_the_baseline_is_in_neither_the_pairs_nor_the_frame(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _sides(
            tmp_path, baseline=(*_BASELINE, _Constraint(2, "re_1 "))
        )

        result, _ = _compare(baseline, candidate)

        assert [p.name for p in result.alignment.pairs] == ["VMINOP_1"]
        [duplicate] = result.alignment.duplicates
        assert duplicate.name_key == "RE_1"
        assert duplicate.baseline_names == ("RE_1", "re_1")
        assert _ids(result.gc_lhs_newave) == [6]
        assert result.differing == ()

    def test_a_case_without_constraints_on_both_sides_gives_empty_frames(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _sides(tmp_path, baseline=(), candidate=())

        result, sink = _compare(baseline, candidate)

        assert result.alignment.pairs == ()
        assert result.gc_constraints == []
        assert result.gc_bounds.is_empty()
        assert result.gc_lhs_newave.is_empty()
        assert result.gc_lhs_newave.schema == _LHS_SCHEMA
        assert result.gc_lhs_novomodelo.is_empty()
        assert result.gc_lhs_novomodelo.schema == _LHS_SCHEMA
        assert result.differing == ()
        assert sink == []

    def test_a_case_without_the_constraint_file_on_both_sides_is_the_same(
        self, tmp_path: Path
    ) -> None:
        baseline = read_side(build_solved_case(tmp_path / "baseline"), "baseline")
        candidate = read_side(build_solved_case(tmp_path / "candidate"), "candidate")

        result, sink = _compare(baseline, candidate)

        assert result.alignment.pairs == ()
        assert result.gc_lhs_newave.is_empty()
        assert result.differing == ()
        assert sink == []

    def test_a_side_only_constraint_list_is_empty_on_the_other(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _sides(tmp_path, baseline=(), candidate=_CANDIDATE)

        result, _ = _compare(baseline, candidate)

        assert result.alignment.only_in_candidate == ("RE_1", "VMINOP_1")
        assert result.gc_lhs_newave.is_empty()
        assert result.gc_lhs_newave.schema == _LHS_SCHEMA
        assert _ids(result.gc_lhs_novomodelo) == [5, 6]

    def test_a_hydro_without_a_pair_makes_the_term_differ(self, tmp_path: Path) -> None:
        baseline, candidate = _sides(tmp_path)
        candidate = replace(
            candidate,
            hydros=[{**h, "name": f"{h['name']}_RENAMED"} for h in candidate.hydros],
        )

        result, _ = _compare(baseline, candidate)

        assert result.differing == (
            DifferingConstraint("RE_1", terms=True, bounds=False),
            DifferingConstraint("VMINOP_1", terms=True, bounds=False),
        )

    def test_reader_diagnostics_are_tagged_with_the_side_that_raised_them(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _sides(tmp_path)
        (baseline.output_dir / _ECHO).unlink()

        _, sink = _compare(baseline, candidate)

        assert [d.code for d in sink] == ["report-file-missing"]
        assert sink[0].summary.startswith("Baseline output: ")

    def test_an_echo_that_cannot_be_read_propagates_the_reader_error(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _sides(tmp_path)
        corrupt_file(candidate.output_dir, _ECHO)

        with pytest.raises(NovomodeloReadError):
            _compare(baseline, candidate)

    def test_the_result_types_are_frozen(self, tmp_path: Path) -> None:
        baseline, candidate = _sides(tmp_path)
        result, _ = _compare(baseline, candidate)

        with pytest.raises(FrozenInstanceError):
            result.differing = ()  # ty: ignore[invalid-assignment] the write is the assertion
        with pytest.raises(FrozenInstanceError):
            DifferingConstraint("RE_1", True, False).terms = False  # ty: ignore[invalid-assignment] the write is the assertion


def _resolved(
    rows: Sequence[tuple[int, int, int | None, str, float, float | None, float | None]],
) -> ResolvedGenericConstraints:
    """Rows are ``(constraint id, stage, block, variable, coefficient, lower,
    upper)``; the variable's kind is its text before the parenthesis."""
    positions: dict[tuple[int, int, int | None], int] = {}
    term_index = []
    for cid, stage, block, *_ in rows:
        key = (cid, stage, block)
        term_index.append(positions.get(key, 0))
        positions[key] = term_index[-1] + 1
    echo = pl.DataFrame(
        {
            "constraint_id": [r[0] for r in rows],
            "stage_id": [r[1] for r in rows],
            "block_id": [r[2] for r in rows],
            "term_index": term_index,
            "variable_kind": [r[3].split("(")[0] for r in rows],
            "variable": [r[3] for r in rows],
            "coefficient": [r[4] for r in rows],
            "bound_lower": [r[5] for r in rows],
            "bound_upper": [r[6] for r in rows],
        },
        schema={
            "constraint_id": pl.Int32,
            "stage_id": pl.Int32,
            "block_id": pl.Int32,
            "term_index": pl.Int32,
            "variable_kind": pl.String,
            "variable": pl.String,
            "coefficient": pl.Float64,
            "bound_lower": pl.Float64,
            "bound_upper": pl.Float64,
        },
    )
    ids = sorted({r[0] for r in rows})
    return resolve_generic_constraints(
        [{"id": i, "name": f"C{i}"} for i in ids], echo, {}
    )


_PAIR = (AlignedPair("RE_1", 0, 5),)
_HYDRO_3_TO_30 = {"hydro": {3: 30}}


def _differences(
    baseline: ResolvedGenericConstraints,
    candidate: ResolvedGenericConstraints,
    entity_ids: dict[str, dict[int, int]] | None = None,
    stages: Sequence[int] = _STAGES,
    pairs: Sequence[AlignedPair] = _PAIR,
) -> tuple[DifferingConstraint, ...]:
    return definition_differences(baseline, candidate, pairs, entity_ids or {}, stages)


def _one(
    constraint: int,
    variable: str,
    coefficient: float = 1.0,
    *,
    stage: int = 0,
    block: int | None = None,
    lower: float | None = 100.0,
    upper: float | None = None,
) -> tuple[int, int, int | None, str, float, float | None, float | None]:
    return (constraint, stage, block, variable, coefficient, lower, upper)


class TestDefinitionDifferences:
    def test_equal_definitions_with_different_entity_ids_do_not_differ_through_the_map(
        self,
    ) -> None:
        baseline = _resolved([_one(0, "hydro_generation(id=3, block=all)")])
        candidate = _resolved([_one(5, "hydro_generation(id=30, block=all)")])

        assert _differences(baseline, candidate, _HYDRO_3_TO_30) == ()

    def test_the_same_entity_ids_differ_when_the_map_pairs_them_with_others(
        self,
    ) -> None:
        baseline = _resolved([_one(0, "hydro_generation(id=3, block=all)")])
        candidate = _resolved([_one(5, "hydro_generation(id=3, block=all)")])

        assert _differences(baseline, candidate, _HYDRO_3_TO_30) == (
            DifferingConstraint("RE_1", terms=True, bounds=False),
        )

    def test_a_term_on_an_unpaired_baseline_plant_never_equals_a_candidate_plant(
        self,
    ) -> None:
        baseline = _resolved([_one(0, "hydro_generation(id=4, block=all)")])
        candidate = _resolved([_one(5, "hydro_generation(id=4, block=all)")])

        assert _differences(baseline, candidate, _HYDRO_3_TO_30) == (
            DifferingConstraint("RE_1", terms=True, bounds=False),
        )

    def test_a_kind_with_no_translation_map_compares_raw_ids(self) -> None:
        same = _differences(
            _resolved([_one(0, "pumping_flow(id=7, block=all)")]),
            _resolved([_one(5, "pumping_flow(id=7, block=all)")]),
            _HYDRO_3_TO_30,
        )
        other = _differences(
            _resolved([_one(0, "pumping_flow(id=7, block=all)")]),
            _resolved([_one(5, "pumping_flow(id=8, block=all)")]),
            _HYDRO_3_TO_30,
        )

        assert same == ()
        assert other == (DifferingConstraint("RE_1", terms=True, bounds=False),)

    @pytest.mark.parametrize(
        ("kind", "entity_type"),
        [
            ("hydro_storage_final", "hydro"),
            ("hydro_useful_volume_final", "hydro"),
            ("thermal_generation", "thermal"),
            ("line_exchange", "line"),
            ("line_direct", "line"),
        ],
    )
    def test_the_entity_type_is_the_kind_up_to_its_first_underscore(
        self, kind: str, entity_type: str
    ) -> None:
        baseline = _resolved([_one(0, f"{kind}(id=2, block=all)")])
        candidate = _resolved([_one(5, f"{kind}(id=20, block=all)")])

        assert _differences(baseline, candidate, {entity_type: {2: 20}}) == ()
        assert _differences(baseline, candidate, {"other": {2: 20}}) == (
            DifferingConstraint("RE_1", terms=True, bounds=False),
        )

    def test_a_term_without_a_bus_does_not_differ_when_buses_are_aligned(self) -> None:
        baseline = _resolved([_one(0, "hydro_generation(id=3, block=all)")])
        candidate = _resolved([_one(5, "hydro_generation(id=30, block=all)")])

        assert (
            _differences(baseline, candidate, {**_HYDRO_3_TO_30, "bus": {1: 11}}) == ()
        )

    def test_a_bus_cell_translates_through_the_bus_map(self) -> None:
        baseline = _resolved([_one(0, "hydro_generation(id=3, bus=1, block=all)")])
        candidate = _resolved([_one(5, "hydro_generation(id=30, bus=11, block=all)")])

        assert (
            _differences(baseline, candidate, {**_HYDRO_3_TO_30, "bus": {1: 11}}) == ()
        )
        assert _differences(
            baseline, candidate, {**_HYDRO_3_TO_30, "bus": {1: 12}}
        ) == (DifferingConstraint("RE_1", terms=True, bounds=False),)
        assert _differences(
            baseline, candidate, {**_HYDRO_3_TO_30, "bus": {2: 11}}
        ) == (DifferingConstraint("RE_1", terms=True, bounds=False),)

    def test_a_bus_compares_raw_when_the_buses_are_not_aligned(self) -> None:
        baseline = _resolved([_one(0, "hydro_generation(id=3, bus=1, block=all)")])
        same = _resolved([_one(5, "hydro_generation(id=30, bus=1, block=all)")])
        other = _resolved([_one(5, "hydro_generation(id=30, bus=11, block=all)")])

        assert _differences(baseline, same, _HYDRO_3_TO_30) == ()
        assert _differences(baseline, other, _HYDRO_3_TO_30) == (
            DifferingConstraint("RE_1", terms=True, bounds=False),
        )

    def test_a_term_block_is_compared_raw(self) -> None:
        baseline = _resolved([_one(0, "hydro_generation(id=3, block=1)")])
        same = _resolved([_one(5, "hydro_generation(id=30, block=1)")])
        other = _resolved([_one(5, "hydro_generation(id=30, block=2)")])

        assert _differences(baseline, same, _HYDRO_3_TO_30) == ()
        assert _differences(baseline, other, _HYDRO_3_TO_30) == (
            DifferingConstraint("RE_1", terms=True, bounds=False),
        )

    @pytest.mark.parametrize(
        ("baseline", "candidate", "differs"),
        [
            (1.0, 1.0, False),
            (1.0, 1.0 + 1e-12, False),
            (1.0 + 1e-12, 1.0, False),
            (1.0, 1.0 + 1e-6, True),
            (1e6, 1e6 * (1 + 1e-12), False),
            (1e-12, 2e-12, True),
            (-1.0, -1.0 - 1e-12, False),
            (1.0, -1.0, True),
            (0.0, 0.0, False),
            (0.0, 1e-300, True),
        ],
    )
    def test_a_coefficient_compares_with_a_relative_tolerance(
        self, baseline: float, candidate: float, differs: bool
    ) -> None:
        base = _resolved([_one(0, "thermal_generation(id=1)", baseline)])
        cand = _resolved([_one(5, "thermal_generation(id=1)", candidate)])

        expected = (DifferingConstraint("RE_1", terms=True, bounds=False),)
        assert _differences(base, cand) == (expected if differs else ())

    def test_a_term_on_one_side_only_differs(self) -> None:
        one_baseline = _resolved([_one(0, "thermal_generation(id=1)")])
        one_candidate = _resolved([_one(5, "thermal_generation(id=1)")])
        two_baseline = _resolved(
            [_one(0, "thermal_generation(id=1)"), _one(0, "thermal_generation(id=2)")]
        )
        two_candidate = _resolved(
            [_one(5, "thermal_generation(id=1)"), _one(5, "thermal_generation(id=2)")]
        )

        expected = (DifferingConstraint("RE_1", terms=True, bounds=False),)
        assert _differences(one_baseline, two_candidate) == expected
        assert _differences(two_baseline, one_candidate) == expected

    def test_a_term_at_a_stage_one_side_lacks_differs(self) -> None:
        both = _resolved(
            [
                _one(0, "thermal_generation(id=1)", stage=0),
                _one(0, "thermal_generation(id=1)", stage=1),
            ]
        )
        first = _resolved([_one(5, "thermal_generation(id=1)", stage=0)])

        assert _differences(both, first) == (
            DifferingConstraint("RE_1", terms=True, bounds=True),
        )

    def test_the_order_of_the_terms_in_a_row_does_not_matter(self) -> None:
        baseline = _resolved(
            [
                _one(0, "thermal_generation(id=1)", 1.0),
                _one(0, "thermal_generation(id=2)", 2.0),
            ]
        )
        candidate = _resolved(
            [
                _one(5, "thermal_generation(id=2)", 2.0),
                _one(5, "thermal_generation(id=1)", 1.0),
            ]
        )

        assert _differences(baseline, candidate) == ()

    def test_a_bound_present_on_one_side_only_differs_in_bounds(self) -> None:
        baseline = _resolved(
            [
                _one(0, "thermal_generation(id=1)", stage=0),
                _one(0, "thermal_generation(id=1)", stage=1),
            ]
        )
        candidate = _resolved(
            [
                _one(5, "thermal_generation(id=1)", stage=0),
                _one(5, "thermal_generation(id=1)", stage=1, lower=None, upper=None),
            ]
        )

        assert _differences(baseline, candidate) == (
            DifferingConstraint("RE_1", terms=False, bounds=True),
        )

    def test_a_row_the_candidate_has_no_bound_row_for_differs_in_bounds_and_terms(
        self,
    ) -> None:
        baseline = _resolved([_one(0, "thermal_generation(id=1)", stage=0)])
        candidate = _resolved(
            [
                _one(5, "thermal_generation(id=1)", stage=0),
                _one(5, "thermal_generation(id=1)", stage=1),
            ]
        )

        assert _differences(baseline, candidate) == (
            DifferingConstraint("RE_1", terms=True, bounds=True),
        )

    @pytest.mark.parametrize(
        ("baseline", "candidate", "differs"),
        [
            ((100.0, None), (100.0, None), False),
            ((100.0, None), (100.0 * (1 + 1e-12), None), False),
            ((100.0, None), (120.0, None), True),
            ((100.0, None), (None, None), True),
            ((None, None), (100.0, None), True),
            ((None, 100.0), (100.0, None), True),
            ((None, 100.0), (None, 100.0), False),
            ((100.0, 200.0), (100.0, 200.0), False),
            ((100.0, 200.0), (100.0, 220.0), True),
            ((100.0, 200.0), (100.0, None), True),
            ((None, None), (None, None), False),
        ],
    )
    def test_an_endpoint_compares_by_nullness_then_with_a_relative_tolerance(
        self,
        baseline: tuple[float | None, float | None],
        candidate: tuple[float | None, float | None],
        differs: bool,
    ) -> None:
        base = _resolved(
            [_one(0, "thermal_generation(id=1)", lower=baseline[0], upper=baseline[1])]
        )
        cand = _resolved(
            [
                _one(
                    5,
                    "thermal_generation(id=1)",
                    lower=candidate[0],
                    upper=candidate[1],
                )
            ]
        )

        expected = (DifferingConstraint("RE_1", terms=False, bounds=True),)
        assert _differences(base, cand) == (expected if differs else ())

    def test_a_stage_level_bound_and_a_block_bound_have_different_keys(self) -> None:
        baseline = _resolved([_one(0, "thermal_generation(id=1, block=0)", block=None)])
        candidate = _resolved([_one(5, "thermal_generation(id=1, block=0)", block=0)])

        assert _differences(baseline, candidate) == (
            DifferingConstraint("RE_1", terms=True, bounds=True),
        )

    def test_a_difference_at_a_stage_outside_the_common_stages_is_ignored(self) -> None:
        baseline = _resolved(
            [
                _one(0, "thermal_generation(id=1)", 1.0, stage=0),
                _one(0, "thermal_generation(id=1)", 1.0, stage=1, lower=100.0),
            ]
        )
        candidate = _resolved(
            [
                _one(5, "thermal_generation(id=1)", 1.0, stage=0),
                _one(5, "thermal_generation(id=1)", 9.0, stage=1, lower=500.0),
            ]
        )

        assert _differences(baseline, candidate, stages=(0,)) == ()
        assert _differences(baseline, candidate, stages=(0, 1)) == (
            DifferingConstraint("RE_1", terms=True, bounds=True),
        )

    def test_no_common_stage_leaves_nothing_to_compare(self) -> None:
        baseline = _resolved([_one(0, "thermal_generation(id=1)", 1.0)])
        candidate = _resolved([_one(5, "thermal_generation(id=1)", 9.0, lower=1.0)])

        assert _differences(baseline, candidate, stages=()) == ()

    def test_terms_and_bounds_can_differ_at_once(self) -> None:
        baseline = _resolved([_one(0, "thermal_generation(id=1)", 1.0, lower=100.0)])
        candidate = _resolved([_one(5, "thermal_generation(id=1)", 2.0, lower=120.0)])

        assert _differences(baseline, candidate) == (
            DifferingConstraint("RE_1", terms=True, bounds=True),
        )

    def test_a_pair_that_neither_side_has_rows_for_is_left_out(self) -> None:
        empty = _resolved([_one(9, "thermal_generation(id=1)")])

        assert _differences(empty, empty) == ()

    def test_a_pair_that_only_one_side_has_rows_for_differs_in_both(self) -> None:
        full = _resolved([_one(0, "thermal_generation(id=1)")])
        other = _resolved([_one(9, "thermal_generation(id=1)")])

        assert _differences(full, other) == (
            DifferingConstraint("RE_1", terms=True, bounds=True),
        )

    def test_each_pair_is_compared_by_its_own_ids_in_the_order_of_the_pairs(
        self,
    ) -> None:
        baseline = _resolved(
            [
                _one(0, "thermal_generation(id=1)", 1.0),
                _one(1, "thermal_generation(id=1)", 1.0),
                _one(2, "thermal_generation(id=1)", 1.0),
            ]
        )
        candidate = _resolved(
            [
                _one(5, "thermal_generation(id=1)", 1.0),
                _one(6, "thermal_generation(id=1)", 7.0),
                _one(7, "thermal_generation(id=1)", 8.0),
            ]
        )
        pairs = (
            AlignedPair("C", 2, 7),
            AlignedPair("A", 0, 5),
            AlignedPair("B", 1, 6),
        )

        assert _differences(baseline, candidate, pairs=pairs) == (
            DifferingConstraint("C", terms=True, bounds=False),
            DifferingConstraint("B", terms=True, bounds=False),
        )

    def test_no_pairs_give_no_differences(self) -> None:
        baseline = _resolved([_one(0, "thermal_generation(id=1)", 1.0)])
        candidate = _resolved([_one(5, "thermal_generation(id=1)", 9.0)])

        assert _differences(baseline, candidate, pairs=()) == ()

    def test_the_expression_text_is_not_compared(self) -> None:
        baseline = _resolved([_one(0, "hydro_generation(id=3, block=all)")])._replace(
            constraints=[
                {"id": 0, "name": "RE_1", "expression": "hydro_generation(id=3)"}
            ]
        )
        candidate = _resolved([_one(5, "hydro_generation(id=30, block=all)")])._replace(
            constraints=[
                {"id": 5, "name": "RE_1", "expression": "hydro_generation(id=30)"}
            ]
        )

        assert _differences(baseline, candidate, _HYDRO_3_TO_30) == ()

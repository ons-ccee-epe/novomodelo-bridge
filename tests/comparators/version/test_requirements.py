"""Tests of ``comparators.version.requirements`` over ``build_solved_case`` trees.

The table tests read the declared mappings; the decision tests damage outputs
with the builder's helpers, never mocks. The table-against-reader test drops one
column at a time from a full fixture and holds every section's readers to the
declared table: a dropped column ends in drift, never in a reader error.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from novomodelo_bridge.comparators.constraints import (
    evaluate_lhs_novomodelo,
    stage_bounds_novomodelo,
)
from novomodelo_bridge.comparators.fpha import novomodelo_plane_arrays
from novomodelo_bridge.comparators.operating_points import novomodelo_points
from novomodelo_bridge.comparators.version.availability import (
    ENTITY_LIST_KEYS,
    SectionAvailability,
    Unavailable,
    unavailable_rows,
    with_entity_lists,
)
from novomodelo_bridge.comparators.version.requirements import (
    BUS,
    BUS_NET_IMPORT,
    COSTS,
    ENERGY_BALANCE,
    GENERIC_CONSTRAINTS,
    HYDRO,
    HYDRO_BOUNDS,
    LINE,
    LINE_LIMITS,
    OPERATING_POINTS,
    OPTIONAL_PARTITIONS,
    PRODUCTION_FUNCTION,
    REQUIREMENTS,
    SYSTEM_SPILLAGE,
    THERMAL,
    decide_sections,
)
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.novomodelo.case_io import case_dir_for
from novomodelo_bridge.novomodelo.cost_categories import (
    NOVOMODELO_COST_COMPONENT_COLUMNS,
)
from novomodelo_bridge.novomodelo.probe import (
    PARQUET_FILES,
    SIMULATION_ENTITIES,
    probe_output,
)
from novomodelo_bridge.novomodelo.readers import (
    read_novomodelo_bus_aggregates,
    read_novomodelo_bus_means,
    read_novomodelo_bus_metadata,
    read_novomodelo_bus_net_import,
    read_novomodelo_bus_percentiles,
    read_novomodelo_cost_breakdown,
    read_novomodelo_fpha_planes,
    read_novomodelo_hydro_bus_labels,
    read_novomodelo_hydro_means,
    read_novomodelo_hydro_per_stage_bounds,
    read_novomodelo_hydro_percentiles,
    read_novomodelo_hydro_total_flows,
    read_novomodelo_hydro_withdrawal,
    read_novomodelo_line_means,
    read_novomodelo_line_percentiles,
    read_novomodelo_lines,
    read_novomodelo_resolved_generic_constraints,
    read_novomodelo_spillage_energy,
    read_novomodelo_stage_costs,
    read_novomodelo_stage_line_bounds,
    read_novomodelo_thermal_means,
    read_novomodelo_thermal_percentiles,
)
from tests.conftest import write_generic_constraint_echo
from tests.novomodelo_output_builders import (
    SolvedCaseSpec,
    build_solved_case,
    drop_column,
    drop_partition,
)

_SECTIONS = [
    "hydro",
    "hydro-bounds",
    "system-spillage",
    "thermal",
    "bus",
    "bus-net-import",
    "line",
    "line-limits",
    "energy-balance",
    "costs",
    "generic-constraints",
    "production-function",
    "operating-points",
]
_ALL_USABLE = {"thermal": True, "line": True, "generic_constraint": True}
_FULL = SolvedCaseSpec(load_stats=True, non_controllables=("WIND_A",))
_WEIGHTED_SECTIONS = [
    HYDRO,
    HYDRO_BOUNDS,
    SYSTEM_SPILLAGE,
    THERMAL,
    BUS,
    BUS_NET_IMPORT,
    LINE,
    ENERGY_BALANCE,
    COSTS,
    GENERIC_CONSTRAINTS,
]
_SCENARIO_SUMMARY = "simulation/scenario_summary.parquet"


def _write_bounds(case_dir: Path) -> None:
    constraints = case_dir / "constraints"
    constraints.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "hydro_id": pa.array([0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "water_withdrawal_m3s": pa.array([1.0, 2.0], type=pa.float64()),
            }
        ),
        constraints / "hydro_bounds.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "line_id": pa.array([0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "block_id": pa.array([None, None], type=pa.int32()),
                "direct_mw": pa.array([1000.0, 1000.0], type=pa.float64()),
                "reverse_mw": pa.array([900.0, 900.0], type=pa.float64()),
            }
        ),
        constraints / "line_bounds.parquet",
    )


def _write_generic_constraint(case_dir: Path, output: Path) -> None:
    constraints = case_dir / "constraints"
    constraints.mkdir(parents=True, exist_ok=True)
    (constraints / "generic_constraints.json").write_text(
        json.dumps({"constraints": [{"id": 0, "name": "GC0"}]}), encoding="utf-8"
    )
    write_generic_constraint_echo(
        output,
        [
            (
                0,
                "GC0",
                stage,
                None,
                "hydro_storage_final(id=0, block=all)",
                1.0,
                2.0e3,
                None,
            )
            for stage in (0, 1)
        ],
    )


def _case(case_dir: Path, spec: SolvedCaseSpec = _FULL, *, bounds: bool = True) -> Path:
    output = build_solved_case(case_dir, spec)
    if bounds:
        _write_bounds(case_dir)
    _write_generic_constraint(case_dir, output)
    return output


def _availability(
    output: Path, usable: Mapping[str, bool] = _ALL_USABLE
) -> dict[str, frozenset[str] | None]:
    return with_entity_lists(probe_output(output), usable)


def _remove(output: Path, key: str) -> None:
    if PARQUET_FILES.get(key):
        (case_dir_for(output) / key).unlink()
    else:
        drop_partition(output, key)


def _drop(output: Path, key: str, column: str) -> None:
    if PARQUET_FILES.get(key):
        path = case_dir_for(output) / key
        pq.write_table(pq.read_table(path).drop_columns([column]), path)
    else:
        drop_column(output, key, column)


def _unavailable(decisions: Mapping[str, SectionAvailability]) -> list[str]:
    return [name for name, decision in decisions.items() if not decision.available]


def _rows(decisions: Mapping[str, SectionAvailability]) -> list[Unavailable]:
    return unavailable_rows(decisions.values())


class TestTables:
    def test_sections_are_declared_in_report_order(self) -> None:
        assert list(REQUIREMENTS) == _SECTIONS

    def test_each_requirement_is_keyed_by_its_section(self) -> None:
        assert all(name == r.section for name, r in REQUIREMENTS.items())

    @pytest.mark.parametrize("table", [REQUIREMENTS, OPTIONAL_PARTITIONS])
    def test_the_tables_are_read_only(self, table: Any) -> None:
        with pytest.raises(TypeError):
            table["x"] = None
        inners: list[Any] = list(OPTIONAL_PARTITIONS.values())
        for inner in inners:
            with pytest.raises(TypeError):
                inner["x"] = None

    @pytest.mark.parametrize(
        ("entity", "sections"),
        [
            ("thermal", {THERMAL, ENERGY_BALANCE}),
            ("line", {LINE, LINE_LIMITS, BUS_NET_IMPORT}),
            ("generic_constraint", {GENERIC_CONSTRAINTS}),
        ],
    )
    def test_an_entity_list_is_required_by_its_sections_alone(
        self, entity: str, sections: set[str]
    ) -> None:
        key = ENTITY_LIST_KEYS[entity]

        holders = {name for name, r in REQUIREMENTS.items() if key in r.partitions}

        assert holders == sections
        assert {REQUIREMENTS[name].partitions[key] for name in holders} == {frozenset()}

    def test_an_entity_list_is_never_optional(self) -> None:
        lists = set(ENTITY_LIST_KEYS.values())

        assert [
            name
            for name, optional in OPTIONAL_PARTITIONS.items()
            if lists & set(optional)
        ] == []

    def test_required_and_optional_partitions_are_disjoint(self) -> None:
        for name, requirement in REQUIREMENTS.items():
            optional = OPTIONAL_PARTITIONS.get(name, {})

            assert set(requirement.partitions).isdisjoint(optional), name

    def test_optional_partitions_belong_to_declared_sections(self) -> None:
        assert set(OPTIONAL_PARTITIONS) <= set(REQUIREMENTS)

    def test_the_scenario_summary_is_optional_where_scenarios_are_averaged(
        self,
    ) -> None:
        assert [
            name
            for name in REQUIREMENTS
            if _SCENARIO_SUMMARY in OPTIONAL_PARTITIONS.get(name, {})
        ] == _WEIGHTED_SECTIONS
        assert {
            OPTIONAL_PARTITIONS[name][_SCENARIO_SUMMARY] for name in _WEIGHTED_SECTIONS
        } == {frozenset({"scenario_id", "probability"})}

    def test_costs_variables_are_the_components_and_the_two_roll_ups(self) -> None:
        variables = REQUIREMENTS[COSTS].variables

        assert list(variables) == [
            *NOVOMODELO_COST_COMPONENT_COLUMNS,
            "immediate_cost",
            "future_cost",
        ]
        assert all(columns == {name} for name, (_, columns) in variables.items())

    def test_every_simulation_partition_is_a_probed_key(self) -> None:
        probed = {f"simulation/{entity}" for entity in SIMULATION_ENTITIES} | set(
            PARQUET_FILES
        )
        keys = {
            key
            for name, requirement in REQUIREMENTS.items()
            for key in (*requirement.partitions, *OPTIONAL_PARTITIONS.get(name, {}))
        } - set(ENTITY_LIST_KEYS.values())

        assert keys <= probed


class TestDecideSections:
    def test_a_full_pair_clears_every_section_and_every_variable_it_holds(
        self, tmp_path: Path
    ) -> None:
        availability = _availability(_case(tmp_path / "case"))

        decisions = decide_sections(availability, availability)

        assert list(decisions) == _SECTIONS
        assert _unavailable(decisions) == []
        assert {n: d.variables for n, d in decisions.items()} == {
            name: tuple(requirement.variables)
            for name, requirement in REQUIREMENTS.items()
        }
        assert _rows(decisions) == []

    def test_a_pair_without_the_bounds_files_lacks_exactly_their_sections(
        self, tmp_path: Path
    ) -> None:
        spec = SolvedCaseSpec(load_stats=True)
        baseline = _case(tmp_path / "baseline", spec, bounds=False)
        candidate = _case(tmp_path / "candidate", spec, bounds=False)

        decisions = decide_sections(_availability(baseline), _availability(candidate))

        assert _unavailable(decisions) == ["hydro-bounds", "line-limits"]
        assert sum(d.available for d in decisions.values()) == 11
        assert _rows(decisions) == [
            Unavailable("hydro-bounds", None, "both"),
            Unavailable("line-limits", None, "both"),
        ]

    def test_the_inputs_are_left_as_they_were(self, tmp_path: Path) -> None:
        output = _case(tmp_path / "case")
        _remove(output, _SCENARIO_SUMMARY)
        availability = _availability(output)
        before = dict(availability)

        decide_sections(availability, availability)

        assert availability == before
        assert availability[_SCENARIO_SUMMARY] is None

    @pytest.mark.parametrize("side", ["baseline", "candidate"])
    def test_a_dropped_partition_is_a_row_for_that_side(
        self, tmp_path: Path, side: str
    ) -> None:
        intact = _availability(_case(tmp_path / "intact"))
        damaged_output = _case(tmp_path / "damaged")
        drop_partition(damaged_output, "simulation/hydro_bus_generation")
        damaged = _availability(damaged_output)
        baseline, candidate = (
            (damaged, intact) if side == "baseline" else (intact, damaged)
        )

        decisions = decide_sections(baseline, candidate)

        assert _unavailable(decisions) == [ENERGY_BALANCE, OPERATING_POINTS]
        assert _rows(decisions) == [
            Unavailable("energy-balance", None, side),
            Unavailable("operating-points", None, side),
        ]

    def test_the_same_dropped_partition_on_both_sides_is_one_row_for_both(
        self, tmp_path: Path
    ) -> None:
        output = _case(tmp_path / "case")
        drop_partition(output, "simulation/hydro_bus_generation")
        availability = _availability(output)

        decisions = decide_sections(availability, availability)

        assert _rows(decisions) == [
            Unavailable("energy-balance", None, "both"),
            Unavailable("operating-points", None, "both"),
        ]

    @pytest.mark.parametrize("side", ["baseline", "candidate"])
    @pytest.mark.parametrize(
        ("entity", "sections"),
        [
            ("thermal", ["thermal", "energy-balance"]),
            ("line", ["bus-net-import", "line", "line-limits"]),
            ("generic_constraint", ["generic-constraints"]),
        ],
    )
    def test_an_unusable_entity_list_makes_its_sections_unavailable_on_that_side(
        self, tmp_path: Path, entity: str, sections: list[str], side: str
    ) -> None:
        output = _case(tmp_path / "case")
        usable = {**_ALL_USABLE, entity: False}
        unusable, usable_side = _availability(output, usable), _availability(output)
        baseline, candidate = (
            (unusable, usable_side) if side == "baseline" else (usable_side, unusable)
        )

        decisions = decide_sections(baseline, candidate)

        assert _unavailable(decisions) == sorted(sections, key=_SECTIONS.index)
        assert _rows(decisions) == [
            Unavailable(section, None, side) for section in sorted(sections)
        ]

    def test_an_unusable_entity_list_on_both_sides_is_one_row_for_both(
        self, tmp_path: Path
    ) -> None:
        availability = _availability(
            _case(tmp_path / "case"), {**_ALL_USABLE, "line": False}
        )

        decisions = decide_sections(availability, availability)

        assert _rows(decisions) == [
            Unavailable("bus-net-import", None, "both"),
            Unavailable("line", None, "both"),
            Unavailable("line-limits", None, "both"),
        ]

    @pytest.mark.parametrize(
        ("key", "unavailable"),
        [
            (_SCENARIO_SUMMARY, []),
            ("simulation/non_controllables", []),
            ("simulation/thermals", ["thermal"]),
            ("simulation/exchanges", ["bus-net-import", "line"]),
            ("scenarios/load_seasonal_stats.parquet", ["bus-net-import"]),
        ],
    )
    def test_an_absent_optional_partition_leaves_the_section_available(
        self, tmp_path: Path, key: str, unavailable: list[str]
    ) -> None:
        baseline = _availability(_case(tmp_path / "baseline"))
        output = _case(tmp_path / "candidate")
        _remove(output, key)

        decisions = decide_sections(baseline, _availability(output))

        optional_in = {
            name for name in REQUIREMENTS if key in OPTIONAL_PARTITIONS.get(name, {})
        }
        assert optional_in
        assert _unavailable(decisions) == unavailable
        assert all(decisions[name].available for name in optional_in - set(unavailable))
        assert _rows(decisions) == [
            Unavailable(name, None, "candidate") for name in sorted(unavailable)
        ]

    def test_an_absent_optional_partition_on_both_sides_is_no_row(
        self, tmp_path: Path
    ) -> None:
        output = _case(tmp_path / "case")
        _remove(output, _SCENARIO_SUMMARY)
        availability = _availability(output)

        decisions = decide_sections(availability, availability)

        assert _rows(decisions) == []

    def test_an_energy_balance_without_thermal_plants_stays_available(
        self, tmp_path: Path
    ) -> None:
        baseline = _availability(_case(tmp_path / "baseline"))
        output = _case(
            tmp_path / "candidate", SolvedCaseSpec(load_stats=True, thermals=())
        )

        decisions = decide_sections(baseline, _availability(output))

        assert decisions[ENERGY_BALANCE].available
        assert decisions[THERMAL].unavailable == (
            Unavailable("thermal", None, "candidate"),
        )

    def test_a_thermal_partition_without_its_id_makes_the_energy_balance_unavailable(
        self, tmp_path: Path
    ) -> None:
        baseline = _availability(_case(tmp_path / "baseline"))
        output = _case(tmp_path / "candidate", SolvedCaseSpec(load_stats=True))
        drop_column(output, "simulation/thermals", "thermal_id")

        decisions = decide_sections(baseline, _availability(output))

        assert decisions[ENERGY_BALANCE].available is False
        assert decisions[ENERGY_BALANCE].unavailable == (
            Unavailable("energy-balance", "thermal_id", "candidate"),
        )

    @pytest.mark.parametrize(
        ("key", "column", "sections"),
        [
            (_SCENARIO_SUMMARY, "probability", _WEIGHTED_SECTIONS),
            (_SCENARIO_SUMMARY, "scenario_id", _WEIGHTED_SECTIONS),
            (
                "simulation/exchanges",
                "line_id",
                [BUS, GENERIC_CONSTRAINTS, LINE, BUS_NET_IMPORT],
            ),
            ("scenarios/load_seasonal_stats.parquet", "mean_mw", [BUS, BUS_NET_IMPORT]),
            ("simulation/non_controllables", "non_controllable_id", [ENERGY_BALANCE]),
        ],
    )
    @pytest.mark.parametrize("sides", ["candidate", "both"])
    def test_an_optional_partition_without_a_listed_column_is_a_column_row(
        self, tmp_path: Path, key: str, column: str, sections: list[str], sides: str
    ) -> None:
        intact = _availability(_case(tmp_path / "intact"))
        output = _case(tmp_path / "damaged")
        _drop(output, key, column)
        damaged = _availability(output)

        decisions = decide_sections(damaged if sides == "both" else intact, damaged)

        assert _rows(decisions) == sorted(
            (Unavailable(name, column, sides) for name in sections),
            key=lambda row: row.section,
        )
        assert all(not decisions[name].available for name in sections)


def _read_hydro(output: Path, cleared: tuple[str, ...]) -> None:
    means = read_novomodelo_hydro_means(output)
    read_novomodelo_hydro_percentiles(output)
    if "total_inflow_m3s" in cleared:
        read_novomodelo_hydro_total_flows(output, means)


def _read_hydro_bounds(output: Path, cleared: tuple[str, ...]) -> None:
    read_novomodelo_hydro_per_stage_bounds(output)
    if "withdrawal_m3s" in cleared:
        read_novomodelo_hydro_withdrawal(output)


def _read_system_spillage(output: Path, cleared: tuple[str, ...]) -> None:
    read_novomodelo_spillage_energy(output)


def _read_thermal(output: Path, cleared: tuple[str, ...]) -> None:
    if "generation_mw" in cleared:
        read_novomodelo_thermal_means(output)
        read_novomodelo_thermal_percentiles(output)


def _read_bus(output: Path, cleared: tuple[str, ...]) -> None:
    read_novomodelo_bus_means(output)
    read_novomodelo_bus_percentiles(output)
    read_novomodelo_bus_metadata(output)


def _read_bus_net_import(output: Path, cleared: tuple[str, ...]) -> None:
    read_novomodelo_bus_net_import(output)


def _read_line(output: Path, cleared: tuple[str, ...]) -> None:
    if "net_flow_mw" in cleared:
        read_novomodelo_line_means(output)
        read_novomodelo_line_percentiles(output)


def _read_line_limits(output: Path, cleared: tuple[str, ...]) -> None:
    read_novomodelo_stage_line_bounds(output, read_novomodelo_lines(output))


def _read_energy_balance(output: Path, cleared: tuple[str, ...]) -> None:
    read_novomodelo_bus_aggregates(output)
    read_novomodelo_hydro_bus_labels(output)


def _read_costs(output: Path, cleared: tuple[str, ...]) -> None:
    read_novomodelo_cost_breakdown(output, max_stage_id=0)
    read_novomodelo_stage_costs(output)


def _read_generic_constraints(output: Path, cleared: tuple[str, ...]) -> None:
    resolved = read_novomodelo_resolved_generic_constraints(
        case_dir_for(output), output
    )
    evaluate_lhs_novomodelo(resolved, output)
    stage_bounds_novomodelo(resolved, output)


def _read_production_function(output: Path, cleared: tuple[str, ...]) -> None:
    planes = read_novomodelo_fpha_planes(output)
    assert planes is not None
    novomodelo_plane_arrays(planes)


def _read_operating_points(output: Path, cleared: tuple[str, ...]) -> None:
    hydros = range(len(_FULL.hydros))
    novomodelo_points(output, {hydro: hydro for hydro in hydros})


#: Each reader call list takes the output and the section's cleared variables,
#: which only the readers of a conditional variable consult.
_READERS: Mapping[str, Callable[[Path, tuple[str, ...]], None]] = {
    HYDRO: _read_hydro,
    HYDRO_BOUNDS: _read_hydro_bounds,
    SYSTEM_SPILLAGE: _read_system_spillage,
    THERMAL: _read_thermal,
    BUS: _read_bus,
    BUS_NET_IMPORT: _read_bus_net_import,
    LINE: _read_line,
    LINE_LIMITS: _read_line_limits,
    ENERGY_BALANCE: _read_energy_balance,
    COSTS: _read_costs,
    GENERIC_CONSTRAINTS: _read_generic_constraints,
    PRODUCTION_FUNCTION: _read_production_function,
    OPERATING_POINTS: _read_operating_points,
}


def _is_hive_entity(key: str) -> bool:
    return key in {f"simulation/{entity}" for entity in SIMULATION_ENTITIES}


def _droppable(key: str, columns: frozenset[str]) -> list[str]:
    """The columns of *key* a test can drop: a simulation entity's hive
    partition column ``scenario_id`` is in no file."""
    return sorted(columns - {"scenario_id"} if _is_hive_entity(key) else columns)


def _damage_cases() -> list[tuple[str, str, str]]:
    """Every ``(section, key, column)`` the table-against-reader test drops: all
    columns of a required partition in the full fixture, and the listed columns
    of an optional one."""
    with tempfile.TemporaryDirectory() as root:
        probed = probe_output(_case(Path(root) / "case"))
    cases: list[tuple[str, str, str]] = []
    for section, requirement in REQUIREMENTS.items():
        sources = [
            *((key, probed.get(key) or frozenset()) for key in requirement.partitions),
            *OPTIONAL_PARTITIONS.get(section, {}).items(),
        ]
        cases += [
            (section, key, column)
            for key, columns in sources
            for column in _droppable(key, columns)
        ]
    return cases


_DAMAGE_CASES = _damage_cases()


@pytest.fixture(scope="module")
def baseline_availability(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, frozenset[str] | None]:
    return _availability(_case(tmp_path_factory.mktemp("baseline")))


def test_every_section_has_a_reader_call_list() -> None:
    assert list(_READERS) == list(REQUIREMENTS)


def test_the_damage_cases_cover_every_section_and_every_optional_column() -> None:
    assert {section for section, _, _ in _DAMAGE_CASES} == set(REQUIREMENTS)
    for section, optional in OPTIONAL_PARTITIONS.items():
        for key, columns in optional.items():
            assert {
                column for s, k, column in _DAMAGE_CASES if (s, k) == (section, key)
            } == set(_droppable(key, columns))


@pytest.mark.parametrize(
    ("section", "key", "column"),
    _DAMAGE_CASES,
    ids=[f"{section}|{key}|{column}" for section, key, column in _DAMAGE_CASES],
)
def test_a_dropped_column_ends_in_drift_or_a_read_that_does_not_raise(
    tmp_path: Path,
    baseline_availability: dict[str, frozenset[str] | None],
    section: str,
    key: str,
    column: str,
) -> None:
    output = _case(tmp_path / "candidate")
    _drop(output, key, column)

    decision = decide_sections(baseline_availability, _availability(output))[section]

    declared = {
        **REQUIREMENTS[section].partitions,
        **OPTIONAL_PARTITIONS.get(section, {}),
    }
    if column in declared.get(key, frozenset()):
        assert Unavailable(section, column, "candidate") in decision.unavailable
        assert not decision.available
    if decision.available:
        with dx.collect():
            _READERS[section](output, decision.variables)

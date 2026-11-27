"""The sections of a version comparison and what each needs from an output.

:data:`REQUIREMENTS` declares, per section, the availability keys and columns
its readers cannot do without and each compared variable with its source
columns; :func:`decide_sections` sets them against the two sides. A section
name is persisted as the ``section`` of an unavailable row, in the facts record
and in the ``section-unavailable`` Diagnostic: a one-way contract.

A key is a probe key or an entity-list key from
:data:`~novomodelo_bridge.comparators.version.availability.ENTITY_LIST_KEYS`. An
entity list is never optional: an unusable one makes its sections unavailable
on that side.

:data:`OPTIONAL_PARTITIONS` holds the partitions a section's readers read only
when they exist, such as the scenario summary or a case's thermal simulation.
Their absence is an ordinary case and cannot be required, but a present one that
lacks its key columns makes the reader raise, so it is declared and fed through
the same availability rule.

A section absent on both sides is reported once, for ``both``; none is optional.
The convergence file is not a section: its reader returns an empty frame and
the comparability classifier reports it as ``bounds-missing``. The functions
emit no Diagnostic and read no file.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from novomodelo_bridge.comparators.version.availability import (
    ENTITY_LIST_KEYS,
    SectionAvailability,
    SectionRequirement,
    section_availability,
)
from novomodelo_bridge.novomodelo.cost_categories import (
    NOVOMODELO_COST_COMPONENT_COLUMNS,
)

HYDRO = "hydro"
HYDRO_BOUNDS = "hydro-bounds"
SYSTEM_SPILLAGE = "system-spillage"
THERMAL = "thermal"
BUS = "bus"
BUS_NET_IMPORT = "bus-net-import"
LINE = "line"
LINE_LIMITS = "line-limits"
ENERGY_BALANCE = "energy-balance"
COSTS = "costs"
GENERIC_CONSTRAINTS = "generic-constraints"
PRODUCTION_FUNCTION = "production-function"
OPERATING_POINTS = "operating-points"

_HYDROS = "simulation/hydros"
_THERMALS = "simulation/thermals"
_BUSES = "simulation/buses"
_EXCHANGES = "simulation/exchanges"
_NON_CONTROLLABLES = "simulation/non_controllables"
_HYDRO_BUS_GENERATION = "simulation/hydro_bus_generation"
_COSTS = "simulation/costs"
_SCENARIO_SUMMARY = "simulation/scenario_summary.parquet"
_FPHA = "hydro_models/fpha_hyperplanes.parquet"
_HYDRO_BOUNDS = "constraints/hydro_bounds.parquet"
_LINE_BOUNDS = "constraints/line_bounds.parquet"
_LOAD_STATS = "scenarios/load_seasonal_stats.parquet"
_ECHO = "generic_constraints/resolved_echo.parquet"
_THERMAL_LIST = ENTITY_LIST_KEYS["thermal"]
_LINE_LIST = ENTITY_LIST_KEYS["line"]
_GENERIC_CONSTRAINT_LIST = ENTITY_LIST_KEYS["generic_constraint"]

_NO_COLUMNS: frozenset[str] = frozenset()

_ROW_KEYS = frozenset({"stage_id", "block_id"})
_HYDRO_ROW = _ROW_KEYS | {"hydro_id"}
_THERMAL_ROW = _ROW_KEYS | {"thermal_id"}
_BUS_ROW = _ROW_KEYS | {"bus_id"}
_LINE_ROW = _ROW_KEYS | {"line_id"}
_NON_CONTROLLABLE_ROW = _ROW_KEYS | {"non_controllable_id"}

_SCENARIO_SUMMARY_COLUMNS = frozenset({"scenario_id", "probability"})
_LOAD_STATS_COLUMNS = frozenset({"bus_id", "mean_mw"})
_HYDRO_BUS_GENERATION_COLUMNS = frozenset(
    {
        "hydro_id",
        "bus_id",
        "stage_id",
        "block_id",
        "turbined_m3s",
        "generation_mw",
        "generation_mwh",
    }
)
_FPHA_COLUMNS = frozenset(
    {"hydro_id", "stage_id", "gamma_0", "gamma_v", "gamma_q", "gamma_s", "kappa"}
)

_Variables = dict[str, tuple[str, frozenset[str]]]


def _own_columns(key: str, names: Iterable[str]) -> _Variables:
    return {name: (key, frozenset({name})) for name in names}


REQUIREMENTS: Mapping[str, SectionRequirement] = MappingProxyType(
    {
        requirement.section: requirement
        for requirement in (
            SectionRequirement(
                HYDRO,
                {_HYDROS: _HYDRO_ROW},
                {
                    **_own_columns(
                        _HYDROS,
                        (
                            "storage_final_hm3",
                            "generation_mw",
                            "turbined_m3s",
                            "spillage_m3s",
                            "inflow_m3s",
                            "water_value_per_hm3",
                            "evaporation_m3s",
                            "outflow_m3s",
                        ),
                    ),
                    "total_inflow_m3s": (
                        _HYDROS,
                        frozenset({"incremental_inflow_m3s", "outflow_m3s"}),
                    ),
                    "productivity_mw_per_m3s": (
                        _HYDROS,
                        frozenset({"generation_mw", "turbined_m3s"}),
                    ),
                },
            ),
            SectionRequirement(
                HYDRO_BOUNDS,
                {
                    _HYDROS: _HYDRO_ROW,
                    _HYDRO_BOUNDS: frozenset({"hydro_id", "stage_id"}),
                },
                {
                    "withdrawal_m3s": (
                        _HYDRO_BOUNDS,
                        frozenset({"water_withdrawal_m3s"}),
                    )
                },
            ),
            SectionRequirement(
                SYSTEM_SPILLAGE,
                {
                    _HYDROS: _HYDRO_ROW
                    | {"spillage_m3s", "equivalent_productivity_mw_per_m3s"}
                },
                dict.fromkeys(
                    (
                        "spill_energy_total_mw",
                        "spill_energy_reservoir_mw",
                        "spill_energy_rorov_mw",
                    ),
                    (_HYDROS, _NO_COLUMNS),
                ),
            ),
            SectionRequirement(
                THERMAL,
                {_THERMAL_LIST: _NO_COLUMNS, _THERMALS: _THERMAL_ROW},
                _own_columns(_THERMALS, ("generation_mw",)),
            ),
            SectionRequirement(
                BUS,
                {_BUSES: _BUS_ROW},
                _own_columns(_BUSES, ("spot_price", "deficit_mw")),
            ),
            SectionRequirement(
                BUS_NET_IMPORT,
                {
                    _LINE_LIST: _NO_COLUMNS,
                    _BUSES: _BUS_ROW,
                    _EXCHANGES: _LINE_ROW | {"net_flow_mw"},
                    _LOAD_STATS: _LOAD_STATS_COLUMNS,
                },
                {"net_import_mw": (_EXCHANGES, _NO_COLUMNS)},
            ),
            SectionRequirement(
                LINE,
                {_LINE_LIST: _NO_COLUMNS, _EXCHANGES: _LINE_ROW},
                _own_columns(_EXCHANGES, ("net_flow_mw",)),
            ),
            SectionRequirement(
                LINE_LIMITS,
                {
                    _LINE_LIST: _NO_COLUMNS,
                    _LINE_BOUNDS: _LINE_ROW | {"direct_mw", "reverse_mw"},
                },
            ),
            SectionRequirement(
                ENERGY_BALANCE,
                {
                    _THERMAL_LIST: _NO_COLUMNS,
                    _BUSES: _BUS_ROW,
                    _HYDRO_BUS_GENERATION: _HYDRO_BUS_GENERATION_COLUMNS,
                },
                {
                    "hydro_gen_mw": (_HYDRO_BUS_GENERATION, _NO_COLUMNS),
                    "thermal_gen_mw": (_THERMAL_LIST, _NO_COLUMNS),
                    **_own_columns(_BUSES, ("deficit_mw", "excess_mw")),
                    "net_load_mw": (_BUSES, frozenset({"load_mw"})),
                },
            ),
            SectionRequirement(
                COSTS,
                {
                    _COSTS: frozenset(
                        {
                            "stage_id",
                            "discount_factor",
                            # read_novomodelo_stage_costs reads the four together.
                            "immediate_cost",
                            "future_cost",
                            "thermal_cost",
                            "anticipated_thermal_cost",
                        }
                    )
                },
                _own_columns(
                    _COSTS,
                    (
                        *NOVOMODELO_COST_COMPONENT_COLUMNS,
                        "immediate_cost",
                        "future_cost",
                    ),
                ),
            ),
            SectionRequirement(
                GENERIC_CONSTRAINTS,
                {
                    _GENERIC_CONSTRAINT_LIST: _NO_COLUMNS,
                    _ECHO: frozenset(
                        {
                            "constraint_id",
                            "stage_id",
                            "block_id",
                            "term_index",
                            "variable_kind",
                            "variable",
                            "coefficient",
                            "bound_lower",
                            "bound_upper",
                        }
                    ),
                },
            ),
            SectionRequirement(PRODUCTION_FUNCTION, {_FPHA: _FPHA_COLUMNS}),
            SectionRequirement(
                OPERATING_POINTS,
                {
                    _FPHA: _FPHA_COLUMNS,
                    _HYDROS: _HYDRO_ROW
                    | {
                        "turbined_m3s",
                        "spillage_m3s",
                        "outflow_m3s",
                        "incremental_inflow_m3s",
                        "inflow_nonnegativity_slack_m3s",
                        "storage_initial_hm3",
                        "storage_final_hm3",
                        "generation_mw",
                        "water_value_per_hm3",
                        "accumulated_productivity_mw_per_m3s",
                    },
                    _BUSES: _BUS_ROW | {"spot_price"},
                    _HYDRO_BUS_GENERATION: _HYDRO_BUS_GENERATION_COLUMNS,
                },
            ),
        )
    }
)

_SCENARIO_WEIGHTS: Mapping[str, frozenset[str]] = MappingProxyType(
    {_SCENARIO_SUMMARY: _SCENARIO_SUMMARY_COLUMNS}
)

OPTIONAL_PARTITIONS: Mapping[str, Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        **dict.fromkeys(
            (
                HYDRO,
                HYDRO_BOUNDS,
                SYSTEM_SPILLAGE,
                THERMAL,
                LINE,
                BUS_NET_IMPORT,
                COSTS,
            ),
            _SCENARIO_WEIGHTS,
        ),
        BUS: MappingProxyType(
            {
                **_SCENARIO_WEIGHTS,
                _EXCHANGES: _LINE_ROW,
                _LOAD_STATS: _LOAD_STATS_COLUMNS,
            }
        ),
        ENERGY_BALANCE: MappingProxyType(
            {
                **_SCENARIO_WEIGHTS,
                _THERMALS: _THERMAL_ROW,
                _NON_CONTROLLABLES: _NON_CONTROLLABLE_ROW,
            }
        ),
        GENERIC_CONSTRAINTS: MappingProxyType(
            {
                **_SCENARIO_WEIGHTS,
                _HYDROS: _HYDRO_ROW,
                _THERMALS: _THERMAL_ROW,
                _EXCHANGES: _LINE_ROW,
            }
        ),
    }
)

_EFFECTIVE: Mapping[str, SectionRequirement] = {
    name: SectionRequirement(
        name,
        {**required.partitions, **OPTIONAL_PARTITIONS.get(name, {})},
        required.variables,
    )
    for name, required in REQUIREMENTS.items()
}


def _satisfy_absent(
    side: Mapping[str, frozenset[str] | None], optional: Mapping[str, frozenset[str]]
) -> dict[str, frozenset[str] | None]:
    return {
        **side,
        **{key: columns for key, columns in optional.items() if side.get(key) is None},
    }


def decide_sections(
    baseline: Mapping[str, frozenset[str] | None],
    candidate: Mapping[str, frozenset[str] | None],
) -> dict[str, SectionAvailability]:
    """One verdict per section, in the order of :data:`REQUIREMENTS`.

    Each side is a probe result extended by ``with_entity_lists``. An optional
    partition that is absent satisfies its requirement; one that is present
    without a listed column yields the column row ``(section, column, side)``.
    """
    return {
        name: section_availability(
            requirement,
            _satisfy_absent(baseline, OPTIONAL_PARTITIONS.get(name, {})),
            _satisfy_absent(candidate, OPTIONAL_PARTITIONS.get(name, {})),
        )
        for name, requirement in _EFFECTIVE.items()
    }

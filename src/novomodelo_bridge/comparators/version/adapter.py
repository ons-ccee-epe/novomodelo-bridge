"""Encoding of a Novomodelo baseline into the reference slots of the report.

The charts read the reference side of a comparison from the ``nw_*`` fields of
``PercentileData`` in the source models' units and vocabulary. This module is the
one place that encodes a Novomodelo baseline into them, each unit conversion
stated once: ``nw_costs`` stays in R$, the cost series of ``nw_sin`` are in 10^6 R$,
its ``EARMF`` in MWmes and ``ENA`` in MW, the bus series in MW and the timings in
seconds. The slots are chart inputs only; the verdict rows are built from the
readers' native values and never read them. Nothing here reads a file or emits a
Diagnostic; an empty frame means that its section was not read.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypedDict

import polars as pl

from novomodelo_bridge.comparators.version.alignment import AlignedPair
from novomodelo_bridge.comparators.version.sides import SideInputs
from novomodelo_bridge.core.units import MEAN_MONTH_HOURS

# A component takes the first source-model label of its cost-breakdown row
# (`charts.costs._COST_MAP`), which the chart sums, so one label lands the
# baseline on that row. Both thermal columns take the delivery-basis label: the
# decision-basis labels make `charts.costs._cost_map` raise when both bases are
# present. The three labels spelled like a column are the Novomodelo-only
# components; they are keys of ``nw_costs`` in the persisted dataset, so one
# renamed orphans every dataset already written.
PARCEL_OF_COST_COLUMN: Mapping[str, str] = MappingProxyType(
    {
        "thermal_cost": "GERACAO TERMICA",
        "anticipated_thermal_cost": "GERACAO TERMICA",
        "contract_cost": "CONTRATOS",
        "deficit_cost": "DEFICIT",
        "excess_cost": "EXCESSO ENERGIA",
        "outflow_violation_below_cost": "VIOLACAO VZMIN",
        "outflow_violation_above_cost": "VIOL. DEFL. MAXIMA",
        "turbined_violation_cost": "VIOL. TURB. MINIMO",
        "generation_violation_cost": "VIOLACAO GHMIN",
        "storage_violation_cost": "storage_violation_cost",
        "filling_target_cost": "VIOLACAO EVMIN",
        "evaporation_violation_cost": "VIOL. EVAP. UHE",
        "withdrawal_violation_cost": "VIOLACAO RETIRADA",
        "fpha_violation_cost": "VIOLACAO FPHA",
        "inflow_penalty_cost": "inflow_penalty_cost",
        "generic_violation_cost": "VIOLACAO CAR",
        "spillage_cost": "VERTIMENTO",
        "turbined_cost": "TURBINAMENTO UHE",
        "curtailment_cost": "CORTE GER. EOLICA",
        "exchange_cost": "INTERCAMBIO",
        "pumping_cost": "pumping_cost",
    }
)

_REAIS_PER_MILLION = 1.0e6
_MS_PER_SECOND = 1000.0

# `nw_sin` variable to (cleared cost variable that admits it, stage-cost column).
_STAGE_COST_VARIABLES = {
    "COPER": ("immediate_cost", "immediate_cost"),
    "CUSTO_FUTURO": ("future_cost", "future_cost"),
    "CTERM": ("thermal_cost", "thermal_cost_total"),
}

#: Stage-cost column to the cleared cost variable that admits it, on either side.
STAGE_COST_GATES: dict[str, str] = {
    column: admitted_by for admitted_by, column in _STAGE_COST_VARIABLES.values()
}

# `nw_market` and `nw_net_load` variable to the cleared balance variable it carries.
_MARKET_VARIABLES = {
    "GHTOT": "hydro_gen_mw",
    "GTERM": "thermal_gen_mw",
    "DEFT": "deficit_mw",
    "EXCESSO": "excess_mw",
}
_NET_LOAD_VARIABLES = {"NET_LOAD": "net_load_mw"}

_STAGE_KEYS = {"stage_id": "stage"}
_BUS_KEYS = {"bus_id": "newave_code", "stage_id": "stage"}

_SLACK_COLUMNS = (
    "water_withdrawal_violation_pos_m3s",
    "water_withdrawal_violation_neg_m3s",
    "evaporation_violation_pos_m3s",
    "evaporation_violation_neg_m3s",
)
_HYDRO_SLACK_SCHEMA = pl.Schema(
    {
        "entity_id": pl.Int64,
        "stage_id": pl.Int64,
        **dict.fromkeys(_SLACK_COLUMNS, pl.Float64),
    }
)

_ITERATION_SCHEMA = {"iteration": pl.Int64, "total_seconds": pl.Float64}
_SPLIT_COLUMNS = {
    "backward_seconds": "time_backward_ms",
    "forward_seconds": "time_forward_ms",
}


@dataclass(frozen=True)
class BaselineFrames:
    """What the baseline's readers returned, each in the reader's native unit.

    An empty frame or mapping means that its section was not read.

    Attributes:
        convergence: ``iteration``, ``lower_bound``, ``upper_bound``.
        cost_breakdown: Cost column to R$ present value.
        stage_costs: ``stage_id`` and per-stage costs in R$.
        hydro_means: Per ``(entity_id, stage_id)`` means; ``stored_energy_final_mwh``
            in MWh, ``incremental_inflow_energy_mw`` in MW, the slacks in m3/s.
        bus_aggregates: ``bus_id``, ``stage_id`` and ``<variable>_p10``, ``_p50``,
            ``_p90`` in MW.
        iteration_timing: ``iteration`` and ``time_*_ms`` in milliseconds.
        training_seconds: Training wall-clock in seconds; 0.0 when unknown.
    """

    convergence: pl.DataFrame = field(default_factory=pl.DataFrame)
    cost_breakdown: Mapping[str, float] = field(default_factory=dict)
    stage_costs: pl.DataFrame = field(default_factory=pl.DataFrame)
    hydro_means: pl.DataFrame = field(default_factory=pl.DataFrame)
    bus_aggregates: pl.DataFrame = field(default_factory=pl.DataFrame)
    iteration_timing: pl.DataFrame = field(default_factory=pl.DataFrame)
    training_seconds: float = 0.0


class BaselineSlots(TypedDict):
    """The reference slots of ``PercentileData`` that a baseline fills."""

    nw_convergence: pl.DataFrame
    nw_costs: dict[str, float]
    nw_sin: pl.DataFrame
    nw_market: pl.DataFrame
    nw_net_load: pl.DataFrame
    nw_bus_names: dict[int, str]
    nw_hydro_names: dict[int, str]
    nw_hydro_slacks: pl.DataFrame
    nw_tim_iterations: pl.DataFrame
    nw_tim_stages: dict[str, float]


def _long_frame(
    wide: pl.DataFrame,
    keys: Mapping[str, str],
    variables: Mapping[str, pl.Expr],
) -> pl.DataFrame:
    """*wide* as long rows: the key columns *keys* (source to slot name, Int64),
    ``variable`` and ``value`` (Float64).

    A variable whose source columns *wide* lacks, and a null value, build no row.
    """
    present = {
        name: expr
        for name, expr in variables.items()
        if set(expr.meta.root_names()) <= set(wide.columns)
    }
    if wide.is_empty() or not present:
        return pl.DataFrame(
            schema={
                **dict.fromkeys(keys.values(), pl.Int64),
                "variable": pl.Utf8,
                "value": pl.Float64,
            }
        )
    return (
        wide.select(
            *(
                pl.col(source).cast(pl.Int64).alias(name)
                for source, name in keys.items()
            ),
            *(expr.cast(pl.Float64).alias(name) for name, expr in present.items()),
        )
        .unpivot(
            index=list(keys.values()), variable_name="variable", value_name="value"
        )
        .drop_nulls("value")
    )


def _costs(
    cost_breakdown: Mapping[str, float], cost_columns: Collection[str]
) -> dict[str, float]:
    costs: dict[str, float] = {}
    for column, value in cost_breakdown.items():
        if column not in cost_columns:
            continue
        if column not in PARCEL_OF_COST_COLUMN:
            raise ValueError(f"cost column {column!r} has no parcel label")
        label = PARCEL_OF_COST_COLUMN[column]
        costs[label] = costs.get(label, 0.0) + value
    return costs


def _system_series(
    stage_costs: pl.DataFrame,
    hydro_means: pl.DataFrame,
    cost_columns: Collection[str],
) -> pl.DataFrame:
    costs = _long_frame(
        stage_costs,
        _STAGE_KEYS,
        {
            variable: pl.col(column) / _REAIS_PER_MILLION
            for variable, (admitted_by, column) in _STAGE_COST_VARIABLES.items()
            if admitted_by in cost_columns
        },
    )
    # Every baseline plant, not the paired ones: the slot is a system total. A
    # stage with a null in the sum has no total, never a smaller one.
    totals = (
        hydro_means
        if hydro_means.is_empty()
        else hydro_means.group_by("stage_id", maintain_order=True).agg(
            [
                pl.when(pl.col(column).null_count() == 0)
                .then(pl.col(column).sum())
                .alias(column)
                for column in (
                    "stored_energy_final_mwh",
                    "incremental_inflow_energy_mw",
                )
                if column in hydro_means.columns
            ]
        )
    )
    energies = _long_frame(
        totals,
        _STAGE_KEYS,
        {
            # MWh to MWmes: divide by the mean month's hours.
            "EARMF": pl.col("stored_energy_final_mwh") / MEAN_MONTH_HOURS,
            "ENA": pl.col("incremental_inflow_energy_mw"),
        },
    )
    return pl.concat([costs, energies])


def _bus_series(
    bus_aggregates: pl.DataFrame,
    variables: Mapping[str, str],
    balance_variables: Collection[str],
) -> pl.DataFrame:
    # The median across scenarios, not the mean: the candidate's own centre line.
    return _long_frame(
        bus_aggregates,
        _BUS_KEYS,
        {
            name: pl.col(f"{variable}_p50")
            for name, variable in variables.items()
            if variable in balance_variables
        },
    )


def _names(entries: Sequence[dict]) -> dict[int, str]:
    return {entry["id"]: entry["name"] for entry in entries}


def _hydro_slacks(
    hydro_means: pl.DataFrame, hydro_pairs: Sequence[AlignedPair]
) -> pl.DataFrame:
    if hydro_means.is_empty():
        return pl.DataFrame(schema=_HYDRO_SLACK_SCHEMA)
    candidate_ids = {pair.baseline_id: pair.candidate_id for pair in hydro_pairs}
    return (
        hydro_means.filter(pl.col("entity_id").is_in(list(candidate_ids)))
        .with_columns(
            pl.col("entity_id").replace_strict(candidate_ids, return_dtype=pl.Int64)
        )
        .match_to_schema(
            _HYDRO_SLACK_SCHEMA, missing_columns="insert", extra_columns="ignore"
        )
    )


def _iteration_timing(iteration_timing: pl.DataFrame) -> pl.DataFrame:
    if iteration_timing.is_empty():
        return pl.DataFrame(schema=_ITERATION_SCHEMA)
    timed = iteration_timing.drop_nulls("time_total_ms")
    seconds = {"total_seconds": "time_total_ms"}
    # The split chart reads the columns' presence as "the split is known".
    if all(
        column in timed.columns and not timed[column].has_nulls()
        for column in _SPLIT_COLUMNS.values()
    ):
        seconds |= _SPLIT_COLUMNS
    return timed.select(
        pl.col("iteration").cast(pl.Int64),
        *(
            (pl.col(column) / _MS_PER_SECOND).alias(name)
            for name, column in seconds.items()
        ),
    )


def _stage_timing(training_seconds: float) -> dict[str, float]:
    return {"Calculo da Politica": training_seconds} if training_seconds > 0 else {}


def encode_baseline(
    frames: BaselineFrames,
    baseline: SideInputs,
    hydro_pairs: Sequence[AlignedPair],
    *,
    cost_columns: Collection[str],
    balance_variables: Collection[str],
) -> BaselineSlots:
    """Encode the baseline's frames into the reference slots, filling all ten.

    Every frame is read as is and left unmutated. A null value builds no row and
    is never filled with zero; a column a frame lacks builds no variable.

    Args:
        frames: What the baseline's readers returned.
        baseline: Read for its ``hydros`` and ``buses`` name maps only.
        hydro_pairs: The hydro alignment; ``nw_hydro_slacks`` keeps the paired
            plants and carries each one's candidate id as ``entity_id``.
        cost_columns: Cleared variables of the ``costs`` section; a cost column,
            ``immediate_cost`` or ``future_cost`` outside it fills no slot.
        balance_variables: Cleared variables of the ``energy-balance`` section.

    Raises:
        ValueError: ``frames.cost_breakdown`` holds a cleared column with no entry
            in :data:`PARCEL_OF_COST_COLUMN`; the message names it.
    """
    return {
        "nw_convergence": frames.convergence,
        "nw_costs": _costs(frames.cost_breakdown, cost_columns),
        "nw_sin": _system_series(frames.stage_costs, frames.hydro_means, cost_columns),
        "nw_market": _bus_series(
            frames.bus_aggregates, _MARKET_VARIABLES, balance_variables
        ),
        "nw_net_load": _bus_series(
            frames.bus_aggregates, _NET_LOAD_VARIABLES, balance_variables
        ),
        "nw_bus_names": _names(baseline.buses),
        "nw_hydro_names": _names(baseline.hydros),
        "nw_hydro_slacks": _hydro_slacks(frames.hydro_means, hydro_pairs),
        "nw_tim_iterations": _iteration_timing(frames.iteration_timing),
        "nw_tim_stages": _stage_timing(frames.training_seconds),
    }

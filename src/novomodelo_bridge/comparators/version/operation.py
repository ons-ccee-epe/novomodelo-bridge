"""Operation rows of a version comparison: hydro plants, thermal plants, buses,
lines and system spillage.

Each side's values come from the readers' native-unit frames, so two identical
runs differ by exactly zero; nothing is converted or rounded. The hydro frame
folds the evaporation and withdrawal slacks into realized flows, a null flow or
slack reading as 0, and derives the productivity the way the track comparators
do. The functions emit no Diagnostic of their own: those of the readers reach the
enclosing sink tagged with the side.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import cast

import polars as pl

from novomodelo_bridge.comparators.model import ResultComparison
from novomodelo_bridge.comparators.version.alignment import (
    AlignedPair,
    NameAlignment,
    name_key,
)
from novomodelo_bridge.comparators.version.availability import SectionAvailability
from novomodelo_bridge.comparators.version.requirements import (
    BUS,
    BUS_NET_IMPORT,
    HYDRO,
    HYDRO_BOUNDS,
    LINE,
    SYSTEM_SPILLAGE,
    THERMAL,
)
from novomodelo_bridge.comparators.version.rows import paired_stage_rows
from novomodelo_bridge.comparators.version.sides import SideInputs, side_tagged
from novomodelo_bridge.novomodelo.readers import (
    read_novomodelo_bus_means,
    read_novomodelo_hydro_means,
    read_novomodelo_hydro_total_flows,
    read_novomodelo_hydro_withdrawal,
    read_novomodelo_line_means,
    read_novomodelo_spillage_energy,
    read_novomodelo_thermal_means,
)

ROW_GROUPS: tuple[str, ...] = (
    "hydro",
    "thermal",
    "bus",
    "line",
    "system_spillage",
)

_KEYS = ["entity_id", "stage_id"]

# Realized flow, its positive slack, its negative slack; every column in m3/s.
_REALIZED_FLOWS = (
    (
        "evaporation_m3s",
        "evaporation_violation_pos_m3s",
        "evaporation_violation_neg_m3s",
    ),
    (
        "withdrawal_m3s",
        "water_withdrawal_violation_pos_m3s",
        "water_withdrawal_violation_neg_m3s",
    ),
)

# m3/s: at or below it generation and turbined flow are both about zero, so
# their ratio is an undefined 0/0 and no productivity is reported.
_MIN_TURBINED_M3S = 1.0e-6

_SPILLAGE_COLUMNS = {
    "total_mw": "spill_energy_total_mw",
    "reservoir_mw": "spill_energy_reservoir_mw",
    "rorov_mw": "spill_energy_rorov_mw",
}


def _only(section: SectionAvailability, variable: str) -> tuple[str, ...]:
    return (variable,) if variable in section.variables else ()


def operation_variables(
    sections: Mapping[str, SectionAvailability],
) -> dict[str, tuple[str, ...]]:
    """The cleared variables of each entity type in :data:`ROW_GROUPS` order.

    An entity type without a cleared variable is left out. ``withdrawal_m3s``
    joins the hydro variables and ``net_import_mw`` the bus variables only when
    their own section cleared them.
    """
    cleared = {
        "hydro": (
            *sections[HYDRO].variables,
            *_only(sections[HYDRO_BOUNDS], "withdrawal_m3s"),
        ),
        "thermal": sections[THERMAL].variables,
        "bus": (
            *sections[BUS].variables,
            *_only(sections[BUS_NET_IMPORT], "net_import_mw"),
        ),
        "line": sections[LINE].variables,
        "system_spillage": sections[SYSTEM_SPILLAGE].variables,
    }
    return {group: variables for group, variables in cleared.items() if variables}


def _keyed(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col(key).cast(pl.Int64) for key in _KEYS)


def _hydro_frame(output_dir: Path, variables: Sequence[str]) -> pl.DataFrame:
    frame = _keyed(read_novomodelo_hydro_means(output_dir))
    if "total_inflow_m3s" in variables:
        total_flows = read_novomodelo_hydro_total_flows(output_dir, frame)
        frame = frame.join(_keyed(total_flows), on=_KEYS, how="left")
    if "withdrawal_m3s" in variables:
        withdrawal = read_novomodelo_hydro_withdrawal(output_dir)
        frame = frame.join(_keyed(withdrawal), on=_KEYS, how="left")
    realized = [
        (
            pl.col(flow).fill_null(0.0)
            + pl.col(positive).fill_null(0.0)
            - pl.col(negative).fill_null(0.0)
        ).alias(flow)
        for flow, positive, negative in _REALIZED_FLOWS
        if flow in frame.columns
    ]
    productivity = (
        pl.when(pl.col("turbined_m3s") > _MIN_TURBINED_M3S)
        .then(pl.col("generation_mw") / pl.col("turbined_m3s"))
        .alias("productivity_mw_per_m3s")
    )
    return frame.with_columns(*realized, productivity)


def _spillage_frame(output_dir: Path) -> pl.DataFrame:
    return (
        read_novomodelo_spillage_energy(output_dir)
        .rename(_SPILLAGE_COLUMNS)
        .with_columns(pl.lit(0, dtype=pl.Int64).alias("entity_id"))
    )


def read_operation_frames(
    side: SideInputs, variables: Mapping[str, Sequence[str]]
) -> dict[str, pl.DataFrame]:
    """One frame of *side* per entity type that is a key of *variables*.

    Each holds ``entity_id``, ``stage_id`` and a column per variable of its type,
    in the readers' native units; the hydro frame also holds the realized
    ``evaporation_m3s`` and ``productivity_mw_per_m3s``. No reader runs for an
    entity type outside *variables*, so a section that is unavailable never
    reaches its reader's warning; the bus frame reads ``system/lines.json`` only
    when ``net_import_mw`` is among its variables, and holds it null otherwise.
    Reader failures propagate unchanged.
    """
    output_dir = side.output_dir
    frames: dict[str, pl.DataFrame] = {}
    with side_tagged(side.side):
        if "hydro" in variables:
            frames["hydro"] = _hydro_frame(output_dir, variables["hydro"])
        if "thermal" in variables:
            frames["thermal"] = read_novomodelo_thermal_means(output_dir)
        if "bus" in variables:
            frames["bus"] = read_novomodelo_bus_means(
                output_dir, include_net_import="net_import_mw" in variables["bus"]
            )
        if "line" in variables:
            frames["line"] = read_novomodelo_line_means(output_dir)
        if "system_spillage" in variables:
            frames["system_spillage"] = _spillage_frame(output_dir)
    return frames


def _line_ends(side: SideInputs) -> dict[int, tuple[str | None, str | None]]:
    names = {bus["id"]: name_key(bus["name"]) for bus in side.buses}
    # Never None: lines are aligned only when both sides' lists are usable.
    lines = cast(list[dict], side.lines)
    return {
        line["id"]: (
            names.get(line.get("source_bus_id")),
            names.get(line.get("target_bus_id")),
        )
        for line in lines
    }


def _baseline_line_frame(
    baseline: SideInputs,
    candidate: SideInputs,
    frame: pl.DataFrame,
    pairs: Sequence[AlignedPair],
) -> pl.DataFrame:
    """*frame* with ``net_flow_mw`` negated on each line the candidate runs the
    other way round.

    ``net_flow_mw`` is positive from the source bus to the target bus. A pair
    whose ends, by bus name, are exchanged between the cases reports the same
    flow with the opposite sign, so the baseline follows the candidate's
    direction. Pairs whose ends are not both known on both sides, or are not
    each other's reverse, keep the flow as read.
    """
    baseline_ends = _line_ends(baseline)
    candidate_ends = _line_ends(candidate)
    reversed_ids = []
    for pair in pairs:
        source, target = baseline_ends[pair.baseline_id]
        if (
            source is not None
            and target is not None
            and source != target
            and candidate_ends[pair.candidate_id] == (target, source)
        ):
            reversed_ids.append(pair.baseline_id)
    return frame.with_columns(
        pl.when(pl.col("entity_id").is_in(reversed_ids))
        .then(-pl.col("net_flow_mw"))
        .otherwise(pl.col("net_flow_mw"))
    )


def operation_rows(
    baseline: SideInputs,
    candidate: SideInputs,
    baseline_frames: Mapping[str, pl.DataFrame],
    candidate_frames: Mapping[str, pl.DataFrame],
    alignments: Mapping[str, NameAlignment],
    variables: Mapping[str, Sequence[str]],
    stages: Collection[int],
) -> list[ResultComparison]:
    """The comparison rows of every entity type that *variables* and both frame
    mappings hold, in :data:`ROW_GROUPS` order and then the order of
    :func:`~novomodelo_bridge.comparators.version.rows.paired_stage_rows`.

    Hydro, thermal, bus and line rows pair the entities of ``alignments[<type>]``
    and build nothing for a type without an alignment; system spillage is the
    single pair ``SIN``, id 0 on both sides. The baseline flow of a line whose
    ends are exchanged between the cases is negated first. *stages* are the
    stages both sides hold.
    """
    rows: list[ResultComparison] = []
    for entity_type in ROW_GROUPS:
        if any(
            entity_type not in held
            for held in (variables, baseline_frames, candidate_frames)
        ):
            continue
        pairs: Sequence[AlignedPair]
        if entity_type == "system_spillage":
            pairs = (AlignedPair("SIN", 0, 0),)
        elif entity_type in alignments:
            pairs = alignments[entity_type].pairs
        else:
            continue
        baseline_frame = baseline_frames[entity_type]
        if entity_type == "line":
            baseline_frame = _baseline_line_frame(
                baseline, candidate, baseline_frame, pairs
            )
        rows += paired_stage_rows(
            entity_type,
            baseline_frame,
            candidate_frames[entity_type],
            pairs,
            variables[entity_type],
            stages,
        )
    return rows

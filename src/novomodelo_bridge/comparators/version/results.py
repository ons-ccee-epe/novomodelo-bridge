"""The dataset of a version comparison, assembled from every module of the package.

:func:`build_version_dataset` reads the two outputs, decides which sections both
support and builds the judged rows and the render inputs of the sections that
are available. The baseline fills the ``newave_*`` row fields and the ``nw_*``
render slots, whichever two runs are compared; the candidate fills every other
render input.

Failures propagate in a fixed order that is part of the exit-code contract: the
baseline directory, the candidate directory, then the read of the baseline and of
the candidate. No exception is caught here, and the Diagnostics a reader emitted
before a failure have already reached the sink, tagged with their side. The
findings are emitted here, once, after every builder has run.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import polars as pl

from novomodelo_bridge.comparators.analyze import build_results_dataset
from novomodelo_bridge.comparators.dataset import ComparisonDataset
from novomodelo_bridge.comparators.model import PercentileData
from novomodelo_bridge.comparators.operating_points import (
    OperatingPointComparison,
    operating_point_fields,
)
from novomodelo_bridge.comparators.version.adapter import (
    STAGE_COST_GATES,
    BaselineFrames,
    encode_baseline,
)
from novomodelo_bridge.comparators.version.alignment import (
    NameAlignment,
    StageAlignment,
    align_by_name,
    align_stages,
)
from novomodelo_bridge.comparators.version.availability import (
    SectionAvailability,
    Unavailable,
    unavailable_rows,
    with_entity_lists,
)
from novomodelo_bridge.comparators.version.constraints import (
    ConstraintComparison,
    compare_generic_constraints,
    constraint_list_usable,
)
from novomodelo_bridge.comparators.version.convergence import (
    BoundsComparison,
    compare_bounds,
    final_bound_rows,
    paired_convergence_rows,
)
from novomodelo_bridge.comparators.version.facts import METADATA_KEY, build_facts
from novomodelo_bridge.comparators.version.findings import (
    differing_constraints,
    differing_iteration_counts,
    differing_stage_sets,
    duplicated_names,
    emit_findings,
    one_sided_entities,
    same_output_compared,
    unavailable_sections,
    upper_bound_not_comparable,
)
from novomodelo_bridge.comparators.version.labels import SIDES, side_labels
from novomodelo_bridge.comparators.version.operating_points import (
    operating_point_comparison,
)
from novomodelo_bridge.comparators.version.operation import (
    operation_rows,
    operation_variables,
    read_operation_frames,
)
from novomodelo_bridge.comparators.version.production import fpha_surface_comparison
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
    PRODUCTION_FUNCTION,
    THERMAL,
    decide_sections,
)
from novomodelo_bridge.comparators.version.sides import (
    SideInputs,
    read_side,
    require_output_dir,
    side_tagged,
)
from novomodelo_bridge.comparators.version.ties import apply_tie_marks
from novomodelo_bridge.novomodelo.readers import (
    read_novomodelo_block_hours,
    read_novomodelo_bus_aggregates,
    read_novomodelo_bus_metadata,
    read_novomodelo_bus_percentiles,
    read_novomodelo_convergence,
    read_novomodelo_convergence_bounds,
    read_novomodelo_cost_breakdown,
    read_novomodelo_hydro_bus_labels,
    read_novomodelo_hydro_metadata,
    read_novomodelo_hydro_per_stage_bounds,
    read_novomodelo_hydro_percentiles,
    read_novomodelo_iteration_timing,
    read_novomodelo_line_percentiles,
    read_novomodelo_stage_costs,
    read_novomodelo_stage_line_bounds,
    read_novomodelo_thermal_percentiles,
    read_novomodelo_training_duration,
)

#: The dataset metadata key holding the convergence bounds of both sides, a frame
#: of :data:`_BOUNDS_SCHEMA`, persisted in ``metadata.json``: a one-way contract.
CONVERGENCE_BOUNDS_KEY: str = "convergence_bounds"

_BASELINE, _CANDIDATE = SIDES

_BOUNDS_SCHEMA = pl.Schema(
    {
        "side": pl.Utf8,
        "iteration": pl.Int64,
        "lower_bound": pl.Float64,
        "upper_bound": pl.Float64,
        "upper_bound_std": pl.Float64,
        "upper_bound_kind": pl.Utf8,
    }
)


def _stage_ids(side: SideInputs) -> list[int]:
    with side_tagged(side.side):
        block_hours = read_novomodelo_block_hours(side.output_dir)
    return block_hours["stage_id"].unique().to_list()


def _availability(side: SideInputs) -> dict[str, frozenset[str] | None]:
    usable = {
        **side.usable_entity_lists,
        "generic_constraint": constraint_list_usable(side),
    }
    return with_entity_lists(side.partitions, usable)


def _entity_alignments(
    baseline: SideInputs, candidate: SideInputs
) -> dict[str, NameAlignment]:
    """The entity types whose list both sides can use, hydro and bus always.

    A ``None`` list is never aligned and never replaced by ``[]``: it reads as
    an unavailable section, not as an entity type without entities.
    """
    lists = {
        "hydro": (baseline.hydros, candidate.hydros),
        "bus": (baseline.buses, candidate.buses),
        "thermal": (baseline.thermals, candidate.thermals),
        "line": (baseline.lines, candidate.lines),
    }
    return {
        entity_type: align_by_name(entity_type, baseline_list, candidate_list)
        for entity_type, (baseline_list, candidate_list) in lists.items()
        if baseline_list is not None and candidate_list is not None
    }


def _bounds(side: SideInputs) -> pl.DataFrame:
    with side_tagged(side.side):
        return read_novomodelo_convergence_bounds(side.output_dir)


def _bounds_frame(
    baseline_bounds: pl.DataFrame, candidate_bounds: pl.DataFrame
) -> pl.DataFrame:
    tagged = [
        bounds.with_columns(pl.lit(side).alias("side"))
        for side, bounds in zip(SIDES, (baseline_bounds, candidate_bounds), strict=True)
    ]
    return pl.concat(tagged).select(list(_BOUNDS_SCHEMA)).cast(_BOUNDS_SCHEMA)


def _side_frames(
    side: SideInputs,
    sections: Mapping[str, SectionAvailability],
    max_stage: int | None,
    operation_frames: Mapping[str, pl.DataFrame],
) -> BaselineFrames:
    """What the readers both sides share return, each read only when its section
    is available; the composed hydro frame is the one ``read_operation_frames`` built."""
    output_dir = side.output_dir
    with side_tagged(side.side):
        frames: dict[str, Any] = {
            "convergence": read_novomodelo_convergence(output_dir),
            "iteration_timing": read_novomodelo_iteration_timing(output_dir),
            "training_seconds": read_novomodelo_training_duration(output_dir),
        }
        if sections[COSTS].available:
            frames["cost_breakdown"] = read_novomodelo_cost_breakdown(
                output_dir, max_stage_id=max_stage
            )
            frames["stage_costs"] = read_novomodelo_stage_costs(output_dir)
        if sections[ENERGY_BALANCE].available:
            frames["bus_aggregates"] = read_novomodelo_bus_aggregates(output_dir)
    if sections[HYDRO].available:
        frames["hydro_means"] = operation_frames["hydro"]
    return BaselineFrames(**frames)


def _cleared_stage_costs(
    stage_costs: pl.DataFrame, cleared: Collection[str]
) -> pl.DataFrame:
    return stage_costs.with_columns(
        pl.lit(None, dtype=pl.Float64).alias(column)
        for column, variable in STAGE_COST_GATES.items()
        if variable not in cleared
    )


def _hydro_meta(
    candidate: SideInputs, sections: Mapping[str, SectionAvailability]
) -> dict[int, dict]:
    with side_tagged(candidate.side):
        meta = read_novomodelo_hydro_metadata(candidate.output_dir)
        if sections[ENERGY_BALANCE].available:
            labels = read_novomodelo_hydro_bus_labels(candidate.output_dir)
            for hydro_id, bus_ids in labels.items():
                if hydro_id in meta:
                    meta[hydro_id]["bus_ids"] = sorted(bus_ids)
    return meta


def _bus_meta(
    candidate: SideInputs, sections: Mapping[str, SectionAvailability]
) -> dict[int, dict]:
    if sections[BUS].available:
        with side_tagged(candidate.side):
            return read_novomodelo_bus_metadata(candidate.output_dir)
    # Not the reader: it raises on load statistics that lack the columns BUS requires.
    return {
        bus["id"]: {"name": bus["name"], "has_load": False} for bus in candidate.buses
    }


def _candidate_render(
    candidate: SideInputs,
    sections: Mapping[str, SectionAvailability],
    frames: BaselineFrames,
) -> dict[str, Any]:
    """The candidate's render inputs; a field whose section is unavailable keeps
    its default."""
    output_dir = candidate.output_dir
    # None only for an unusable list, which makes LINE and LINE_LIMITS unavailable.
    lines = cast(list[dict], candidate.lines)
    fields: dict[str, Any] = {
        "novomodelo_convergence": frames.convergence,
        "novomodelo_iteration_timing": frames.iteration_timing,
        "novomodelo_training_seconds": frames.training_seconds,
    }
    with side_tagged(candidate.side):
        if sections[COSTS].available:
            cleared = sections[COSTS].variables
            fields["novomodelo_costs"] = {
                column: value
                for column, value in frames.cost_breakdown.items()
                if column in cleared
            }
            fields["novomodelo_stage_costs"] = _cleared_stage_costs(
                frames.stage_costs, cleared
            )
        if sections[HYDRO].available:
            fields["novomodelo_hydro_means"] = frames.hydro_means
            fields["hydro"] = read_novomodelo_hydro_percentiles(output_dir)
        if sections[HYDRO_BOUNDS].available:
            fields["novomodelo_hydro_per_stage_bounds"] = (
                read_novomodelo_hydro_per_stage_bounds(output_dir)
            )
        if "generation_mw" in sections[THERMAL].variables:
            fields["thermal"] = read_novomodelo_thermal_percentiles(output_dir)
        if sections[BUS].variables:
            fields["bus"] = read_novomodelo_bus_percentiles(
                output_dir, include_net_import=sections[BUS_NET_IMPORT].available
            )
        if "net_flow_mw" in sections[LINE].variables:
            fields["line"] = read_novomodelo_line_percentiles(output_dir)
            fields["line_meta"] = lines
        if sections[LINE_LIMITS].available:
            fields["line_bounds"] = read_novomodelo_stage_line_bounds(output_dir, lines)
        if sections[ENERGY_BALANCE].available:
            fields["bus_aggregates"] = frames.bus_aggregates
    fields["novomodelo_hydro_meta"] = _hydro_meta(candidate, sections)
    fields["novomodelo_bus_meta"] = _bus_meta(candidate, sections)
    return fields


def _constraint_fields(constraints: ConstraintComparison | None) -> dict[str, Any]:
    if constraints is None:
        return {}
    return {
        "gc_constraints": constraints.gc_constraints,
        "gc_bounds": constraints.gc_bounds,
        "gc_lhs_newave": constraints.gc_lhs_newave,
        "gc_lhs_novomodelo": constraints.gc_lhs_novomodelo,
    }


def _production(
    baseline: SideInputs,
    candidate: SideInputs,
    sections: Mapping[str, SectionAvailability],
    alignment: NameAlignment,
    stages: StageAlignment,
) -> dict[str, Any]:
    """The fitted-surface and operating-point fields, whichever of the two
    sections is available."""
    surface = sections[PRODUCTION_FUNCTION].available
    points = sections[OPERATING_POINTS].available
    if not (surface or points):
        return {}
    comparison = OperatingPointComparison()
    if points:
        comparison = operating_point_comparison(
            baseline, candidate, alignment.pairs, stages.common
        )
    if surface:
        metrics, grid, spill = fpha_surface_comparison(
            baseline, candidate, alignment.pairs, stages.common
        )
        comparison = replace(comparison, metrics=metrics, surface=grid, spill=spill)
    if not sections[HYDRO].available:
        # masked_hydro refines the hydro rows, which are then absent.
        comparison = replace(
            comparison, masked_hydro=OperatingPointComparison().masked_hydro
        )
    return dict(operating_point_fields(comparison))


def _findings(
    baseline_output_dir: Path,
    candidate_output_dir: Path,
    alignments: Mapping[str, NameAlignment],
    stages: StageAlignment,
    unavailable: Sequence[Unavailable],
    constraints: ConstraintComparison | None,
    bounds: BoundsComparison,
) -> None:
    aligned = list(alignments.values())
    emit_findings(
        [
            same_output_compared(baseline_output_dir, candidate_output_dir),
            one_sided_entities(aligned),
            duplicated_names(aligned),
            differing_stage_sets(stages),
            unavailable_sections(unavailable),
            differing_constraints(constraints.differing if constraints else ()),
            upper_bound_not_comparable(bounds),
            differing_iteration_counts(bounds),
        ]
    )


def build_version_dataset(
    baseline_output_dir: Path,
    candidate_output_dir: Path,
    *,
    tolerance: float,
    labels: tuple[str | None, str | None] = (None, None),
) -> ComparisonDataset:
    """Compare two Novomodelo outputs, the directories being as the user typed
    them and possibly the same.

    Returns a validated dataset whose ``render.results`` holds the judged rows
    and whose metadata holds the facts record under
    :data:`~novomodelo_bridge.comparators.version.facts.METADATA_KEY` and the
    convergence bounds of both sides under :data:`CONVERGENCE_BOUNDS_KEY`. Runs
    inside the caller's ``diagnostics.collect()`` sink and emits every finding
    into it; the caller must not emit them again. *labels* are the labels the
    user chose for the baseline and the candidate, ``None`` for the default.

    Raises:
        SourceFileError: a directory is missing, or ``hydros.json`` or
            ``buses.json`` is absent.
        NovomodeloOutputError: a directory is not a Novomodelo output.
        FieldParseError: ``hydros.json`` or ``buses.json`` is malformed.
        NovomodeloReadError: a present file cannot be read.
    """
    require_output_dir(baseline_output_dir, _BASELINE)
    require_output_dir(candidate_output_dir, _CANDIDATE)
    baseline = read_side(baseline_output_dir, _BASELINE)
    candidate = read_side(candidate_output_dir, _CANDIDATE)

    stages = align_stages(_stage_ids(baseline), _stage_ids(candidate))
    sections = decide_sections(_availability(baseline), _availability(candidate))
    unavailable = unavailable_rows(sections.values())

    alignments = _entity_alignments(baseline, candidate)
    constraints = None
    if sections[GENERIC_CONSTRAINTS].available:
        constraints = compare_generic_constraints(
            baseline, candidate, alignments, stages.common
        )
        alignments = {**alignments, "generic_constraint": constraints.alignment}

    variables = operation_variables(sections)
    if sections[HYDRO].available:
        variables.setdefault("hydro", ())
    baseline_operation = read_operation_frames(baseline, variables)
    candidate_operation = read_operation_frames(candidate, variables)
    rows = apply_tie_marks(
        operation_rows(
            baseline,
            candidate,
            baseline_operation,
            candidate_operation,
            alignments,
            variables,
            stages.common,
        ),
        baseline,
        candidate,
    )
    baseline_bounds, candidate_bounds = _bounds(baseline), _bounds(candidate)
    bounds = compare_bounds(
        baseline_bounds,
        candidate_bounds,
        baseline.risk_aversion,
        candidate.risk_aversion,
    )
    rows += final_bound_rows(bounds)
    rows += paired_convergence_rows(baseline_bounds, candidate_bounds)

    max_stage = max(stages.common, default=None)
    baseline_frames = _side_frames(baseline, sections, max_stage, baseline_operation)
    candidate_frames = _side_frames(candidate, sections, max_stage, candidate_operation)
    slots = encode_baseline(
        baseline_frames,
        baseline,
        alignments["hydro"].pairs,
        cost_columns=sections[COSTS].variables,
        balance_variables=sections[ENERGY_BALANCE].variables,
    )
    percentiles = PercentileData(
        **slots,
        **_candidate_render(candidate, sections, candidate_frames),
        **_constraint_fields(constraints),
        **_production(baseline, candidate, sections, alignments["hydro"], stages),
        nw_offset=0,
        nw_max_stage=max_stage,
    )

    dataset = build_results_dataset(rows, percentiles, tolerance)
    dataset.metadata[METADATA_KEY] = build_facts(
        baseline,
        candidate,
        list(alignments.values()),
        unavailable,
        bounds,
        side_labels(
            (baseline.version, candidate.version),
            (baseline.output_dir, candidate.output_dir),
            labels,
        ),
    ).to_dict()
    dataset.metadata[CONVERGENCE_BOUNDS_KEY] = _bounds_frame(
        baseline_bounds, candidate_bounds
    )
    _findings(
        baseline_output_dir,
        candidate_output_dir,
        alignments,
        stages,
        unavailable,
        constraints,
        bounds,
    )
    return dataset

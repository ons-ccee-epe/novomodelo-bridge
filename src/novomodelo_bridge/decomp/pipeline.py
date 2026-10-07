"""Conversion orchestrator for DECOMP-like decks.

``convert_decomp_case(src, dst)`` discovers the deck (``caso.dat`` names
the revision extension; the ``rvN`` index file names the data files),
parses it once, and writes a Cobre case directory. A deck feature the
conversion leaves out (water travel time behind a tracked cobre gap, the
register term types their emitters name) is reported as a diagnostic by the
emitter that detects it, never dropped silently.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl
import pyarrow as pa
from idecomp.decomp import Dadger, Vazoes

from cobre_bridge.cobre import schemas as cobre_schemas
from cobre_bridge.cobre.case_writer import CaseWriter
from cobre_bridge.core import diagnostics as dx
from cobre_bridge.core import emission_checks
from cobre_bridge.core.bound_merge import merge_bound_tables
from cobre_bridge.core.conversion import (
    ClearedArtifacts,
    ConversionReport,
    clear_dst_contents,
)
from cobre_bridge.core.generic_constraint_builder import ConstraintIdAllocator
from cobre_bridge.decomp import (
    bounds_accumulator,
    constraint_registers,
)
from cobre_bridge.decomp import group_bounds as group_bounds_conv
from cobre_bridge.decomp import load as load_conv
from cobre_bridge.decomp import scenarios as scenarios_conv
from cobre_bridge.decomp import temporal as temporal_conv
from cobre_bridge.decomp.case import DecompCase
from cobre_bridge.decomp.converters import anticipated as anticipated_conv
from cobre_bridge.decomp.converters import bounds as bounds_conv
from cobre_bridge.decomp.converters import cadastro as cadastro_conv
from cobre_bridge.decomp.converters import config as config_conv
from cobre_bridge.decomp.converters import constraints as constraints_conv
from cobre_bridge.decomp.converters import contracts as contracts_conv
from cobre_bridge.decomp.converters import fpha as fpha_conv
from cobre_bridge.decomp.converters import hydro as hydro_conv
from cobre_bridge.decomp.converters import libs_electrical as libs_electrical_conv
from cobre_bridge.decomp.converters import (
    libs_electrical_emit,
    single_term_bounds,
)
from cobre_bridge.decomp.converters import ncs as ncs_conv
from cobre_bridge.decomp.converters import network as network_conv
from cobre_bridge.decomp.converters import thermal as thermal_conv
from cobre_bridge.decomp.converters import travel_time as travel_time_conv
from cobre_bridge.decomp.converters.scalar_parameters import (
    build_decomp_scalar_parameters,
    write_scalar_parameters,
)
from cobre_bridge.decomp.id_map import DecompIdMap

_LOG = logging.getLogger(__name__)

DECOMP_CLEARED_ARTIFACTS = ClearedArtifacts(
    subdirs=("system", "scenarios", "constraints", "boundary"),
    files=(
        "config.json",
        "stages.json",
        "penalties.json",
        "initial_conditions.json",
        "conversion_manifest.json",
        # DECOMP-only, conditional, root-level artifacts a re-run may not
        # reproduce; the NEWAVE set never lists these.
        "post_study_stages.json",
    ),
)

#: Coarse conversion phases reported to an optional ``on_phase`` callback (drives the
#: CLI progress bar). The order matches the boundaries in
#: :func:`_convert_decomp_case_impl`; the count is the progress-bar total.
DECOMP_CONVERSION_PHASE_LABELS: tuple[str, ...] = (
    "Discovering deck",
    "Converting entities",
    "Converting scenarios",
    "Resolving bounds",
    "Converting constraints",
    "Writing outputs",
)


@dataclass
class DecompCaseArtifacts:
    """In-memory artifacts threaded across the six conversion phases.

    ``_discover`` populates every field through ``tx``; ``_convert_core_entities``
    and ``_convert_scenarios`` populate the rest of the entity-level fields;
    ``_resolve_bounds`` and ``_convert_constraints`` populate the resolved
    bound/constraint fields that ``_emit_and_write`` consumes. Every phase
    reads its inputs back off the same bundle rather than its own locals, so
    no value crosses a phase boundary through anything but *artifacts*. A
    field not yet populated by the phase that owns it defaults to ``None``.
    """

    case: DecompCase
    id_map: DecompIdMap
    dadger: Dadger
    calendar: list[temporal_conv.OperativeStage]
    vazoes: Vazoes
    effective: cadastro_conv.EffectiveCadastro
    itaipu_operated: bool
    fan_probabilities: list[float]
    tx: float
    config: dict | None = None
    stages_dict: dict | None = None
    hydros_dict: dict | None = None
    thermals_dict: dict | None = None
    buses_doc: dict | None = None
    lines_doc: dict | None = None
    line_bounds: pa.Table | None = None
    initial_conditions: dict | None = None
    deficit_cost: float | None = None
    has_travel_time: bool | None = None
    fpha_codes: set[int] | None = None
    # Populated by `_resolve_bounds`.
    census: constraint_registers.ConstraintCensus | None = None
    pumping_ids: dict[int, int] | None = None
    block_counts: dict[int, int] | None = None
    hydro_bounds: pa.Table | None = None
    thermal_bounds_table: pa.Table | None = None
    pumping_bounds_table: pa.Table | None = None
    # Populated by `_convert_constraints`.
    availability_values: (
        dict[tuple[int, int, int], group_bounds_conv.GroupBoundEntry] | None
    ) = None
    group_bounds: pa.Table | None = None
    contract_bounds_table: pa.Table | None = None
    energy_contracts_doc: dict | None = None
    contracts: list[contracts_conv.Contract] | None = None


@dataclass
class FcfInputs:
    """Optional out-parameter carrying the pipeline's in-memory ``config``/
    ``initial_conditions`` dicts back to a caller that needs them post-return
    (the boundary-FCF importer's ``config``/``initial_conditions`` params —
    see ``fcf/importer.py::import_boundary_fcf``).

    Deliberately not :class:`DecompCaseArtifacts`: that bundle carries several
    mandatory fields (the parsed case, id map, calendar, and more) that are
    unavailable to a caller before ``convert_decomp_case`` runs, so it cannot
    be constructed ahead of the call the way an empty ``FcfInputs()`` can.
    ``convert_decomp_case`` assigns the same dict objects it wrote onto the
    passed instance, leaving both fields ``None`` when not requested (the
    default) or on a failed conversion.
    """

    config: dict | None = None
    initial_conditions: dict | None = None


#: Sentinel for a null ``block_id`` when joining two bound frames in polars:
#: `nulls_equal` defaults to False, so two null block ids never match each
#: other on their own; fill to this real value for the join, then restore
#: null on the result. See :func:`_fill_block_sentinel` /
#: :func:`_restore_block_null`.
_NULL_BLOCK_SENTINEL = -1

#: The ``(stage_id, block_id)`` tail every resolved bound table sorts by,
#: after its own leading entity-id column(s) -- shared by :func:`_resolve_bounds`
#: and :func:`_convert_constraints` so the two phases can't drift onto
#: different orderings for the same table shape.
_BOUND_SORT_KEYS = [("stage_id", "ascending"), ("block_id", "ascending")]


def _fill_block_sentinel(frame: pl.DataFrame) -> pl.DataFrame:
    """Map a null ``block_id`` to :data:`_NULL_BLOCK_SENTINEL` ahead of a join."""
    return frame.with_columns(pl.col("block_id").fill_null(_NULL_BLOCK_SENTINEL))


def _restore_block_null(frame: pl.DataFrame) -> pl.DataFrame:
    """Undo :func:`_fill_block_sentinel` on a merged/joined result."""
    return frame.with_columns(
        pl.when(pl.col("block_id") == _NULL_BLOCK_SENTINEL)
        .then(None)
        .otherwise(pl.col("block_id"))
        .alias("block_id")
    )


def _rejoin_thermal_cost(thermal_bounds: pa.Table, cost_table: pa.Table) -> pa.Table:
    """Fold *cost_table* (``thermal.py``'s ``cost_per_mwh`` side-table) onto
    *thermal_bounds* (the accumulator's resolved ``THERMAL_BOUNDS_SCHEMA``
    table), restoring the ``cost_per_mwh`` column ``convert_thermal_bounds``
    carries alongside rather than through its bound contributions.

    Merged via :func:`~cobre_bridge.core.bound_merge.merge_bound_tables`
    (``precedence="base"``, immaterial here since the two frames share no
    non-key column) on ``(thermal_id, stage_id, block_id)`` — not a
    left-join keyed off *thermal_bounds* — because a ``(thermal, stage)``
    whose generation bound resolved entirely to per-block contributions
    (every block non-uniform, so ``resolve()`` never materializes a
    ``block_id = None`` row for it, per the module's replace-vs-intersect
    discipline) still needs a row to carry that stage's cost: a strict left
    join would have nothing in *thermal_bounds* to attach it to and silently
    drop the value. The outer merge instead adds a ``block_id = None`` row
    with every bound column ``null`` for exactly that case. Per-block rows
    always get ``null`` cost, since the cost table never carries a
    non-``None`` ``block_id``.
    """
    key = ["thermal_id", "stage_id", "block_id"]
    bounds_df = thermal_bounds.to_pandas()
    cost_df = cost_table.to_pandas()
    for frame in (bounds_df, cost_df):
        frame["block_id"] = frame["block_id"].astype("Int64")

    bounds_pl = _fill_block_sentinel(pl.from_pandas(bounds_df))
    cost_pl = _fill_block_sentinel(pl.from_pandas(cost_df))
    merged_pl = _restore_block_null(
        merge_bound_tables(bounds_pl, cost_pl, on=key, precedence="base")
    )
    merged = merged_pl.to_pandas()

    schema = pa.schema(
        [
            *bounds_accumulator.THERMAL_BOUNDS_SCHEMA,
            pa.field("cost_per_mwh", pa.float64(), nullable=True),
        ]
    )
    return pa.table(
        {field.name: pa.array(merged[field.name], type=field.type) for field in schema},
        schema=schema,
    )


def _row_group_contributions(
    rows: Sequence[Mapping[str, object]],
    *,
    family: str,
    id_column: str,
    axes: Sequence[str],
    contributor: str,
) -> list[bounds_accumulator.BoundContribution]:
    """Recover the accumulator's contribution discipline from an already
    base-plus-sparse-per-block *rows* sequence (the ``convert_lines``/
    ``convert_contract_bounds``/``convert_hydro_unit_group_bounds``
    convention: one ``block_id = None`` base row per ``(id_column,
    stage_id)``, plus one row per block -- every block, not just the
    differing ones -- wherever that key is not block-uniform).

    Per *axis*, independently per side (lower/upper): a side no per-block row
    overrides contributes the base row's own value once, at ``block_id =
    None`` -- ``resolve()`` then folds it onto every materialized block (this
    is what lets ``hydro_unit_group``'s base-only ``max_generation_mw``
    ceiling survive alongside a per-block-varying ``min_generation_mw``,
    rather than being dropped with the rest of the base row). A side any
    per-block row overrides drops the base row's own value for that side
    instead: it is an hours-weighted average
    (``convert_hydro_unit_group_bounds``'s/``convert_contract_bounds``'s own
    fold), not a genuine additional bound, and intersecting it against the
    per-block values it was averaged from could raise a lower or lower an
    upper the source data never asked for.
    """
    groups: dict[tuple[object, object], list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((row[id_column], row["stage_id"]), []).append(row)

    contribs: list[bounds_accumulator.BoundContribution] = []
    for (entity_id, stage_id), group_rows in groups.items():
        base_row = next((r for r in group_rows if r["block_id"] is None), None)
        block_rows = [r for r in group_rows if r["block_id"] is not None]

        for axis in axes:
            spec = bounds_accumulator.axis_spec(family, axis)
            lower_overridden = spec.lower_column is not None and any(
                r[spec.lower_column] is not None for r in block_rows
            )
            upper_overridden = spec.upper_column is not None and any(
                r[spec.upper_column] is not None for r in block_rows
            )

            if base_row is not None:
                base_lower = (
                    base_row[spec.lower_column]
                    if spec.lower_column and not lower_overridden
                    else None
                )
                base_upper = (
                    base_row[spec.upper_column]
                    if spec.upper_column and not upper_overridden
                    else None
                )
                if base_lower is not None or base_upper is not None:
                    contribs.append(
                        bounds_accumulator.BoundContribution(
                            family=family,
                            entity_id=entity_id,
                            stage_id=stage_id,
                            block_id=None,
                            axis=axis,
                            lower=base_lower,
                            upper=base_upper,
                            contributor=contributor,
                        )
                    )

            for row in block_rows:
                contribs.append(
                    bounds_accumulator.BoundContribution(
                        family=family,
                        entity_id=entity_id,
                        stage_id=stage_id,
                        block_id=row["block_id"],
                        axis=axis,
                        lower=row[spec.lower_column] if spec.lower_column else None,
                        upper=row[spec.upper_column] if spec.upper_column else None,
                        contributor=contributor,
                    )
                )
    return contribs


def _fan_resolved_rows(
    entries: Sequence[tuple[tuple[object, ...], bounds_accumulator.ResolvedRow]],
    *,
    schema: pa.Schema,
    key_columns: Sequence[str],
) -> pa.Table:
    """Fan ``(key, ResolvedRow)`` pairs into *schema*, one row per distinct
    *key* and one column per axis side -- the per-cell fan-out
    ``bounds_accumulator.build_bound_tables`` performs for its own three
    built-in families (hydro/thermal/pumping), reimplemented here for a
    family outside that registry: line/hydro_unit_group/contract bounds
    resolve through the same ``resolve()`` primitive but fan into their own
    tables per that function's own docstring, since widening its registry
    is out of this change's scope.

    *key* names each output row's index-column values, in *key_columns*
    order -- the caller composes it, since a family whose entity id is not
    globally unique on its own (a hydro-unit-group row's key prepends the
    owning ``hydro_id``, which :class:`~bounds_accumulator.ResolvedRow`
    itself does not carry) needs a wider key than ``ResolvedRow`` supplies.
    """
    cells: dict[tuple[object, ...], dict[str, float]] = {}
    for key, row in entries:
        cell = cells.setdefault(key, {})
        spec = bounds_accumulator.axis_spec(row.family, row.axis)
        if spec.lower_column is not None and row.lower is not None:
            cell[spec.lower_column] = row.lower
        if spec.upper_column is not None and row.upper is not None:
            cell[spec.upper_column] = row.upper

    if not cells:
        return pa.table(
            {field.name: pa.array([], type=field.type) for field in schema},
            schema=schema,
        )

    value_columns = [f.name for f in schema if f.name not in key_columns]
    columns: dict[str, list[object]] = {f.name: [] for f in schema}
    for key, cell in cells.items():
        for name, value in zip(key_columns, key, strict=True):
            columns[name].append(value)
        for name in value_columns:
            columns[name].append(cell.get(name))
    return pa.table(
        {
            name: pa.array(values, type=schema.field(name).type)
            for name, values in columns.items()
        },
        schema=schema,
    )


_CONTRACT_POWER_SCHEMA = pa.schema(
    [
        pa.field("contract_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("min_mw", pa.float64(), nullable=True),
        pa.field("max_mw", pa.float64(), nullable=True),
    ]
)


def _rejoin_contract_price(
    contract_bounds: pa.Table, price_table: pa.Table
) -> pa.Table:
    """Fold *price_table*'s ``price_per_mwh`` (``contracts.py``'s own
    materialized table, read only for that column) onto *contract_bounds*
    (the accumulator's resolved, bound-only table), restoring the column
    ``convert_contract_bounds`` carries alongside its bound contributions
    rather than through one -- mirrors :func:`_rejoin_thermal_cost`.

    A plain left join, not :func:`_rejoin_thermal_cost`'s outer merge: every
    ``(contract, stage)`` key *contract_bounds* carries also has a matching
    row in *price_table* at the same ``block_id`` (the two are derived from
    the same base-vs-per-block decision in ``convert_contract_bounds``), so
    there is no gap to fill from the other side -- only *price_table*'s
    now-redundant base row (dropped by :func:`_row_group_contributions`
    wherever per-block rows exist) to leave out. The ``-1`` block sentinel
    mirrors :func:`_rejoin_thermal_cost`'s own (polars' join never matches
    null keys against each other).
    """
    key = ["contract_id", "stage_id", "block_id"]
    bounds_df = contract_bounds.to_pandas()
    price_df = price_table.select([*key, "price_per_mwh"]).to_pandas()
    for frame in (bounds_df, price_df):
        frame["block_id"] = frame["block_id"].astype("Int64")

    bounds_pl = _fill_block_sentinel(pl.from_pandas(bounds_df))
    price_pl = _fill_block_sentinel(pl.from_pandas(price_df))
    merged_pl = _restore_block_null(bounds_pl.join(price_pl, on=key, how="left"))
    merged = merged_pl.to_pandas()

    schema = contracts_conv._CONTRACT_BOUNDS_SCHEMA
    return pa.table(
        {field.name: pa.array(merged[field.name], type=field.type) for field in schema},
        schema=schema,
    )


def _topology_relink_diagnostic(
    effective: cadastro_conv.EffectiveCadastro,
    id_map: DecompIdMap,
) -> dx.Diagnostic | None:
    """The ``cadastro-topology-relinked`` INFO diagnostic,
    or ``None`` when no operated plant's effective downstream link/inflow
    gauge differs from the base registry (the byte-identical no-override
    case).

    Lists every operated plant whose stage-0 effective downstream link
    (``AC NUMJUS``) or inflow gauge (``AC NUMPOS``) differs from the base
    ``hidr`` value — the direct, human-readable relink; rv3's own 5
    ``AC NUMJUS`` rows all take this form (34/43 -> 45, 127/181 -> 129,
    117 -> 108).

    As a defensive backstop — a precise relink check — also walks the
    incremental-inflow cascade
    (:func:`~cobre_bridge.decomp.scenarios._incremental_context`) under
    both *effective* and a base-only cadastro (same registry, no override
    maps) and compares the resolved ``parents`` maps: an ``AC NUMJUS`` on a
    non-operated intermediate can change which operated plant receives an
    upstream plant's water without any operated plant's own link differing,
    so the direct per-code diff above would miss it. No rv3 row exercises
    that case, so it is not itemized by name here, but it is never silently
    dropped either — a tracked-gap warning fires instead of a
    diagnostic-less, byte-different conversion.
    """
    base = cadastro_conv.EffectiveCadastro(
        base=effective.base, n_stages=effective.n_stages, stage_varying={}
    )
    _, base_parents = scenarios_conv._incremental_context(base, id_map)
    _, effective_parents = scenarios_conv._incremental_context(effective, id_map)

    rows: list[list[object]] = []
    affected_codes: set[int] = set()
    for code in id_map.hydro_codes:
        name = str(effective.base.loc[code, "nome_usina"]).strip()
        base_downstream = int(effective.base.loc[code, "codigo_usina_jusante"])
        eff_downstream = effective.downstream_plant(code, 0)
        if eff_downstream != base_downstream:
            rows.append([name, code, "downstream", base_downstream, eff_downstream])
            affected_codes.add(code)
        base_gauge = int(effective.base.loc[code, "posto"])
        eff_gauge = effective.inflow_gauge(code, 0)
        if eff_gauge != base_gauge:
            rows.append([name, code, "gauge", base_gauge, eff_gauge])
            affected_codes.add(code)

    if not rows:
        if base_parents != effective_parents:
            _LOG.warning(
                "the effective and base incremental-inflow cascades differ, "
                "but no operated plant's own downstream link or inflow "
                "gauge changed directly; likely an AC NUMJUS relink on a "
                "non-operated intermediate -- not itemized by the "
                "cadastro-topology-relinked diagnostic"
            )
        return None

    return dx.Diagnostic(
        code="cadastro-topology-relinked",
        severity=dx.Severity.INFO,
        category="Cadastro overrides",
        title=f"Topology/gauge relinked for {len(affected_codes)} plant(s)",
        summary=(
            f"{len(affected_codes)} operated plant(s) carry an AC NUMJUS/"
            "NUMPOS override that changes their effective downstream link "
            "or inflow gauge from the base registry"
        ),
        table=dx.DiagnosticTable(
            columns=["Plant", "Code", "Kind", "Base", "Effective"],
            rows=rows,
            justify=["left", "right", "left", "right", "right"],
        ),
    )


#: The withdrawal axis's own side-table schema -- mirrors
#: :func:`~cobre_bridge.decomp.converters.bounds.convert_irrigation_withdrawal`'s shape
#: plus the ``block_id`` column :func:`_fan_resolved_rows` needs as a fan-out
#: key (always ``None``: the axis is registered ``block_eligible=False``).
_HYDRO_WITHDRAWAL_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("water_withdrawal_m3s", pa.float64(), nullable=True),
    ]
)


def _water_withdrawal_contributions(
    withdrawal: pa.Table,
) -> list[bounds_accumulator.BoundContribution]:
    """Base-only ``("hydro", "water_withdrawal")`` contributions from
    *withdrawal*
    (:func:`~cobre_bridge.decomp.converters.bounds.convert_irrigation_withdrawal`'s
    per-(hydro, stage) table) -- one ``block_id=None`` contribution per row,
    since irrigation withdrawal has no per-block dimension.
    """
    return [
        bounds_accumulator.BoundContribution(
            family="hydro",
            entity_id=row["hydro_id"],
            stage_id=row["stage_id"],
            block_id=None,
            axis="water_withdrawal",
            lower=row["water_withdrawal_m3s"],
            upper=None,
            contributor="convert_irrigation_withdrawal",
        )
        for row in withdrawal.to_pylist()
    ]


def _attach_water_withdrawal(
    hydro_bounds: pa.Table,
    withdrawal: pa.Table | None,
    block_counts: Mapping[int, int],
) -> pa.Table:
    """Fold the ``TI`` irrigation withdrawal into ``hydro_bounds``.

    Routes *withdrawal*
    (:func:`~cobre_bridge.decomp.converters.bounds.convert_irrigation_withdrawal`)
    through the same ``BoundContribution`` -> :func:`bounds_accumulator.resolve`
    primitive as every other hydro bound (so a withdrawal colliding with another
    contributor on the axis loud-fails instead of one silently overwriting the
    other), then attaches the resolved rows onto *hydro_bounds*.
    ``water_withdrawal_m3s`` is not a ``HYDRO_BOUNDS_SCHEMA`` column (its axis
    rides its own side-table, per that schema's own comment), so the resolved
    rows fan out into their own table (:func:`_fan_resolved_rows`) and attach via
    :func:`~cobre_bridge.core.bound_merge.merge_bound_tables` — mirroring
    :func:`_rejoin_thermal_cost`'s ``-1`` block-id sentinel dance (polars' join
    never matches null keys against each other), restored to null afterward.
    Returns ``hydro_bounds`` unchanged when the deck declares no irrigation.
    """
    if withdrawal is None or withdrawal.num_rows == 0:
        return hydro_bounds

    resolved = bounds_accumulator.resolve(
        _water_withdrawal_contributions(withdrawal), block_counts
    )
    withdrawal_bounds = _fan_resolved_rows(
        [((row.entity_id, row.stage_id, row.block_id), row) for row in resolved],
        schema=_HYDRO_WITHDRAWAL_SCHEMA,
        key_columns=("hydro_id", "stage_id", "block_id"),
    )

    key = ["hydro_id", "stage_id", "block_id"]
    hb = _fill_block_sentinel(pl.from_arrow(hydro_bounds))
    w = _fill_block_sentinel(pl.from_arrow(withdrawal_bounds))
    merged = _restore_block_null(
        merge_bound_tables(hb, w, on=key, precedence="base")
    ).sort(["hydro_id", "stage_id", "block_id"], nulls_last=False)
    return merged.to_arrow()


def _first_diversion_channel(
    effective: cadastro_conv.EffectiveCadastro, code: int
) -> cadastro_conv.DiversionChannel | None:
    """The first stage's non-``None`` diversion channel for *code*, or ``None``
    if it never has one -- shared by :func:`_diversion_channels` and
    :func:`_base_diversion_channels`.
    """
    for stage in range(effective.n_stages):
        channel = effective.diversion(code, stage)
        if channel is not None:
            return channel
    return None


def _diversion_channels(
    hydro_bounds: pa.Table,
    id_map: DecompIdMap,
    effective: cadastro_conv.EffectiveCadastro,
) -> tuple[dict[int, dict], list[int]]:
    """Cobre ``diversion`` channels for hydros carrying a positive diversion floor.

    A single-term QDES limit lowers to a ``min_diversion_m3s`` row; cobre couples
    a positive floor to a declared diversion channel (without one, diversion is
    pinned ``[0, 0]`` and any positive floor is infeasible). The source deck
    already declares the channel — read into ``effective.diversions`` — so this
    returns, keyed by cobre hydro id, the ``{downstream_id, max_flow_m3s}`` entry
    for every floored hydro whose channel resolves (a numeric limit and a
    downstream plant that survives into the case). Its second element lists the
    cobre ids of any floored hydro whose channel does NOT resolve — a genuine
    gap the caller surfaces rather than emitting an invalid case silently.

    Only floored hydros get a channel: an unconstrained diverter keeps
    ``diversion = null`` (its diversion stays pinned ``[0, 0]``, unchanged), so
    no plant's LP feasible region moves except the ones the floor already binds.
    """
    if "min_diversion_m3s" not in hydro_bounds.column_names:
        return {}, []
    columns = hydro_bounds.select(["hydro_id", "min_diversion_m3s"]).to_pydict()
    floored_ids = {
        int(hydro_id)
        for hydro_id, floor in zip(
            columns["hydro_id"], columns["min_diversion_m3s"], strict=True
        )
        if floor is not None and floor > 0.0
    }
    if not floored_ids:
        return {}, []
    code_by_id = {id_map.hydro_id(code): code for code in id_map.hydro_codes}
    operated_codes = set(id_map.hydro_codes)
    channels: dict[int, dict] = {}
    unresolved: list[int] = []
    for hydro_id in sorted(floored_ids):
        code = code_by_id.get(hydro_id)
        channel = (
            _first_diversion_channel(effective, code) if code is not None else None
        )
        if (
            channel is None
            or channel.limit is None
            or channel.downstream not in operated_codes
        ):
            unresolved.append(hydro_id)
            continue
        channels[hydro_id] = {
            "downstream_id": id_map.hydro_id(channel.downstream),
            "max_flow_m3s": float(channel.limit),
        }
    return channels, unresolved


def _base_diversion_channels(
    effective: cadastro_conv.EffectiveCadastro,
    id_map: DecompIdMap,
    hydro_capacities: Mapping[int, single_term_bounds.HydroCapacities],
    already_bounded_ids: set[int],
    stage_ids: Sequence[int],
) -> tuple[dict[int, dict], list[bounds_accumulator.BoundContribution]]:
    """Diversion channels + ``max_diversion`` bounds for BASE ``desvio`` diverters.

    A base ``desvio`` (the ``hidr`` field) with neither an ``AC DESVIO`` limit
    nor a QDES single-term bound leaves cobre's diversion column pinned
    ``[0, 0]`` (``max_diversion_m3s`` defaults to 0), stranding any downstream
    plant fed *solely* by that diversion -- e.g. MOXOTO's spill-side split to
    P.AFONSO 4, which otherwise receives no water and generates 0 MW. This
    declares the channel and opens the diversion column up to the receiving
    plant's turbine capacity (the flow it can actually pass to generation), so
    the diverted water reaches the downstream plant.

    Skips any diverter whose diversion axis is already bounded
    (``already_bounded_ids`` -- a QDES/``AC DESVIO`` floor already handled by
    :func:`_diversion_channels`), so this only *adds* the previously-unmodelled
    base channels and never overrides an explicit source limit. Returns the
    ``{downstream_id, max_flow_m3s}`` channel per source cobre id and the
    per-stage ``diversion`` bound contributions to fold into the accumulator.
    """
    operated = set(id_map.hydro_codes)
    channels: dict[int, dict] = {}
    contribs: list[bounds_accumulator.BoundContribution] = []
    for code in sorted(id_map.hydro_codes):
        if not effective.has_diversion(code):
            continue
        source_id = id_map.hydro_id(code)
        if source_id in already_bounded_ids:
            continue
        channel = _first_diversion_channel(effective, code)
        if channel is None or channel.downstream not in operated:
            continue
        downstream_id = id_map.hydro_id(channel.downstream)
        # A base `desvio` carries no explicit flow limit; cap the channel at the
        # receiving plant's turbine capacity (what it can turn into generation).
        cap = channel.limit
        if cap is None:
            cap = hydro_capacities[downstream_id].max_turbined_m3s
        if cap is None or cap <= 0.0:
            continue
        cap = float(cap)
        channels[source_id] = {"downstream_id": downstream_id, "max_flow_m3s": cap}
        contribs.extend(
            bounds_accumulator.BoundContribution(
                family="hydro",
                entity_id=source_id,
                stage_id=stage_id,
                block_id=None,
                axis="diversion",
                lower=0.0,
                upper=cap,
                contributor="base-desvio",
            )
            for stage_id in stage_ids
        )
    return channels, contribs


def _libs_electrical_census_diagnostic(
    result: libs_electrical_emit.LibsElectricalResult,
) -> dx.Diagnostic:
    """The ``decomp-libs-electrical-converted`` INFO diagnostic:
    the authoritative per-restriction census for the deck's LIBs-era
    long-form electrical file -- how many restrictions converted to a cobre
    generic constraint, and how many were dropped, broken down by
    :class:`~cobre_bridge.decomp.converters.libs_electrical_emit.LibsElectricalResult`'s
    four drop reasons (``inactive``/``unrecognized-token``/
    ``unresolved-bucket-bc``/``unresolved-bucket-a``, each already carrying
    its own per-restriction WARNING/INFO diagnostic).

    This supersedes the flat ``decomp-libs-electrical-present`` warning for
    the converted subset (:func:`_emit_and_write` suppresses that warning
    once a model converts) -- this is the one place a caller reads
    how much of the entry actually converted, so no "not converted" claim
    survives over it.
    """
    n_converted = len(result.converted_codes)
    n_dropped = sum(len(codes) for codes in result.deferred.values())
    rows: list[list[object]] = [["Converted", n_converted]]
    rows.extend(
        [reason, len(codes)] for reason, codes in sorted(result.deferred.items())
    )
    return dx.Diagnostic(
        code="decomp-libs-electrical-converted",
        severity=dx.Severity.INFO,
        category="Special constraints",
        title=(
            f"LIBs electrical constraints converted "
            f"({n_converted}/{n_converted + n_dropped})"
        ),
        summary=(
            f"{n_converted} of {n_converted + n_dropped} LIBs-era long-form "
            "electrical restriction(s) converted to a cobre generic "
            "constraint; the rest were dropped (see table for the "
            "per-reason breakdown)."
        ),
        table=dx.DiagnosticTable(
            columns=["Outcome", "Count"],
            rows=rows,
            justify=["left", "right"],
        ),
    )


def _discover(src: Path) -> DecompCaseArtifacts:
    """The ``step("Discovering deck")`` phase: parse *src* via one
    :class:`DecompCase`, resolve the cadastro override view (emitting its
    overrides-applied/topology-relinked diagnostics), and compute the scalars
    every later phase needs -- the Itaipu-operated flag, the terminal fan
    probabilities, and the annual discount rate.
    """
    case = DecompCase.from_directory(src)
    id_map = case.id_map
    dadger = case.dadger
    calendar = case.calendar
    _LOG.info("converting %s (revision %s)", src, case.files.revision)

    vazoes = Vazoes.read(str(case.files.vazoes))

    # Itaipu's split-plant topology needs a
    # synthesized SE<->IV line and, only when the deck carries the data,
    # the ANDE load on the IV bus -- both driven by this single
    # Itaipu(66)-operated check, computed once and reused by both wiring
    # sites below.
    itaipu_operated = hydro_conv._ITAIPU_CODE in id_map.hydro_codes
    # Cadastro overrides: the per-stage-effective view of
    # the registry, folding in any temporal AC VOLMIN/VOLMAX/VAZMIN override —
    # read once and threaded to every consumer below (initial storage, the
    # entity reservoir envelope, the per-stage storage and minimum-outflow
    # bounds) so they all agree on the same effective values.
    effective, cadastro_report = cadastro_conv.build_effective_cadastro(
        case.dadger, case.hidr, case.calendar
    )
    if cadastro_report.applied or cadastro_report.out_of_horizon:
        applied_desc = ", ".join(
            f"{count} {param}"
            for param, count in sorted(cadastro_report.applied.items())
        )
        summary = f"cadastro overrides applied: {applied_desc or 'none'}"
        if cadastro_report.out_of_horizon:
            summary += (
                f"; {len(cadastro_report.out_of_horizon)} override(s) fall "
                "outside the study horizon and were not applied"
            )
        dx.emit(
            dx.Diagnostic(
                code="cadastro-overrides-applied",
                severity=dx.Severity.INFO,
                category="Cadastro overrides",
                title="Cadastro overrides applied",
                summary=summary,
            ),
            logger=_LOG,
        )
    # The topology/gauge relink is its own diagnostic
    # (not folded into the summary above) so an automated consumer can key
    # on "cadastro-topology-relinked" specifically — the resolved
    # downstream/gauge links, not just an override count.
    relink_diagnostic = _topology_relink_diagnostic(effective, id_map)
    if relink_diagnostic is not None:
        dx.emit(relink_diagnostic, logger=_LOG)
    fan_probabilities = scenarios_conv.terminal_fan_probabilities(vazoes, calendar)

    tx = float(dadger.tx.taxa) / 100.0

    return DecompCaseArtifacts(
        case=case,
        id_map=id_map,
        dadger=dadger,
        calendar=calendar,
        vazoes=vazoes,
        effective=effective,
        itaipu_operated=itaipu_operated,
        fan_probabilities=fan_probabilities,
        tx=tx,
    )


def _convert_core_entities(artifacts: DecompCaseArtifacts, writer: CaseWriter) -> None:
    """The ``step("Converting entities")`` phase: CVaR, ``config``/``stages``,
    FPHA configs, productivity, penalties, the ``initial_conditions`` doc,
    buses, hydros (kept on *artifacts*, not written -- the bounds phase defers
    ``hydros.json`` for the diversion-channel coupling), lines (+ the Itaipu
    SE<->IV line), pumping, thermals + the GNL splice, production models,
    energy-productivity parquet, geometry, tailrace, and the
    non-controllable-source registry. Populates *artifacts*' entity fields.
    """
    case = artifacts.case
    id_map = artifacts.id_map
    dadger = artifacts.dadger
    effective = artifacts.effective
    itaipu_operated = artifacts.itaipu_operated
    tx = artifacts.tx
    fan_probabilities = artifacts.fan_probabilities

    # DECOMP risk aversion (CVaR): resolve the AR register / FCF-header CVaR
    # into a per-stage cobre risk measure, emitted uniformly across all stages
    # (temporal.stage_records) so cobre admits the gap stopping rule under CVaR
    # with enumerated forwards (it computes the exact risk-adjusted upper bound).
    cvar = temporal_conv.resolve_cvar(dadger, case.files.cortesh)
    if cvar is not None:
        dx.emit(
            dx.Diagnostic(
                code="decomp-cvar-risk-measure",
                severity=dx.Severity.INFO,
                category="Risk",
                title="CVaR risk measure converted",
                summary=(
                    f"The deck runs CVaR (alpha={cvar.alpha:.4g}, "
                    f"lambda={cvar.lambda_:.4g}); DECOMP starts it at stage "
                    f"{cvar.from_stage_index} but it is emitted uniformly on every "
                    "stage (it collapses to expectation on the deterministic trunk), "
                    "so cobre's gap stopping rule stays admissible under the "
                    "risk-adjusted enumerated upper bound."
                ),
            ),
            logger=_LOG,
        )
    artifacts.config = config_conv.convert_config(case)
    writer.write_json("config.json", artifacts.config)
    stages_dict = temporal_conv.convert_stages(
        case,
        annual_discount_rate=tx,
        fan_probabilities=fan_probabilities,
        cvar=cvar,
    )
    artifacts.stages_dict = stages_dict
    writer.write_json("stages.json", stages_dict)

    # FPHA-anchor fidelity: the source model fits each plant's hydro production
    # function around its initial reservoir volume. FPHA-eligible reservoirs get
    # cobre's computed-FPHA model (geometry + tailrace families, fit over a
    # ±window around the initial volume); the rest keep constant productivity,
    # whose ρ_eq is likewise anchored at the initial volume (not the full-range mean) —
    # see hydro._equivalent_productivity_mw_per_m3s and decomp/converters/fpha.py.
    initial_volumes = hydro_conv._operated_initial_volumes(case, effective=effective)
    fpha_codes = fpha_conv.fpha_eligible_codes(effective, id_map)
    artifacts.fpha_codes = fpha_codes
    fpha_configs: dict[int, dict] = {}
    reference_volumes: dict[int, dict] = {}
    for code in fpha_codes:
        vmin, vmax = cadastro_conv.effective_storage_range(effective, code, 0)
        v_init = initial_volumes.get(code, vmin)
        fpha_configs[code] = {
            "source": "computed",
            "fitting_window": fpha_conv.fitting_window(v_init, vmin, vmax),
        }
        reference_volumes[code] = {"volume_hm3": v_init}

    # penalties.json takes the full per-plant ρ_eq list (system ρ_avg/ρ_max);
    # the parquet omits FPHA plants (cobre computes their ρ_eq, so a parquet row
    # would double-supply it).
    productivity = hydro_conv.convert_energy_productivity(
        effective, id_map, initial_volumes
    )
    productivity_parquet = hydro_conv.convert_energy_productivity(
        effective, id_map, initial_volumes, exclude_codes=fpha_codes
    )
    deficit_costs = network_conv._bus_deficit_costs(dadger)
    deficit_cost = max(deficit_costs.values()) if deficit_costs else 0.0
    artifacts.deficit_cost = deficit_cost
    writer.write_json(
        "penalties.json",
        config_conv.convert_penalties(
            deficit_cost,
            productivity["equivalent_productivity_mw_per_m3s"].to_pylist(),
        ),
    )
    # TRACKED COBRE-GAP WORKAROUND (VI water travel time): the converter
    # (decomp/travel_time.py) is complete and `cobre validate` accepts the
    # emitted travel_time_hours + past_defluences, but cobre's transit-bucket
    # cut generation currently produces an invalid cut that blows the lower
    # bound past the upper bound (a large negative gap that cobre clamps to
    # zero and mistakes for convergence, stopping on a corrupted policy). So VI
    # is DEFERRED alongside GNL and the boundary FCF: detected for the deferral
    # warning below, but not emitted. Restore by wiring convert_travel_time back
    # into hydros.json + initial_conditions.past_defluences once the cobre bug
    # is fixed. Removal condition tracked in the cobre repository's
    # conversion-found-improvements registry (C-travel-time).
    has_travel_time = bool(travel_time_conv.read_travel_times(dadger))
    artifacts.has_travel_time = has_travel_time
    # Built here but written below (after GNL): an anticipated GNL fleet extends
    # it with the left/right temporal-boundary arrays, and those need the thermal
    # ids assigned when thermals.json is built.
    initial_conditions_doc: dict = {
        "$schema": cobre_schemas.schema_url_for("initial_conditions.json"),
        "storage": hydro_conv.convert_initial_storage(
            case, id_map, effective=effective
        ),
        "filling_storage": [],
    }
    artifacts.initial_conditions = initial_conditions_doc

    buses_doc = network_conv.convert_buses(case, id_map)
    artifacts.buses_doc = buses_doc
    writer.write_json("system/buses.json", buses_doc)
    hydros_dict = hydro_conv.convert_hydros(
        case, id_map, effective=effective, fpha_codes=fpha_codes
    )
    artifacts.hydros_dict = hydros_dict
    # hydros.json is written after the bound tables are resolved (below): a
    # plant carrying a positive QDES diversion floor (min_diversion_m3s > 0)
    # must also declare its diversion channel, or cobre rejects the floor as
    # infeasible against the channel-less [0, 0] pin. That coupling can only be
    # applied once the resolved hydro_bounds are in hand.
    lines_doc, line_bounds = network_conv.convert_lines(case, id_map)
    if itaipu_operated:
        # When Itaipu is operated (the split-plant topology relocates its 60 Hz
        # group to the IV bus), the IV bus needs a connection to SE. append_iv_se_line
        # synthesizes an SE<->IV line only when the deck's own IA register does
        # NOT already wire IV<->SE; when it does (as on most Itaipu decks), the
        # helper reuses that line -- its real operational capacity -- and adding
        # a second one would make cobre reject the duplicate.
        itaipu_submercado = int(case.hidr.loc[hydro_conv._ITAIPU_CODE, "submercado"])
        lines_doc, line_bounds = network_conv.append_iv_se_line(
            case,
            lines_doc=lines_doc,
            line_bounds=line_bounds,
            source_bus_id=id_map.transhipment_bus_id,
            target_bus_id=id_map.bus_id(itaipu_submercado),
            capacity_mw=network_conv._itaipu_60hz_capacity_mw(dadger),
        )
    artifacts.lines_doc = lines_doc
    artifacts.line_bounds = line_bounds
    writer.write_json("system/lines.json", lines_doc)
    writer.write_json(
        "system/pumping_stations.json",
        network_conv.convert_pumping_stations(case, id_map),
    )
    thermals_dict = thermal_conv.convert_thermals(case, id_map)
    # GNL (fuel-constrained) thermals are declared in dadgnl, absent from CT, so
    # the thermal converter never sees them. Read the commitment model and emit
    # the created thermals plus their anticipated ring: the mandatory left
    # boundary (past_anticipated_commitments, including any já-comandada
    # post-horizon run) and, when the deck declares a GS calendar, the
    # post-study thermal_bounds carrier (post_study_stages.json).
    gnl_model = (
        anticipated_conv.read_gnl_model(case.dadgnl)
        if case.dadgnl is not None
        else None
    )
    if gnl_model is not None:
        gnl = anticipated_conv.convert_gnl(
            gnl_model,
            first_thermal_id=max(t["id"] for t in thermals_dict["thermals"]) + 1,
            bus_id_of=id_map.bus_id,
            calendar=case.calendar,
        )
        thermals_dict["thermals"].extend(gnl.thermals)
        thermals_dict["thermals"].sort(key=lambda t: t["id"])
        if gnl.past_anticipated_commitments:
            initial_conditions_doc["past_anticipated_commitments"] = (
                gnl.past_anticipated_commitments
            )
        if gnl.post_study_stages is not None:
            writer.write_json("post_study_stages.json", gnl.post_study_stages)
        _LOG.info(
            "emitted %d GNL anticipated thermal(s) from dadgnl",
            len(gnl.thermals),
        )
    artifacts.thermals_dict = thermals_dict
    writer.write_json("system/thermals.json", thermals_dict)
    # The PAR inflow-lag seed (initial_conditions.recent_observations) is NOT
    # written here. The boundary cuts price the inflow-lag *deviation from the
    # seasonal mean* (MLT), not the absolute inflow, so a raw observation seed is
    # only correct once the cut RHS is folded by that same mean. Both the mean
    # fold and the raw seed are authored together by the boundary-FCF importer
    # (`fcf/importer.py::import_boundary_fcf`, which has the mlt.dat the fold
    # needs), patched onto this file after conversion -- so they can never ship
    # apart. A plain conversion (no boundary FCF) keeps the lag state at 0: its
    # inflow model is order 0 (no autoregression), so the lags are inert anyway.
    writer.write_json("initial_conditions.json", initial_conditions_doc)
    writer.write_json(
        "system/hydro_production_models.json",
        hydro_conv.convert_production_models(id_map, fpha_configs, reference_volumes),
    )
    writer.write_parquet(
        "system/hydro_energy_productivity.parquet", productivity_parquet
    )
    # FPHA geometry + tailrace inputs cobre fits the production function from.
    writer.write_parquet(
        "system/hydro_geometry.parquet",
        fpha_conv.convert_hydro_geometry(effective, id_map),
    )
    tailrace_table = fpha_conv.convert_tailrace_curves(case, id_map)
    if tailrace_table is not None:
        writer.write_parquet("system/tailrace_curves.parquet", tailrace_table)

    ncs_registry = ncs_conv.convert_non_controllable_sources(case, id_map)
    writer.write_json("system/non_controllable_sources.json", ncs_registry)


def _convert_scenarios(artifacts: DecompCaseArtifacts, writer: CaseWriter) -> None:
    """The ``step("Converting scenarios")`` phase: inflow stats, external
    inflows, the Itaipu ``carga_ande`` extra bus loads, load stats/factors,
    non-controllable-source stats/factors, and the deterministic external
    non-controllable-source/load scenarios.
    """
    case = artifacts.case
    id_map = artifacts.id_map
    dadger = artifacts.dadger
    calendar = artifacts.calendar
    vazoes = artifacts.vazoes
    effective = artifacts.effective
    itaipu_operated = artifacts.itaipu_operated
    fan_probabilities = artifacts.fan_probabilities

    writer.write_parquet(
        "scenarios/inflow_seasonal_stats.parquet",
        scenarios_conv.convert_inflow_stats_identity(case, id_map),
    )
    writer.write_parquet(
        "scenarios/external_inflow_scenarios.parquet",
        scenarios_conv.convert_external_inflows(
            case, id_map, vazoes=vazoes, effective=effective
        ),
    )
    # Itaipu's carga_ande load nets into its own SE bus (the HVDC Elo
    # delivers the 50 Hz group's output, and the ANDE demand, directly there
    # -- see hydro.py's convert_hydros docstring) and is gated on the dadger
    # RI register being present -- independent of (and additional to) the
    # unconditional Itaipu-operated line above. An Itaipu deck with no RI
    # register (read_carga_ande returns {}) converts gracefully: SE carries
    # only its own DP demand.
    extra_bus_loads: dict[tuple[int, int], list[float]] | None = None
    if itaipu_operated:
        carga_ande = libs_electrical_conv.read_carga_ande(dadger, calendar)
        if carga_ande:
            ande_bus = id_map.bus_id(
                int(case.hidr.loc[hydro_conv._ITAIPU_CODE, "submercado"])
            )
            extra_bus_loads = {
                (ande_bus, stage_index): values
                for stage_index, values in carga_ande.items()
            }
    load_stats = load_conv.convert_load_stats(
        case, id_map, extra_bus_loads=extra_bus_loads
    )
    writer.write_parquet("scenarios/load_seasonal_stats.parquet", load_stats)
    writer.write_json(
        "scenarios/load_factors.json",
        load_conv.convert_load_factors(case, id_map, extra_bus_loads=extra_bus_loads),
    )
    ncs_stats = ncs_conv.convert_ncs_stats(case, id_map)
    writer.write_parquet("scenarios/non_controllable_stats.parquet", ncs_stats)
    writer.write_json(
        "scenarios/non_controllable_factors.json",
        ncs_conv.convert_ncs_factors(case, id_map),
    )
    # Under the node-native explicit tree every stochastic class is sourced
    # externally: inflow (the tree), NCS (renewables), and load. A
    # deterministic (std = 0) external load class still occupies its noise slot
    # and standardizes to eta = 0 (value == mean) under cobre's scheme-aware
    # load membership (`std > 0 OR load_scheme == External`). Each deterministic
    # per-(entity, stage) mean fans out unchanged across the same per-stage
    # scenario columns as the inflow library (1 on the trunk, the terminal fan
    # width at the last stage) for joint scenario_id coherence.
    scenario_counts = [1] * (len(calendar) - 1) + [len(fan_probabilities)]
    writer.write_parquet(
        "scenarios/external_ncs_scenarios.parquet",
        scenarios_conv.deterministic_external_scenarios(
            ncs_stats,
            entity_column="ncs_id",
            value_in="mean",
            # cobre's clean break (0.14) removed the legacy ``value`` alias for
            # the NCS availability column; the sole accepted spelling is now
            # ``availability_factor`` (external inflow/load keep value_m3s/value_mw).
            value_out="availability_factor",
            scenario_counts=scenario_counts,
        ),
    )
    writer.write_parquet(
        "scenarios/external_load_scenarios.parquet",
        scenarios_conv.deterministic_external_scenarios(
            load_stats,
            entity_column="bus_id",
            value_in="mean_mw",
            value_out="value_mw",
            scenario_counts=scenario_counts,
        ),
    )


def _resolve_bounds(artifacts: DecompCaseArtifacts, writer: CaseWriter) -> None:
    """The ``step("Resolving bounds")`` phase: resolve every hydro/thermal/
    pumping/line bound, fold in the TI irrigation withdrawal and the
    diversion channels, and write the diversion-coupled ``system/hydros.json``
    (deferred from ``_convert_core_entities``'s system-file block for exactly
    that coupling). Stores the resolved bound tables and the
    ``census``/``pumping_ids``/``block_counts`` maps the later phases need back
    onto *artifacts*.
    """
    case = artifacts.case
    dadger = artifacts.dadger
    calendar = artifacts.calendar
    id_map = artifacts.id_map
    effective = artifacts.effective
    hydros_dict = artifacts.hydros_dict
    assert hydros_dict is not None
    line_bounds = artifacts.line_bounds
    assert line_bounds is not None

    census = constraint_registers.read_constraints(dadger)
    pumping_ids = network_conv.pumping_station_id_map(dadger)
    thermal_generation_contribs, thermal_cost_table = (
        thermal_conv.convert_thermal_bounds(case, id_map)
    )
    # The RE `FU` single-hydro-generation and the RHQ `QTUR`/turbined bound
    # producers each clamp their emitted upper to the plant's own declared
    # max_generation_mw/max_turbined_m3s, so a looser source-declared ceiling
    # on either axis (a real cross-source mismatch — see BELO MONTE's RE
    # ceiling on both real decks) never trips cobre rule 43
    # (emission_checks.check_hydro_bounds_no_raising, self-checked below).
    hydro_capacities: dict[int, single_term_bounds.HydroCapacities] = {
        hydro["id"]: single_term_bounds.HydroCapacities(
            max_generation_mw=hydro["generation"]["max_generation_mw"],
            max_turbined_m3s=hydro["generation"]["max_turbined_m3s"],
        )
        for hydro in hydros_dict["hydros"]
    }
    # Run-of-river plants from a cascade headwater down to (but excluding) the
    # first reservoir have their RQ/UH minimum-outflow default released, to
    # avoid infeasibility on a zero-inflow stage/scenario. Only the RQ/UH
    # default floor (convert_hydro_bounds) is released; an explicit RHQ QDEF
    # window (single_term_bound_contributions) is a user-declared constraint
    # and is kept.
    unregulated_codes = cadastro_conv.unregulated_runofriver_codes(
        effective, id_map.hydro_codes
    )
    for code in sorted(unregulated_codes):
        if effective.downstream_plant_varies(code):
            _LOG.warning(
                "plant %d's downstream link (AC NUMJUS) varies across stages; "
                "the run-of-river minimum-outflow relaxation uses the stage-0 "
                "effective link for the whole horizon",
                code,
            )
    if unregulated_codes:
        dx.emit(
            dx.Diagnostic(
                code="decomp-runofriver-outflow-relaxed",
                severity=dx.Severity.INFO,
                category="Minimum outflow",
                title="Run-of-river minimum-outflow restriction released",
                summary=(
                    f"{len(unregulated_codes)} run-of-river plant(s) "
                    f"({sorted(unregulated_codes)}) sit above the first cascade "
                    "reservoir; their RQ/UH minimum-defluence default is "
                    "released. An explicit RHQ QDEF window, if any, is kept."
                ),
            ),
            logger=_LOG,
        )
    # Every per-entity bound — the legacy RQ/UH minimum-outflow, the per-stage
    # storage envelope, the CT thermal generation bounds, and the RE/RHQ/RHV
    # single-term special-constraint bounds — is collected as
    # bounds_accumulator.BoundContribution objects and resolved through
    # exactly ONE resolve() + build_bound_tables() pass, so a new special
    # constraint colliding with a legacy bound on the same (entity, stage,
    # block) cell correctly intersects instead of producing the
    # two-rows-same-column parquet cobre rejects.
    contribs = [
        *bounds_conv.convert_hydro_bounds(
            case, id_map, effective=effective, unregulated_codes=unregulated_codes
        ),
        *bounds_conv.convert_storage_bounds(case, id_map, effective=effective),
        *bounds_conv.convert_volume_espera_bounds(case, id_map, effective=effective),
        *thermal_generation_contribs,
        *single_term_bounds.single_term_bound_contributions(
            case,
            id_map,
            census=census,
            pumping_station_ids=pumping_ids,
            effective=effective,
            hydro_capacities=hydro_capacities,
        ),
    ]
    # Base `desvio` diverters carry no QDES flow bound, so cobre pins their
    # diversion column to [0, 0] and strands any plant fed solely by the
    # diversion (e.g. MOXOTO -> P.AFONSO 4, which otherwise generates 0 MW).
    # Give the still-unbounded ones a max_diversion bound + a routed channel.
    _diversion_bounded_ids = {
        contrib.entity_id
        for contrib in contribs
        if contrib.family == "hydro" and contrib.axis == "diversion"
    }
    base_diversion_channels, base_diversion_contribs = _base_diversion_channels(
        effective,
        id_map,
        hydro_capacities,
        _diversion_bounded_ids,
        [stage.index for stage in calendar],
    )
    contribs.extend(base_diversion_contribs)
    block_counts = {stage.index: len(stage.block_hours) for stage in calendar}
    bound_tables = bounds_accumulator.build_bound_tables(
        bounds_accumulator.resolve(contribs, block_counts)
    )
    hydro_bounds = bound_tables.hydro.sort_by(
        [("hydro_id", "ascending"), *_BOUND_SORT_KEYS]
    )
    # Fold in the TI irrigation withdrawal (a consumptive water use cobre reads
    # from hydro_bounds.water_withdrawal_m3s): water lost to irrigation is
    # unavailable for generation, so omitting it lets cobre turbine the extra
    # flow and over-generate. Mirrors the source model's dsvagua path.
    hydro_bounds = _attach_water_withdrawal(
        hydro_bounds,
        bounds_conv.convert_irrigation_withdrawal(case, id_map),
        block_counts,
    )
    # Attach the diversion channel to every hydro that carries a positive
    # diversion floor, then write hydros.json (its write was deferred from
    # `_convert_core_entities`'s system-file block for exactly this). cobre
    # couples the two: a `min_diversion_m3s > 0` row is infeasible unless the
    # hydro also declares a `diversion` channel. The source deck already
    # carries the channel (read into `effective.diversions`); this is the
    # only place both the resolved floor and the channel are in hand.
    diversion_channels, diversion_floor_no_channel = _diversion_channels(
        hydro_bounds, id_map, effective
    )
    for hydro in hydros_dict["hydros"]:
        channel = diversion_channels.get(hydro["id"]) or base_diversion_channels.get(
            hydro["id"]
        )
        if channel is not None:
            hydro["diversion"] = channel
    if base_diversion_channels:
        dx.emit(
            dx.Diagnostic(
                code="decomp-base-diversion-modeled",
                severity=dx.Severity.INFO,
                category="Diversion",
                title="Base diversion channels modeled",
                summary=(
                    f"{len(base_diversion_channels)} hydro(s) "
                    f"({sorted(base_diversion_channels)}) carry a base diversion "
                    "channel with no explicit flow limit; each was given a "
                    "diversion channel capped at the receiving plant's turbine "
                    "capacity so the diverted water reaches the downstream plant "
                    "(otherwise that plant is stranded with no inflow)."
                ),
            ),
            logger=_LOG,
        )
    if diversion_floor_no_channel:
        dx.emit(
            dx.Diagnostic(
                code="decomp-diversion-floor-no-channel",
                severity=dx.Severity.WARNING,
                category="Diversion",
                title="Diversion floor without a resolvable channel",
                summary=(
                    f"{len(diversion_floor_no_channel)} hydro(s) "
                    f"({sorted(diversion_floor_no_channel)}) carry a positive "
                    "min_diversion_m3s floor but no resolvable diversion channel "
                    "(absent/limitless channel or an unmapped downstream plant); "
                    "cobre rejects the case until the channel is supplied."
                ),
            ),
            logger=_LOG,
        )
    writer.write_json("system/hydros.json", hydros_dict)
    thermal_bounds_table = _rejoin_thermal_cost(
        bound_tables.thermal, thermal_cost_table
    ).sort_by([("thermal_id", "ascending"), *_BOUND_SORT_KEYS])
    pumping_bounds_table = bound_tables.pumping.sort_by(
        [("pumping_station_id", "ascending"), *_BOUND_SORT_KEYS]
    )
    # Line bounds now route through the same BoundContribution -> resolve()
    # primitive as hydro/thermal/pumping above: a colliding pair of
    # contributions on the direct/reverse axis intersects (or loud-fails on a
    # genuine conflict) instead of one silently overwriting the other.
    # bounds_accumulator.build_bound_tables only fans into its own three
    # built-in tables, so line bounds fan out through _fan_resolved_rows
    # instead (see its docstring).
    line_contribs = _row_group_contributions(
        line_bounds.to_pylist(),
        family="line",
        id_column="line_id",
        axes=("direct", "reverse"),
        contributor="convert_lines",
    )
    line_bounds = _fan_resolved_rows(
        [
            ((row.entity_id, row.stage_id, row.block_id), row)
            for row in bounds_accumulator.resolve(line_contribs, block_counts)
        ],
        schema=network_conv._LINE_BOUNDS_SCHEMA,
        key_columns=("line_id", "stage_id", "block_id"),
    ).sort_by([("line_id", "ascending"), *_BOUND_SORT_KEYS])

    artifacts.census = census
    artifacts.pumping_ids = pumping_ids
    artifacts.block_counts = block_counts
    artifacts.hydro_bounds = hydro_bounds
    artifacts.thermal_bounds_table = thermal_bounds_table
    artifacts.pumping_bounds_table = pumping_bounds_table
    artifacts.line_bounds = line_bounds


def _convert_constraints(artifacts: DecompCaseArtifacts, writer: CaseWriter) -> None:
    """The ``step("Converting constraints")`` phase: build per-group
    per-stage availability (+ the Itaipu frequency-floor overlay), resolve
    the per-hydro ``group_bounds``, and build the ``contract_bounds_table``
    via :func:`_rejoin_contract_price`. Stores ``availability_values``,
    ``group_bounds``, ``contract_bounds_table``, ``energy_contracts_doc``,
    and ``contracts`` onto *artifacts* for :func:`_emit_and_write`.
    """
    case = artifacts.case
    dadger = artifacts.dadger
    calendar = artifacts.calendar
    id_map = artifacts.id_map
    effective = artifacts.effective
    itaipu_operated = artifacts.itaipu_operated
    block_counts = artifacts.block_counts
    assert block_counts is not None

    # Per-group per-stage availability (installed × MP ×
    # FD, capped by the ρ_eq·q_max hydraulic ceiling) for every plant,
    # single-group and Itaipu's own two per-frequency groups (code 66) alike.
    # Sparse by construction — a value only lands here where it lowers that
    # group's own declared envelope — so cobre rule 45 (a group-bound max_*
    # may not exceed that group's declared max) is now reachable; its mirror
    # is wired into the self-check block below.
    availability_values = hydro_conv.convert_hydro_group_availability(
        case, id_map, effective=effective
    )
    # Itaipu's RI per-frequency must-run floors
    # (geracao_minima_50/60_hz) overlay the same per-group table as the
    # availability max caps. The 50 Hz floor binds in DECOMP (its 50 Hz half
    # sits exactly at it), so without it the converted case under-runs Itaipu's
    # 50 Hz half and backfills the ANDE load through SE's own lines. Merged
    # into the availability entries (min on the same (hydro, group, stage) row
    # as the availability max) rather than emitted as a second table.
    if itaipu_operated:
        for key, min_generation in hydro_conv.convert_itaipu_frequency_min_generation(
            case, id_map
        ).items():
            existing = availability_values.get(key)
            availability_values[key] = (
                replace(existing, min_generation_mw=min_generation)
                if existing is not None
                else group_bounds_conv.GroupBoundEntry(min_generation_mw=min_generation)
            )
    group_bounds = group_bounds_conv.convert_hydro_unit_group_bounds(
        case, values=availability_values
    )
    # hydro_unit_group_id is scoped WITHIN its own hydro
    # (_build_split_unit_groups/build_mirror_unit_group number each plant's
    # own groups from 0), so one resolve() call spanning every hydro's rows
    # at once would wrongly intersect hydro A's group 0 against hydro B's
    # group 0; partition by hydro_id and resolve() each hydro's own groups
    # independently instead.
    group_rows_by_hydro: defaultdict[int, list[dict]] = defaultdict(list)
    for row in group_bounds.to_pylist():
        group_rows_by_hydro[row["hydro_id"]].append(row)
    resolved_group_entries: list[
        tuple[tuple[int, int, int, int | None], bounds_accumulator.ResolvedRow]
    ] = []
    for hydro_id, rows in group_rows_by_hydro.items():
        per_hydro_contribs = _row_group_contributions(
            rows,
            family="hydro_unit_group",
            id_column="hydro_unit_group_id",
            axes=("turbined", "generation"),
            contributor="group_bounds",
        )
        for resolved in bounds_accumulator.resolve(per_hydro_contribs, block_counts):
            resolved_group_entries.append(
                (
                    (
                        hydro_id,
                        resolved.entity_id,
                        resolved.stage_id,
                        resolved.block_id,
                    ),
                    resolved,
                )
            )
    group_bounds = _fan_resolved_rows(
        resolved_group_entries,
        schema=group_bounds_conv._HYDRO_UNIT_GROUP_BOUNDS_SCHEMA,
        key_columns=("hydro_id", "hydro_unit_group_id", "stage_id", "block_id"),
    ).sort_by(
        [
            ("hydro_id", "ascending"),
            ("hydro_unit_group_id", "ascending"),
            *_BOUND_SORT_KEYS,
        ]
    )

    # The CI/CE energy-contract model, read once and
    # shared by both emitters. Real decks carry no usable contract data
    # (rv0's single placeholder is skipped by D6, rv3 has none), so the
    # visible effect on a real conversion is: no contract files written, no
    # self-check finding, no regression.
    contracts = contracts_conv.read_contracts(dadger, calendar)
    contracts_conv.warn_nonnull_loss_factor(contracts)
    energy_contracts_doc = contracts_conv.convert_energy_contracts(
        case, id_map, contracts=contracts
    )
    contract_bounds_table = contracts_conv.convert_contract_bounds(
        case, contracts=contracts
    )
    # Contract power bounds now route through the same BoundContribution ->
    # resolve() primitive as the hydro/thermal/pumping/line bounds in
    # `_resolve_bounds`; price_per_mwh is not a bound axis (it rides its own
    # side-table, per bounds_accumulator's own AXES docstring), so it is
    # rejoined afterward.
    contract_power_contribs = _row_group_contributions(
        contract_bounds_table.to_pylist(),
        family="contract",
        id_column="contract_id",
        axes=("power",),
        contributor="convert_contract_bounds",
    )
    resolved_contract_power = _fan_resolved_rows(
        [
            ((row.entity_id, row.stage_id, row.block_id), row)
            for row in bounds_accumulator.resolve(contract_power_contribs, block_counts)
        ],
        schema=_CONTRACT_POWER_SCHEMA,
        key_columns=("contract_id", "stage_id", "block_id"),
    )
    contract_bounds_table = _rejoin_contract_price(
        resolved_contract_power, contract_bounds_table
    ).sort_by([("contract_id", "ascending"), *_BOUND_SORT_KEYS])

    artifacts.availability_values = availability_values
    artifacts.group_bounds = group_bounds
    artifacts.contract_bounds_table = contract_bounds_table
    artifacts.energy_contracts_doc = energy_contracts_doc
    artifacts.contracts = contracts


def _emit_and_write(
    artifacts: DecompCaseArtifacts, writer: CaseWriter
) -> ConversionReport:
    """The ``step("Writing outputs")`` phase: run the post-emission self-checks
    (``emission_checks.run_and_gate``) before any constraint parquet is
    written, write the bound parquets, emit the RE -> RHQ/RHV -> RHE ->
    LIBs-electrical generic constraints over one shared
    :class:`ConstraintIdAllocator`, write the generic-constraint artifacts and
    scalar parameters, emit the detection diagnostics and the travel-time
    deferral warning, and return the
    :class:`~cobre_bridge.core.conversion.ConversionReport` built from
    ``writer.would_write``.
    """
    case = artifacts.case
    dadger = artifacts.dadger
    calendar = artifacts.calendar
    id_map = artifacts.id_map
    effective = artifacts.effective
    fan_probabilities = artifacts.fan_probabilities
    hydros_dict = artifacts.hydros_dict
    assert hydros_dict is not None
    thermals_dict = artifacts.thermals_dict
    assert thermals_dict is not None
    buses_doc = artifacts.buses_doc
    assert buses_doc is not None
    lines_doc = artifacts.lines_doc
    assert lines_doc is not None
    stages_dict = artifacts.stages_dict
    assert stages_dict is not None
    deficit_cost = artifacts.deficit_cost
    assert deficit_cost is not None
    has_travel_time = artifacts.has_travel_time
    hydro_bounds = artifacts.hydro_bounds
    assert hydro_bounds is not None
    thermal_bounds_table = artifacts.thermal_bounds_table
    assert thermal_bounds_table is not None
    pumping_bounds_table = artifacts.pumping_bounds_table
    assert pumping_bounds_table is not None
    line_bounds = artifacts.line_bounds
    assert line_bounds is not None
    group_bounds = artifacts.group_bounds
    assert group_bounds is not None
    contract_bounds_table = artifacts.contract_bounds_table
    assert contract_bounds_table is not None
    census = artifacts.census
    assert census is not None
    pumping_ids = artifacts.pumping_ids
    assert pumping_ids is not None
    availability_values = artifacts.availability_values
    assert availability_values is not None
    contracts = artifacts.contracts
    assert contracts is not None
    energy_contracts_doc = artifacts.energy_contracts_doc
    assert energy_contracts_doc is not None

    # Post-emission self-checks: mirror cheap cobre load invariants (rules 43,
    # 41, 45, 38, 36, and the block_id-range rule) over the in-memory artifacts
    # before the constraint tables are written. rule 43 is reachable because
    # hydro_bounds carries max_turbined_m3s/max_generation_mw whenever a
    # single-term special constraint (e.g. an RE FU generation ceiling) lowers
    # to one; it raises when such a bound exceeds the entity's own declared
    # capacity. See cobre_bridge.core.emission_checks for the rule scope.
    bound_families = [
        emission_checks.BoundFamily("Hydro", "hydro_id", hydro_bounds),
        emission_checks.BoundFamily("Thermal", "thermal_id", thermal_bounds_table),
        emission_checks.BoundFamily("Line", "line_id", line_bounds),
        emission_checks.BoundFamily(
            "HydroUnitGroup",
            "hydro_id",
            group_bounds,
            group_column="hydro_unit_group_id",
        ),
        emission_checks.BoundFamily("Contract", "contract_id", contract_bounds_table),
        emission_checks.BoundFamily(
            "Pumping", "pumping_station_id", pumping_bounds_table
        ),
    ]

    def _run_emission_checks() -> None:
        emission_checks.check_hydro_bounds_no_raising(hydros_dict, hydro_bounds)
        emission_checks.check_unit_group_envelope(hydros_dict)
        emission_checks.check_group_bound_envelope(hydros_dict, group_bounds)
        emission_checks.check_bound_row_uniqueness(bound_families)
        emission_checks.check_bound_block_id_range(stages_dict, bound_families)
        emission_checks.check_block_id_not_on_anticipated_thermal(
            thermals_dict, thermal_bounds_table
        )

    emission_checks.run_and_gate(_run_emission_checks)

    writer.write_parquet("constraints/thermal_bounds.parquet", thermal_bounds_table)
    writer.write_parquet("constraints/line_bounds.parquet", line_bounds)
    if hydro_bounds.num_rows:
        writer.write_parquet("constraints/hydro_bounds.parquet", hydro_bounds)
    if pumping_bounds_table.num_rows:
        writer.write_parquet("constraints/pumping_bounds.parquet", pumping_bounds_table)
    if group_bounds.num_rows:
        writer.write_parquet(
            "constraints/hydro_unit_group_bounds.parquet", group_bounds
        )
    if contracts:
        writer.write_json("system/energy_contracts.json", energy_contracts_doc)
    if contract_bounds_table.num_rows:
        writer.write_parquet(
            "constraints/contract_bounds.parquet", contract_bounds_table
        )

    # Emit every RE/RHQ/RHV/RHE special constraint that did NOT lower to an
    # entity bound (`census.to_generic`, resolved once in `_resolve_bounds`)
    # as a Cobre generic constraint, over one shared 0-based constraint-id
    # allocator so ids never collide across emitters regardless of which
    # family produced them. The three emitters run in this fixed order (E7);
    # `big_m`, `line_map`, and `hydro_to_ree` are shared inputs every emitter
    # needs but none of them reads the deck to build for itself.
    big_m = constraints_conv.big_m_penalty(deficit_cost)
    line_map = constraints_conv.build_fi_line_map(lines_doc["lines"])
    uh = dadger.uh(df=True)
    operated_uh = uh[uh["volume_inicial"].notna()]
    hydro_to_ree = {
        int(row["codigo_usina"]): int(row["codigo_ree"])
        for _, row in operated_uh.iterrows()
    }

    # Read the deck's LIBs-era electrical-constraint file (the indices.csv
    # RESTRICAO-ELETRICA-ESPECIAL entry, resolved by discover_decomp_files)
    # once: both the narrowed detect_libs_electrical decision and the
    # generic-constraint emitter below thread this same `libs_electrical_model`,
    # so neither can disagree about whether the deck's long-form subset
    # converted. `read_libs_electrical` returns `None` for a short-form-only
    # (RE/RE-* or date-indexed) file -- that fallback stays on
    # detect_libs_electrical's flat warning (OQ-4).
    libs_electrical_model: libs_electrical_conv.LibsElectricalModel | None = None
    if case.libs_restricao_eletrica is not None:
        libs_electrical_model = libs_electrical_conv.read_libs_electrical(
            case.libs_restricao_eletrica, calendar
        )

    generic_constraints: list[dict] = []
    generic_bound_tables: list[pa.Table] = []
    allocator = ConstraintIdAllocator()

    re_generics = constraints_conv.emit_re_generics(
        case,
        id_map,
        census=census,
        line_map=line_map,
        big_m=big_m,
        allocator=allocator,
    )
    if re_generics is not None:
        generic_constraints.extend(re_generics.constraints)
        generic_bound_tables.append(re_generics.bounds)

    rhq_rhv_generics = constraints_conv.emit_rhq_rhv_generics(
        case,
        id_map,
        census=census,
        pumping_station_ids=pumping_ids,
        effective=effective,
        big_m=big_m,
        allocator=allocator,
    )
    if rhq_rhv_generics is not None:
        generic_constraints.extend(rhq_rhv_generics.constraints)
        generic_bound_tables.append(rhq_rhv_generics.bounds)

    rhe_generics = constraints_conv.emit_rhe_generics(
        case,
        id_map,
        census=census,
        effective=effective,
        hydro_to_ree=hydro_to_ree,
        allocator=allocator,
    )
    if rhe_generics.result is not None:
        generic_constraints.extend(rhe_generics.result.constraints)
        generic_bound_tables.append(rhe_generics.result.bounds)

    # The 4th generic-constraint emitter link -- every
    # LIBs-era long-form electrical restriction that resolved
    # (`libs_electrical_model`, read once above), spliced after RHE so the
    # unconditional write block below picks it up automatically, identical in
    # shape to the three links above. Built only when the deck actually
    # carries a long-form model, since none of these inputs are otherwise
    # needed.
    libs_electrical_result: libs_electrical_emit.LibsElectricalResult | None = None
    if libs_electrical_model is not None:
        context_factory = libs_electrical_conv.build_data_context(
            case, id_map, model=libs_electrical_model
        )
        a_h = libs_electrical_conv.build_available_power(
            case, id_map, overlay=availability_values, effective=effective
        )
        ncs_id_by_pee_code = ncs_conv.build_pee_ncs_id_map(case, id_map)
        conjh_bus_by_code_group = {
            (id_map.hydro_codes[entry["id"]], group_index): group["bus_id"]
            for entry in hydros_dict["hydros"]
            for group_index, group in enumerate(entry["unit_groups"], start=1)
        }
        libs_electrical_result = libs_electrical_emit.emit_libs_electrical_generics(
            libs_electrical_model,
            id_map,
            context_factory,
            a_h,
            calendar,
            ncs_id_by_pee_code,
            conjh_bus_by_code_group,
            line_map,
            big_m,
            allocator,
        )
        if libs_electrical_result.generic is not None:
            generic_constraints.extend(libs_electrical_result.generic.constraints)
            generic_bound_tables.append(libs_electrical_result.generic.bounds)

    if generic_constraints:
        writer.write_json(
            "constraints/generic_constraints.json",
            {
                "$schema": cobre_schemas.schema_url_for(
                    "constraints/generic_constraints.json"
                ),
                "constraints": generic_constraints,
            },
        )
        writer.write_parquet(
            "constraints/generic_constraint_bounds.parquet",
            pa.concat_tables(generic_bound_tables),
        )

    # Every `@rho_acum_h{id}` sigil a surviving RHE
    # expression references must resolve, or cobre fails to load the case --
    # write generic_parameters.json whenever any RHE constraint survives,
    # never only when another generic family also happens to be present.
    if rhe_generics.rho_acum_overrides:
        write_scalar_parameters(
            writer,
            build_decomp_scalar_parameters(
                [id_map.hydro_id(code) for code in id_map.hydro_codes],
                rhe_generics.rho_acum_overrides,
            ),
        )

    # E1: the FE/RHA/LIBs-electrical detection diagnostics each surface
    # once, through the structured sink, alongside the flat deferral warning
    # below (the matching clause is deduped out of that warning).
    for detection in constraint_registers.detect_unreadable_electrical(
        case.files.dadger
    ):
        dx.emit(detection, logger=_LOG)
    # The flat presence warning is narrowed to the subset
    # `libs_electrical_model` (read once above) did NOT convert -- a
    # short-form-only (RE/RE-*, date-indexed) file, or no long-form card at
    # all despite the indices.csv entry (OQ-4). Once the long-form subset
    # converts, the census INFO below is authoritative and this flat warning
    # would otherwise be a stale "not converted" claim over that subset.
    # `discover_decomp_files` always resolves every deck file (including
    # `dadger`) as `src / name`, so `case.files.dadger.parent` recovers the
    # deck root without a dedicated `src` field on the bundle.
    libs_detection = constraint_registers.detect_libs_electrical(
        case.files.dadger.parent
    )
    if libs_detection is not None and libs_electrical_model is None:
        dx.emit(libs_detection, logger=_LOG)
    if libs_electrical_result is not None:
        dx.emit(_libs_electrical_census_diagnostic(libs_electrical_result), logger=_LOG)

    # Milestone-deferral warning. Boundary FCF is imported by default now (the
    # CLI runs the importer after this converter returns), reservoir evaporation
    # is converted (cobre >= 0.14's C11 fix), and windowed inflow inputs are a
    # source-model concept that does not apply to the external explicit tree the
    # DECOMP path emits — so none of those are deferred. Only VI water travel
    # time remains deferred (a tracked cobre-gap; see the note in
    # `_convert_core_entities` where it is detected), and only when the deck
    # actually carries it.
    if has_travel_time:
        _LOG.warning("deferred at this milestone: water travel time (VI present)")
    _LOG.info(
        "converted %d buses, %d hydros, %d thermals, %d stages, terminal fan %d",
        id_map.n_buses,
        len(id_map.hydro_codes),
        len(id_map.thermal_codes),
        len(calendar),
        len(fan_probabilities),
    )

    return ConversionReport(
        hydro_count=len(hydros_dict["hydros"]),
        thermal_count=len(thermals_dict["thermals"]),
        bus_count=len(buses_doc["buses"]),
        line_count=len(lines_doc["lines"]),
        stage_count=len(calendar),
        would_write_paths=[str(p) for p in writer.would_write],
    )


def convert_decomp_case(
    src: Path,
    dst: Path,
    *,
    force: bool = False,
    on_phase: Callable[[str], None] | None = None,
    dry_run: bool = False,
    fcf_inputs_out: FcfInputs | None = None,
) -> ConversionReport:
    """Convert one deck revision into a Cobre case directory.

    Mirrors the NEWAVE twin ``convert_newave_case``'s return contract: wraps
    the conversion in a top-level :func:`cobre_bridge.core.diagnostics.collect`
    sink and a package-logger ``dx.WarningCollector`` so every structured
    ``dx.emit`` finding *and* every residual ``logger.warning`` string is
    captured, then returns them as one de-duplicated
    :class:`~cobre_bridge.core.conversion.ConversionReport`.

    Parameters
    ----------
    dry_run:
        When ``True``, run the full in-memory conversion but write nothing to
        *dst* (no files, no subdirectories). The would-write paths are still
        recorded in
        :attr:`~cobre_bridge.core.conversion.ConversionReport.would_write_paths`.
    fcf_inputs_out:
        Optional :class:`FcfInputs` out-parameter. When given, its
        ``config``/``initial_conditions`` fields are assigned the same
        in-memory dict objects this run wrote to ``config.json``/
        ``initial_conditions.json``, so a caller (the CLI's boundary-FCF
        step) can thread them into ``import_boundary_fcf`` without
        re-reading either file. ``None`` (the default) leaves it unused.

    Returns
    -------
    ConversionReport
        Summary of what was converted: the five entity/stage counts, a
        ``diagnostics`` list of every structured finding, and a ``warnings``
        list of the WARNING-severity summaries.

    Raises
    ------
    ValueError
        If the post-emission self-checks (cobre 0.13 rules 43/41/45/38/36 and
        the ``block_id``-range rule; see :mod:`cobre_bridge.core.emission_checks`)
        find an ``ERROR``-severity violation in the converted artifacts. An
        ``INFO`` finding (e.g. rule 43's "not applicable" report, emitted when
        no hydro-bounds capacity column is populated) never raises. No report
        is returned on this path.
    FileExistsError
        If *dst* already exists, is non-empty, and ``force`` is not set.
    """
    # This refusal must run before the clearing try/except below: it is a
    # "do not touch dst" guard, not a mid-write failure, so it must never
    # reach ``clear_dst_contents`` and delete a pre-existing, unrelated case
    # (mirrors the source model's own NEWAVE-twin ordering, which performs
    # this check ahead of its own write/clear try block).
    if dst.exists() and any(dst.iterdir()):
        if not force:
            raise FileExistsError(
                f"{dst} already contains files; pass force to overwrite"
            )
        # --force over a populated dst: pre-clear the previous case's full
        # artifact set before rewriting, so a conditional artifact this run
        # does not reproduce (post_study_stages.json, boundary/) cannot
        # survive. Skipped under a dry run, which never mutates dst.
        if not dry_run:
            clear_dst_contents(dst, DECOMP_CLEARED_ARTIFACTS)

    collector = dx.WarningCollector()
    pkg_logger = logging.getLogger("cobre_bridge")
    pkg_logger.addHandler(collector)
    try:
        # Structured diagnostics emitted by converters (and the inner
        # emission-self-check re-emit loop below) land in ``collected``; any
        # remaining ``logger.warning`` strings (the deferral warning, the
        # topology non-itemized warning) are picked up by ``collector`` and
        # bridged below, so every warning still surfaces.
        with dx.collect() as collected:
            report = _convert_decomp_case_impl(
                src,
                dst,
                force=force,
                on_phase=on_phase,
                dry_run=dry_run,
                fcf_inputs_out=fcf_inputs_out,
            )
    except BaseException:
        # The write phase is a sequence of independent file writes with no
        # rollback, so a failure partway through can leave a subset of the
        # output files behind. Remove the known pipeline outputs so a
        # half-written case is never mistaken for a complete one and a plain
        # (no --force) re-run is not refused as "destination not empty".
        # Skipped under a dry run: nothing was written, and dst may be a
        # pre-existing populated directory the user never asked to clear.
        if not dry_run:
            clear_dst_contents(dst, DECOMP_CLEARED_ARTIFACTS)
        raise
    finally:
        pkg_logger.removeHandler(collector)
    report.diagnostics = dx.finalize_diagnostics(collected, collector.messages)
    # Backward-compatible flat WARNING strings derived from the diagnostics.
    report.warnings = [
        d.summary for d in report.diagnostics if d.severity is dx.Severity.WARNING
    ]
    return report


def _convert_decomp_case_impl(
    src: Path,
    dst: Path,
    *,
    force: bool = False,
    on_phase: Callable[[str], None] | None = None,
    dry_run: bool = False,
    fcf_inputs_out: FcfInputs | None = None,
) -> ConversionReport:
    """Run the DECOMP conversion pipeline (warning capture handled by the wrapper).

    ``on_phase`` (when given) is called once at each boundary in
    :data:`DECOMP_CONVERSION_PHASE_LABELS`, so the CLI can advance a progress bar
    without the pipeline knowing anything about rendering.

    When ``dry_run`` is ``True``, the build phases run exactly as for a real run,
    but the write phase creates no directories and writes no files; the paths that
    would have been written are still accumulated and recorded on the report.

    ``fcf_inputs_out`` (when given) receives the run's in-memory ``config``/
    ``initial_conditions`` dicts — see :func:`convert_decomp_case`.
    """
    step = on_phase if on_phase is not None else (lambda _label: None)
    step("Discovering deck")
    artifacts = _discover(src)

    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    # Every output path is routed through one ``CaseWriter`` so the
    # would-write listing, the byte format, and the dry-run gate live in
    # exactly one place, rather than guarding ~24 individual write sites.
    writer = CaseWriter(dst, dry_run=dry_run)

    step("Converting entities")
    _convert_core_entities(artifacts, writer)

    step("Converting scenarios")
    _convert_scenarios(artifacts, writer)

    step("Resolving bounds")
    _resolve_bounds(artifacts, writer)

    step("Converting constraints")
    _convert_constraints(artifacts, writer)

    step("Writing outputs")
    report = _emit_and_write(artifacts, writer)
    if fcf_inputs_out is not None:
        fcf_inputs_out.config = artifacts.config
        fcf_inputs_out.initial_conditions = artifacts.initial_conditions
    return report

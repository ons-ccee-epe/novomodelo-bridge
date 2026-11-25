"""Conversion orchestrator for DECOMP-like decks.

``convert_decomp_case(src, dst)`` discovers the deck (``caso.dat`` names
the revision extension; the ``rvN`` index file names the data files),
parses it once, and writes a Novomodelo case directory. A deck feature the
conversion leaves out (an unconverted special-constraint family, the
register term types their emitters name) is reported as a diagnostic by the
emitter that detects it, never dropped silently.
"""

from __future__ import annotations

import logging
from calendar import monthrange
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import NamedTuple

import polars as pl
import pyarrow as pa
from idecomp.decomp import Dadger, Vazoes
from inewave.newave import Cortesh

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core import emission_checks
from novomodelo_bridge.core.bound_merge import merge_bound_tables
from novomodelo_bridge.core.conversion import (
    ClearedArtifacts,
    ConversionReport,
    run_conversion,
)
from novomodelo_bridge.core.diversion import (
    channel_flow_limit,
    is_unlimited_flow,
    unlimited_channels_diagnostic,
)
from novomodelo_bridge.core.errors import FieldParseError, UnsupportedSourceError
from novomodelo_bridge.core.generic_constraint_builder import (
    ConstraintIdAllocator,
    GenericConstraintResult,
    is_bounded,
)
from novomodelo_bridge.core.libs_index import unread_lines_diagnostic
from novomodelo_bridge.core.pandas_utils import as_float, as_frame, as_int
from novomodelo_bridge.core.source_text import tab_character_diagnostic
from novomodelo_bridge.core.tolerances import beyond_written_precision
from novomodelo_bridge.decomp import (
    bounds_accumulator,
    constraint_registers,
    plants,
)
from novomodelo_bridge.decomp import group_bounds as group_bounds_conv
from novomodelo_bridge.decomp import load as load_conv
from novomodelo_bridge.decomp import scenarios as scenarios_conv
from novomodelo_bridge.decomp import temporal as temporal_conv
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.case_identity import require_registered_hydros
from novomodelo_bridge.decomp.converters import anticipated as anticipated_conv
from novomodelo_bridge.decomp.converters import bounds as bounds_conv
from novomodelo_bridge.decomp.converters import cadastro as cadastro_conv
from novomodelo_bridge.decomp.converters import config as config_conv
from novomodelo_bridge.decomp.converters import constraints as constraints_conv
from novomodelo_bridge.decomp.converters import contracts as contracts_conv
from novomodelo_bridge.decomp.converters import fpha as fpha_conv
from novomodelo_bridge.decomp.converters import hydro as hydro_conv
from novomodelo_bridge.decomp.converters import libs_electrical as libs_electrical_conv
from novomodelo_bridge.decomp.converters import (
    libs_electrical_emit,
    single_term_bounds,
)
from novomodelo_bridge.decomp.converters import ncs as ncs_conv
from novomodelo_bridge.decomp.converters import network as network_conv
from novomodelo_bridge.decomp.converters import penalties as penalties_conv
from novomodelo_bridge.decomp.converters import rhe_shares as rhe_shares_conv
from novomodelo_bridge.decomp.converters import tailrace_flow as tailrace_flow_conv
from novomodelo_bridge.decomp.converters import thermal as thermal_conv
from novomodelo_bridge.decomp.converters import travel_time as travel_time_conv
from novomodelo_bridge.decomp.converters.scalar_parameters import (
    build_decomp_scalar_parameters,
    write_scalar_parameters,
)
from novomodelo_bridge.decomp.fcf import auxiliary as auxiliary_conv
from novomodelo_bridge.decomp.fcf.cortes import read_cortesh, read_patamar_fractions
from novomodelo_bridge.decomp.fcf.coupling import (
    CouplingAuxiliary,
    complexo_components,
    plan_coupling_auxiliaries,
)
from novomodelo_bridge.decomp.files import (
    REGISTER_COMMENT_PREFIX,
    unindexed_libs_diagnostic,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas
from novomodelo_bridge.novomodelo.case_writer import DEFAULT_FLOAT_DIGITS, CaseWriter

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
    coupling_auxiliaries: tuple[CouplingAuxiliary, ...] = ()
    config: dict | None = None
    stages_dict: dict | None = None
    hydros_dict: dict | None = None
    thermals_dict: dict | None = None
    gnl_model: anticipated_conv.GnlCommitmentModel | None = None
    buses_doc: dict | None = None
    lines_doc: dict | None = None
    line_bounds: pa.Table | None = None
    initial_conditions: dict | None = None
    deficit_cost: float | None = None
    fpha_codes: set[int] | None = None
    turbine_limits: hydro_conv.TurbineLimits | None = None
    #: `(bus_id, {stage_index: [MW per block]})` of the `RI` ANDE load.
    ande_bus_load: tuple[int, dict[int, list[float]]] | None = None
    # Populated by `_resolve_bounds`.
    census: constraint_registers.ConstraintCensus | None = None
    pumping_ids: dict[int, int] | None = None
    block_counts: dict[int, int] | None = None
    availability_values: (
        dict[tuple[int, int, int], group_bounds_conv.GroupBoundEntry] | None
    ) = None
    hydro_bounds: pa.Table | None = None
    thermal_bounds_table: pa.Table | None = None
    pumping_bounds_table: pa.Table | None = None
    kept_bound_conflicts: list[_KeptConflict] | None = None
    # Populated by `_convert_constraints`.
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
    coupling_auxiliaries: tuple[CouplingAuxiliary, ...] = ()


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

    Merged via :func:`~novomodelo_bridge.core.bound_merge.merge_bound_tables`
    (``precedence="base"``, immaterial here since the two frames share no
    non-key column) on ``(thermal_id, stage_id, block_id)`` — not a
    left-join keyed off *thermal_bounds* — because a ``(thermal, stage)``
    whose generation bound resolved entirely to per-block contributions
    (every block non-uniform, so ``resolve()`` never materializes a
    ``block_id = None`` row for it, per the module's replace-vs-intersect
    discipline) still needs a row to carry that stage's cost: a strict left
    join would have nothing in *thermal_bounds* to attach it to and silently
    drop the value. The outer merge instead adds a ``block_id = None`` row
    with every bound column ``null`` for exactly that case. A per-block cost
    row lands on the block's existing bound row (non-uniform ``CT``, or a
    per-block single-term ``RE``) or else becomes a cost-only block row; the
    stage row likewise. The column-by-column contract is in
    :mod:`novomodelo_bridge.decomp.converters.thermal`'s module docstring.
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


def _optional_int(value: object) -> int | None:
    return None if value is None else as_int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else as_float(value)


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
    groups: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (as_int(row[id_column]), as_int(row["stage_id"]))
        groups.setdefault(key, []).append(row)

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
                    _optional_float(base_row[spec.lower_column])
                    if spec.lower_column and not lower_overridden
                    else None
                )
                base_upper = (
                    _optional_float(base_row[spec.upper_column])
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
                        block_id=_optional_int(row["block_id"]),
                        axis=axis,
                        lower=(
                            _optional_float(row[spec.lower_column])
                            if spec.lower_column
                            else None
                        ),
                        upper=(
                            _optional_float(row[spec.upper_column])
                            if spec.upper_column
                            else None
                        ),
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
    """The ``cadastro-topology-relinked`` INFO, or ``None`` when no operated
    plant's first-stage cascade differs from ``hidr.dat``.

    Lists each converted plant whose own ``AC NUMJUS`` or ``NUMPOS`` changes
    its downstream plant or gauge, and each whose next converted plant
    downstream changes only through an ``AC NUMJUS`` on a plant left out of
    the case, which the per-plant comparison cannot see.
    """
    base = cadastro_conv.EffectiveCadastro(
        base=effective.base, n_stages=effective.n_stages, stage_varying={}
    )
    _, base_parents = scenarios_conv._incremental_context(base, id_map)
    _, effective_parents = scenarios_conv._incremental_context(effective, id_map)
    base_next = _next_operated(base_parents)
    effective_next = _next_operated(effective_parents)

    rows: list[list[object]] = []
    affected_codes: set[int] = set()
    for code in id_map.hydro_codes:
        name = str(effective.base.loc[code, "nome_usina"]).strip()
        base_downstream = int(effective.base.loc[code, "codigo_usina_jusante"])
        eff_downstream = effective.downstream_plant(code, 0)
        if eff_downstream != base_downstream:
            rows.append(
                [code, name, "downstream plant", base_downstream, eff_downstream]
            )
            affected_codes.add(code)
        elif base_next.get(code, 0) != effective_next.get(code, 0):
            rows.append(
                [
                    code,
                    name,
                    "next operated plant",
                    base_next.get(code, 0),
                    effective_next.get(code, 0),
                ]
            )
            affected_codes.add(code)
        base_gauge = int(effective.base.loc[code, "posto"])
        eff_gauge = effective.inflow_gauge(code, 0)
        if eff_gauge != base_gauge:
            rows.append([code, name, "gauge", base_gauge, eff_gauge])
            affected_codes.add(code)

    if not rows:
        return None

    return dx.Diagnostic(
        code="cadastro-topology-relinked",
        severity=dx.Severity.INFO,
        category="Cadastro overrides",
        title=(
            "Downstream plants and gauges changed by AC registers "
            f"({len(affected_codes)})"
        ),
        summary=(
            "AC NUMJUS and NUMPOS registers change the downstream plant or "
            "inflow gauge of the listed plants from hidr.dat, directly or "
            "through a plant left out of the case, and the conversion follows "
            "them, as in DECOMP."
        ),
        table=dx.DiagnosticTable(
            columns=["Code", "Name", "Changes", "hidr.dat", "Effective"],
            rows=rows,
            justify=["right", "left", "left", "right", "right"],
            caption="Effective values are those of the first stage.",
        ),
    )


def _next_operated(parents: Mapping[int, Sequence[int]]) -> dict[int, int]:
    return {child: parent for parent, children in parents.items() for child in children}


#: The withdrawal axis's own side-table schema -- mirrors
#: :func:`~novomodelo_bridge.decomp.converters.bounds.convert_irrigation_withdrawal`'s shape
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
    (:func:`~novomodelo_bridge.decomp.converters.bounds.convert_irrigation_withdrawal`'s
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
    (:func:`~novomodelo_bridge.decomp.converters.bounds.convert_irrigation_withdrawal`)
    through the same ``BoundContribution`` -> :func:`bounds_accumulator.resolve`
    primitive as every other hydro bound (so a withdrawal colliding with another
    contributor on the axis loud-fails instead of one silently overwriting the
    other), then attaches the resolved rows onto *hydro_bounds*.
    ``water_withdrawal_m3s`` is not a ``HYDRO_BOUNDS_SCHEMA`` column (its axis
    rides its own side-table, per that schema's own comment), so the resolved
    rows fan out into their own table (:func:`_fan_resolved_rows`) and attach via
    :func:`~novomodelo_bridge.core.bound_merge.merge_bound_tables` — mirroring
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
    hb = _fill_block_sentinel(pl.DataFrame(hydro_bounds))
    w = _fill_block_sentinel(pl.DataFrame(withdrawal_bounds))
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


@dataclass(frozen=True)
class DiversionChannels:
    """The ``diversion`` channels coupled to the resolved diversion bounds.

    ``channels`` maps a novomodelo hydro id to its ``{downstream_id, max_flow_m3s}``
    entry. ``dropped`` lists the hydros with only a positive diversion ceiling
    whose channel does not resolve; their ceiling must be closed to 0, or the
    diverted water reaches no plant. A floored hydro whose channel does not
    resolve is in neither: the floor cannot be closed, so it stays without a
    channel and fails the emission check
    :func:`~novomodelo_bridge.core.emission_checks.check_diversion_has_channel`.
    """

    channels: dict[int, dict]
    dropped: list[int]


def _diversion_channels(
    hydro_bounds: pa.Table,
    id_map: DecompIdMap,
    effective: cadastro_conv.EffectiveCadastro,
) -> DiversionChannels:
    """Channels for every hydro whose resolved diversion can be positive.

    novomodelo takes the diversion out of the plant's water balance but credits it
    to a receiving plant only through the ``diversion`` channel, so a positive
    ``min_diversion_m3s`` or ``max_diversion_m3s`` needs one. The source model
    sends ``QDES`` to the plant's ``desvio`` target (``AC DESVIO``, else
    ``hidr.dat``), and the conversion refuses a ``QDES`` term on a plant
    without one (``screen_constraints``), so the target comes from
    ``effective.diversions``. The channel carries the
    ``AC DESVIO`` limit, or none when the deck gives none or its "no limit"
    value; the resolved bounds still cap each stage. A target outside the case
    leaves the hydro without a channel.
    """
    floored: set[int] = set()
    opened: set[int] = set()
    columns = hydro_bounds.column_names
    rows = len(hydro_bounds)
    hydro_ids = hydro_bounds.column("hydro_id").to_pylist()
    floors = (
        hydro_bounds.column("min_diversion_m3s").to_pylist()
        if "min_diversion_m3s" in columns
        else [None] * rows
    )
    uppers = (
        hydro_bounds.column("max_diversion_m3s").to_pylist()
        if "max_diversion_m3s" in columns
        else [None] * rows
    )
    for hydro_id, floor, upper in zip(hydro_ids, floors, uppers, strict=True):
        if floor is not None and floor > 0.0:
            floored.add(int(hydro_id))
        if upper is not None and upper > 0.0:
            opened.add(int(hydro_id))
    code_by_id = {id_map.hydro_id(code): code for code in id_map.hydro_codes}
    converted_codes = set(id_map.hydro_codes)
    channels: dict[int, dict] = {}
    dropped: list[int] = []
    for hydro_id in sorted(floored | opened):
        code = code_by_id.get(hydro_id)
        channel = (
            _first_diversion_channel(effective, code) if code is not None else None
        )
        if (
            channel is None
            or channel.downstream not in converted_codes
            or (channel.limit is not None and channel.limit <= 0.0)
        ):
            if hydro_id not in floored:
                dropped.append(hydro_id)
            continue
        channels[hydro_id] = {
            "downstream_id": id_map.hydro_id(channel.downstream),
            "max_flow_m3s": channel_flow_limit(channel.limit),
        }
    return DiversionChannels(channels, dropped)


def _close_diversion(hydro_bounds: pa.Table, hydro_ids: Sequence[int]) -> pa.Table:
    """*hydro_bounds* with the ``max_diversion_m3s`` of *hydro_ids* set to 0."""
    if not hydro_ids or "max_diversion_m3s" not in hydro_bounds.column_names:
        return hydro_bounds
    closed = set(hydro_ids)
    upper = [
        0.0 if hydro_id in closed and value is not None else value
        for hydro_id, value in zip(
            hydro_bounds.column("hydro_id").to_pylist(),
            hydro_bounds.column("max_diversion_m3s").to_pylist(),
            strict=True,
        )
    ]
    index = hydro_bounds.column_names.index("max_diversion_m3s")
    return hydro_bounds.set_column(
        index,
        hydro_bounds.schema.field(index),
        pa.array(upper, type=hydro_bounds.schema.field(index).type),
    )


def _diversion_channel_dropped_diagnostic(
    hydro_bounds: pa.Table,
    dropped: Sequence[int],
    id_map: DecompIdMap,
    effective: cadastro_conv.EffectiveCadastro,
    hydros_dict: dict,
) -> dx.Diagnostic:
    """The WARNING for diversion ceilings closed because no channel resolves."""
    names = {hydro["id"]: hydro["name"] for hydro in hydros_dict["hydros"]}
    ceilings: dict[int, float] = defaultdict(float)
    for hydro_id, upper in zip(
        hydro_bounds.column("hydro_id").to_pylist(),
        hydro_bounds.column("max_diversion_m3s").to_pylist(),
        strict=True,
    ):
        if hydro_id in dropped and upper is not None:
            ceilings[hydro_id] = max(ceilings[hydro_id], upper)
    rows: list[list[object]] = []
    for hydro_id in dropped:
        code = id_map.hydro_codes[hydro_id]
        channel = _first_diversion_channel(effective, code)
        rows.append(
            [
                code,
                names.get(hydro_id, ""),
                channel.downstream if channel is not None else "-",
                ceilings[hydro_id],
            ]
        )
    return dx.Diagnostic(
        code="decomp-diversion-channel-dropped",
        severity=dx.Severity.WARNING,
        category="Diversion",
        title=f"Diversion closed: target plant not in the case ({len(rows)})",
        summary=(
            "The listed plants allow a positive diverted flow, but the plant "
            "that receives it is not in the case. The conversion closes each "
            "diversion, so the water stays in the plant's own river."
        ),
        table=dx.DiagnosticTable(
            columns=["Code", "Name", "Target code", "Max diversion (m³/s)"],
            rows=rows,
            justify=["right", "left", "right", "right"],
        ),
        remediation=(
            "Declare each listed target plant with a UH record in dadger.rvN "
            "to model its diversion."
        ),
    )


def _base_diversion_channels(
    effective: cadastro_conv.EffectiveCadastro,
    id_map: DecompIdMap,
    already_bounded_ids: set[int],
) -> dict[int, dict]:
    """Diversion channels for the ``desvio`` plants without a ``QDES`` bound.

    Without a channel novomodelo closes a plant's diversion column, stranding any
    plant fed *solely* by that diversion -- e.g. MOXOTO's spill-side split to
    P.AFONSO 4, which otherwise receives no water and generates 0 MW. The
    channel carries the ``AC DESVIO`` limit, or none when the deck gives none
    or its "no limit" value, as in the source model.

    Skips any diverter whose diversion axis is already bounded
    (``already_bounded_ids`` -- a ``QDES`` bound, whose channel
    :func:`_diversion_channels` resolves). Returns the ``{downstream_id,
    max_flow_m3s}`` channel per source novomodelo id.
    """
    converted = set(id_map.hydro_codes)
    channels: dict[int, dict] = {}
    for code in id_map.hydro_codes:
        if not effective.has_diversion(code):
            continue
        source_id = id_map.hydro_id(code)
        if source_id in already_bounded_ids:
            continue
        channel = _first_diversion_channel(effective, code)
        if channel is None or channel.downstream not in converted:
            continue
        if channel.limit is not None and channel.limit <= 0.0:
            continue
        channels[source_id] = {
            "downstream_id": id_map.hydro_id(channel.downstream),
            "max_flow_m3s": channel_flow_limit(channel.limit),
        }
    return channels


_LIBS_LEFT_OUT = {
    "inactive": "Left out: never active",
    "unrecognized-token": "Left out: undeclared name",
    "unresolved-bucket-a": "Left out: no case variable for a term",
    "unresolved-bucket-bc": "Left out: data term not evaluated",
}


def _libs_electrical_census_diagnostic(
    result: libs_electrical_emit.LibsElectricalResult,
) -> dx.Diagnostic:
    """The ``decomp-libs-electrical-converted`` INFO census: the converted
    restrictions plus one row per
    :attr:`~novomodelo_bridge.decomp.converters.libs_electrical_emit.LibsElectricalResult.deferred`
    reason, labelled by :data:`_LIBS_LEFT_OUT`."""
    n_converted = len(result.converted_codes)
    total = n_converted + sum(len(codes) for codes in result.deferred.values())
    rows: list[list[object]] = [["Converted", n_converted]]
    rows.extend(
        [_LIBS_LEFT_OUT[reason], len(codes)]
        for reason, codes in sorted(result.deferred.items())
    )
    return dx.Diagnostic(
        code="decomp-libs-electrical-converted",
        severity=dx.Severity.INFO,
        category="Special constraints",
        title=f"LIBS electrical restrictions converted ({n_converted} of {total})",
        summary=(
            f"Of the {dx.counted(total, 'electrical restriction')} in the LIBS "
            f"file, the conversion turns {n_converted} into generic "
            "constraints. The table counts the rest by the reason each is left "
            "out; the other electrical-restriction panels list them."
        ),
        table=dx.DiagnosticTable(
            columns=["Outcome", "Count"],
            rows=rows,
            justify=["left", "right"],
        ),
    )


def _gnl_thermal_codes(
    gnl_model: anticipated_conv.GnlCommitmentModel | None, id_map: DecompIdMap
) -> frozenset[int]:
    """The ``dadgnl`` GNL plant codes, whose LIBs ``ger_usit`` is input data.

    Raises
    ------
    ValueError
        If a code is declared both in ``CT`` and in ``TG``: its ``ger_usit``
        would be read as a commanded GNL generation and leave the equation.
    """
    codes = frozenset(t.code for t in gnl_model.thermals) if gnl_model else frozenset()
    shared = sorted(codes.intersection(id_map.thermal_codes))
    if shared:
        raise ValueError(
            f"the thermal codes {shared} are declared both in the dadger CT "
            "registry and in the dadgnl TG registry; an electrical restriction's "
            "ger_usit for them cannot be told apart as a decision or a "
            "commanded GNL generation"
        )
    return codes


def _water_bucket_stages(
    horizon_end: date, travel_time_hours: Mapping[int, float]
) -> list[dict] | None:
    """The post-study water delivery calendar of a deck with a travel-time arc.

    One stage starts at the study horizon end and lasts the days left in that
    month (its day included). ``None`` when the deck declares no travel time.

    Raises
    ------
    UnsupportedSourceError
        If the largest travel time exceeds the stage's hours, which would need
        a water calendar of more than one post-study stage.
    """
    if not travel_time_hours:
        return None
    days_left = monthrange(horizon_end.year, horizon_end.month)[1] - horizon_end.day + 1
    stage_hours = days_left * 24.0
    code = max(travel_time_hours, key=travel_time_hours.__getitem__)
    if travel_time_hours[code] > stage_hours:
        raise UnsupportedSourceError(
            f"dadger.rvN gives plant {code} a VI travel time of "
            f"{travel_time_hours[code]:g} h, longer than the {stage_hours:g} h left "
            f"in the month of the study horizon end {horizon_end.isoformat()}. "
            "Water still in transit after that month is not converted yet, so "
            f"give the plant a VI travel time of at most {stage_hours:g} h.",
            path="dadger.rvN",
            field="VI duracao",
        )
    return [{"start_date": horizon_end.isoformat(), "duration_hours": stage_hours}]


def _discover(src: Path, *, boundary_fcf: bool) -> DecompCaseArtifacts:
    """The ``step("Discovering deck")`` phase: parse *src* via one
    :class:`DecompCase`, resolve the cadastro override view (emitting its
    AC-register diagnostics), and compute the scalars
    every later phase needs -- the Itaipu-operated flag, the terminal fan
    probabilities, the annual discount rate, and, when *boundary_fcf* and the
    deck has cut files, the auxiliary coupling plants.
    """
    case = DecompCase.from_directory(src)
    tabs = tab_character_diagnostic(
        case.files.fixed_column_files, comment_prefix=REGISTER_COMMENT_PREFIX
    )
    if tabs is not None:
        dx.emit(tabs, logger=_LOG)
    unindexed = unindexed_libs_diagnostic(case.files)
    if unindexed is not None:
        dx.emit(unindexed, logger=_LOG)
    id_map = case.id_map
    dadger = case.dadger
    calendar = case.calendar
    _LOG.info("converting %s (revision %s)", src, case.files.revision)
    require_registered_hydros(case)

    vazoes = Vazoes.read(str(case.files.vazoes))

    # Itaipu's split-plant topology needs a
    # synthesized SE<->IV line and, only when the deck carries the data,
    # the ANDE load on the IV bus -- both driven by this single
    # Itaipu(66)-operated check, computed once and reused by both wiring
    # sites below.
    itaipu_operated = hydro_conv._ITAIPU_CODE in id_map.operated_hydro_codes
    # Cadastro overrides: the per-stage-effective view of
    # the registry, folding in any temporal AC VOLMIN/VOLMAX/VAZMIN override —
    # read once and threaded to every consumer below (initial storage, the
    # entity reservoir envelope, the per-stage storage and minimum-outflow
    # bounds) so they all agree on the same effective values.
    effective, cadastro_report = cadastro_conv.build_effective_cadastro(
        case.dadger, case.hidr, case.calendar
    )
    if cadastro_report.applied or cadastro_report.out_of_horizon:
        dx.emit(_cadastro_overrides_diagnostic(cadastro_report), logger=_LOG)
    deferred_diagnostic = cadastro_conv.ac_overrides_deferred_diagnostic(dadger)
    if deferred_diagnostic is not None:
        dx.emit(deferred_diagnostic, logger=_LOG)
    # The topology/gauge relink is its own diagnostic
    # (not folded into the summary above) so an automated consumer can key
    # on "cadastro-topology-relinked" specifically — the resolved
    # downstream/gauge links, not just an override count.
    relink_diagnostic = _topology_relink_diagnostic(effective, id_map)
    if relink_diagnostic is not None:
        dx.emit(relink_diagnostic, logger=_LOG)
    varies_diagnostic = cadastro_conv.topology_varies_diagnostic(effective)
    if varies_diagnostic is not None:
        dx.emit(varies_diagnostic, logger=_LOG)
    fan_probabilities = scenarios_conv.terminal_fan_probabilities(vazoes, calendar)

    tx_register = dadger.tx
    if tx_register is None or tx_register.taxa is None:
        raise ValueError(
            "the deck has no TX record with a discount rate; cannot convert"
        )
    tx = float(tx_register.taxa) / 100.0

    coupling_auxiliaries: tuple[CouplingAuxiliary, ...] = ()
    if (
        boundary_fcf
        and case.files.cortesh is not None
        and case.files.cortes is not None
    ):
        coupling_auxiliaries = _plan_coupling_auxiliaries(
            case, vazoes, effective, case.files.cortesh
        )

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
        coupling_auxiliaries=coupling_auxiliaries,
    )


def _plan_coupling_auxiliaries(
    case: DecompCase,
    vazoes: Vazoes,
    effective: cadastro_conv.EffectiveCadastro,
    cortesh: Path,
) -> tuple[CouplingAuxiliary, ...]:
    """The auxiliary coupling plants the boundary cuts need, and their note."""
    # TRACKED NOVOMODELO-GAP (C26): novomodelo's lag state follows each hydro's
    # own inflow, so a cut row's last-stage coupling inflow that differs from
    # it rides on an auxiliary plant's inflow.
    auxiliaries = plan_coupling_auxiliaries(
        vazoes,
        effective,
        case.id_map,
        case.calendar,
        read_cortesh(cortesh).plant_codes,
        complexo_components(case.dadger),
    )
    if auxiliaries:
        names = {
            int(code): str(name).strip()
            for code, name in effective.base["nome_usina"].items()
        }
        dx.emit(
            auxiliary_conv.coupling_auxiliary_diagnostic(auxiliaries, names),
            logger=_LOG,
        )
    return auxiliaries


#: ``AC`` registers whose override count is per unit set, not per plant.
_UNIT_SET_AC_REGISTERS = ("NUMMAQ", "POTEFE", "VAZEFE", "ALTEFE")


def _cadastro_overrides_diagnostic(
    report: cadastro_conv.CadastroResolutionReport,
) -> dx.Diagnostic:
    rows = sorted(
        [cadastro_conv.AC_REGISTER_BY_PARAM[param], count]
        for param, count in report.applied.items()
    )
    late = len(report.out_of_horizon)
    late_sentence = (
        f"{dx.counted(late, 'AC override')} "
        f"{'takes' if late == 1 else 'take'} effect after the study "
        f"horizon and {'changes' if late == 1 else 'change'} nothing"
    )
    if not rows:
        return dx.Diagnostic(
            code="cadastro-overrides-applied",
            severity=dx.Severity.INFO,
            category="Cadastro overrides",
            title=f"AC registry overrides after the study horizon ({late})",
            summary=f"{late_sentence}, as in DECOMP.",
        )
    summary = (
        "The deck's AC registers change the hidr.dat registry of the plants "
        "below from the stage each one takes effect, as in DECOMP."
    )
    if late:
        summary += f" {late_sentence}."
    present = {row[0] for row in rows}
    per_unit_set = [name for name in _UNIT_SET_AC_REGISTERS if name in present]
    notes = (
        [f"{', '.join(per_unit_set)} count unit sets, not plants."]
        if per_unit_set
        else []
    )
    return dx.Diagnostic(
        code="cadastro-overrides-applied",
        severity=dx.Severity.INFO,
        category="Cadastro overrides",
        title=f"AC registry overrides applied ({len(rows)})",
        summary=summary,
        table=dx.DiagnosticTable(
            columns=["AC register", "Plants"],
            rows=rows,
            justify=["left", "right"],
        ),
        notes=notes,
    )


def _convert_config_and_stages(
    artifacts: DecompCaseArtifacts, writer: CaseWriter
) -> None:
    """Write ``config.json`` and ``stages.json``, carrying the deck's CVaR risk
    measure onto every stage, and store both dicts on *artifacts*."""
    case = artifacts.case
    # DECOMP risk aversion (CVaR): resolve the AR register / FCF-header CVaR
    # into a per-stage novomodelo risk measure, emitted uniformly across all stages
    # (temporal.stage_records) so novomodelo admits the gap stopping rule under CVaR
    # with enumerated forwards (it computes the exact risk-adjusted upper bound).
    cvar = temporal_conv.resolve_cvar(artifacts.dadger, case.files.cortesh)
    if cvar is not None:
        measure = f"CVaR (α {cvar.alpha:.4g}, λ {cvar.lambda_:.4g})"
        dx.emit(
            dx.Diagnostic(
                code="decomp-cvar-risk-measure",
                severity=dx.Severity.INFO,
                category="Risk",
                title="CVaR risk measure converted",
                summary=(
                    f"The deck applies {measure} and the converted case applies "
                    "it on every stage, as DECOMP does."
                    if cvar.from_stage_index == 0
                    else f"The deck applies {measure} from stage "
                    f"{cvar.from_stage_index} (0-based); the converted case "
                    "applies it on every stage. The earlier stages have a single "
                    "scenario, where CVaR equals the expected value, so the "
                    "results are the same."
                ),
            ),
            logger=_LOG,
        )
    artifacts.config = config_conv.convert_config(case)
    writer.write_json("config.json", artifacts.config)
    stages_dict = temporal_conv.convert_stages(
        case,
        annual_discount_rate=artifacts.tx,
        fan_probabilities=artifacts.fan_probabilities,
        cvar=cvar,
    )
    artifacts.stages_dict = stages_dict
    writer.write_json("stages.json", stages_dict)


def _fpha_stage_models(
    artifacts: DecompCaseArtifacts,
    fpha_codes: set[int],
    initial_volumes: dict[int, float],
    turbine_limits: hydro_conv.TurbineLimits,
) -> dict[int, list[dict]]:
    """Each FPHA-eligible plant's per-stage production model: the FPHA fit
    anchored at its initial volume, or the outage model on a stage whose
    plant turbine limit is zero."""
    stage_models: dict[int, list[dict]] = {}
    if not fpha_codes:
        return stage_models
    case = artifacts.case
    effective = artifacts.effective
    if case.mlt is None:
        dx.emit(fpha_conv.mlt_absent_diagnostic(len(fpha_codes)), logger=_LOG)
    fpha_windows = fpha_conv.volume_window_rule(artifacts.dadger, artifacts.calendar)
    for code in fpha_codes:
        vmin, _ = cadastro_conv.effective_storage_range(effective, code, 0)
        reference_volume = {"volume_hm3": initial_volumes.get(code, vmin)}
        stage_models[code] = [
            hydro_conv.outage_stage_model()
            if turbine_limits.plant_limit(code, stage.index) <= 0.0
            else {
                "model": "fpha",
                "fpha_config": fpha_conv.fpha_config(
                    effective,
                    case.mlt,
                    code,
                    stage.index,
                    fpha_windows,
                    turbined_fit_max_m3s=turbine_limits.fit_maxima[code][stage.index],
                ),
                "reference_volume": reference_volume,
            }
            for stage in artifacts.calendar
        ]
    outside_tailrace = fpha_conv.spill_outside_tailrace(effective, stage_models)
    if outside_tailrace is not None:
        dx.emit(outside_tailrace, logger=_LOG)
    return stage_models


def _convert_lines_with_itaipu_link(
    artifacts: DecompCaseArtifacts,
) -> tuple[dict, pa.Table]:
    """The deck's lines and line bounds, plus the SE<->IV connection an
    operated Itaipu needs."""
    case = artifacts.case
    id_map = artifacts.id_map
    lines_doc, line_bounds = network_conv.convert_lines(case, id_map)
    if artifacts.itaipu_operated:
        # When Itaipu is operated (the split-plant topology relocates its 60 Hz
        # group to the IV bus), the IV bus needs a connection to SE. append_iv_se_line
        # synthesizes an SE<->IV line only when the deck's own IA register does
        # NOT already wire IV<->SE; when it does (as on most Itaipu decks), the
        # helper reuses that line -- its real operational capacity -- and adding
        # a second one would make novomodelo reject the duplicate.
        itaipu_submercado = int(case.hidr.loc[hydro_conv._ITAIPU_CODE, "submercado"])
        lines_doc, line_bounds = network_conv.append_iv_se_line(
            case,
            lines_doc=lines_doc,
            line_bounds=line_bounds,
            source_bus_id=id_map.transhipment_bus_id,
            target_bus_id=id_map.bus_id(itaipu_submercado),
            capacity_mw=network_conv._itaipu_60hz_capacity_mw(artifacts.dadger),
        )
    return lines_doc, line_bounds


def _convert_thermals_with_gnl(
    artifacts: DecompCaseArtifacts,
    writer: CaseWriter,
    initial_conditions_doc: dict,
    travel_time_hours: Mapping[int, float],
) -> tuple[dict, anticipated_conv.GnlCommitmentModel | None]:
    """The thermal registry with the dadgnl GNL fleet spliced in, and the GNL
    commitment model (``None`` when the deck has no dadgnl file).

    Extends *initial_conditions_doc* with the GNL left boundary and writes
    ``post_study_stages.json`` once: the GNL thermal calendar when the deck
    declares a GS calendar, plus the water calendar of a deck with a
    travel-time arc.
    """
    case = artifacts.case
    id_map = artifacts.id_map
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
    post_study_stages: dict = {}
    if gnl_model is not None:
        gnl_model = anticipated_conv.fill_undeclared_block_hours(
            gnl_model,
            case.calendar,
            read_patamar_fractions(Cortesh.read(str(case.files.cortesh)))
            if case.files.cortesh is not None
            else None,
        )
        gnl = anticipated_conv.convert_gnl(
            gnl_model,
            first_thermal_id=max(
                (t["id"] for t in thermals_dict["thermals"]), default=-1
            )
            + 1,
            bus_id_of=id_map.bus_id,
            calendar=case.calendar,
        )
        thermals_dict["thermals"].extend(gnl.thermals)
        thermals_dict["thermals"].sort(key=lambda t: t["id"])
        if gnl.past_anticipated_commitments:
            initial_conditions_doc["past_anticipated_commitments"] = (
                gnl.past_anticipated_commitments
            )
        post_study_stages = dict(gnl.post_study_stages or {})
        _LOG.info(
            "GNL anticipated thermals emitted from dadgnl: %d",
            len(gnl.thermals),
        )
    water_bucket_stages = _water_bucket_stages(
        case.calendar[-1].end_date, travel_time_hours
    )
    if water_bucket_stages is not None:
        post_study_stages["water_bucket_stages"] = water_bucket_stages
    if post_study_stages:
        writer.write_json("post_study_stages.json", post_study_stages)
    return thermals_dict, gnl_model


def _write_hydro_production_files(
    artifacts: DecompCaseArtifacts,
    writer: CaseWriter,
    stage_models: dict[int, list[dict]],
    productivity_parquet: pa.Table,
) -> None:
    """Write the per-stage production models, the energy-productivity table,
    and the geometry and tailrace inputs of the FPHA fit."""
    case = artifacts.case
    id_map = artifacts.id_map
    auxiliaries = artifacts.coupling_auxiliaries
    production_models = hydro_conv.convert_production_models(id_map, stage_models)
    production_models["production_models"].extend(
        auxiliary_conv.auxiliary_production_models(auxiliaries)
    )
    writer.write_json("system/hydro_production_models.json", production_models)
    writer.write_parquet(
        "system/hydro_energy_productivity.parquet",
        auxiliary_conv.with_auxiliary_productivity(productivity_parquet, auxiliaries),
    )
    # FPHA geometry + tailrace inputs novomodelo fits the production function from.
    writer.write_parquet(
        "system/hydro_geometry.parquet",
        fpha_conv.convert_hydro_geometry(artifacts.effective, id_map),
    )
    tailrace_table = fpha_conv.convert_tailrace_curves(case, id_map)
    if tailrace_table is not None:
        writer.write_parquet("system/tailrace_curves.parquet", tailrace_table)


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

    _convert_config_and_stages(artifacts, writer)

    # Both the FPHA reference volume and the constant-productivity ρ_eq are
    # anchored at the initial volume, not the full-range mean — see
    # hydro._equivalent_productivity_mw_per_m3s.
    initial_volumes = hydro_conv._converted_initial_volumes(case, effective=effective)
    turbine_limits = hydro_conv.compute_turbine_limits(
        effective,
        id_map,
        artifacts.calendar,
        initial_volumes=initial_volumes,
        incremental_inflows=scenarios_conv.expected_incremental_inflows(
            artifacts.vazoes,
            effective,
            id_map,
            artifacts.calendar,
            artifacts.fan_probabilities,
        ),
        tailrace=hydro_conv.tailrace_families(case, id_map),
        availability_factors=hydro_conv.stage_availability_factors(
            case, id_map, effective=effective
        ),
    )
    artifacts.turbine_limits = turbine_limits
    fpha_codes = fpha_conv.fpha_eligible_codes(effective, id_map)
    artifacts.fpha_codes = fpha_codes
    stage_models = _fpha_stage_models(
        artifacts, fpha_codes, initial_volumes, turbine_limits
    )
    fpha_stage_codes = {
        code
        for code, models in stage_models.items()
        if any(m["model"] == "fpha" for m in models)
    }
    resolution = tailrace_flow_conv.resolve_tailrace_flow(
        dadger,
        effective,
        id_map.hydro_codes,
        tailrace_flow_conv.piecewise_tailrace_codes(case, id_map),
    )

    productivity = hydro_conv.convert_energy_productivity(
        effective, id_map, initial_volumes
    )
    productivity_parquet = hydro_conv.convert_hydro_energy_productivity(
        effective, id_map, initial_volumes, fpha_codes
    )
    deficit_costs = network_conv.bus_deficit_costs(dadger)
    deficit_cost = max(deficit_costs.values()) if deficit_costs else 0.0
    artifacts.deficit_cost = deficit_cost
    writer.write_json(
        "penalties.json",
        config_conv.convert_penalties(
            deficit_cost,
            productivity["equivalent_productivity_mw_per_m3s"].to_pylist(),
        ),
    )
    # week_hours is stage 0's own duration, never a hardcoded 168: a deck whose
    # first stage is not a full operative week still tiles past_defluences exactly.
    calendar = case.calendar
    travel_time_hours, past_defluences = travel_time_conv.convert_travel_time(
        dadger, id_map, effective, case.start_date, calendar[0].total_hours
    )
    # Built here but written below (after GNL): an anticipated GNL fleet extends
    # it with the left/right temporal-boundary arrays, and those need the thermal
    # ids assigned when thermals.json is built.
    initial_conditions_doc: dict = {
        "$schema": novomodelo_schemas.schema_url_for("initial_conditions.json"),
        "storage": [
            *hydro_conv.convert_initial_storage(case, id_map, effective=effective),
            *auxiliary_conv.auxiliary_initial_storage(artifacts.coupling_auxiliaries),
        ],
        "filling_storage": [],
    }
    if past_defluences:
        initial_conditions_doc["past_defluences"] = past_defluences
    artifacts.initial_conditions = initial_conditions_doc

    buses_doc = network_conv.convert_buses(case, id_map)
    artifacts.buses_doc = buses_doc
    writer.write_json("system/buses.json", buses_doc)
    artifacts.hydros_dict = hydro_conv.convert_hydros(
        case,
        id_map,
        effective=effective,
        turbine_limits=turbine_limits,
        travel_time_hours=travel_time_hours,
        fpha_codes=fpha_codes,
        tailrace_flows=tailrace_flow_conv.convert_tailrace_flow(
            resolution,
            fpha_stage_codes,
            id_map,
            fpha_codes=fpha_codes,
            effective=effective,
        ),
    )
    # hydros.json is written after the bound tables are resolved (below): its
    # diversion channels follow the resolved diversion bounds.
    lines_doc, line_bounds = _convert_lines_with_itaipu_link(artifacts)
    lines_doc = penalties_conv.with_fictitious_exchange_cost(
        lines_doc, network_conv.fictitious_bus_ids(deficit_costs, id_map)
    )
    artifacts.lines_doc = lines_doc
    artifacts.line_bounds = line_bounds
    if lines_doc["lines"]:
        writer.write_json("system/lines.json", lines_doc)
    writer.write_json(
        "system/pumping_stations.json",
        network_conv.convert_pumping_stations(case, id_map),
    )
    thermals_dict, gnl_model = _convert_thermals_with_gnl(
        artifacts, writer, initial_conditions_doc, travel_time_hours
    )
    artifacts.thermals_dict = thermals_dict
    artifacts.gnl_model = gnl_model
    if thermals_dict["thermals"]:
        writer.write_json("system/thermals.json", thermals_dict)
    # The PAR inflow-lag seed (initial_conditions.recent_observations) is
    # written by `fcf/importer.py::import_boundary_fcf`, not here: it is read on
    # the boundary cut's coupling postos and feeds only that cut's inflow-lag
    # terms. Without a boundary FCF the inflow model is order 0, so the lags
    # are inert.
    writer.write_json("initial_conditions.json", initial_conditions_doc)
    _write_hydro_production_files(artifacts, writer, stage_models, productivity_parquet)

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
    calendar = artifacts.calendar
    vazoes = artifacts.vazoes
    effective = artifacts.effective
    fan_probabilities = artifacts.fan_probabilities

    auxiliaries = artifacts.coupling_auxiliaries
    writer.write_parquet(
        "scenarios/inflow_seasonal_stats.parquet",
        auxiliary_conv.with_auxiliary_inflow_stats(
            scenarios_conv.convert_inflow_stats_identity(case, id_map), auxiliaries
        ),
    )
    writer.write_parquet(
        "scenarios/external_inflow_scenarios.parquet",
        auxiliary_conv.with_auxiliary_external_inflows(
            scenarios_conv.convert_external_inflows(
                case, id_map, vazoes=vazoes, effective=effective
            ),
            auxiliaries,
            len(calendar),
        ),
    )
    # Itaipu's carga_ande load nets into its own SE bus (the HVDC Elo
    # delivers the 50 Hz group's output, and the ANDE demand, directly there
    # -- see hydro.py's convert_hydros docstring) and is gated on the dadger
    # RI register being present -- independent of (and additional to) the
    # unconditional Itaipu-operated line above. An Itaipu deck with no RI
    # register (ande_load returns None) converts gracefully: SE carries only
    # its own DP demand.
    extra_bus_loads: dict[tuple[int, int], list[float]] | None = None
    ande = libs_electrical_conv.ande_load(case)
    if ande is not None:
        ande_submarket, carga_ande = ande
        ande_bus = id_map.bus_id(ande_submarket)
        artifacts.ande_bus_load = (ande_bus, carga_ande)
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
    # and standardizes to eta = 0 (value == mean) under novomodelo's scheme-aware
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
            # novomodelo reads the NCS availability column only as
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


def _read_constraint_census(
    dadger: constraint_registers.ConstraintRegisters,
    effective: cadastro_conv.EffectiveCadastro,
    id_map: DecompIdMap,
    pumping_ids: Mapping[int, int],
    gnl_generation: Mapping[int, Sequence[Sequence[float] | None]],
    *,
    gnl_submarkets: Mapping[int, int],
    ct_submarkets: Mapping[int, Collection[int]],
    dadger_path: str,
    dadgnl_path: str | None,
) -> constraint_registers.ConstraintCensus:
    """The deck's special constraints split by lowering target, with the
    commanded generation of GNL thermals moved into their limits and without
    the ones on an entity the case lacks, reporting those, any conflicting
    bounds and the GNL generations folded that differ by patamar as
    diagnostics."""
    census, gnl_only = constraints_conv.fold_commanded_gnl_terms(
        constraint_registers.read_constraints(dadger, dadger_path=dadger_path),
        gnl_generation,
        gnl_submarkets=gnl_submarkets,
        ct_submarkets=ct_submarkets,
        dadger_path=dadger_path,
        dadgnl_path=dadgnl_path,
    )
    census, screened = constraint_registers.screen_constraints(
        census,
        hydro_codes=id_map.operated_hydro_codes,
        coupling_only_codes=id_map.idle_hydro_codes,
        thermal_codes=id_map.thermal_codes,
        pumping_codes=pumping_ids.keys(),
        diversion_codes=effective.diversions.keys(),
        plant_names={
            int(code): str(name).strip()
            for code, name in effective.base["nome_usina"].items()
        },
        dadger_path=dadger_path,
    )
    patamar_values = constraints_conv.re_gnl_patamar_diagnostic(census, gnl_generation)
    for diagnostic in (*gnl_only, *screened):
        dx.emit(diagnostic, logger=_LOG)
    if patamar_values is not None:
        dx.emit(patamar_values, logger=_LOG)
    return census


def _released_runofriver_codes(
    effective: cadastro_conv.EffectiveCadastro, id_map: DecompIdMap
) -> set[int]:
    """The run-of-river plants whose minimum-outflow default is released,
    reported as a diagnostic."""
    # Run-of-river plants from a cascade headwater down to (but excluding) the
    # first reservoir have their RQ/UH minimum-outflow default released, to
    # avoid infeasibility on a zero-inflow stage/scenario. Only the RQ/UH
    # default floor (convert_hydro_bounds) is released; an explicit RHQ QDEF
    # window (single_term_bound_contributions) is a user-declared constraint
    # and is kept.
    unregulated_codes = cadastro_conv.unregulated_runofriver_codes(
        effective, id_map.operated_hydro_codes
    )
    if unregulated_codes:
        dx.emit(
            dx.Diagnostic(
                code="decomp-runofriver-outflow-relaxed",
                severity=dx.Severity.INFO,
                category="Minimum outflow",
                title=(
                    "Run-of-river plants above the first reservoir without a "
                    f"minimum outflow ({len(unregulated_codes)})"
                ),
                summary=(
                    "The listed run-of-river plants sit upstream "
                    "of the first reservoir of their cascade, so, as in DECOMP, "
                    "they take no RQ/UH minimum outflow. A single-plant RHQ QDEF "
                    "floor still applies."
                ),
                table=dx.DiagnosticTable(
                    columns=["Code", "Name"],
                    rows=[
                        [code, str(effective.base.loc[code, "nome_usina"]).strip()]
                        for code in sorted(unregulated_codes)
                    ],
                    justify=["right", "left"],
                ),
            ),
            logger=_LOG,
        )
    return unregulated_codes


def _couple_diversion_channels(
    hydro_bounds: pa.Table,
    hydros_dict: dict,
    base_diversion_channels: dict[int, dict],
    id_map: DecompIdMap,
    effective: cadastro_conv.EffectiveCadastro,
) -> pa.Table:
    """Give each hydro in *hydros_dict* the diversion channel its resolved
    diversion bounds need, else its base channel.

    Returns *hydro_bounds* with the diversion ceiling of every hydro whose
    channel does not resolve closed to 0.
    """
    diversion = _diversion_channels(hydro_bounds, id_map, effective)
    if diversion.dropped:
        dx.emit(
            _diversion_channel_dropped_diagnostic(
                hydro_bounds, diversion.dropped, id_map, effective, hydros_dict
            ),
            logger=_LOG,
        )
        hydro_bounds = _close_diversion(hydro_bounds, diversion.dropped)
    names = {hydro["id"]: hydro["name"] for hydro in hydros_dict["hydros"]}
    unlimited: list[tuple[int, str, int, str, float]] = []
    for hydro in hydros_dict["hydros"]:
        channel = diversion.channels.get(hydro["id"]) or base_diversion_channels.get(
            hydro["id"]
        )
        if channel is None:
            continue
        hydro["diversion"] = channel
        code = id_map.hydro_codes[hydro["id"]]
        source = _first_diversion_channel(effective, code)
        if (
            source is not None
            and source.limit is not None
            and is_unlimited_flow(source.limit)
        ):
            unlimited.append(
                (
                    code,
                    hydro["name"],
                    source.downstream,
                    names.get(id_map.hydro_id(source.downstream), ""),
                    source.limit,
                )
            )
    if unlimited:
        dx.emit(
            unlimited_channels_diagnostic(unlimited, source="AC DESVIO"),
            logger=_LOG,
        )
    if base_diversion_channels:
        dx.emit(
            dx.Diagnostic(
                code="decomp-base-diversion-modeled",
                severity=dx.Severity.INFO,
                category="Diversion",
                title=f"Diversions without a QDES restriction ({len(base_diversion_channels)})",
                summary=(
                    "The listed plants divert water through "
                    "their hidr.dat or AC DESVIO desvio with no QDES restriction. "
                    "Each diverts through a channel to the receiving plant, "
                    "limited only by the AC DESVIO flow, as in DECOMP."
                ),
                table=dx.DiagnosticTable(
                    columns=[
                        "Code",
                        "Name",
                        "Target code",
                        "Target name",
                        "Limit (m³/s)",
                    ],
                    rows=[
                        [
                            id_map.hydro_codes[hydro_id],
                            names.get(hydro_id, ""),
                            id_map.hydro_codes[channel["downstream_id"]],
                            names.get(channel["downstream_id"], ""),
                            "none"
                            if channel["max_flow_m3s"] is None
                            else channel["max_flow_m3s"],
                        ]
                        for hydro_id, channel in sorted(base_diversion_channels.items())
                    ],
                    justify=["right", "left", "right", "left", "right"],
                ),
            ),
            logger=_LOG,
        )
    return hydro_bounds


def _resolve_line_bounds(
    line_bounds: pa.Table, block_counts: dict[int, int]
) -> pa.Table:
    """*line_bounds* resolved to one row per ``(line, stage, block)`` cell."""
    # Line bounds route through the same BoundContribution -> resolve()
    # primitive as hydro/thermal/pumping in `_resolve_bounds`: a colliding
    # pair of contributions on the direct/reverse axis intersects (or
    # loud-fails on a genuine conflict) instead of one silently overwriting
    # the other.
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
    return _fan_resolved_rows(
        [
            ((row.entity_id, row.stage_id, row.block_id), row)
            for row in bounds_accumulator.resolve(line_contribs, block_counts)
        ],
        schema=network_conv._LINE_BOUNDS_SCHEMA,
        key_columns=("line_id", "stage_id", "block_id"),
    ).sort_by([("line_id", "ascending"), *_BOUND_SORT_KEYS])


#: Contributor prefix of a special-constraint bound -> the register its
#: report names, followed by the constraint id.
_SPECIAL_CONSTRAINT_REGISTERS: dict[str, str] = {"RE": "RE", "HQ": "RHQ", "HV": "RHV"}

#: Every other bound contributor that can meet another -> the deck record or
#: limit a report names.
_BOUND_SOURCE_LABELS: dict[str, str] = {
    "RQ": "RQ",
    "UH": "UH",
    "UH_vert": "UH",
    "VE": "VE",
    "CT": "CT",
    "storage-envelope": "AC VOLMIN/VOLMAX",
    "turbine_limit": "turbined-flow limit",
    "split_plant_availability": "available power",
    "group_bounds": "available power or RI",
    "convert_contract_bounds": "CI/CE",
}


def _bound_source_label(contributor: str) -> str:
    """The deck constraint or limit a bound *contributor* stands for, as a
    report names it."""
    prefix, _, constraint_id = contributor.partition("_")
    register = _SPECIAL_CONSTRAINT_REGISTERS.get(prefix)
    if register is not None and constraint_id.isdigit():
        return f"{register} {constraint_id}"
    return _BOUND_SOURCE_LABELS[contributor]


def _hydro_bound_labels(
    contribs: Sequence[bounds_accumulator.BoundContribution],
) -> emission_checks.BoundLabels:
    """The deck constraints and limits behind each ``hydro_bounds`` cell, each
    with the value it sets, for the capacity clamp's report."""
    labels: dict[emission_checks.BoundKey, list[tuple[str, float]]] = {}
    for contrib in contribs:
        if contrib.family != "hydro":
            continue
        source = _bound_source_label(contrib.contributor)
        spec = bounds_accumulator.axis_spec("hydro", contrib.axis)
        for column, value in (
            (spec.lower_column, contrib.lower),
            (spec.upper_column, contrib.upper),
        ):
            if column is not None and is_bounded(value):
                labels.setdefault(
                    (contrib.entity_id, contrib.stage_id, contrib.block_id, column),
                    [],
                ).append((source, value))
    return labels


def _source_list(sources: Iterable[tuple[str, float]]) -> str:
    """The deck constraints and limits behind *sources*' contributors, each
    named once."""
    return ", ".join(
        dict.fromkeys(_bound_source_label(contributor) for contributor, _ in sources)
    )


#: Two-sided bound axes novomodelo enforces on both sides -> the noun and unit
#: a refusal names the variable by.
_ENFORCED_AXIS_NOUNS: dict[tuple[str, str], tuple[str, str]] = {
    ("hydro", "spillage"): ("spillage", "m³/s"),
    ("hydro", "diversion"): ("diversion", "m³/s"),
    ("thermal", "generation"): ("generation", "MW"),
    ("pumping", "flow"): ("pumped flow", "m³/s"),
    ("contract", "power"): ("contracted power", "MW"),
}


def _resolve_or_refuse(
    contribs: Sequence[bounds_accumulator.BoundContribution],
    block_counts: Mapping[int, int],
    entity: Callable[[bounds_accumulator.ResolvedRow], str],
    *,
    dadger: Path,
) -> list[bounds_accumulator.ResolvedRow]:
    """:func:`bounds_accumulator.resolve`, refusing a floor above the ceiling
    on an axis novomodelo enforces on both sides.

    Raises
    ------
    FieldParseError
        Naming the entity, as *entity* names a row's, the stage, both limits
        and the *dadger* records that set them.
    """
    try:
        return bounds_accumulator.resolve(contribs, block_counts)
    except bounds_accumulator.FloorAboveCeilingError as exc:
        row, conflict = exc.row, exc.conflict
        noun, unit = _ENFORCED_AXIS_NOUNS[(row.family, row.axis)]
        floors = _source_list(conflict.floors)
        ceilings = _source_list(conflict.ceilings)
        block = "," if row.block_id is None else f", load block {row.block_id + 1},"
        raise FieldParseError(
            f"The deck sets the {noun} of {entity(row)} in its stage "
            f"{row.stage_id + 1}{block} a floor of {round(conflict.lower, 6)} "
            f"{unit} ({floors}) above its ceiling of {round(conflict.upper, 6)} "
            f"{unit} ({ceilings}). novomodelo holds the {noun} within both "
            "limits, so the case would have no solution.",
            path=str(dadger),
            field=_source_list((*conflict.floors, *conflict.ceilings)),
        ) from exc


#: Soft bound axes whose shortfall novomodelo prices, which the merge keeps a
#: floor above the ceiling on and reports -> the limit a report names. A storage
#: pair kept past the written precision is refused by
#: :func:`~novomodelo_bridge.core.emission_checks.clamp_hydro_bounds_to_declared`
#: instead.
_SOFT_LIMIT_LABELS: dict[tuple[str, str], str] = {
    ("hydro", "turbined"): "Turbined flow (m³/s)",
    ("hydro", "generation"): "Generation (MW)",
    ("hydro", "outflow"): "Outflow (m³/s)",
    ("hydro_unit_group", "turbined"): "turbined flow (m³/s)",
    ("hydro_unit_group", "generation"): "generation (MW)",
}


@dataclass(frozen=True)
class _KeptConflict:
    """A floor above the ceiling written as the deck has it: the plant's code
    and name, the limit, and the stage."""

    code: int
    name: str
    limit: str
    stage_id: int
    conflict: bounds_accumulator.FloorAboveCeiling


def _keep_soft_conflicts(
    rows: Sequence[bounds_accumulator.ResolvedRow],
    plant: Callable[[bounds_accumulator.ResolvedRow], tuple[int, str, str]],
    kept: list[_KeptConflict],
) -> list[bounds_accumulator.ResolvedRow]:
    """*rows* with each floor above its ceiling settled: lowered to the ceiling
    within the deck's written precision, else kept and, on an axis of
    :data:`_SOFT_LIMIT_LABELS`, recorded in *kept* under the ``(code, name,
    limit prefix)`` *plant* gives its row."""
    settled: list[bounds_accumulator.ResolvedRow] = []
    for row in rows:
        conflict = row.conflict
        if conflict is None:
            settled.append(row)
        elif beyond_written_precision(conflict.lower, conflict.upper):
            limit = _SOFT_LIMIT_LABELS.get((row.family, row.axis))
            if limit is not None:
                code, name, prefix = plant(row)
                kept.append(
                    _KeptConflict(code, name, prefix + limit, row.stage_id, conflict)
                )
            settled.append(row)
        else:
            settled.append(replace(row, lower=row.upper, conflict=None))
    return settled


def _kept_hydro_floors(
    rows: Iterable[bounds_accumulator.ResolvedRow],
) -> set[emission_checks.BoundKey]:
    """The ``hydro_bounds`` floors :func:`_keep_soft_conflicts` kept above
    their ceiling, which the merge warning already reports."""
    floors: set[emission_checks.BoundKey] = set()
    for row in rows:
        if row.family != "hydro" or row.conflict is None:
            continue
        column = bounds_accumulator.axis_spec(row.family, row.axis).lower_column
        if column is not None:
            floors.add((row.entity_id, row.stage_id, row.block_id, column))
    return floors


def _kept_conflicts_diagnostic(kept: Sequence[_KeptConflict]) -> dx.Diagnostic:
    """WARNING listing the floors above the ceiling written as the deck has
    them, one row per plant and limit."""
    grouped: dict[tuple[int, str, str], list[_KeptConflict]] = {}
    for item in kept:
        grouped.setdefault((item.code, item.name, item.limit), []).append(item)
    rows: list[list[object]] = []
    for (code, name, limit), items in sorted(grouped.items()):
        widest = max(items, key=lambda i: i.conflict.lower - i.conflict.upper)
        rows.append(
            [
                code,
                name,
                limit,
                _source_list(s for i in items for s in i.conflict.floors),
                _source_list(s for i in items for s in i.conflict.ceilings),
                dx.format_stage_ranges(i.stage_id for i in items),
                round(widest.conflict.lower, 6),
                round(widest.conflict.upper, 6),
            ]
        )
    return dx.Diagnostic(
        code="decomp-bound-floor-above-ceiling",
        severity=dx.Severity.WARNING,
        category="Special constraints",
        title=f"Stage floors above the stage ceiling ({len(rows)})",
        summary=(
            "The deck sets the listed floors above the ceilings beside them on "
            "the listed stages. Both limits were written as the deck has them: "
            "DECOMP violates whichever is cheaper, and novomodelo charges the "
            "violation at its penalty, which raises the cost of those stages."
        ),
        table=dx.DiagnosticTable(
            columns=[
                "Code",
                "Name",
                "Limit",
                "Floor set by",
                "Ceiling set by",
                "Stages",
                "Floor",
                "Ceiling",
            ],
            rows=rows,
            justify=["right", "left", "left", "left", "left", "left", "right", "right"],
            caption=(
                "Stage and unit-group ids are 0-based; the values shown are the "
                "stage with the widest gap."
            ),
        ),
    )


def _bound_entity(
    row: bounds_accumulator.ResolvedRow,
    id_map: DecompIdMap,
    hydro_names: Mapping[int, str],
    thermals: dict | None,
    pumping_ids: Mapping[int, int],
) -> str:
    """How a refusal names the plant or pumping station of *row*."""
    if row.family == "hydro":
        return (
            f"plant {id_map.hydro_codes[row.entity_id]} ({hydro_names[row.entity_id]})"
        )
    if row.family == "thermal":
        assert thermals is not None
        (name,) = [t["name"] for t in thermals["thermals"] if t["id"] == row.entity_id]
        return f"thermal plant {id_map.thermal_codes[row.entity_id]} ({name})"
    (code,) = [
        code for code, station in pumping_ids.items() if station == row.entity_id
    ]
    return f"pumping station {code}"


def _contract_entity(contract: contracts_conv.Contract) -> str:
    """How a refusal names *contract*: its register, number and name."""
    register = contracts_conv.register_label(contract.kind)
    return f"{register} {contract.numero} ({contract.name})"


def _resolve_bounds(artifacts: DecompCaseArtifacts, writer: CaseWriter) -> None:
    """The ``step("Resolving bounds")`` phase: resolve every hydro/thermal/
    pumping/line bound, fold in the TI irrigation withdrawal and the
    diversion channels, and write the diversion-coupled ``system/hydros.json``
    (deferred from ``_convert_core_entities``'s system-file block for exactly
    that coupling). Stores the resolved bound tables, the per-group
    ``availability_values``, and the ``census``/``pumping_ids``/
    ``block_counts`` maps the later phases need back onto *artifacts*.
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
    turbine_limits = artifacts.turbine_limits
    assert turbine_limits is not None

    pumping_ids = network_conv.pumping_station_id_map(dadger)
    gnl_model = artifacts.gnl_model
    census = _read_constraint_census(
        dadger,
        effective,
        id_map,
        pumping_ids,
        anticipated_conv.gnl_block_commitments(gnl_model, calendar)
        if gnl_model is not None
        else {},
        gnl_submarkets=(
            anticipated_conv.gnl_submarkets(gnl_model) if gnl_model is not None else {}
        ),
        ct_submarkets=thermal_conv.ct_submarkets(dadger),
        dadger_path=str(case.files.dadger),
        dadgnl_path=str(case.files.dadgnl) if case.files.dadgnl is not None else None,
    )
    thermal_generation_contribs, thermal_cost_table = (
        thermal_conv.convert_thermal_bounds(case, id_map)
    )
    unregulated_codes = _released_runofriver_codes(effective, id_map)
    availability_values = hydro_conv.convert_hydro_group_availability(
        case, id_map, effective=effective, turbine_limits=turbine_limits
    )
    # Every per-entity bound — the RQ/UH minimum-outflow, the per-stage
    # storage envelope, a split plant's turbined-flow limit and available
    # power, the CT thermal
    # generation bounds, and the RE/RHQ/RHV single-term special-constraint
    # bounds — is collected as
    # bounds_accumulator.BoundContribution objects and resolved through
    # exactly ONE resolve() + build_bound_tables() pass, so a special
    # constraint colliding with another bound on the same (entity, stage,
    # block) cell correctly intersects instead of producing the
    # two-rows-same-column parquet novomodelo rejects. The UH spillage
    # ceiling is the exception: hydros.json carries it, so it is added below
    # only on cells another contribution already bounds.
    contribs = [
        *bounds_conv.convert_hydro_bounds(
            case, id_map, effective=effective, unregulated_codes=unregulated_codes
        ),
        *bounds_conv.convert_storage_bounds(case, id_map, effective=effective),
        *bounds_conv.convert_volume_espera_bounds(case, id_map, effective=effective),
        *hydro_conv.plant_turbine_limit_contributions(turbine_limits, id_map),
        *hydro_conv.split_plant_generation_contributions(availability_values),
        *thermal_generation_contribs,
        *single_term_bounds.single_term_bound_contributions(
            case,
            id_map,
            census=census,
            pumping_station_ids=pumping_ids,
            effective=effective,
        ),
    ]
    declared_ceilings = {
        hydro["id"]: hydro["spillage"]["max_spillage_m3s"]
        for hydro in hydros_dict["hydros"]
        if "spillage" in hydro
    }
    contribs.extend(
        bounds_conv.convert_spillage_upper_bounds(declared_ceilings, contribs)
    )
    base_diversion_channels = _base_diversion_channels(
        effective,
        id_map,
        {
            contrib.entity_id
            for contrib in contribs
            if contrib.family == "hydro" and contrib.axis == "diversion"
        },
    )
    block_counts = {stage.index: len(stage.block_hours) for stage in calendar}
    hydro_names = {hydro["id"]: hydro["name"] for hydro in hydros_dict["hydros"]}
    kept: list[_KeptConflict] = []
    rows = _keep_soft_conflicts(
        _resolve_or_refuse(
            contribs,
            block_counts,
            lambda row: _bound_entity(
                row, id_map, hydro_names, artifacts.thermals_dict, pumping_ids
            ),
            dadger=case.files.dadger,
        ),
        lambda row: (
            id_map.hydro_codes[row.entity_id],
            hydro_names[row.entity_id],
            "",
        ),
        kept,
    )
    bound_tables = bounds_accumulator.build_bound_tables(rows)
    # An RE or RHQ QTUR ceiling can sit above the plant's declared capacity
    # (BELO MONTE's RE does on both real decks); novomodelo rule 43 rejects it.
    # An RHV VARM ceiling can sit above the reservoir, which novomodelo takes
    # as written. A ceiling below the stage's floor is not raised: the source
    # model treats both limits as soft and violates whichever is cheaper, so the
    # case keeps both and novomodelo prices the shortfall, except on storage,
    # whose floor novomodelo enforces, so the clamp refuses the pair.
    hydro_bounds = emission_checks.clamp_hydro_bounds_to_declared(
        hydros_dict,
        bound_tables.hydro.sort_by([("hydro_id", "ascending"), *_BOUND_SORT_KEYS]),
        source_codes={id_map.hydro_id(code): code for code in id_map.hydro_codes},
        raise_to_minimum=False,
        labels=_hydro_bound_labels(contribs),
        reported_floors=_kept_hydro_floors(rows),
        deck_file=case.files.dadger,
    )
    assert hydro_bounds is not None
    # Fold in the TI irrigation withdrawal (a consumptive water use novomodelo reads
    # from hydro_bounds.water_withdrawal_m3s): water lost to irrigation is
    # unavailable for generation, so omitting it lets novomodelo turbine the extra
    # flow and over-generate. Mirrors the source model's dsvagua path.
    hydro_bounds = _attach_water_withdrawal(
        hydro_bounds,
        bounds_conv.convert_irrigation_withdrawal(case, id_map),
        block_counts,
    )
    hydro_bounds = _couple_diversion_channels(
        hydro_bounds, hydros_dict, base_diversion_channels, id_map, effective
    )
    writer.write_json(
        "system/hydros.json",
        auxiliary_conv.with_auxiliary_hydros(
            hydros_dict,
            artifacts.coupling_auxiliaries,
            id_map=id_map,
            submercados={
                auxiliary.plant_code: as_int(
                    effective.base.at[auxiliary.plant_code, "submercado"]
                )
                for auxiliary in artifacts.coupling_auxiliaries
            },
            operational_start_date=case.start_date.isoformat(),
        ),
    )
    thermal_bounds_table = _rejoin_thermal_cost(
        bound_tables.thermal, thermal_cost_table
    ).sort_by([("thermal_id", "ascending"), *_BOUND_SORT_KEYS])
    pumping_bounds_table = bound_tables.pumping.sort_by(
        [("pumping_station_id", "ascending"), *_BOUND_SORT_KEYS]
    )
    line_bounds = _resolve_line_bounds(line_bounds, block_counts)

    artifacts.census = census
    artifacts.pumping_ids = pumping_ids
    artifacts.block_counts = block_counts
    artifacts.kept_bound_conflicts = kept
    artifacts.availability_values = availability_values
    artifacts.hydro_bounds = hydro_bounds
    artifacts.thermal_bounds_table = thermal_bounds_table
    artifacts.pumping_bounds_table = pumping_bounds_table
    artifacts.line_bounds = line_bounds


def _convert_constraints(artifacts: DecompCaseArtifacts, writer: CaseWriter) -> None:
    """The ``step("Converting constraints")`` phase: overlay Itaipu's ``RI``
    frequency limits on the per-group availability, resolve the per-hydro
    ``group_bounds``, and build the ``contract_bounds_table`` via
    :func:`_rejoin_contract_price`. Stores ``group_bounds``,
    ``contract_bounds_table``, ``energy_contracts_doc``, and ``contracts``
    onto *artifacts* for :func:`_emit_and_write`.
    """
    case = artifacts.case
    dadger = artifacts.dadger
    calendar = artifacts.calendar
    id_map = artifacts.id_map
    block_counts = artifacts.block_counts
    assert block_counts is not None
    census = artifacts.census
    assert census is not None
    hydros_dict = artifacts.hydros_dict
    assert hydros_dict is not None
    availability_values = artifacts.availability_values
    assert availability_values is not None
    kept = artifacts.kept_bound_conflicts
    assert kept is not None
    hydro_names = {hydro["id"]: hydro["name"] for hydro in hydros_dict["hydros"]}

    group_values = hydro_conv.itaipu_frequency_group_values(
        case, id_map, availability_values
    )
    group_bounds = group_bounds_conv.convert_hydro_unit_group_bounds(
        case, values=group_values
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
    frequency_contribs = single_term_bounds.split_plant_group_contributions(
        case,
        id_map,
        census=census,
        group_capacities={
            (hydro["id"], group["id"]): group["max_generation_mw"]
            for hydro in hydros_dict["hydros"]
            for group in hydro["unit_groups"]
        },
        plant_names=hydro_names,
    )
    resolved_group_entries: list[
        tuple[tuple[int, int, int, int | None], bounds_accumulator.ResolvedRow]
    ] = []
    for hydro_id in sorted(group_rows_by_hydro.keys() | frequency_contribs.keys()):
        per_hydro_contribs = [
            *_row_group_contributions(
                group_rows_by_hydro[hydro_id],
                family="hydro_unit_group",
                id_column="hydro_unit_group_id",
                axes=("turbined", "generation"),
                contributor="group_bounds",
            ),
            *frequency_contribs.get(hydro_id, []),
        ]
        for resolved in _keep_soft_conflicts(
            bounds_accumulator.resolve(per_hydro_contribs, block_counts),
            lambda row, hydro_id=hydro_id: (
                id_map.hydro_codes[hydro_id],
                hydro_names[hydro_id],
                f"Unit group {row.entity_id} ",
            ),
            kept,
        ):
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
    # (rv0's single placeholder row is skipped, rv3 has none), so the
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
    # Contract power bounds route through the same BoundContribution ->
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
    contract_by_id = {contract.id: contract for contract in contracts}
    resolved_contract_power = _fan_resolved_rows(
        [
            ((row.entity_id, row.stage_id, row.block_id), row)
            for row in _resolve_or_refuse(
                contract_power_contribs,
                block_counts,
                lambda row: _contract_entity(contract_by_id[row.entity_id]),
                dadger=case.files.dadger,
            )
        ],
        schema=_CONTRACT_POWER_SCHEMA,
        key_columns=("contract_id", "stage_id", "block_id"),
    )
    contract_bounds_table = _rejoin_contract_price(
        resolved_contract_power, contract_bounds_table
    ).sort_by([("contract_id", "ascending"), *_BOUND_SORT_KEYS])
    if kept:
        dx.emit(_kept_conflicts_diagnostic(kept), logger=_LOG)

    artifacts.group_bounds = group_bounds
    artifacts.contract_bounds_table = contract_bounds_table
    artifacts.energy_contracts_doc = energy_contracts_doc
    artifacts.contracts = contracts


class _BoundTables(NamedTuple):
    """The resolved bound tables the write phase self-checks and writes."""

    hydro: pa.Table
    thermal: pa.Table
    pumping: pa.Table
    line: pa.Table
    group: pa.Table
    contract: pa.Table

    @classmethod
    def from_artifacts(cls, artifacts: DecompCaseArtifacts) -> _BoundTables:
        """The tables ``_resolve_bounds`` and ``_convert_constraints`` stored."""
        hydro = artifacts.hydro_bounds
        assert hydro is not None
        thermal = artifacts.thermal_bounds_table
        assert thermal is not None
        pumping = artifacts.pumping_bounds_table
        assert pumping is not None
        line = artifacts.line_bounds
        assert line is not None
        group = artifacts.group_bounds
        assert group is not None
        contract = artifacts.contract_bounds_table
        assert contract is not None
        return cls(hydro, thermal, pumping, line, group, contract)


def _run_emission_self_checks(
    bounds: _BoundTables, *, hydros_dict: dict, thermals_dict: dict, stages_dict: dict
) -> None:
    """Raise on an ``ERROR``-severity self-check finding, via
    ``emission_checks.run_and_gate``."""
    # Post-emission self-checks: mirror cheap novomodelo load invariants (rules 43,
    # 41, 45, 38, 36, and the block_id-range rule) over the in-memory artifacts
    # before the constraint tables are written. rule 43 is reachable because
    # hydro_bounds carries max_turbined_m3s/max_generation_mw whenever a
    # single-term special constraint (e.g. an RE FU generation ceiling) lowers
    # to one; it raises when such a bound exceeds the entity's own declared
    # capacity. See novomodelo_bridge.core.emission_checks for the rule scope.
    bound_families = [
        emission_checks.BoundFamily("Hydro", "hydro_id", bounds.hydro),
        emission_checks.BoundFamily("Thermal", "thermal_id", bounds.thermal),
        emission_checks.BoundFamily("Line", "line_id", bounds.line),
        emission_checks.BoundFamily(
            "HydroUnitGroup",
            "hydro_id",
            bounds.group,
            group_column="hydro_unit_group_id",
        ),
        emission_checks.BoundFamily("Contract", "contract_id", bounds.contract),
        emission_checks.BoundFamily("Pumping", "pumping_station_id", bounds.pumping),
    ]

    def _run_emission_checks() -> None:
        emission_checks.check_hydro_bounds_no_raising(hydros_dict, bounds.hydro)
        emission_checks.check_unit_group_envelope(hydros_dict)
        emission_checks.check_group_bound_envelope(hydros_dict, bounds.group)
        emission_checks.check_bound_row_uniqueness(bound_families)
        emission_checks.check_bound_block_id_range(stages_dict, bound_families)
        emission_checks.check_block_id_not_on_anticipated_thermal(
            thermals_dict, bounds.thermal
        )
        emission_checks.check_diversion_has_channel(hydros_dict, bounds.hydro)

    emission_checks.run_and_gate(_run_emission_checks)


def _write_bound_tables(
    writer: CaseWriter,
    bounds: _BoundTables,
    *,
    contracts: list[contracts_conv.Contract],
    energy_contracts_doc: dict,
) -> None:
    """Write the bound parquets, skipping an empty optional family, and the
    energy-contract registry when the deck declares contracts."""
    writer.write_parquet("constraints/thermal_bounds.parquet", bounds.thermal)
    writer.write_parquet("constraints/line_bounds.parquet", bounds.line)
    if bounds.hydro.num_rows:
        writer.write_parquet("constraints/hydro_bounds.parquet", bounds.hydro)
    if bounds.pumping.num_rows:
        writer.write_parquet("constraints/pumping_bounds.parquet", bounds.pumping)
    if bounds.group.num_rows:
        writer.write_parquet(
            "constraints/hydro_unit_group_bounds.parquet", bounds.group
        )
    if contracts:
        writer.write_json("system/energy_contracts.json", energy_contracts_doc)
    if bounds.contract.num_rows:
        writer.write_parquet("constraints/contract_bounds.parquet", bounds.contract)


class _SharedEmitterInputs(NamedTuple):
    """The inputs the generic-constraint emitters share, built once."""

    big_m: float
    line_map: dict[tuple[int, int], int]
    hydro_to_ree: dict[int, int]
    allocator: ConstraintIdAllocator


def _shared_emitter_inputs(
    dadger: Dadger, deficit_cost: float, lines_doc: dict
) -> _SharedEmitterInputs:
    """The big-M penalty, the FI line map, the operated hydro -> REE map, and a
    fresh 0-based constraint-id allocator.

    Raises
    ------
    ValueError
        If the deck has no ``UH`` records.
    """
    big_m = constraints_conv.big_m_penalty(deficit_cost)
    line_map = constraints_conv.build_fi_line_map(lines_doc["lines"])
    uh = as_frame(dadger.uh(df=True))
    if uh is None:
        raise ValueError("the deck has no UH records; cannot map hydros to REEs")
    operated_uh = plants.operated_uh(uh)
    hydro_to_ree = {
        int(row["codigo_usina"]): int(row["codigo_ree"])
        for _, row in operated_uh.iterrows()
    }
    return _SharedEmitterInputs(big_m, line_map, hydro_to_ree, ConstraintIdAllocator())


def _emit_special_constraint_generics(
    artifacts: DecompCaseArtifacts,
    shared: _SharedEmitterInputs,
    *,
    census: constraint_registers.ConstraintCensus,
    pumping_ids: dict[int, int],
) -> list[constraints_conv.DecompGenericResult]:
    """The RE, RHQ/RHV and RHE generic constraints, in that emitter order,
    leaving out an emitter with nothing to emit."""
    case = artifacts.case
    id_map = artifacts.id_map
    contracts = artifacts.contracts
    assert contracts is not None
    re_generics = constraints_conv.emit_re_generics(
        case,
        id_map,
        census=census,
        line_map=shared.line_map,
        contract_tokens=constraints_conv.build_contract_token_map(contracts),
        big_m=shared.big_m,
        allocator=shared.allocator,
    )
    rhq_rhv_generics = constraints_conv.emit_rhq_rhv_generics(
        case,
        id_map,
        census=census,
        pumping_station_ids=pumping_ids,
        effective=artifacts.effective,
        big_m=shared.big_m,
        allocator=shared.allocator,
    )
    rhe_generics = constraints_conv.emit_rhe_generics(
        case,
        id_map,
        census=census,
        effective=artifacts.effective,
        hydro_to_ree=shared.hydro_to_ree,
        ez_fractions=rhe_shares_conv.read_ez_fractions(artifacts.dadger),
        big_m=shared.big_m,
        allocator=shared.allocator,
    )
    return [
        result
        for result in (re_generics, rhq_rhv_generics, rhe_generics)
        if result is not None
    ]


def _emit_libs_electrical_generics(
    artifacts: DecompCaseArtifacts,
    model: libs_electrical_conv.LibsElectricalModel,
    shared: _SharedEmitterInputs,
    *,
    source: Path,
    availability_values: dict[tuple[int, int, int], group_bounds_conv.GroupBoundEntry],
    hydros_dict: dict,
) -> libs_electrical_emit.LibsElectricalResult:
    """Every long-form LIBs electrical restriction of *model*, read from
    *source*, that resolves, as a generic constraint."""
    case = artifacts.case
    id_map = artifacts.id_map
    context_factory = libs_electrical_conv.build_data_context(
        case, id_map, model=model, gnl_model=artifacts.gnl_model
    )
    a_h = libs_electrical_conv.build_available_power(
        id_map, overlay=availability_values
    )
    ncs_id_by_pee_code = ncs_conv.build_pee_ncs_id_map(case, id_map)
    conjh_bus_by_code_group = {
        (id_map.hydro_codes[entry["id"]], group_index): group["bus_id"]
        for entry in hydros_dict["hydros"]
        for group_index, group in enumerate(entry["unit_groups"], start=1)
    }
    return libs_electrical_emit.emit_libs_electrical_generics(
        model,
        id_map,
        context_factory,
        a_h,
        artifacts.calendar,
        ncs_id_by_pee_code,
        conjh_bus_by_code_group,
        shared.line_map,
        shared.big_m,
        shared.allocator,
        source=source,
    )


def _emit_itaipu_ande_generic(
    artifacts: DecompCaseArtifacts, shared: _SharedEmitterInputs
) -> GenericConstraintResult | None:
    """Itaipu's 50 Hz generation held at or above the `RI` ANDE load, or
    `None` when the deck carries no ANDE load."""
    if artifacts.ande_bus_load is None:
        return None
    bus_id, carga_ande = artifacts.ande_bus_load
    return constraints_conv.emit_itaipu_ande_generic(
        artifacts.id_map.hydro_id(hydro_conv._ITAIPU_CODE),
        bus_id,
        carga_ande,
        big_m=shared.big_m,
        allocator=shared.allocator,
    )


def _write_generic_constraints(
    writer: CaseWriter,
    id_map: DecompIdMap,
    special_generics: list[constraints_conv.DecompGenericResult],
    libs_electrical_result: libs_electrical_emit.LibsElectricalResult | None,
    ande_generic: GenericConstraintResult | None,
) -> None:
    """Write the generic constraints with their bounds, and the scalar
    parameters their expressions reference."""
    emitted: list[constraints_conv.DecompGenericResult | GenericConstraintResult] = [
        *special_generics
    ]
    if (
        libs_electrical_result is not None
        and libs_electrical_result.generic is not None
    ):
        emitted.append(libs_electrical_result.generic)
    if ande_generic is not None:
        emitted.append(ande_generic)
    generic_constraints = [
        constraint for result in emitted for constraint in result.constraints
    ]
    generic_bound_tables = [result.bounds for result in emitted]
    if generic_constraints:
        writer.write_json(
            "constraints/generic_constraints.json",
            {
                "$schema": novomodelo_schemas.schema_url_for(
                    "constraints/generic_constraints.json"
                ),
                "constraints": generic_constraints,
            },
        )
        writer.write_parquet(
            "constraints/generic_constraint_bounds.parquet",
            pa.concat_tables(generic_bound_tables),
        )

    # Every `@name` a surviving expression references must resolve, or novomodelo
    # fails to load the case.
    if any("@" in constraint["expression"] for constraint in generic_constraints):
        write_scalar_parameters(
            writer,
            build_decomp_scalar_parameters(
                [id_map.hydro_id(code) for code in id_map.hydro_codes],
                per_stage={
                    name: values
                    for result in special_generics
                    for name, values in result.per_stage_parameters.items()
                },
                expressions={
                    name: body
                    for result in special_generics
                    for name, body in result.expression_parameters.items()
                },
            ),
        )


def _emit_constraint_detection_diagnostics(
    case: DecompCase,
    libs_electrical_model: libs_electrical_conv.LibsElectricalModel | None,
    libs_electrical_result: libs_electrical_emit.LibsElectricalResult | None,
) -> None:
    """Report the electrical-constraint files the conversion cannot read, and
    the LIBs electrical census."""
    rha_detection = constraint_registers.detect_unconverted_rha(case.dadger)
    if rha_detection is not None:
        dx.emit(rha_detection, logger=_LOG)
    # The flat presence warning is narrowed to the subset
    # `libs_electrical_model` (read once by `_emit_and_write`) did NOT
    # convert -- a short-form-only (RE/RE-*, date-indexed) file, or no long-form card at
    # all despite the indices.csv entry. Once the long-form subset
    # converts, the census INFO below is authoritative and this flat warning
    # would otherwise be a stale "not converted" claim over that subset.
    libs_detection = constraint_registers.detect_libs_electrical(
        case.files.libs_restricao_eletrica
    )
    if libs_detection is not None and libs_electrical_model is None:
        dx.emit(libs_detection, logger=_LOG)
    if libs_electrical_result is not None:
        dx.emit(_libs_electrical_census_diagnostic(libs_electrical_result), logger=_LOG)


def _emit_and_write(
    artifacts: DecompCaseArtifacts, writer: CaseWriter
) -> ConversionReport:
    """The ``step("Writing outputs")`` phase: run the post-emission self-checks
    (``emission_checks.run_and_gate``) before any constraint parquet is
    written, write the bound parquets, emit the RE -> RHQ/RHV -> RHE ->
    LIBs-electrical -> RI ANDE generic constraints over one shared
    :class:`ConstraintIdAllocator`, write the generic-constraint artifacts and
    scalar parameters, emit the detection diagnostics, and return the
    :class:`~novomodelo_bridge.core.conversion.ConversionReport` built from
    ``writer.would_write``.
    """
    case = artifacts.case
    dadger = artifacts.dadger
    calendar = artifacts.calendar
    id_map = artifacts.id_map
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
    bounds = _BoundTables.from_artifacts(artifacts)
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

    _run_emission_self_checks(
        bounds,
        hydros_dict=hydros_dict,
        thermals_dict=thermals_dict,
        stages_dict=stages_dict,
    )
    _write_bound_tables(
        writer, bounds, contracts=contracts, energy_contracts_doc=energy_contracts_doc
    )

    # Emit every RE/RHQ/RHV/RHE special constraint that did NOT lower to an
    # entity bound (`census.to_generic`, resolved once in `_resolve_bounds`)
    # as a Novomodelo generic constraint, over one shared 0-based constraint-id
    # allocator so ids never collide across emitters regardless of which
    # family produced them. The emitters run in this fixed order;
    # `big_m`, `line_map`, and `hydro_to_ree` are shared inputs every emitter
    # needs but none of them reads the deck to build for itself.
    shared = _shared_emitter_inputs(dadger, deficit_cost, lines_doc)

    # Read the deck's LIBs-era electrical-constraint file (the LIBS index's
    # RESTRICAO-ELETRICA-ESPECIAL entry, resolved by discover_decomp_files)
    # once: both the narrowed detect_libs_electrical decision and the
    # generic-constraint emitter below thread this same `libs_electrical_model`,
    # so neither can disagree about whether the deck's long-form subset
    # converted. `read_libs_electrical` returns `None` for a short-form-only
    # (RE/RE-* or date-indexed) file -- that fallback stays on
    # detect_libs_electrical's flat warning.
    libs_electrical_model: libs_electrical_conv.LibsElectricalModel | None = None
    libs_file = case.files.libs_restricao_eletrica
    restricoes = case.libs_restricao_eletrica
    if libs_file is not None and restricoes is not None:
        unread = unread_lines_diagnostic(libs_file, restricoes.data)
        if unread is not None:
            dx.emit(unread, logger=_LOG)
        libs_electrical_model = libs_electrical_conv.read_libs_electrical(
            restricoes,
            calendar,
            source=libs_file,
            gnl_thermal_codes=_gnl_thermal_codes(artifacts.gnl_model, id_map),
        )

    special_generics = _emit_special_constraint_generics(
        artifacts, shared, census=census, pumping_ids=pumping_ids
    )

    # The 4th generic-constraint emitter link -- every
    # LIBs-era long-form electrical restriction that resolved
    # (`libs_electrical_model`, read once above), spliced after RHE so the
    # unconditional write block below picks it up automatically, identical in
    # shape to the three links above. Built only when the deck actually
    # carries a long-form model, since none of these inputs are otherwise
    # needed.
    libs_electrical_result: libs_electrical_emit.LibsElectricalResult | None = None
    if libs_electrical_model is not None and libs_file is not None:
        libs_electrical_result = _emit_libs_electrical_generics(
            artifacts,
            libs_electrical_model,
            shared,
            source=libs_file,
            availability_values=availability_values,
            hydros_dict=hydros_dict,
        )

    # Last, so the ids of the deck's own restrictions do not move.
    ande_generic = _emit_itaipu_ande_generic(artifacts, shared)
    _write_generic_constraints(
        writer, id_map, special_generics, libs_electrical_result, ande_generic
    )
    _emit_constraint_detection_diagnostics(
        case, libs_electrical_model, libs_electrical_result
    )

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
    boundary_fcf: bool = False,
    float_digits: int | None = DEFAULT_FLOAT_DIGITS,
) -> ConversionReport:
    """Convert one deck revision into a Novomodelo case directory.

    Runs under :func:`~novomodelo_bridge.core.conversion.run_conversion`,
    which refuses an occupied output and rolls back a failed run.

    Parameters
    ----------
    dry_run:
        When ``True``, run the full in-memory conversion but write nothing to
        *dst* (no files, no subdirectories). The would-write paths are still
        recorded in
        :attr:`~novomodelo_bridge.core.conversion.ConversionReport.would_write_paths`.
    fcf_inputs_out:
        Optional :class:`FcfInputs` out-parameter. When given, its
        ``config``/``initial_conditions`` fields are assigned the same
        in-memory dict objects this run wrote to ``config.json``/
        ``initial_conditions.json``, so a caller (the CLI's boundary-FCF
        step) can thread them into ``import_boundary_fcf`` without
        re-reading either file. ``None`` (the default) leaves it unused.
        Its ``coupling_auxiliaries`` field receives the auxiliary coupling
        plants the run added.
    boundary_fcf:
        When ``True`` and the deck has cut files, add the auxiliary coupling
        plants the boundary cuts need, for ``import_boundary_fcf`` to price.
    float_digits:
        Significant digits of the floats in the case's JSON files; ``None``
        keeps full precision.

    Returns
    -------
    ConversionReport
        Summary of what was converted: the five entity/stage counts, a
        ``diagnostics`` list of every structured finding, and a ``warnings``
        list of the WARNING-severity summaries.

    Raises
    ------
    ~novomodelo_bridge.core.errors.EmissionCheckError
        A ``ValueError``: a post-emission self-check in
        :mod:`novomodelo_bridge.core.emission_checks` found an ``ERROR``-severity
        violation in the converted artifacts. No report is returned.
    ~novomodelo_bridge.core.errors.OutputExistsError
        A ``FileExistsError``: *dst* already holds files and ``force`` is not
        set.
    """
    return run_conversion(
        lambda: _convert_decomp_case_impl(
            src,
            dst,
            on_phase=on_phase,
            dry_run=dry_run,
            fcf_inputs_out=fcf_inputs_out,
            boundary_fcf=boundary_fcf,
            float_digits=float_digits,
        ),
        dst,
        DECOMP_CLEARED_ARTIFACTS,
        force=force,
        dry_run=dry_run,
    )


def _convert_decomp_case_impl(
    src: Path,
    dst: Path,
    *,
    on_phase: Callable[[str], None] | None = None,
    dry_run: bool = False,
    fcf_inputs_out: FcfInputs | None = None,
    boundary_fcf: bool = False,
    float_digits: int | None = DEFAULT_FLOAT_DIGITS,
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
    artifacts = _discover(src, boundary_fcf=boundary_fcf)

    # Every output path is routed through one ``CaseWriter`` so the
    # would-write listing, the byte format, and the dry-run gate live in
    # exactly one place, rather than guarding ~24 individual write sites.
    writer = CaseWriter(dst, dry_run=dry_run, float_digits=float_digits)

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
        fcf_inputs_out.coupling_auxiliaries = artifacts.coupling_auxiliaries
    return report

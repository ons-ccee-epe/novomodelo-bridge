"""Per-entity bound contribution model and the per-family cobre axis registry.

Single-term special constraints and the existing minimum-outflow/thermal/head
folds all land on the **same** per-entity bound parquet rows. cobre rejects two
rows setting the same column for the same ``(entity, stage, block)`` and
*replaces* (not merges) a base row with a per-block override, so the bridge
must merge every contributor per axis into one consistent row set itself. This
module lays the accumulator's foundation: a typed contribution record
(:class:`BoundContribution`) and the static registry (:data:`AXES`) of which
cobre bound axes exist per entity family — ``hydro``, ``thermal``,
``pumping``, ``line``, ``hydro_unit_group``, and ``contract`` — which are
two-sided vs upper-only/lower-only, and which are block-eligible vs
stage-level, and the per-group
:func:`intersect` primitive that reduces a homogeneous group of
contributions to one lower/upper pair.

This module also resolves the grouped contributions into the two emit
shapes cobre's replace-not-merge semantics require: :func:`resolve` groups
by ``(family, entity_id, stage_id, axis)`` and, per group, either intersects
the base contributions into one all-blocks row or fully materializes one
row per block (base contributions carried into every block) — never both.
:func:`resolve` is generic across every registered family.

Finally, :func:`build_bound_tables` fans the resolved rows out into the
three widened parquet tables cobre reads (``hydro_bounds``,
``thermal_bounds``, ``pumping_bounds``): every axis resolved for the same
``(entity, stage, block)`` cell lands in that cell's **one** row, across
different columns, because cobre rejects two rows for the same cell.
``line``, ``hydro_unit_group``, and ``contract`` rows resolve through the
same :func:`resolve` primitive but fan into their own family-specific
tables elsewhere — :func:`build_bound_tables` knows only the three families
above, so handing it a resolved row from another family silently drops
that row from every output table.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import pyarrow as pa

from cobre_bridge.core.generic_constraint_builder import is_bounded
from cobre_bridge.core.tolerances import relative_tolerance

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class BoundContribution:
    """One contributor's proposed lower/upper bound on one entity/axis/slot.

    ``family`` is one of the family strings :data:`AXES` keys on (``"hydro"``,
    ``"thermal"``, ``"pumping"``, ``"line"``, ``"hydro_unit_group"``,
    ``"contract"``); ``entity_id`` is the
    cobre 0-based id; ``block_id`` is ``None`` for a base/all-blocks row or the
    0-based block index for a per-block override; ``axis`` names the cobre
    bound axis (a key alongside ``family`` into :data:`AXES`); ``contributor``
    is a diagnostic label identifying the source (e.g. ``"RQ"``, ``"RE_12"``,
    ``"HQ_164"``) for conflict reporting.
    """

    family: str
    entity_id: int
    stage_id: int
    block_id: int | None
    axis: str
    lower: float | None
    upper: float | None
    contributor: str


@dataclass(frozen=True)
class AxisSpec:
    """Static description of one cobre bound axis for one entity family.

    ``lower_column``/``upper_column`` are the exact cobre parquet column names
    for that side, or ``None`` when the axis has no bound on that side — a
    one-sided axis, e.g. ``line``'s ``direct``/``reverse`` (upper-only) or
    ``hydro``'s ``water_withdrawal`` (lower-only); :func:`intersect` enforces
    the one-sided guard for every axis registered that way.
    ``block_eligible`` is ``False`` for axes cobre rejects a ``block_id`` on
    (stage-level only, e.g. storage and water_withdrawal).
    """

    family: str
    lower_column: str | None
    upper_column: str | None
    block_eligible: bool


#: The cobre bound axes this accumulator knows about, keyed by
#: ``(family, axis)``, across ``hydro``, ``thermal``, ``pumping``, ``line``,
#: ``hydro_unit_group``, and ``contract``. Covers exactly the verified cobre
#: axes — do not invent an axis cobre lacks (e.g. a single net-exchange axis
#: for ``line``: cobre has no such column, only the two directional
#: ``direct_mw``/``reverse_mw`` capacities registered below). A price/cost
#: column (``contract``'s ``price_per_mwh``, ``thermal``'s
#: ``cost_per_mwh``) is never registered here — it is not a bound, so it
#: rides its own side-table outside this accumulator.
AXES: Mapping[tuple[str, str], AxisSpec] = {
    ("hydro", "turbined"): AxisSpec(
        family="hydro",
        lower_column="min_turbined_m3s",
        upper_column="max_turbined_m3s",
        block_eligible=True,
    ),
    ("hydro", "outflow"): AxisSpec(
        family="hydro",
        lower_column="min_outflow_m3s",
        upper_column="max_outflow_m3s",
        block_eligible=True,
    ),
    ("hydro", "generation"): AxisSpec(
        family="hydro",
        lower_column="min_generation_mw",
        upper_column="max_generation_mw",
        block_eligible=True,
    ),
    ("hydro", "storage"): AxisSpec(
        family="hydro",
        lower_column="min_storage_hm3",
        upper_column="max_storage_hm3",
        block_eligible=False,
    ),
    ("hydro", "diversion"): AxisSpec(
        family="hydro",
        lower_column="min_diversion_m3s",
        upper_column="max_diversion_m3s",
        block_eligible=True,
    ),
    ("hydro", "spillage"): AxisSpec(
        family="hydro",
        lower_column="min_spillage_m3s",
        upper_column="max_spillage_m3s",
        block_eligible=True,
    ),
    # The `TI` irrigation-withdrawal register (`convert_irrigation_withdrawal`)
    # supplies a compulsory removal, not a movable range — modeled lower-only
    # (like `outflow`'s RQ/VAZMIN floor), never upper.
    # `water_withdrawal_m3s` is not a column of `HYDRO_BOUNDS_SCHEMA` (mirrors
    # `thermal`'s `cost_per_mwh`): it rides its own side-table joined onto
    # `hydro_bounds` outside `build_bound_tables`, whose hydro fan-out covers
    # only the axes `HYDRO_BOUNDS_SCHEMA` declares.
    ("hydro", "water_withdrawal"): AxisSpec(
        family="hydro",
        lower_column="water_withdrawal_m3s",
        upper_column=None,
        block_eligible=False,
    ),
    ("thermal", "generation"): AxisSpec(
        family="thermal",
        lower_column="min_generation_mw",
        upper_column="max_generation_mw",
        block_eligible=True,
    ),
    ("pumping", "flow"): AxisSpec(
        family="pumping",
        lower_column="min_m3s",
        upper_column="max_m3s",
        block_eligible=True,
    ),
    # `convert_lines`/`decomp.converters.network.convert_lines` emit one capacity per
    # flow direction, not a min/max pair on one column — each direction is
    # its own upper-only axis (a lower contribution on either loud-fails).
    ("line", "direct"): AxisSpec(
        family="line",
        lower_column=None,
        upper_column="direct_mw",
        block_eligible=True,
    ),
    ("line", "reverse"): AxisSpec(
        family="line",
        lower_column=None,
        upper_column="reverse_mw",
        block_eligible=True,
    ),
    # `group_bounds.convert_hydro_unit_group_bounds` emits two two-sided axes
    # (mirroring hydro's own `turbined`/`generation` naming, scoped to this
    # family so the two never collide in `AXES`).
    ("hydro_unit_group", "turbined"): AxisSpec(
        family="hydro_unit_group",
        lower_column="min_turbined_m3s",
        upper_column="max_turbined_m3s",
        block_eligible=True,
    ),
    ("hydro_unit_group", "generation"): AxisSpec(
        family="hydro_unit_group",
        lower_column="min_generation_mw",
        upper_column="max_generation_mw",
        block_eligible=True,
    ),
    # `contracts.convert_contract_bounds` also emits a `price_per_mwh` column
    # alongside `min_mw`/`max_mw` — a signed price, not a bound, so (per the
    # module comment above) it is never registered here.
    ("contract", "power"): AxisSpec(
        family="contract",
        lower_column="min_mw",
        upper_column="max_mw",
        block_eligible=True,
    ),
}


def axis_spec(family: str, axis: str) -> AxisSpec:
    """The registered :class:`AxisSpec` for ``(family, axis)``.

    Raises
    ------
    ValueError
        When ``(family, axis)`` is not a registered cobre bound axis; the
        message names the unknown pair.
    """
    try:
        return AXES[(family, axis)]
    except KeyError as err:
        raise ValueError(
            f"unregistered bound axis: family={family!r}, axis={axis!r}"
        ) from err


def _effective(value: float | None) -> float | None:
    """*value* as an effective bound, or ``None`` for "no bound on that side".

    ``None`` and a magnitude at or past
    :data:`~cobre_bridge.core.generic_constraint_builder.UNBOUNDED` both mean
    unbounded (per :func:`is_bounded`). A genuine ``0.0`` is a real bound
    (e.g. a zeroed pumping minimum) and passes through unchanged.
    """
    return value if is_bounded(value) else None


def intersect(
    contribs: Sequence[BoundContribution], spec: AxisSpec
) -> tuple[float | None, float | None]:
    """Intersect a homogeneous group of contributions into one bound pair.

    ``contribs`` must all share one full key (family, entity, stage, block,
    axis) — this is a pure per-group reduction, not a grouping primitive.
    Returns ``(max of the effective lowers, min of
    the effective uppers)``, each ``None`` when no contribution bounds that
    side.

    Raises
    ------
    ValueError
        When ``spec.lower_column`` is ``None`` (an upper-only axis, e.g.
        ``line``'s ``direct``/``reverse``) and some contribution supplies an
        effective lower, or symmetrically when ``spec.upper_column`` is
        ``None`` (a lower-only axis, e.g. ``hydro``'s ``water_withdrawal``)
        and some contribution supplies an effective upper — a bound cannot
        land on the side a one-sided axis has no column for; the message
        names the axis, the missing side, and the offending contributors.
        Also when an effective lower and an effective upper both survive and
        the lower exceeds the upper past float noise — the message names
        every contributing label and the conflicting values.
    """
    lowers: list[tuple[float, str]] = []
    uppers: list[tuple[float, str]] = []
    for contrib in contribs:
        eff_lower = _effective(contrib.lower)
        eff_upper = _effective(contrib.upper)
        if eff_lower is not None:
            lowers.append((eff_lower, contrib.contributor))
        if eff_upper is not None:
            uppers.append((eff_upper, contrib.contributor))

    if spec.lower_column is None and lowers:
        offenders = ", ".join(f"{name}={value}" for value, name in lowers)
        raise ValueError(
            f"{spec.family} axis has no lower_column (upper-only) but "
            f"{offenders} supplied a lower bound"
        )

    if spec.upper_column is None and uppers:
        offenders = ", ".join(f"{name}={value}" for value, name in uppers)
        raise ValueError(
            f"{spec.family} axis has no upper_column (lower-only) but "
            f"{offenders} supplied an upper bound"
        )

    lower = max((value for value, _ in lowers), default=None)
    upper = min((value for value, _ in uppers), default=None)

    if lower is not None and upper is not None:
        tolerance = relative_tolerance(upper)
        if lower - upper > tolerance:
            lower_desc = ", ".join(f"{name}={value}" for value, name in lowers)
            upper_desc = ", ".join(f"{name}={value}" for value, name in uppers)
            raise ValueError(
                f"empty intersection on {spec.family} axis: lower {lower} "
                f"(from {lower_desc}) exceeds upper {upper} "
                f"(from {upper_desc})"
            )

    return lower, upper


@dataclass(frozen=True)
class ResolvedRow:
    """One materialized bound row, resolved from a group of contributions.

    ``block_id`` is ``None`` for an all-blocks (base) row, or the 0-based
    block index for a materialized per-block row. :func:`resolve` never
    emits both a base row and per-block rows for the same ``(family,
    entity_id, stage_id, axis)`` — see its docstring for why.
    """

    family: str
    entity_id: int
    stage_id: int
    block_id: int | None
    axis: str
    lower: float | None
    upper: float | None


def resolve(
    contribs: Sequence[BoundContribution], block_counts: Mapping[int, int]
) -> list[ResolvedRow]:
    """Group *contribs* and resolve each group to cobre's two emit shapes.

    Groups ``contribs`` by ``(family, entity_id, stage_id, axis)``. Per
    group, looks up the :class:`AxisSpec` and then either:

    - **fully materializes per-block**, when any contribution in the group
      carries a non-``None`` ``block_id``: for every ``b in
      range(block_counts[stage_id])``, :func:`intersect` is called on the
      group's base contributions (``block_id is None``) *together with*
      block ``b``'s contributions, emitting one :class:`ResolvedRow` per
      block and no base row. This is the only way to honour cobre's
      replace-not-merge column semantics without silently dropping a base
      contributor's bound on the blocks nobody overrides; or
    - **stays all-base**, when every contribution is base granularity:
      :func:`intersect` is called once over the whole group, emitting one
      ``block_id=None`` row.

    A group whose resolved intersection is ``(None, None)`` (no effective
    bound on either side) is skipped and contributes no row.

    Raises
    ------
    ValueError
        When ``axis_spec(family, axis).block_eligible`` is ``False`` (a
        stage-level axis, e.g. storage) but some contribution in the group
        carries a non-``None`` ``block_id`` — the message names the axis,
        the entity, and the stage. Also when a group needs materialization
        but ``block_counts`` has no entry for its ``stage_id``. ``intersect``'s
        own raises (upper-only violation, empty intersection) propagate
        unchanged.
    """
    groups: defaultdict[tuple[str, int, int, str], list[BoundContribution]] = (
        defaultdict(list)
    )
    for contrib in contribs:
        key = (contrib.family, contrib.entity_id, contrib.stage_id, contrib.axis)
        groups[key].append(contrib)

    rows: list[ResolvedRow] = []
    for (family, entity_id, stage_id, axis), group in groups.items():
        spec = axis_spec(family, axis)
        block_ids = sorted({c.block_id for c in group if c.block_id is not None})

        if not spec.block_eligible and block_ids:
            raise ValueError(
                f"stage-level axis {family}.{axis} rejects a block_id, but "
                f"entity_id={entity_id} at stage_id={stage_id} has "
                f"block_id(s) {block_ids}"
            )

        base_contribs = [c for c in group if c.block_id is None]

        if not block_ids:
            lower, upper = intersect(base_contribs, spec)
            if lower is None and upper is None:
                continue
            rows.append(
                ResolvedRow(
                    family=family,
                    entity_id=entity_id,
                    stage_id=stage_id,
                    block_id=None,
                    axis=axis,
                    lower=lower,
                    upper=upper,
                )
            )
            continue

        if stage_id not in block_counts:
            raise ValueError(
                f"no block_counts entry for stage_id={stage_id}, needed to "
                f"materialize {family}.{axis} for entity_id={entity_id}"
            )
        n_blocks = block_counts[stage_id]
        # A contribution carrying a block_id outside the stage's declared block
        # range matches no materialization iteration, so it would vanish with no
        # row. Raise instead of silently dropping the bound (the same fail-loud
        # contract as the stage-level and missing-block-count guards above).
        out_of_range = sorted(b for b in block_ids if not 0 <= b < n_blocks)
        if out_of_range:
            raise ValueError(
                f"block_id(s) {out_of_range} out of range [0, {n_blocks}) for "
                f"{family}.{axis} at entity_id={entity_id}, stage_id={stage_id}; "
                f"a contribution would otherwise be silently dropped"
            )
        for block_id in range(n_blocks):
            block_contribs = [c for c in group if c.block_id == block_id]
            lower, upper = intersect(base_contribs + block_contribs, spec)
            if lower is None and upper is None:
                continue
            rows.append(
                ResolvedRow(
                    family=family,
                    entity_id=entity_id,
                    stage_id=stage_id,
                    block_id=block_id,
                    axis=axis,
                    lower=lower,
                    upper=upper,
                )
            )

    return rows


#: The three cobre bound-table schemas this accumulator fans resolved rows
#: into. Every non-key column is float64-nullable: a cell that resolves no
#: contribution on some axis simply carries ``null`` there, distinct from a
#: genuine ``0.0`` bound. ``HYDRO_BOUNDS_SCHEMA`` covers every hydro axis in
#: :data:`AXES`, including the two-sided diversion and spillage axes that
#: cobre's generic-constraint-authoring support added.
HYDRO_BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("min_outflow_m3s", pa.float64(), nullable=True),
        pa.field("max_outflow_m3s", pa.float64(), nullable=True),
        pa.field("min_turbined_m3s", pa.float64(), nullable=True),
        pa.field("max_turbined_m3s", pa.float64(), nullable=True),
        pa.field("min_generation_mw", pa.float64(), nullable=True),
        pa.field("max_generation_mw", pa.float64(), nullable=True),
        pa.field("min_storage_hm3", pa.float64(), nullable=True),
        pa.field("max_storage_hm3", pa.float64(), nullable=True),
        pa.field("min_diversion_m3s", pa.float64(), nullable=True),
        pa.field("max_diversion_m3s", pa.float64(), nullable=True),
        pa.field("min_spillage_m3s", pa.float64(), nullable=True),
        pa.field("max_spillage_m3s", pa.float64(), nullable=True),
    ]
)

THERMAL_BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("thermal_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("min_generation_mw", pa.float64(), nullable=True),
        pa.field("max_generation_mw", pa.float64(), nullable=True),
    ]
)

PUMPING_BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("pumping_station_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("min_m3s", pa.float64(), nullable=True),
        pa.field("max_m3s", pa.float64(), nullable=True),
    ]
)

#: Per-family schema and id-column name, keyed by the same ``family`` string
#: :class:`BoundContribution`/:class:`ResolvedRow` carry.
_FAMILY_SCHEMAS: Mapping[str, pa.Schema] = {
    "hydro": HYDRO_BOUNDS_SCHEMA,
    "thermal": THERMAL_BOUNDS_SCHEMA,
    "pumping": PUMPING_BOUNDS_SCHEMA,
}

_FAMILY_ID_COLUMNS: Mapping[str, str] = {
    "hydro": "hydro_id",
    "thermal": "thermal_id",
    "pumping": "pumping_station_id",
}


class BoundTables(NamedTuple):
    """The three fanned-out cobre bound tables :func:`build_bound_tables` emits."""

    hydro: pa.Table
    thermal: pa.Table
    pumping: pa.Table


def _empty(schema: pa.Schema) -> pa.Table:
    """A 0-row table honouring *schema* exactly (mirrors
    ``decomp/converters/bounds.py``)."""
    return pa.table(
        {field.name: pa.array([], type=field.type) for field in schema},
        schema=schema,
    )


def build_bound_tables(rows: Sequence[ResolvedRow]) -> BoundTables:
    """Fan *rows* out into the three per-family cobre bound tables.

    Groups ``rows`` by cell key ``(family, entity_id, stage_id, block_id)``.
    Every :class:`ResolvedRow` sharing a cell contributes its axis's
    ``lower``/``upper`` into that cell's *one* row, via
    :func:`axis_spec`'s ``lower_column``/``upper_column``; every axis column
    the cell's rows never touch stays ``None``. This is the fan-out that
    keeps two axes on the same ``(entity, stage, block)`` — e.g. an outflow
    bound and a turbined bound on the same plant/stage/block — from ever
    becoming two parquet rows, which cobre rejects as a duplicate cell.

    A family with no rows returns a 0-row table with the correct schema
    (see :func:`_empty`).

    Raises
    ------
    ValueError
        Propagated from :func:`axis_spec` when a row names an unregistered
        ``(family, axis)`` pair.
    """
    cells: dict[tuple[str, int, int, int | None], dict[str, float]] = {}
    for row in rows:
        key = (row.family, row.entity_id, row.stage_id, row.block_id)
        cell = cells.setdefault(key, {})
        spec = axis_spec(row.family, row.axis)
        if spec.lower_column is not None and row.lower is not None:
            cell[spec.lower_column] = row.lower
        if spec.upper_column is not None and row.upper is not None:
            cell[spec.upper_column] = row.upper

    per_family: defaultdict[
        str, list[tuple[tuple[str, int, int, int | None], dict[str, float]]]
    ] = defaultdict(list)
    for key, cell in cells.items():
        per_family[key[0]].append((key, cell))

    tables: dict[str, pa.Table] = {}
    for family, schema in _FAMILY_SCHEMAS.items():
        items = per_family[family]
        if not items:
            tables[family] = _empty(schema)
            continue

        id_column = _FAMILY_ID_COLUMNS[family]
        columns: dict[str, list[object]] = {field.name: [] for field in schema}
        for (_, entity_id, stage_id, block_id), cell in items:
            columns[id_column].append(entity_id)
            columns["stage_id"].append(stage_id)
            columns["block_id"].append(block_id)
            for field in schema:
                if field.name in (id_column, "stage_id", "block_id"):
                    continue
                columns[field.name].append(cell.get(field.name))
        tables[family] = pa.table(
            {
                name: pa.array(values, type=schema.field(name).type)
                for name, values in columns.items()
            },
            schema=schema,
        )

    return BoundTables(
        hydro=tables["hydro"], thermal=tables["thermal"], pumping=tables["pumping"]
    )

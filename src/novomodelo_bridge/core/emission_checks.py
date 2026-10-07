"""Post-emission self-checks mirroring cobre 0.13's new hard validation rules.

Both pipelines build every artifact in memory before writing the Cobre case
directory. This module inspects those in-memory artifacts — the ``hydros.json``
document, the ``thermals.json`` document, the ``stages.json`` document, and the
``*_bounds`` Parquet tables — for the cobre 0.13 rules the bridge is most
likely to violate silently, and reports a match as an ``ERROR``
:class:`~cobre_bridge.core.diagnostics.Diagnostic` through the sink the pipeline
already runs converters inside. A failure here therefore surfaces with
bridge-side context (entity, stage, column, declared vs. offending value) in
milliseconds, instead of at ``cobre validate``/``cobre run`` load time.

This is a **courtesy mirror**, not a substitute: cobre remains the authority.
Each rule is scoped to match cobre's own implementation exactly (cobre's
semantic validation layer):

- Rule 43 — :func:`check_hydro_bounds_no_raising`
  (``block_bounds.rs::check_bound_raises_declared_capacity``).
- Rule 41 — :func:`check_unit_group_envelope`
  (``hydro.rs::check_hydro_unit_groups``, ceiling half only; the floor half,
  rule 44, is out of scope here).
- Rule 45 — :func:`check_group_bound_envelope`
  (``block_bounds.rs::check_group_bound_raises_declared_capacity``) — the
  group-scoped counterpart to rule 43: no ``hydro_unit_group_bounds`` row may
  raise a *group's own* declared max above what ``hydros.json``'s
  ``unit_groups[]`` entry for that ``(hydro_id, hydro_unit_group_id)``
  declares.
- Rule 36 — :func:`check_bound_row_uniqueness`
  (``block_bounds.rs::check_duplicate_bound_rows``).
- The ``block_id``-range rule — :func:`check_bound_block_id_range`
  (``block_bounds.rs::check_bound_block_id_range``).
- Rule 38 — :func:`check_block_id_not_on_anticipated_thermal`
  (``block_bounds.rs::check_block_id_on_anticipated_thermal``).

This module must not import either pipeline — they import it — so it stays
unit-testable against hand-built artifacts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import pyarrow as pa

from cobre_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    collect,
    emit,
    format_stage_ranges,
)
from cobre_bridge.core.tolerances import relative_tolerance

_LOG = logging.getLogger(__name__)

_CATEGORY = "Emission self-checks"

#: Both guarded columns for rule 43 / rule 41, checked independently.
_ENVELOPE_COLUMNS: tuple[str, ...] = ("max_turbined_m3s", "max_generation_mw")

#: Per-stage ``hydro_bounds`` MAX columns clamped into the plant's declared
#: envelope, each as ``(bounds column, hydros block, declared-min key,
#: declared-max key)``. See :func:`clamp_hydro_bounds_to_declared`.
_CLAMP_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    ("max_turbined_m3s", "generation", "min_turbined_m3s", "max_turbined_m3s"),
    ("max_generation_mw", "generation", "min_generation_mw", "max_generation_mw"),
    ("max_outflow_m3s", "outflow", "min_outflow_m3s", "max_outflow_m3s"),
)


class EmissionCheckError(ValueError):
    """A post-emission self-check reported at least one ERROR-severity diagnostic."""


def _tolerance(declared: float) -> float:
    """``ENVELOPE_TOLERANCE * max(|declared|, 1.0)`` — cobre's envelope tolerance."""
    return relative_tolerance(declared)


@dataclass(frozen=True)
class BoundFamily:
    """One emitted bound-override table plus the metadata the row-level checks
    (:func:`check_bound_row_uniqueness`, :func:`check_bound_block_id_range`) need
    to name it in a finding.

    *table* is ``None`` when the pipeline did not build that family for this
    case (e.g. a source-model case with no thermal capacity overrides); the row-level
    checks skip an absent family without a finding, exactly as they would skip
    an empty one.

    *group_column* names an optional secondary key column (e.g.
    ``hydro_unit_group_id`` for the ``hydro_unit_group_bounds`` family) that,
    when set, both row-level checks fold into the **key** alongside
    *entity_column*/``stage_id``/``block_id`` instead of treating it as a
    value column — mirroring cobre rule 36's own widened key ``(hydro_id,
    hydro_unit_group_id, stage_id, block_id, column)``.
    ``None`` (the default) reproduces the three pre-existing families'
    unchanged, single-entity-key behaviour.
    """

    label: str
    entity_column: str
    table: pa.Table | None
    group_column: str | None = None


def _hydro_declared(hydros: Mapping[str, object]) -> dict[int, dict[str, float]]:
    """``{hydro_id: {column: declared_value}}`` from the ``hydros.json`` document.

    Only columns whose value is present and numeric are included, so a hydro
    entry with no ``generation`` block (or missing a column) simply has no
    entry for it — the callers treat that as "nothing to check", not a defect
    of this module.
    """
    declared: dict[int, dict[str, float]] = {}
    raw_hydros = hydros.get("hydros")
    if not isinstance(raw_hydros, list):
        return declared
    for hydro in raw_hydros:
        if not isinstance(hydro, Mapping):
            continue
        hydro_id = hydro.get("id")
        if not isinstance(hydro_id, int):
            continue
        generation = hydro.get("generation")
        if not isinstance(generation, Mapping):
            continue
        entry: dict[str, float] = {}
        for column in _ENVELOPE_COLUMNS:
            value = generation.get(column)
            if isinstance(value, int | float):
                entry[column] = float(value)
        if entry:
            declared[hydro_id] = entry
    return declared


def check_hydro_bounds_no_raising(
    hydros: Mapping[str, object],
    hydro_bounds: pa.Table | None,
) -> None:
    """Cobre rule 43 mirror: no ``hydro_bounds`` row may raise ``max_turbined_m3s``
    or ``max_generation_mw`` above the hydro's own declared value in
    ``hydros.json`` (``generation.max_turbined_m3s`` /
    ``generation.max_generation_mw``), each column checked independently.
    Lowering is fine; every other column is untouched by this rule.

    Structurally inapplicable when *hydro_bounds* is ``None`` or carries
    neither guarded column — DECOMP writes only ``min_outflow_m3s`` there, so
    its exposure is nil by construction. That is reported as an explicit INFO
    finding, never a silent pass.
    """
    columns = (
        [c for c in _ENVELOPE_COLUMNS if c in hydro_bounds.column_names]
        if hydro_bounds is not None
        else []
    )

    if not columns:
        emit(
            Diagnostic(
                code="hydro-bounds-raising-not-applicable",
                severity=Severity.INFO,
                category=_CATEGORY,
                title="Hydro-bounds raising check (rule 43) not applicable",
                summary=(
                    "hydro_bounds carries neither max_turbined_m3s nor "
                    "max_generation_mw for this case, so cobre rule 43 (no row "
                    "may raise a hydro's declared capacity) has nothing to scan"
                ),
            ),
            logger=_LOG,
        )
        return

    declared = _hydro_declared(hydros)
    hydro_ids = hydro_bounds["hydro_id"].to_pylist()
    stage_ids = hydro_bounds["stage_id"].to_pylist()
    column_values = {c: hydro_bounds[c].to_pylist() for c in columns}

    violations: dict[tuple[int, str], list[tuple[int, float]]] = {}
    for i, hydro_id in enumerate(hydro_ids):
        entry = declared.get(hydro_id)
        if entry is None:
            continue
        stage_id = stage_ids[i]
        for column in columns:
            value = column_values[column][i]
            if value is None:
                continue
            declared_value = entry.get(column)
            if declared_value is None:
                continue
            if value > declared_value + _tolerance(declared_value):
                violations.setdefault((hydro_id, column), []).append((stage_id, value))

    if not violations:
        return

    rows: list[list[object]] = []
    for (hydro_id, column), occurrences in sorted(violations.items()):
        declared_value = declared[hydro_id][column]
        worst = max(value for _, value in occurrences)
        rows.append(
            [
                hydro_id,
                column,
                format_stage_ranges(stage for stage, _ in occurrences),
                round(declared_value, 6),
                round(worst, 6),
            ]
        )

    emit(
        Diagnostic(
            code="hydro-bounds-raises-declared-capacity",
            severity=Severity.ERROR,
            category=_CATEGORY,
            title=f"hydro_bounds raises declared capacity ({len(rows)} finding(s))",
            summary=(
                f"{len(rows)} (hydro, column) combination(s) have a hydro_bounds "
                "row that raises max_turbined_m3s or max_generation_mw above the "
                "plant's own declared value in system/hydros.json (cobre rule 43)"
            ),
            table=DiagnosticTable(
                columns=["Hydro ID", "Column", "Stages", "Declared", "Worst offending"],
                rows=rows,
                justify=["right", "left", "left", "right", "right"],
            ),
            remediation=(
                "Declare the plant at its final (post-uprate) value instead and "
                "add hydro_bounds rows tightening the earlier stages down to "
                "their true value."
            ),
        ),
        logger=_LOG,
    )


def _hydro_declared_bounds(
    hydros: Mapping[str, object],
) -> dict[int, dict[str, tuple[float | None, float | None]]]:
    """``{hydro_id: {bounds_column: (declared_min, declared_max)}}`` for every
    column in :data:`_CLAMP_COLUMNS`.

    A bound is ``None`` when the plant declares no value for it (e.g. a
    run-of-river plant with ``outflow.max_outflow_m3s = null`` has no ceiling to
    clamp against). Columns with neither a min nor a max are omitted, so callers
    treat a missing entry as "no declared envelope for this variable".
    """
    declared: dict[int, dict[str, tuple[float | None, float | None]]] = {}
    raw_hydros = hydros.get("hydros")
    if not isinstance(raw_hydros, list):
        return declared
    for hydro in raw_hydros:
        if not isinstance(hydro, Mapping):
            continue
        hydro_id = hydro.get("id")
        if not isinstance(hydro_id, int):
            continue
        entry: dict[str, tuple[float | None, float | None]] = {}
        for column, block_name, min_key, max_key in _CLAMP_COLUMNS:
            block = hydro.get(block_name)
            if not isinstance(block, Mapping):
                continue
            raw_min = block.get(min_key)
            raw_max = block.get(max_key)
            declared_min = float(raw_min) if isinstance(raw_min, int | float) else None
            declared_max = float(raw_max) if isinstance(raw_max, int | float) else None
            if declared_min is not None or declared_max is not None:
                entry[column] = (declared_min, declared_max)
        if entry:
            declared[hydro_id] = entry
    return declared


def clamp_hydro_bounds_to_declared(
    hydros: Mapping[str, object],
    hydro_bounds: pa.Table | None,
) -> pa.Table | None:
    """Clamp every per-stage ``hydro_bounds`` MAX column into the owning plant's
    declared ``[min, max]`` envelope from ``hydros.json``, returning the clamped
    table.

    A per-stage override (TURBMAXT, VAZMAXT, an RE ceiling) can raise a MAX
    column above the plant's declaration, which cobre rejects on load (rule 43
    and its outflow analogue). Rather than raise the declaration to fit the
    override — which would let a bound *loosen* the declared capacity — this
    ceils each value at the declared max and floors it at the declared min, so
    the row stays a valid tightening; the declared value is the authority. A
    WARNING reports what was clamped. Returns *hydro_bounds* unchanged (same
    object) when it is ``None``, carries no clampable column, or nothing fell
    outside its envelope.
    """
    if hydro_bounds is None:
        return None
    present = [
        column for column, *_ in _CLAMP_COLUMNS if column in hydro_bounds.column_names
    ]
    if not present:
        return hydro_bounds
    declared = _hydro_declared_bounds(hydros)
    if not declared:
        return hydro_bounds

    hydro_ids = hydro_bounds["hydro_id"].to_pylist()
    stage_ids = hydro_bounds["stage_id"].to_pylist()
    column_values = {column: hydro_bounds[column].to_pylist() for column in present}

    # (hydro_id, column) -> [(stage, original, clamped, declared_bound), ...]
    clamped: dict[tuple[int, str], list[tuple[int, float, float, float]]] = {}
    for i, hydro_id in enumerate(hydro_ids):
        entry = declared.get(hydro_id)
        if entry is None:
            continue
        for column in present:
            envelope = entry.get(column)
            if envelope is None:
                continue
            declared_min, declared_max = envelope
            value = column_values[column][i]
            if value is None:
                continue
            new_value = value
            bound: float | None = None
            if declared_max is not None and value > declared_max + _tolerance(
                declared_max
            ):
                new_value = declared_max
                bound = declared_max
            elif declared_min is not None and value < declared_min - _tolerance(
                declared_min
            ):
                new_value = declared_min
                bound = declared_min
            if bound is not None:
                column_values[column][i] = new_value
                clamped.setdefault((hydro_id, column), []).append(
                    (stage_ids[i], value, new_value, bound)
                )

    if not clamped:
        return hydro_bounds

    clamped_columns = {column for _, column in clamped}
    table = hydro_bounds
    for column in clamped_columns:
        index = table.column_names.index(column)
        table = table.set_column(
            index, column, pa.array(column_values[column], type=pa.float64())
        )

    rows: list[list[object]] = []
    for (hydro_id, column), occurrences in sorted(clamped.items()):
        worst = max(occurrences, key=lambda occ: abs(occ[1] - occ[2]))
        rows.append(
            [
                hydro_id,
                column,
                format_stage_ranges(stage for stage, *_ in occurrences),
                round(worst[3], 6),
                round(worst[1], 6),
            ]
        )

    emit(
        Diagnostic(
            code="hydro-bounds-clamped-to-declared-capacity",
            severity=Severity.WARNING,
            category=_CATEGORY,
            title=f"hydro_bounds clamped to declared capacity ({len(rows)} finding(s))",
            summary=(
                f"{len(rows)} (hydro, column) combination(s) had a per-stage "
                "bound outside the plant's declared envelope in system/hydros.json "
                "(e.g. a TURBMAXT/VAZMAXT/RE override above the reference-head "
                "declaration); each was clamped to the declared bound so the row "
                "stays a valid tightening (cobre rule 43 and its outflow analogue)"
            ),
            table=DiagnosticTable(
                columns=["Hydro ID", "Column", "Stages", "Declared", "Clamped from"],
                rows=rows,
                justify=["right", "left", "left", "right", "right"],
            ),
            remediation=(
                "No action required — the bound was clamped to the plant's own "
                "declared value. If the higher value is intended, raise the "
                "plant's declared capacity in the source deck instead."
            ),
        ),
        logger=_LOG,
    )
    return table


def check_unit_group_envelope(hydros: Mapping[str, object]) -> None:
    """Cobre rule 41 mirror: the sum of a hydro's unit-group maxima must not
    exceed the hydro's own declared value in ``hydros.json``, each of
    ``max_turbined_m3s`` / ``max_generation_mw`` checked independently, against
    the entity declaration only — never a per-stage ``hydro_bounds`` override.

    Always structurally applicable: both pipelines declare a non-empty
    ``unit_groups`` on every hydro (cobre requires it), so there is no
    "nothing to scan" case to report here.
    """
    raw_hydros = hydros.get("hydros")
    if not isinstance(raw_hydros, list):
        return

    rows: list[list[object]] = []
    for hydro in raw_hydros:
        if not isinstance(hydro, Mapping):
            continue
        hydro_id = hydro.get("id")
        generation = hydro.get("generation")
        groups = hydro.get("unit_groups")
        if not isinstance(hydro_id, int) or not isinstance(generation, Mapping):
            continue
        if not isinstance(groups, list):
            continue

        for column in _ENVELOPE_COLUMNS:
            declared_value = generation.get(column)
            if not isinstance(declared_value, int | float):
                continue
            declared_value = float(declared_value)

            group_sum = 0.0
            for group in groups:
                if not isinstance(group, Mapping):
                    continue
                group_value = group.get(column)
                if isinstance(group_value, int | float):
                    group_sum += float(group_value)

            if group_sum > declared_value + _tolerance(declared_value):
                rows.append(
                    [hydro_id, column, round(declared_value, 6), round(group_sum, 6)]
                )

    if not rows:
        return

    rows.sort(key=lambda r: (r[0], r[1]))
    emit(
        Diagnostic(
            code="hydro-unit-group-envelope-exceeded",
            severity=Severity.ERROR,
            category=_CATEGORY,
            title=(
                f"Unit-group maxima exceed the plant envelope ({len(rows)} finding(s))"
            ),
            summary=(
                f"{len(rows)} hydro/column combination(s) have unit groups whose "
                "max_turbined_m3s or max_generation_mw sums above the plant's "
                "own declared value in system/hydros.json (cobre rule 41)"
            ),
            table=DiagnosticTable(
                columns=["Hydro ID", "Column", "Declared", "Group sum"],
                rows=rows,
                justify=["right", "left", "right", "right"],
            ),
            remediation=(
                "Lower the unit groups' maxima so their sum does not exceed the "
                "plant's own declared value; the plant value is the envelope."
            ),
        ),
        logger=_LOG,
    )


def _hydro_group_declared(
    hydros: Mapping[str, object],
) -> dict[tuple[int, int], dict[str, float]]:
    """``{(hydro_id, group_id): {column: declared_value}}`` from each hydro's
    declared ``unit_groups[]`` in the ``hydros.json`` document.

    Only columns whose value is present and numeric are included, mirroring
    :func:`_hydro_declared`'s own "nothing to check" treatment of a missing
    column.
    """
    declared: dict[tuple[int, int], dict[str, float]] = {}
    raw_hydros = hydros.get("hydros")
    if not isinstance(raw_hydros, list):
        return declared
    for hydro in raw_hydros:
        if not isinstance(hydro, Mapping):
            continue
        hydro_id = hydro.get("id")
        if not isinstance(hydro_id, int):
            continue
        groups = hydro.get("unit_groups")
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, Mapping):
                continue
            group_id = group.get("id")
            if not isinstance(group_id, int):
                continue
            entry: dict[str, float] = {}
            for column in _ENVELOPE_COLUMNS:
                value = group.get(column)
                if isinstance(value, int | float):
                    entry[column] = float(value)
            if entry:
                declared[(hydro_id, group_id)] = entry
    return declared


def check_group_bound_envelope(
    hydros: Mapping[str, object],
    group_bounds: pa.Table | None,
) -> None:
    """Cobre rule 45 mirror: no ``hydro_unit_group_bounds`` row may raise
    ``max_turbined_m3s`` or ``max_generation_mw`` above *that group's own*
    declared value in ``hydros.json`` (``unit_groups[].max_turbined_m3s`` /
    ``unit_groups[].max_generation_mw``), each column checked independently
    (``block_bounds.rs::check_group_bound_raises_declared_capacity``).
    Lowering is fine; the check is keyed by ``(hydro_id,
    hydro_unit_group_id)``, not by the plant's own envelope (that is rule 43,
    :func:`check_hydro_bounds_no_raising`).

    Structurally inapplicable, without a finding, when *group_bounds* is
    ``None``/empty or carries neither guarded column — unlike rule 43's mirror
    this is not reported as an explicit INFO finding, since (unlike
    ``hydro_bounds``, which DECOMP always writes with only
    ``min_outflow_m3s``) an absent/empty ``hydro_unit_group_bounds`` table is
    the ordinary, expected shape whenever no plant's availability actually
    needed an override.
    """
    if group_bounds is None or group_bounds.num_rows == 0:
        return
    columns = [c for c in _ENVELOPE_COLUMNS if c in group_bounds.column_names]
    if not columns:
        return

    declared = _hydro_group_declared(hydros)
    hydro_ids = group_bounds["hydro_id"].to_pylist()
    group_ids = group_bounds["hydro_unit_group_id"].to_pylist()
    stage_ids = group_bounds["stage_id"].to_pylist()
    column_values = {c: group_bounds[c].to_pylist() for c in columns}

    violations: dict[tuple[int, int, str], list[tuple[int, float]]] = {}
    for i, hydro_id in enumerate(hydro_ids):
        group_id = group_ids[i]
        entry = declared.get((hydro_id, group_id))
        if entry is None:
            continue
        stage_id = stage_ids[i]
        for column in columns:
            value = column_values[column][i]
            if value is None:
                continue
            declared_value = entry.get(column)
            if declared_value is None:
                continue
            if value > declared_value + _tolerance(declared_value):
                violations.setdefault((hydro_id, group_id, column), []).append(
                    (stage_id, value)
                )

    if not violations:
        return

    rows: list[list[object]] = []
    for (hydro_id, group_id, column), occurrences in sorted(violations.items()):
        declared_value = declared[(hydro_id, group_id)][column]
        worst = max(value for _, value in occurrences)
        rows.append(
            [
                hydro_id,
                group_id,
                column,
                format_stage_ranges(stage for stage, _ in occurrences),
                round(declared_value, 6),
                round(worst, 6),
            ]
        )

    emit(
        Diagnostic(
            code="hydro-group-bounds-raises-declared-capacity",
            severity=Severity.ERROR,
            category=_CATEGORY,
            title=(
                f"hydro_unit_group_bounds raises declared capacity "
                f"({len(rows)} finding(s))"
            ),
            summary=(
                f"{len(rows)} (hydro, group, column) combination(s) have a "
                "hydro_unit_group_bounds row that raises max_turbined_m3s or "
                "max_generation_mw above that group's own declared value in "
                "system/hydros.json (cobre rule 45)"
            ),
            table=DiagnosticTable(
                columns=[
                    "Hydro ID",
                    "Group ID",
                    "Column",
                    "Stages",
                    "Declared",
                    "Worst offending",
                ],
                rows=rows,
                justify=["right", "right", "left", "left", "right", "right"],
            ),
            remediation=(
                "Lower the overlay row(s) below the group's own declared "
                "value, or raise the group's declared value if it should be "
                "higher."
            ),
        ),
        logger=_LOG,
    )


class _RowUniquenessFinding(NamedTuple):
    """One duplicate-row finding, before the ``Group ID`` column's
    presence is known — decided only once every family has been scanned
    (see :func:`check_bound_row_uniqueness`)."""

    label: str
    entity_id: int
    group_id: int | None
    has_group: bool
    stage_id: int
    block_id: int | None
    column: str


def check_bound_row_uniqueness(families: Sequence[BoundFamily]) -> None:
    """Cobre rule 36 mirror: at most one row per ``(entity, [group,] stage,
    block, column)`` within a family, checked independently for every
    non-key column.

    A family's ``group_column`` (e.g. ``hydro_unit_group_id`` for
    ``hydro_unit_group_bounds``), when set, folds into the **key** rather
    than being treated as a value column — two groups of the same plant
    setting the same column at the same ``(stage, block)`` are distinct
    keys, not a collision, mirroring cobre rule 36's own widened key
    ``(hydro_id, hydro_unit_group_id, stage_id, block_id, column)``.
    A family with ``group_column=None`` (the
    three pre-existing families) is unaffected: the finding's rendered
    ``Group ID`` column reflects whether any *actual finding* came from a
    group-scoped family, not merely whether one was present in *families* —
    a group family registered alongside the others but contributing no
    finding of its own (e.g. still empty) leaves the other families'
    findings byte-identical to before this change, even when passed
    together in one call, exactly as the live pipeline does.

    ``block_id = None`` is a distinct key from ``Some(b)``: a stage-wide row
    and its per-block override rows are legal together — exactly what the
    ``line_bounds`` base-plus-per-block-override convention emits. Two rows
    setting *different* columns for the same ``(entity, [group,] stage,
    block)`` are also legal (sparse input), since the key is scoped per
    column.
    """
    raw_findings: list[_RowUniquenessFinding] = []
    for family in families:
        table = family.table
        if table is None or table.num_rows == 0:
            continue

        key_columns = {family.entity_column, "stage_id", "block_id"}
        if family.group_column is not None:
            key_columns.add(family.group_column)
        value_columns = [c for c in table.column_names if c not in key_columns]
        if not value_columns:
            continue

        entity_ids = table[family.entity_column].to_pylist()
        stage_ids = table["stage_id"].to_pylist()
        block_ids = (
            table["block_id"].to_pylist()
            if "block_id" in table.column_names
            else [None] * table.num_rows
        )
        group_ids = (
            table[family.group_column].to_pylist()
            if family.group_column is not None
            else [None] * table.num_rows
        )
        column_values = {c: table[c].to_pylist() for c in value_columns}

        seen: set[tuple[int, int | None, int, int | None, str]] = set()
        for i in range(table.num_rows):
            entity_id = entity_ids[i]
            stage_id = stage_ids[i]
            block_id = block_ids[i]
            group_id = group_ids[i]
            for column in value_columns:
                value = column_values[column][i]
                if value is None:
                    continue
                key = (entity_id, group_id, stage_id, block_id, column)
                if key in seen:
                    raw_findings.append(
                        _RowUniquenessFinding(
                            label=family.label,
                            entity_id=entity_id,
                            group_id=group_id,
                            has_group=family.group_column is not None,
                            stage_id=stage_id,
                            block_id=block_id,
                            column=column,
                        )
                    )
                    continue
                seen.add(key)

    if not raw_findings:
        return

    show_group = any(finding.has_group for finding in raw_findings)
    columns = ["Family", "Entity ID"]
    if show_group:
        columns.append("Group ID")
    columns.extend(["Stage", "Block", "Column"])

    findings: list[list[object]] = []
    for finding in raw_findings:
        row: list[object] = [finding.label, finding.entity_id]
        if show_group:
            row.append(finding.group_id if finding.group_id is not None else "-")
        row.extend(
            [
                finding.stage_id,
                finding.block_id if finding.block_id is not None else "stage-wide",
                finding.column,
            ]
        )
        findings.append(row)

    emit(
        Diagnostic(
            code="bound-row-duplicate",
            severity=Severity.ERROR,
            category=_CATEGORY,
            title=f"Duplicate bound rows ({len(findings)} finding(s))",
            summary=(
                f"{len(findings)} bound row(s) set the same column twice for the "
                "same (entity, stage, block) key (cobre rule 36)"
            ),
            table=DiagnosticTable(
                columns=columns,
                rows=findings,
            ),
            remediation=(
                "Remove the duplicate row or merge the overrides into a single row."
            ),
        ),
        logger=_LOG,
    )


def _stage_block_counts(stages: Mapping[str, object]) -> dict[int, int]:
    """``{stage_id: declared block count}`` from the ``stages.json`` document."""
    counts: dict[int, int] = {}
    raw_stages = stages.get("stages")
    if not isinstance(raw_stages, list):
        return counts
    for stage in raw_stages:
        if not isinstance(stage, Mapping):
            continue
        stage_id = stage.get("id")
        if not isinstance(stage_id, int):
            continue
        blocks = stage.get("blocks")
        counts[stage_id] = len(blocks) if isinstance(blocks, list) else 0
    return counts


class _BlockIdRangeFinding(NamedTuple):
    """One out-of-range finding, before the ``Group ID`` column's presence
    is known — decided only once every family has been scanned (see
    :func:`check_bound_block_id_range`)."""

    label: str
    entity_id: int
    group_id: int | None
    has_group: bool
    stage_id: int
    block_id: int
    declared_count: int


def check_bound_block_id_range(
    stages: Mapping[str, object],
    families: Sequence[BoundFamily],
) -> None:
    """``block_id``-range mirror (matrix row 18): every non-null ``block_id`` in
    a bound-override row must index a block the row's stage actually declares
    in ``stages.json``. A row referencing a ``stage_id`` that is not a declared
    study stage is skipped without a finding — that is a stage-axis defect this
    rule does not own.

    A family's ``group_column`` does not change the range check itself
    (``block_id`` is still validated only against the row's own stage) — it
    only enriches which row an out-of-range finding names: the rendered
    ``Group ID`` column appears only when an *actual finding* came from a
    group-scoped family (not merely because one was present in *families*),
    so a ``hydro_unit_group_bounds`` finding names its ``(hydro, group,
    stage, block_id)`` row while an unrelated family's own finding stays
    byte-identical to before this change, even in a mixed call.
    """
    stage_block_counts = _stage_block_counts(stages)

    raw_findings: list[_BlockIdRangeFinding] = []
    for family in families:
        table = family.table
        if table is None or table.num_rows == 0:
            continue
        if "block_id" not in table.column_names:
            continue

        entity_ids = table[family.entity_column].to_pylist()
        stage_ids = table["stage_id"].to_pylist()
        block_ids = table["block_id"].to_pylist()
        group_ids = (
            table[family.group_column].to_pylist()
            if family.group_column is not None
            else [None] * table.num_rows
        )
        for entity_id, group_id, stage_id, block_id in zip(
            entity_ids, group_ids, stage_ids, block_ids, strict=True
        ):
            if block_id is None:
                continue
            declared_count = stage_block_counts.get(stage_id)
            if declared_count is None:
                continue
            if block_id < 0 or block_id >= declared_count:
                raw_findings.append(
                    _BlockIdRangeFinding(
                        label=family.label,
                        entity_id=entity_id,
                        group_id=group_id,
                        has_group=family.group_column is not None,
                        stage_id=stage_id,
                        block_id=block_id,
                        declared_count=declared_count,
                    )
                )

    if not raw_findings:
        return

    show_group = any(finding.has_group for finding in raw_findings)
    columns = ["Family", "Entity ID"]
    justify = ["left", "right"]
    if show_group:
        columns.append("Group ID")
        justify.append("right")
    columns.extend(["Stage", "block_id", "Declared blocks"])
    justify.extend(["right", "right", "right"])

    findings: list[list[object]] = []
    for finding in raw_findings:
        row: list[object] = [finding.label, finding.entity_id]
        if show_group:
            row.append(finding.group_id if finding.group_id is not None else "-")
        row.extend([finding.stage_id, finding.block_id, finding.declared_count])
        findings.append(row)

    emit(
        Diagnostic(
            code="bound-block-id-out-of-range",
            severity=Severity.ERROR,
            category=_CATEGORY,
            title=f"block_id out of range ({len(findings)} finding(s))",
            summary=(
                f"{len(findings)} bound row(s) carry a block_id outside the "
                "range the row's own stage declares in stages.json"
            ),
            table=DiagnosticTable(
                columns=columns,
                rows=findings,
                justify=justify,
            ),
            remediation="Fix the block_id or the stage's declared block count.",
        ),
        logger=_LOG,
    )


def _thermal_anticipated(thermals: Mapping[str, object]) -> dict[int, bool]:
    """``{thermal_id: declares anticipated_config}`` from ``thermals.json``."""
    declared: dict[int, bool] = {}
    raw_thermals = thermals.get("thermals")
    if not isinstance(raw_thermals, list):
        return declared
    for thermal in raw_thermals:
        if not isinstance(thermal, Mapping):
            continue
        thermal_id = thermal.get("id")
        if not isinstance(thermal_id, int):
            continue
        declared[thermal_id] = thermal.get("anticipated_config") is not None
    return declared


def check_block_id_not_on_anticipated_thermal(
    thermals: Mapping[str, object],
    thermal_bounds: pa.Table | None,
) -> None:
    """Cobre rule 38 mirror: no ``thermal_bounds`` row may carry a non-null
    ``block_id`` for a thermal whose ``thermals.json`` entry declares a
    non-null ``anticipated_config``
    (``block_bounds.rs::check_block_id_on_anticipated_thermal``). An
    anticipated thermal's dispatch is committed ahead of the block axis, so
    cobre rejects any block-scoped bound on it outright.

    Structurally inapplicable, without a finding, when *thermal_bounds* is
    absent/empty, carries no ``block_id`` column, or no thermal declares
    ``anticipated_config``.
    """
    if thermal_bounds is None or thermal_bounds.num_rows == 0:
        return
    if "block_id" not in thermal_bounds.column_names:
        return

    anticipated = _thermal_anticipated(thermals)
    if not any(anticipated.values()):
        return

    thermal_ids = thermal_bounds["thermal_id"].to_pylist()
    stage_ids = thermal_bounds["stage_id"].to_pylist()
    block_ids = thermal_bounds["block_id"].to_pylist()

    rows: list[list[object]] = []
    for thermal_id, stage_id, block_id in zip(
        thermal_ids, stage_ids, block_ids, strict=True
    ):
        if block_id is None:
            continue
        if anticipated.get(thermal_id):
            rows.append([thermal_id, stage_id, block_id])

    if not rows:
        return

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    emit(
        Diagnostic(
            code="thermal-bound-block-id-on-anticipated",
            severity=Severity.ERROR,
            category=_CATEGORY,
            title=f"block_id on an anticipated thermal ({len(rows)} finding(s))",
            summary=(
                f"{len(rows)} thermal_bounds row(s) carry a non-null block_id "
                "for a thermal whose thermals.json entry declares "
                "anticipated_config (cobre rule 38)"
            ),
            table=DiagnosticTable(
                columns=["Thermal ID", "Stage", "block_id"],
                rows=rows,
                justify=["right", "right", "right"],
            ),
            remediation=(
                "Drop the per-block override row(s) for this thermal; an "
                "anticipated thermal's bounds must stay stage-level."
            ),
        ),
        logger=_LOG,
    )


def _describe_emission_check_errors(errors: Sequence[Diagnostic]) -> str:
    """Render *errors* (every entry ``ERROR`` severity) as a multi-line,
    human-readable block naming the failing rule and the offending entities —
    the message :func:`run_and_gate` raises when any is found. Each
    diagnostic's own detail table (hydro/thermal/line id, stage, column) is
    included, not just its one-line summary, so the exception is actionable
    on its own without re-running with a diagnostics sink attached.
    """
    lines: list[str] = []
    for diagnostic in errors:
        lines.append(f"- {diagnostic.code}: {diagnostic.summary}")
        if diagnostic.table is not None:
            for row in diagnostic.table.rows:
                lines.append("    " + ", ".join(str(value) for value in row))
    return "\n".join(lines)


def run_and_gate(run_checks: Callable[[], None]) -> None:
    """Run *run_checks* inside a diagnostics sink, re-emit every captured
    diagnostic on this module's logger, and raise :class:`EmissionCheckError`
    when any of them is ``ERROR`` severity.

    *run_checks* is a zero-argument callable invoking whichever check
    functions the caller's pipeline needs, so this gate stays track-agnostic.
    """
    with collect() as check_diagnostics:
        run_checks()
    for diagnostic in check_diagnostics:
        emit(diagnostic, logger=_LOG)
    check_errors = [d for d in check_diagnostics if d.severity is Severity.ERROR]
    if check_errors:
        raise EmissionCheckError(
            f"{len(check_errors)} post-emission self-check error(s):\n"
            f"{_describe_emission_check_errors(check_errors)}"
        )

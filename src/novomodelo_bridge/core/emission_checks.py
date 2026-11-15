"""Post-emission self-checks mirroring novomodelo 0.13's new hard validation rules.

Both pipelines build every artifact in memory before writing the Novomodelo case
directory. This module inspects those in-memory artifacts — the ``hydros.json``
document, the ``thermals.json`` document, the ``stages.json`` document, and the
``*_bounds`` Parquet tables — for the novomodelo 0.13 rules the bridge is most
likely to violate silently, and reports a match as an ``ERROR``
:class:`~novomodelo_bridge.core.diagnostics.Diagnostic` through the sink the pipeline
already runs converters inside. A failure here therefore surfaces with
bridge-side context (entity, stage, column, declared vs. offending value) in
milliseconds, instead of at ``novomodelo validate``/``novomodelo run`` load time.

This is a **courtesy mirror**, not a substitute: novomodelo remains the authority.
Each rule is scoped to match novomodelo's own implementation exactly (novomodelo's
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
- Rule 46 — :func:`check_diversion_has_channel`
  (``hydro.rs::check_diversion_requires_channel``).

This module must not import either pipeline — they import it — so it stays
unit-testable against hand-built artifacts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import pyarrow as pa

from novomodelo_bridge.core.diagnostics import (
    ColumnJustify,
    Diagnostic,
    DiagnosticTable,
    Severity,
    collect,
    emit,
    format_stage_ranges,
)
from novomodelo_bridge.core.errors import EmissionCheckError
from novomodelo_bridge.core.tolerances import (
    beyond_written_precision,
    relative_tolerance,
)

_LOG = logging.getLogger(__name__)

_CATEGORY = "Converted case checks"

_REJECTED = "Novomodelo rejects such a case when it loads it, so the conversion stops."

#: For a finding only a conversion defect can produce: the deck is not at fault.
_DEFECT_REMEDIATION = (
    "The conversion should never write such a case: report it to the "
    "novomodelo-bridge maintainers with the deck."
)

#: Both guarded columns for rule 43 / rule 41, checked independently.
_ENVELOPE_COLUMNS: tuple[str, ...] = ("max_turbined_m3s", "max_generation_mw")

#: Per-stage ``hydro_bounds`` limits held to the plant's declared envelope,
#: each as ``(ceiling column, hydros block, floor column)``; the
#: ``hydros.json`` block declares the max and the min under the columns' own
#: names. See :func:`clamp_hydro_bounds_to_declared`.
_CLAMP_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("max_turbined_m3s", "generation", "min_turbined_m3s"),
    ("max_generation_mw", "generation", "min_generation_mw"),
    ("max_outflow_m3s", "outflow", "min_outflow_m3s"),
)

_CLAMP_LIMIT_LABELS: dict[str, str] = {
    "max_turbined_m3s": "Max turbined flow (m³/s)",
    "max_generation_mw": "Max generation (MW)",
    "max_outflow_m3s": "Max outflow (m³/s)",
    "min_turbined_m3s": "Min turbined flow (m³/s)",
    "min_generation_mw": "Min generation (MW)",
    "min_outflow_m3s": "Min outflow (m³/s)",
}

#: The deck constraints and limits behind each ``hydro_bounds`` cell, keyed
#: ``(hydro_id, stage_id, block_id, column)``, each with the value it sets in
#: that column.
#: A ``None`` block_id covers every block of the stage.
type BoundLabels = Mapping[
    tuple[int, int, int | None, str], Sequence[tuple[str, float]]
]

#: A ``hydro_bounds`` row's ``(hydro_id, stage_id, block_id)``.
type _Cell = tuple[int, int, int | None]


def _tolerance(declared: float) -> float:
    """``ENVELOPE_TOLERANCE * max(|declared|, 1.0)`` — novomodelo's envelope tolerance."""
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
    value column — mirroring novomodelo rule 36's own widened key ``(hydro_id,
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
    """Novomodelo rule 43 mirror: no ``hydro_bounds`` row may raise ``max_turbined_m3s``
    or ``max_generation_mw`` above the hydro's own declared value in
    ``hydros.json`` (``generation.max_turbined_m3s`` /
    ``generation.max_generation_mw``), each column checked independently.
    Lowering is fine; every other column is untouched by this rule.

    Structurally inapplicable when *hydro_bounds* is ``None`` or carries
    neither guarded column. That is reported as an explicit INFO finding,
    never a silent pass.
    """
    columns = (
        [c for c in _ENVELOPE_COLUMNS if c in hydro_bounds.column_names]
        if hydro_bounds is not None
        else []
    )

    if hydro_bounds is None or not columns:
        emit(
            Diagnostic(
                code="hydro-bounds-raising-not-applicable",
                severity=Severity.INFO,
                category=_CATEGORY,
                title="Per-stage capacity check not needed",
                summary=(
                    "The case sets no per-stage turbined-flow or generation limit "
                    "in constraints/hydro_bounds.parquet, so no stage can exceed "
                    "a plant's capacity."
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
            title=f"Stage limits above the plant capacity ({len(rows)})",
            summary=(
                "The listed per-stage limits in "
                "constraints/hydro_bounds.parquet sit above the capacity "
                f"system/hydros.json declares for the plant. {_REJECTED}"
            ),
            table=DiagnosticTable(
                columns=["Hydro ID", "Column", "Stages", "Capacity", "Highest value"],
                rows=rows,
                justify=["right", "left", "left", "right", "right"],
            ),
            remediation=_DEFECT_REMEDIATION,
        ),
        logger=_LOG,
    )


def _hydro_declared_bounds(
    hydros: Mapping[str, object],
) -> dict[int, dict[str, tuple[float | None, float | None]]]:
    """``{hydro_id: {ceiling column: (declared_min, declared_max)}}`` for every
    limit in :data:`_CLAMP_COLUMNS`.

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
        for column, block_name, floor_column in _CLAMP_COLUMNS:
            block = hydro.get(block_name)
            if not isinstance(block, Mapping):
                continue
            raw_min = block.get(floor_column)
            raw_max = block.get(column)
            declared_min = float(raw_min) if isinstance(raw_min, int | float) else None
            declared_max = float(raw_max) if isinstance(raw_max, int | float) else None
            if declared_min is not None or declared_max is not None:
                entry[column] = (declared_min, declared_max)
        if entry:
            declared[hydro_id] = entry
    return declared


@dataclass
class _EnvelopeFinding:
    """A plant limit outside its envelope: each ``(stage, deck value,
    reference)`` past it, and the deck constraints that set them."""

    occurrences: list[tuple[int, float, float]] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def add(
        self, stage_id: int, value: float, reference: float, constraints: Iterable[str]
    ) -> None:
        self.occurrences.append((stage_id, value, reference))
        for constraint in constraints:
            if constraint not in self.constraints:
                self.constraints.append(constraint)


#: Findings keyed ``(hydro_id, column)``.
type _Findings = dict[tuple[int, str], _EnvelopeFinding]

#: ``{hydro_id: {ceiling column: (declared_min, declared_max)}}``.
type _DeclaredBounds = Mapping[int, Mapping[str, tuple[float | None, float | None]]]


@dataclass
class _ClampFindings:
    """What :func:`clamp_hydro_bounds_to_declared` reports, by kind."""

    capped: _Findings = field(default_factory=dict)
    raised: _Findings = field(default_factory=dict)
    floors: _Findings = field(default_factory=dict)


def clamp_hydro_bounds_to_declared(
    hydros: Mapping[str, object],
    hydro_bounds: pa.Table | None,
    *,
    source_codes: Mapping[int, int],
    raise_to_minimum: bool,
    labels: BoundLabels | None = None,
) -> pa.Table | None:
    """Hold every per-stage ``hydro_bounds`` turbined, generation and outflow
    limit to the owning plant's declared ``[min, max]`` envelope from
    ``hydros.json``, returning the clamped table.

    Per cell and limit, in order:

    1. A ceiling above the declared max is capped at it. novomodelo rejects a
       turbined or generation ceiling above the declared value on load, and
       the plant cannot exceed its capacity anyway, so an INFO note reports
       the cap.
    2. With *raise_to_minimum*, a ceiling below the cell's floor (the cell's
       own floor where it sets one, else the declared min) is raised to that
       floor, or to the declared max when the floor is above it. That loosens
       the deck's limit, so a WARNING reports it. A source model that treats
       the two limits as soft and violates whichever is cheaper passes
       ``False``: the case then writes both and novomodelo prices the
       shortfall, and the caller reports the pair where it meets them.
    3. A floor above the declared max is written as the deck has it:
       novomodelo prices the shortfall instead of rejecting it, and a WARNING
       reports it. A floor above a lower ceiling is step 2's, or the
       caller's.

    Each finding is reported only past the deck's written precision
    (:func:`~novomodelo_bridge.core.tolerances.beyond_written_precision`);
    nearer, steps 1 and 2 move the ceiling silently and step 3 lowers the
    floor to the declared max.

    *source_codes* maps each hydro id to its source-model code for those
    reports. *labels*, when given, names the deck constraints and limits behind
    each cell, and each report then lists, under "Set by", those that put the
    cell outside its envelope.
    Returns *hydro_bounds* unchanged (same object) when it is ``None``, carries
    no clamped column, or nothing fell outside its envelope.
    """
    if hydro_bounds is None:
        return None
    axes = [
        (ceiling, floor)
        for ceiling, _, floor in _CLAMP_COLUMNS
        if ceiling in hydro_bounds.column_names or floor in hydro_bounds.column_names
    ]
    if not axes:
        return hydro_bounds

    declared = _hydro_declared_bounds(hydros)
    cells = _cells(hydro_bounds)
    found = _ClampFindings()
    table = hydro_bounds
    for ceiling_column, floor_column in axes:
        columns = {
            column: _column_values(table, column)
            for column in (ceiling_column, floor_column)
        }
        ceilings, floors = list(columns[ceiling_column]), list(columns[floor_column])
        _clamp_axis(
            (ceiling_column, floor_column),
            ceilings,
            floors,
            cells,
            declared,
            labels,
            found,
            raise_to_minimum=raise_to_minimum,
        )
        for column, values in ((ceiling_column, ceilings), (floor_column, floors)):
            if values != columns[column]:
                table = table.set_column(
                    table.column_names.index(column),
                    column,
                    pa.array(values, type=pa.float64()),
                )

    names = _hydro_names(hydros)
    labelled = labels is not None
    for findings, build in (
        (found.capped, _capped_to_capacity_diagnostic),
        (found.raised, _raised_to_floor_diagnostic),
        (found.floors, _floor_above_capacity_diagnostic),
    ):
        if findings:
            emit(build(findings, names, source_codes, labelled), logger=_LOG)
    return table


def _column_values(table: pa.Table, column: str) -> list[float | None]:
    """*column*'s values, all ``None`` when *table* lacks it."""
    if column not in table.column_names:
        return [None] * table.num_rows
    return table[column].to_pylist()


def _cells(table: pa.Table) -> list[_Cell]:
    """Each row's ``(hydro_id, stage_id, block_id)``; ``block_id`` is ``None``
    throughout a table without that column."""
    blocks = (
        table["block_id"].to_pylist()
        if "block_id" in table.column_names
        else [None] * table.num_rows
    )
    return list(
        zip(
            table["hydro_id"].to_pylist(),
            table["stage_id"].to_pylist(),
            blocks,
            strict=True,
        )
    )


def _cell_labels(
    labels: BoundLabels | None, cell: _Cell, column: str
) -> list[tuple[str, float]]:
    """The ``(constraint, value)`` pairs *labels* gives *column* of *cell*; a
    stage-wide (``None`` block) entry counts for every block."""
    if labels is None:
        return []
    hydro_id, stage_id, block_id = cell
    entries = list(labels.get((hydro_id, stage_id, block_id, column), ()))
    if block_id is not None:
        entries.extend(labels.get((hydro_id, stage_id, None, column), ()))
    return entries


def _labels_past(
    labels: BoundLabels | None,
    cell: _Cell,
    column: str,
    reference: float,
    *,
    above: bool,
) -> list[str]:
    """The constraints *labels* gives *column* of *cell* whose value sits past
    the written precision above (or, unless *above*, below) *reference*."""
    return [
        constraint
        for constraint, value in _cell_labels(labels, cell, column)
        if (value > reference) is above and beyond_written_precision(value, reference)
    ]


def _record(
    found: _Findings,
    cell: _Cell,
    column: str,
    value: float,
    reference: float,
    constraints: Iterable[str],
) -> None:
    """Record *value* against *reference* in *found* when past the written
    precision."""
    if beyond_written_precision(value, reference):
        found.setdefault((cell[0], column), _EnvelopeFinding()).add(
            cell[1], value, reference, constraints
        )


def _clamp_axis(
    columns: tuple[str, str],
    ceilings: list[float | None],
    floors: list[float | None],
    cells: Sequence[_Cell],
    declared: _DeclaredBounds,
    labels: BoundLabels | None,
    found: _ClampFindings,
    *,
    raise_to_minimum: bool,
) -> None:
    """Apply the steps of :func:`clamp_hydro_bounds_to_declared` to one
    limit's *ceilings* and *floors* in place, keyed by its ``(ceiling column,
    floor column)``, recording what it reports in *found*."""
    ceiling_column, floor_column = columns
    for row, cell in enumerate(cells):
        declared_min, declared_max = declared.get(cell[0], {}).get(
            ceiling_column, (None, None)
        )
        ceiling, floor = ceilings[row], floors[row]
        if (
            ceiling is not None
            and declared_max is not None
            and ceiling > declared_max + _tolerance(declared_max)
        ):
            _record(
                found.capped,
                cell,
                ceiling_column,
                ceiling,
                declared_max,
                _labels_past(labels, cell, ceiling_column, declared_max, above=True),
            )
            ceiling = declared_max
        stage_floor = floor if floor is not None else declared_min
        if (
            raise_to_minimum
            and ceiling is not None
            and stage_floor is not None
            and ceiling < stage_floor - _tolerance(stage_floor)
        ):
            target = (
                stage_floor if declared_max is None else min(stage_floor, declared_max)
            )
            if target > ceiling:
                _record(
                    found.raised,
                    cell,
                    ceiling_column,
                    ceiling,
                    target,
                    [
                        *_labels_past(
                            labels, cell, ceiling_column, target, above=False
                        ),
                        *_labels_past(labels, cell, floor_column, ceiling, above=True),
                    ],
                )
                ceiling = target
        ceilings[row] = ceiling
        if (
            floor is None
            or declared_max is None
            or floor <= declared_max + _tolerance(declared_max)
        ):
            continue
        if beyond_written_precision(floor, declared_max):
            _record(
                found.floors,
                cell,
                floor_column,
                floor,
                declared_max,
                _labels_past(labels, cell, floor_column, declared_max, above=True),
            )
        else:
            floors[row] = declared_max


def _hydro_names(hydros: Mapping[str, object]) -> dict[int, str]:
    """``{hydro_id: name}`` for every ``hydros.json`` entry that has both."""
    raw_hydros = hydros.get("hydros")
    if not isinstance(raw_hydros, list):
        return {}
    return {
        hydro["id"]: hydro["name"]
        for hydro in raw_hydros
        if isinstance(hydro, Mapping)
        and isinstance(hydro.get("id"), int)
        and isinstance(hydro.get("name"), str)
    }


def _clamp_table(
    found: _Findings,
    names: Mapping[int, str],
    source_codes: Mapping[int, int],
    labelled: bool,
    *,
    headers: tuple[str, str],
    caption: str,
) -> DiagnosticTable:
    """One ``[code, name, limit, constraints, stages, deck value, reference]``
    row per ``(hydro, column)`` in *found*, the constraints only when
    *labelled* and the last two columns headed *headers*; the values are the
    stage's whose deck value sits furthest from its reference."""
    rows: list[list[object]] = []
    for (hydro_id, column), finding in sorted(
        found.items(), key=lambda item: (source_codes[item[0][0]], item[0][1])
    ):
        _, value, reference = max(
            finding.occurrences,
            key=lambda occurrence: abs(occurrence[1] - occurrence[2]),
        )
        rows.append(
            [
                source_codes[hydro_id],
                names.get(hydro_id, ""),
                _CLAMP_LIMIT_LABELS[column],
                *([", ".join(finding.constraints)] if labelled else []),
                format_stage_ranges(stage for stage, _, _ in finding.occurrences),
                round(value, 6),
                round(reference, 6),
            ]
        )
    constraints = ["Set by"] if labelled else []
    justify: list[ColumnJustify] = ["right", "left", "left"]
    justify += ["left"] * len(constraints)
    justify += ["left", "right", "right"]
    return DiagnosticTable(
        columns=["Code", "Name", "Limit", *constraints, "Stages", *headers],
        rows=rows,
        justify=justify,
        caption=caption,
    )


def _capped_to_capacity_diagnostic(
    found: _Findings,
    names: Mapping[int, str],
    source_codes: Mapping[int, int],
    labelled: bool,
) -> Diagnostic:
    table = _clamp_table(
        found,
        names,
        source_codes,
        labelled,
        headers=("Deck value", "Capacity"),
        caption="Stage ids are 0-based; the deck value shown is the highest.",
    )
    return Diagnostic(
        code="hydro-bounds-clamped-to-declared-capacity",
        severity=Severity.INFO,
        category="Hydro capacity",
        title=f"Stage limits above the plant capacity capped ({len(table.rows)})",
        summary=(
            "The listed per-stage limits in the deck sit above the plant's "
            "own capacity and were capped at it. The plant cannot exceed its "
            "capacity, so the cap cannot bind."
        ),
        table=table,
    )


def _raised_to_floor_diagnostic(
    found: _Findings,
    names: Mapping[int, str],
    source_codes: Mapping[int, int],
    labelled: bool,
) -> Diagnostic:
    table = _clamp_table(
        found,
        names,
        source_codes,
        labelled,
        headers=("Deck value", "Raised to"),
        caption="Stage ids are 0-based; the values shown are the largest raise.",
    )
    return Diagnostic(
        code="hydro-bounds-raised-to-declared-minimum",
        severity=Severity.WARNING,
        category="Hydro capacity",
        title=f"Stage limits below the stage floor raised ({len(table.rows)})",
        summary=(
            "The listed per-stage limits in the deck sit below the stage's "
            "floor, which is the floor the deck sets on that stage or else "
            "the plant's own minimum. They were raised to that floor, or to "
            "the plant's capacity where the floor is above it, so on those "
            "stages the case allows more than the deck does."
        ),
        table=table,
    )


def _floor_above_capacity_diagnostic(
    found: _Findings,
    names: Mapping[int, str],
    source_codes: Mapping[int, int],
    labelled: bool,
) -> Diagnostic:
    table = _clamp_table(
        found,
        names,
        source_codes,
        labelled,
        headers=("Deck value", "Capacity"),
        caption="Stage ids are 0-based; the deck value shown is the highest.",
    )
    return Diagnostic(
        code="hydro-bounds-floor-above-declared-capacity",
        severity=Severity.WARNING,
        category="Hydro capacity",
        title=f"Stage floors above the plant capacity ({len(table.rows)})",
        summary=(
            "The listed per-stage floors in the deck sit above the plant's "
            "own capacity, so no dispatch meets them. They were written as "
            "the deck has them: novomodelo holds the plant to its capacity "
            "and charges the shortfall below the floor at the violation "
            "penalty, which raises the cost of those stages."
        ),
        table=table,
    )


def check_unit_group_envelope(hydros: Mapping[str, object]) -> None:
    """Novomodelo rule 41 mirror: the sum of a hydro's unit-group maxima must not
    exceed the hydro's own declared value in ``hydros.json``, each of
    ``max_turbined_m3s`` / ``max_generation_mw`` checked independently, against
    the entity declaration only — never a per-stage ``hydro_bounds`` override.

    Always structurally applicable: both pipelines declare a non-empty
    ``unit_groups`` on every hydro (novomodelo requires it), so there is no
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
            title=f"Unit groups above the plant capacity ({len(rows)})",
            summary=(
                "The listed plant limits in system/hydros.json sit below "
                f"the sum of the plant's unit groups. {_REJECTED}"
            ),
            table=DiagnosticTable(
                columns=["Hydro ID", "Column", "Capacity", "Group sum"],
                rows=rows,
                justify=["right", "left", "right", "right"],
            ),
            remediation=_DEFECT_REMEDIATION,
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
    """Novomodelo rule 45 mirror: no ``hydro_unit_group_bounds`` row may raise
    ``max_turbined_m3s`` or ``max_generation_mw`` above *that group's own*
    declared value in ``hydros.json`` (``unit_groups[].max_turbined_m3s`` /
    ``unit_groups[].max_generation_mw``), each column checked independently
    (``block_bounds.rs::check_group_bound_raises_declared_capacity``).
    Lowering is fine; the check is keyed by ``(hydro_id,
    hydro_unit_group_id)``, not by the plant's own envelope (that is rule 43,
    :func:`check_hydro_bounds_no_raising`).

    Structurally inapplicable, without a finding, when *group_bounds* is
    ``None``/empty or carries neither guarded column — unlike rule 43's mirror
    this is not reported as an explicit INFO finding, since an absent/empty
    ``hydro_unit_group_bounds`` table is the ordinary, expected shape whenever
    no plant's availability actually needed an override.
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
            title=f"Stage limits above the unit-group capacity ({len(rows)})",
            summary=(
                "The listed per-stage limits in "
                "constraints/hydro_unit_group_bounds.parquet sit above the "
                "capacity system/hydros.json declares for the unit group. "
                f"{_REJECTED}"
            ),
            table=DiagnosticTable(
                columns=[
                    "Hydro ID",
                    "Group ID",
                    "Column",
                    "Stages",
                    "Capacity",
                    "Highest value",
                ],
                rows=rows,
                justify=["right", "right", "left", "left", "right", "right"],
            ),
            remediation=_DEFECT_REMEDIATION,
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
    """Novomodelo rule 36 mirror: at most one row per ``(entity, [group,] stage,
    block, column)`` within a family, checked independently for every
    non-key column.

    A family's ``group_column`` (e.g. ``hydro_unit_group_id`` for
    ``hydro_unit_group_bounds``), when set, folds into the **key** rather
    than being treated as a value column — two groups of the same plant
    setting the same column at the same ``(stage, block)`` are distinct
    keys, not a collision, mirroring novomodelo rule 36's own widened key
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
            title=f"Bound rows set twice ({len(findings)})",
            summary=(
                "The listed rows of the case's bound files set the same "
                "limit twice for the same entity, stage and load block. "
                f"{_REJECTED}"
            ),
            table=DiagnosticTable(
                columns=columns,
                rows=findings,
            ),
            remediation=_DEFECT_REMEDIATION,
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
    justify: list[ColumnJustify] = ["left", "right"]
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
            title=f"Bound rows on a load block the stage lacks ({len(findings)})",
            summary=(
                "The listed rows of the case's bound files name a load "
                f"block that their stage does not declare in stages.json. "
                f"{_REJECTED}"
            ),
            table=DiagnosticTable(
                columns=columns,
                rows=findings,
                justify=justify,
            ),
            remediation=_DEFECT_REMEDIATION,
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
    """Novomodelo rule 38 mirror: no ``thermal_bounds`` row may carry a non-null
    ``block_id`` for a thermal whose ``thermals.json`` entry declares a
    non-null ``anticipated_config``
    (``block_bounds.rs::check_block_id_on_anticipated_thermal``). An
    anticipated thermal's dispatch is committed ahead of the block axis, so
    novomodelo rejects any block-scoped bound on it outright.

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
            title=f"Per-block limits on anticipated thermals ({len(rows)})",
            summary=(
                "The listed rows of constraints/thermal_bounds.parquet set a "
                "per-block limit for a thermal that system/thermals.json "
                "declares as anticipated, whose limits apply to the whole "
                f"stage. {_REJECTED}"
            ),
            table=DiagnosticTable(
                columns=["Thermal ID", "Stage", "block_id"],
                rows=rows,
                justify=["right", "right", "right"],
            ),
            remediation=_DEFECT_REMEDIATION,
        ),
        logger=_LOG,
    )


#: The ``hydro_bounds`` columns that let a hydro divert water.
_DIVERSION_COLUMNS: tuple[str, ...] = ("min_diversion_m3s", "max_diversion_m3s")


def check_diversion_has_channel(
    hydros: Mapping[str, object],
    hydro_bounds: pa.Table | None,
) -> None:
    """No hydro may divert water without a ``diversion`` channel to receive it.

    novomodelo takes a hydro's diversion out of that hydro's water balance and adds
    it to another plant only through the hydro's ``diversion`` channel in
    ``hydros.json``. A positive ``min_diversion_m3s`` or ``max_diversion_m3s``
    row on a hydro without one is an ERROR, as novomodelo rejects it (rule 46).
    """
    if hydro_bounds is None or hydro_bounds.num_rows == 0:
        return
    columns = [c for c in _DIVERSION_COLUMNS if c in hydro_bounds.column_names]
    if not columns:
        return
    names: dict[int, str] = {}
    with_channel: set[int] = set()
    raw_hydros = hydros.get("hydros")
    for hydro in raw_hydros if isinstance(raw_hydros, list) else []:
        if not isinstance(hydro, Mapping) or not isinstance(hydro.get("id"), int):
            continue
        names[hydro["id"]] = str(hydro.get("name", ""))
        if hydro.get("diversion") is not None:
            with_channel.add(hydro["id"])

    hydro_ids = hydro_bounds["hydro_id"].to_pylist()
    stage_ids = hydro_bounds["stage_id"].to_pylist()
    violations: dict[tuple[int, str], list[tuple[int, float]]] = {}
    for column in columns:
        for hydro_id, stage_id, value in zip(
            hydro_ids, stage_ids, hydro_bounds[column].to_pylist(), strict=True
        ):
            if value is not None and value > 0.0 and hydro_id not in with_channel:
                violations.setdefault((hydro_id, column), []).append((stage_id, value))
    if not violations:
        return

    rows = [
        [
            hydro_id,
            names.get(hydro_id, ""),
            column,
            format_stage_ranges(stage for stage, _ in occurrences),
            round(max(value for _, value in occurrences), 6),
        ]
        for (hydro_id, column), occurrences in sorted(
            violations.items(),
            key=lambda item: (item[0][0], _DIVERSION_COLUMNS.index(item[0][1])),
        )
    ]
    plants = len({hydro_id for hydro_id, _ in violations})
    emit(
        Diagnostic(
            code="hydro-diversion-without-channel",
            severity=Severity.ERROR,
            category=_CATEGORY,
            title=f"Diversions without a receiving plant ({plants})",
            summary=(
                "The listed plants may divert water in "
                "constraints/hydro_bounds.parquet but declare no diversion "
                "channel in system/hydros.json, so the diverted water would "
                f"reach no plant. {_REJECTED}"
            ),
            table=DiagnosticTable(
                columns=["Hydro ID", "Name", "Column", "Stages", "Largest value"],
                rows=rows,
                justify=["right", "left", "left", "left", "right"],
            ),
            remediation=(
                "Include in the case the plant that receives each listed "
                "plant's diversion, or remove the diversion restriction from "
                "the deck."
            ),
        ),
        logger=_LOG,
    )


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
    failed = list(
        dict.fromkeys(d.code for d in check_diagnostics if d.severity is Severity.ERROR)
    )
    if failed:
        checks = "self-check" if len(failed) == 1 else "self-checks"
        raise EmissionCheckError(
            f"The converted case failed {len(failed)} {checks} "
            f"({', '.join(failed)}), so the conversion stopped."
        )

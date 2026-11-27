"""Which sections of a version comparison both outputs can support.

Availability is decided by columns alone, never by a version string:
:func:`section_availability` sets what a :class:`SectionRequirement` declares
against each side's availability mapping and reports what it skipped. This
module never sees an unreadable result file, because the probe raises for it.
It reports two things: drift, an absent partition or column, and an optional
entity list that could not be used, which its reader has already reported with a
side-tagged warning naming the file. An entity list enters as a key of the same
mapping as a partition (:func:`with_entity_lists`), so both reach the rows
through one path.

An item missing on both sides is one row whose side is :data:`BOTH_SIDES`, never
two: equal versions, or one directory given twice, are ordinary inputs. The
functions emit no Diagnostic and read no file.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from novomodelo_bridge.comparators.version.labels import SIDES

BOTH_SIDES: str = "both"

ENTITY_LIST_KEYS: Mapping[str, str] = {
    "thermal": "system/thermals.json",
    "line": "system/lines.json",
    "generic_constraint": "constraints/generic_constraints.json",
}

_BASELINE, _CANDIDATE = SIDES

Availability = Mapping[str, frozenset[str] | None]


@dataclass(frozen=True)
class Unavailable:
    """One skipped item: the whole section when *column* is ``None``, else a column.

    *side* is ``"baseline"`` or ``"candidate"`` when only that side lacks the
    item and :data:`BOTH_SIDES` when both do. The three values are persisted.
    """

    section: str
    column: str | None
    side: str


@dataclass(frozen=True)
class SectionRequirement:
    """What one report section needs from each side.

    Attributes:
        section: The section's name; unique across the requirements in use.
        partitions: Availability key to the columns the section cannot do
            without. A key is a probe key or one of :data:`ENTITY_LIST_KEYS`'s
            values; an entity list that is required at all takes ``frozenset()``.
        variables: Variable name to ``(availability key, source columns)``.

    Raises:
        ValueError: a variable names a key outside *partitions*.
    """

    section: str
    partitions: Mapping[str, frozenset[str]]
    variables: Mapping[str, tuple[str, frozenset[str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, (key, _) in self.variables.items():
            if key not in self.partitions:
                raise ValueError(
                    f"{self.section}: variable {name!r} reads {key!r}, "
                    "which the section does not require"
                )


@dataclass(frozen=True)
class SectionAvailability:
    """The verdict on one section.

    Attributes:
        variables: Available variables in the requirement's order; empty when
            the section is unavailable.
        unavailable: The section-level row (column ``None``) first, then the
            column rows sorted by column; no two rows share a column.
    """

    section: str
    available: bool
    variables: tuple[str, ...]
    unavailable: tuple[Unavailable, ...]


def _side_of(on_baseline: bool, on_candidate: bool) -> str:
    if on_baseline and on_candidate:
        return BOTH_SIDES
    return _BASELINE if on_baseline else _CANDIDATE


def _has_absent_key(side: Availability, keys: Iterable[str]) -> bool:
    return any(side.get(key) is None for key in keys)


def _missing_columns(
    side: Availability, needs: Iterable[tuple[str, frozenset[str]]]
) -> frozenset[str]:
    """Columns of *needs* missing from the keys *side* holds; an absent key adds none."""
    return frozenset(
        column
        for key, columns in needs
        if (present := side.get(key)) is not None
        for column in columns - present
    )


def _column_rows(
    section: str, baseline: frozenset[str], candidate: frozenset[str]
) -> list[Unavailable]:
    return [
        Unavailable(section, column, _side_of(column in baseline, column in candidate))
        for column in sorted(baseline | candidate)
    ]


def _available_variables(
    requirement: SectionRequirement, baseline: Availability, candidate: Availability
) -> tuple[tuple[str, ...], list[Unavailable]]:
    kept: list[str] = []
    missing_baseline: set[str] = set()
    missing_candidate: set[str] = set()
    for name, need in requirement.variables.items():
        on_baseline = _missing_columns(baseline, [need])
        on_candidate = _missing_columns(candidate, [need])
        missing_baseline |= on_baseline
        missing_candidate |= on_candidate
        if not (on_baseline or on_candidate):
            kept.append(name)
    rows = _column_rows(
        requirement.section, frozenset(missing_baseline), frozenset(missing_candidate)
    )
    return tuple(kept), rows


def section_availability(
    requirement: SectionRequirement, baseline: Availability, candidate: Availability
) -> SectionAvailability:
    """Decide *requirement* against the two sides' availability mappings.

    A key that is missing from a mapping counts as absent, like ``None``; an
    empty column set is present. Variable-level rows are reported only for a
    section that is available.
    """
    section = requirement.section
    absent_baseline = _has_absent_key(baseline, requirement.partitions)
    absent_candidate = _has_absent_key(candidate, requirement.partitions)
    rows: list[Unavailable] = []
    if absent_baseline or absent_candidate:
        rows.append(
            Unavailable(section, None, _side_of(absent_baseline, absent_candidate))
        )
    rows += _column_rows(
        section,
        _missing_columns(baseline, requirement.partitions.items()),
        _missing_columns(candidate, requirement.partitions.items()),
    )
    if rows:
        return SectionAvailability(section, False, (), tuple(rows))
    variables, variable_rows = _available_variables(requirement, baseline, candidate)
    return SectionAvailability(section, True, variables, tuple(variable_rows))


def with_entity_lists(
    partitions: Mapping[str, frozenset[str] | None], usable: Mapping[str, bool]
) -> dict[str, frozenset[str] | None]:
    """A copy of *partitions* plus one key per entity type of *usable*.

    The key holds ``frozenset()`` for a usable list and ``None`` for one that is
    not, so an unusable list reads as an absent partition.

    Raises:
        ValueError: *usable* holds an entity type outside :data:`ENTITY_LIST_KEYS`.
    """
    extended = dict(partitions)
    for entity, ok in usable.items():
        if entity not in ENTITY_LIST_KEYS:
            raise ValueError(f"unknown entity type: {entity!r}")
        extended[ENTITY_LIST_KEYS[entity]] = frozenset() if ok else None
    return extended


def unavailable_rows(
    availabilities: Iterable[SectionAvailability],
) -> list[Unavailable]:
    """Every row of *availabilities*, sorted by section, then column with ``None`` first.

    Raises:
        ValueError: two rows share a section and a column, which only two
            requirements of one section name produce; the pair is a row's key.
    """
    rows = [row for availability in availabilities for row in availability.unavailable]
    seen: set[tuple[str, str | None]] = set()
    for row in rows:
        if (row.section, row.column) in seen:
            raise ValueError(
                f"{row.section}: two rows for column {row.column!r}; "
                "section names must be unique"
            )
        seen.add((row.section, row.column))
    return sorted(
        rows, key=lambda row: (row.section, row.column is not None, row.column or "")
    )

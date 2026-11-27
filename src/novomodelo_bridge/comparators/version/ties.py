"""Tie marks of a version comparison: the union over the two cases.

Between two Novomodelo solves either case can hold a degenerate tie, so a row
moves to its unjudged variable when either case ties it. Each case is marked by
the shared :func:`~novomodelo_bridge.comparators.line_ties.mark_tied_lines` and
:func:`~novomodelo_bridge.comparators.plant_buses.merge_shared_plant_buses`
unchanged; they read ``novomodelo_id`` as the id in the directory they read, so the
baseline's pass runs on mirrored rows that carry the baseline's ids there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from novomodelo_bridge.comparators.line_ties import mark_tied_lines
from novomodelo_bridge.comparators.model import ResultComparison
from novomodelo_bridge.comparators.plant_buses import merge_shared_plant_buses
from novomodelo_bridge.comparators.version.alignment import name_key
from novomodelo_bridge.comparators.version.sides import SideInputs, side_tagged


def _mirrored(row: ResultComparison) -> ResultComparison:
    return replace(
        row,
        newave_code=row.novomodelo_id,
        novomodelo_id=row.newave_code,
        newave_value=row.novomodelo_value,
        novomodelo_value=row.newave_value,
    )


def _union_variables(
    rows: Sequence[ResultComparison],
    first: Sequence[ResultComparison],
    second: Sequence[ResultComparison],
) -> list[ResultComparison]:
    """*rows* with the variable that *first*, else *second*, moved each to.

    Both passes hold one row per input row in its position.
    """
    if len(first) != len(rows) or len(second) != len(rows):
        raise ValueError(
            f"a tie-marking pass returned {len(first)} and {len(second)} rows "
            f"for {len(rows)} input rows"
        )
    return [
        replace(
            row,
            variable=a.variable if a.variable != row.variable else b.variable,
        )
        for row, a, b in zip(rows, first, second, strict=True)
    ]


def _group_key(row: ResultComparison) -> tuple[frozenset[str], int]:
    names = frozenset(name_key(part) for part in row.entity_name.split("+"))
    return names, row.stage


def apply_tie_marks(
    rows: Sequence[ResultComparison], baseline: SideInputs, candidate: SideInputs
) -> list[ResultComparison]:
    """*rows* with every line or bus row that ties in either case moved to its
    unjudged variable, then the group net-import rows both cases build.

    *rows* carry the baseline's id in ``newave_code`` and the candidate's in
    ``novomodelo_id``; only the side and directory of *baseline* and *candidate*
    are read. A group row is kept when both cases build one for the same bus names
    at the same stage, with the candidate's ids and values under the baseline's
    entity name. A group of one case only moves its members' rows and adds none:
    the boundary lines of the two cases are then not the same lines.

    The shared functions' ``line-routes-tied`` and ``buses-share-plant`` warnings
    reach the enclosing sink once per case, tagged with its side.

    Raises:
        ValueError: a marking pass returned fewer rows than it was given, or
            ``mark_tied_lines`` a different number; an internal invariant.
    """
    with side_tagged(candidate.side):
        by_candidate = mark_tied_lines(rows, candidate.output_dir)
    with side_tagged(baseline.side):
        by_baseline = mark_tied_lines(
            [_mirrored(row) for row in rows], baseline.output_dir
        )
    tied = _union_variables(rows, by_candidate, by_baseline)

    with side_tagged(candidate.side):
        merged_candidate = merge_shared_plant_buses(tied, candidate.output_dir)
    with side_tagged(baseline.side):
        merged_baseline = merge_shared_plant_buses(
            [_mirrored(row) for row in tied], baseline.output_dir
        )
    count = len(tied)
    marked = _union_variables(tied, merged_candidate[:count], merged_baseline[:count])
    baseline_names = {
        _group_key(row): row.entity_name for row in merged_baseline[count:]
    }
    kept = [
        replace(row, entity_name=baseline_names[_group_key(row)])
        for row in merged_candidate[count:]
        if _group_key(row) in baseline_names
    ]
    return [*marked, *kept]

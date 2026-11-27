"""The generic constraints of a version comparison.

The constraints drawn are the candidate's: its definitions, its bounds and its
left-hand side, beside the baseline's left-hand side re-keyed to the
candidate's ids by name. A pair whose terms or bounds differ stays paired and
is reported once by ``findings.differing_constraints``. The constraint list of
each side is validated by :func:`constraint_list_usable` before it is aligned,
never from a captured Diagnostic. The only Diagnostic this module emits is that
function's warning.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from novomodelo_bridge.comparators.constraints import (
    evaluate_lhs_novomodelo,
    stage_bounds_novomodelo,
)
from novomodelo_bridge.comparators.version.alignment import (
    AlignedPair,
    NameAlignment,
    align_by_name,
)
from novomodelo_bridge.comparators.version.sides import SideInputs, side_tagged
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.errors import FieldParseError, SourceFileError
from novomodelo_bridge.novomodelo.case_io import case_dir_for
from novomodelo_bridge.novomodelo.constraint_expr import ResolvedGenericConstraints
from novomodelo_bridge.novomodelo.readers import (
    read_novomodelo_required_generic_constraints,
    read_novomodelo_resolved_generic_constraints,
)

_LOG = logging.getLogger(__name__)

_FILE = "constraints/generic_constraints.json"
_EFFECT = "the comparison leaves out the generic constraints of this output."

# Two solver versions may resolve one coefficient a unit in the last place apart.
_RELATIVE_TOLERANCE = 1e-9

_Entity = int | tuple[str, int] | None
_TermKey = tuple[int, int | None, str, _Entity, int | None, _Entity]
_BoundKey = tuple[int, int | None]
_Endpoints = tuple[float | None, float | None]


@dataclass(frozen=True)
class DifferingConstraint:
    """A paired constraint, by the baseline's spelling of its name, whose terms or
    bounds differ over the common stages."""

    name: str
    terms: bool
    bounds: bool


@dataclass(frozen=True)
class ConstraintComparison:
    """What the Constraints section draws.

    Attributes:
        alignment: The constraints of the two sides, aligned by name.
        gc_constraints: The candidate's declared constraints.
        gc_bounds: The candidate's hours-weighted stage bounds.
        gc_lhs_newave: The baseline's left-hand side under the paired
            candidate's constraint ids; unpaired constraints have no row. The
            field is named for the render slot the baseline occupies.
        gc_lhs_novomodelo: The candidate's left-hand side.
        differing: The paired constraints defined differently in the two sides.
    """

    alignment: NameAlignment
    gc_constraints: list[dict]
    gc_bounds: pl.DataFrame
    gc_lhs_newave: pl.DataFrame
    gc_lhs_novomodelo: pl.DataFrame
    differing: tuple[DifferingConstraint, ...]


def constraint_list_usable(side: SideInputs) -> bool:
    """Whether *side*'s generic constraints can be aligned by name.

    An absent file is a case without generic constraints, so usable. A file
    that does not parse, or holds an entry without an id or a name, emits the
    side-tagged ``report-file-unreadable`` warning and is not usable.
    """
    with side_tagged(side.side):
        try:
            read_novomodelo_required_generic_constraints(case_dir_for(side.output_dir))
        except SourceFileError:
            return True
        except FieldParseError as exc:
            assert exc.path is not None  # the strict reader always sets it
            dx.emit(
                dx.report_file_unreadable_diagnostic(
                    _FILE, Path(exc.path), _EFFECT, exc
                ),
                logger=_LOG,
            )
            return False
    return True


def _translated(ids: Mapping[int, int] | None, value: int | None) -> _Entity:
    if ids is None or value is None:
        return value
    return ids.get(value, ("baseline", value))


def _term_tables(
    resolved: ResolvedGenericConstraints,
    stages: Collection[int],
    entity_ids: Mapping[str, Mapping[int, int]],
) -> dict[int, dict[_TermKey, float]]:
    tables: dict[int, dict[_TermKey, float]] = {}
    in_stages = resolved.terms.filter(pl.col("stage_id").is_in(list(stages)))
    for row in in_stages.iter_rows(named=True):
        kind = row["variable_kind"]
        key = (
            row["stage_id"],
            row["block_id"],
            kind,
            _translated(entity_ids.get(kind.split("_", 1)[0]), row["entity_id"]),
            row["term_block_id"],
            _translated(entity_ids.get("bus"), row["bus_id"]),
        )
        tables.setdefault(row["constraint_id"], {})[key] = row["coefficient"]
    return tables


def _bound_tables(
    resolved: ResolvedGenericConstraints, stages: Collection[int]
) -> dict[int, dict[_BoundKey, _Endpoints]]:
    tables: dict[int, dict[_BoundKey, _Endpoints]] = {}
    in_stages = resolved.bounds.filter(pl.col("stage_id").is_in(list(stages)))
    for row in in_stages.iter_rows(named=True):
        tables.setdefault(row["constraint_id"], {})[
            (row["stage_id"], row["block_id"])
        ] = (row["bound_lower"], row["bound_upper"])
    return tables


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_RELATIVE_TOLERANCE, abs_tol=0.0)


def _terms_differ(
    baseline: Mapping[_TermKey, float], candidate: Mapping[_TermKey, float]
) -> bool:
    return baseline.keys() != candidate.keys() or any(
        not _close(coefficient, candidate[key]) for key, coefficient in baseline.items()
    )


def _endpoints_differ(baseline: float | None, candidate: float | None) -> bool:
    if baseline is None or candidate is None:
        return (baseline is None) != (candidate is None)
    return not _close(baseline, candidate)


def _bounds_differ(
    baseline: Mapping[_BoundKey, _Endpoints], candidate: Mapping[_BoundKey, _Endpoints]
) -> bool:
    return baseline.keys() != candidate.keys() or any(
        _endpoints_differ(base_endpoint, candidate_endpoint)
        for key, endpoints in baseline.items()
        for base_endpoint, candidate_endpoint in zip(
            endpoints, candidate[key], strict=True
        )
    )


def definition_differences(
    baseline: ResolvedGenericConstraints,
    candidate: ResolvedGenericConstraints,
    pairs: Sequence[AlignedPair],
    entity_ids: Mapping[str, Mapping[int, int]],
    stages: Collection[int],
) -> tuple[DifferingConstraint, ...]:
    """The paired constraints whose terms or bounds differ over *stages*, in the
    order of *pairs*.

    *entity_ids* maps an entity type to the baseline id and the candidate id of
    each entity paired by name. A term compares by stage, block, kind, entity,
    term block and bus, the baseline's entities and buses translated through it;
    a baseline entity without a pair equals no candidate entity. Coefficients
    and bounds compare with a relative tolerance of ``1e-9``, never by absolute
    difference. The ``expression`` text is not compared: it embeds entity ids.
    """
    baseline_terms = _term_tables(baseline, stages, entity_ids)
    candidate_terms = _term_tables(candidate, stages, {})
    baseline_bounds = _bound_tables(baseline, stages)
    candidate_bounds = _bound_tables(candidate, stages)
    differing = []
    for pair in pairs:
        terms = _terms_differ(
            baseline_terms.get(pair.baseline_id, {}),
            candidate_terms.get(pair.candidate_id, {}),
        )
        bounds = _bounds_differ(
            baseline_bounds.get(pair.baseline_id, {}),
            candidate_bounds.get(pair.candidate_id, {}),
        )
        if terms or bounds:
            differing.append(DifferingConstraint(pair.name, terms, bounds))
    return tuple(differing)


def compare_generic_constraints(
    baseline: SideInputs,
    candidate: SideInputs,
    alignments: Mapping[str, NameAlignment],
    stages: Collection[int],
) -> ConstraintComparison:
    """Read both sides' generic constraints and compare them.

    Both sides must have passed :func:`constraint_list_usable`. *alignments* holds
    the entity types aligned so far and is read only to translate the entity ids
    of the baseline's terms; *stages* are the common stage ids. Reader
    Diagnostics reach the enclosing sink tagged with their side. The left-hand
    side frames are not restricted to *stages*.

    Raises:
        NovomodeloReadError: a resolved-constraint echo is present and cannot
            be read.
    """
    with side_tagged(baseline.side):
        baseline_resolved = read_novomodelo_resolved_generic_constraints(
            case_dir_for(baseline.output_dir), baseline.output_dir
        )
        baseline_lhs = evaluate_lhs_novomodelo(baseline_resolved, baseline.output_dir)
    with side_tagged(candidate.side):
        candidate_resolved = read_novomodelo_resolved_generic_constraints(
            case_dir_for(candidate.output_dir), candidate.output_dir
        )
        candidate_lhs = evaluate_lhs_novomodelo(
            candidate_resolved, candidate.output_dir
        )
        candidate_bounds = stage_bounds_novomodelo(
            candidate_resolved, candidate.output_dir
        )
    alignment = align_by_name(
        "generic_constraint",
        baseline_resolved.constraints,
        candidate_resolved.constraints,
    )
    candidate_id = {pair.baseline_id: pair.candidate_id for pair in alignment.pairs}
    entity_ids = {
        entity_type: {pair.baseline_id: pair.candidate_id for pair in aligned.pairs}
        for entity_type, aligned in alignments.items()
    }
    return ConstraintComparison(
        alignment=alignment,
        gc_constraints=candidate_resolved.constraints,
        gc_bounds=candidate_bounds,
        gc_lhs_newave=baseline_lhs.with_columns(
            pl.col("constraint_id").replace_strict(
                candidate_id, default=None, return_dtype=pl.Int32
            )
        ).drop_nulls("constraint_id"),
        gc_lhs_novomodelo=candidate_lhs,
        differing=definition_differences(
            baseline_resolved,
            candidate_resolved,
            alignment.pairs,
            entity_ids,
            stages,
        ),
    )

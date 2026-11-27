"""The facts of a version comparison, as one record every surface reads.

The Diagnostics, the report, the ``--json`` summary and ``comparison.json`` print
the same facts, so they are built once by :func:`build_facts` and stored in the
dataset's metadata under :data:`METADATA_KEY`; no surface recomputes one.
:meth:`VersionFacts.to_dict` holds JSON-native values only, which is what lets
the record survive ``ComparisonDataset.to_dir`` and ``from_dir``. The field
names and :data:`METADATA_KEY` are persisted: a one-way contract.

The functions emit no Diagnostic and read no file.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any, Self

from novomodelo_bridge.comparators.version.alignment import NameAlignment
from novomodelo_bridge.comparators.version.availability import Unavailable
from novomodelo_bridge.comparators.version.convergence import (
    BoundsComparison,
    FinalBounds,
)
from novomodelo_bridge.comparators.version.sides import SideInputs

if TYPE_CHECKING:
    from novomodelo_bridge.comparators.dataset import ComparisonDataset

#: The dataset metadata key holding :meth:`VersionFacts.to_dict`.
METADATA_KEY: str = "version_comparison"


@dataclass(frozen=True)
class VersionFacts:
    """The facts of one version comparison, in the order they are persisted.

    ``BoundsComparison.risk_measures_differ`` is not a field: it is read in
    memory only and never persisted.

    Attributes:
        baseline_output_dir: The directory as the caller typed it, never
            resolved, so the record holds no absolute path the caller did not type.
        candidate_output_dir: As *baseline_output_dir*.
        baseline_version: ``None`` when the output does not say.
        candidate_version: As *baseline_version*.
        baseline_label: The side's label, as ``side_labels`` gives it.
        candidate_label: As *baseline_label*.
        only_in_baseline: One key per aligned entity type, in the alignments'
            order, each the stripped names only the baseline holds; ``[]`` when
            none. An entity type that was not aligned has no key: its list was
            unusable, and the matching row of *unavailable* says so.
        only_in_candidate: As *only_in_baseline*, for the candidate.
        unavailable: One ``{"section", "column", "side"}`` per skipped item, as
            ``unavailable_rows`` gave them; *column* is ``None`` for a whole
            section and *side* is ``"baseline"``, ``"candidate"`` or ``"both"``.
        upper_bound_comparable: Whether *upper_bound_comparability* is
            ``"comparable"``.
        upper_bound_comparability: A member of ``convergence.UB_COMPARABILITY``.
        baseline_upper_bound_kind: ``None`` when the baseline has no final bounds
            or its output does not say.
        candidate_upper_bound_kind: As *baseline_upper_bound_kind*.
        baseline_gap_percent: The baseline's optimality gap, in percent; ``None``
            when it has none. Never NaN or infinity.
        candidate_gap_percent: As *baseline_gap_percent*, for the candidate.
        gap_delta_pp: Candidate minus baseline gap, in percentage points;
            ``None`` unless both gaps exist.
    """

    baseline_output_dir: str
    candidate_output_dir: str
    baseline_version: str | None
    candidate_version: str | None
    baseline_label: str
    candidate_label: str
    only_in_baseline: dict[str, list[str]]
    only_in_candidate: dict[str, list[str]]
    unavailable: list[dict[str, str | None]]
    upper_bound_comparable: bool
    upper_bound_comparability: str
    baseline_upper_bound_kind: str | None
    candidate_upper_bound_kind: str | None
    baseline_gap_percent: float | None
    candidate_gap_percent: float | None
    gap_delta_pp: float | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-native values only, keys in field order; it serializes with
        ``json.dumps(..., allow_nan=False)``."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """The inverse of :meth:`to_dict`; the record shares no container with *data*.

        Raises:
            KeyError: *data* lacks a field; nothing defaults.
        """
        return cls(
            **{field.name: copy.deepcopy(data[field.name]) for field in fields(cls)}
        )


def _kind(side: FinalBounds | None) -> str | None:
    return None if side is None else side.kind


def build_facts(
    baseline: SideInputs,
    candidate: SideInputs,
    alignments: Sequence[NameAlignment],
    unavailable: Sequence[Unavailable],
    bounds: BoundsComparison,
    labels: tuple[str, str],
) -> VersionFacts:
    """Assemble the record from the parts of a comparison, adding, merging and
    dropping nothing.

    *alignments* holds one entry per entity type whose list is usable on both
    sides; *unavailable* is the output of ``unavailable_rows``; *labels* is what
    ``side_labels`` returned.

    Raises:
        ValueError: two alignments share an ``entity_type``; the message names it.
    """
    seen: set[str] = set()
    for alignment in alignments:
        if alignment.entity_type in seen:
            raise ValueError(f"two alignments of entity type {alignment.entity_type!r}")
        seen.add(alignment.entity_type)
    return VersionFacts(
        baseline_output_dir=str(baseline.output_dir),
        candidate_output_dir=str(candidate.output_dir),
        baseline_version=baseline.version,
        candidate_version=candidate.version,
        baseline_label=labels[0],
        candidate_label=labels[1],
        only_in_baseline={a.entity_type: list(a.only_in_baseline) for a in alignments},
        only_in_candidate={
            a.entity_type: list(a.only_in_candidate) for a in alignments
        },
        unavailable=[
            {"section": row.section, "column": row.column, "side": row.side}
            for row in unavailable
        ],
        upper_bound_comparable=bounds.upper_bound_comparable,
        upper_bound_comparability=bounds.comparability,
        baseline_upper_bound_kind=_kind(bounds.baseline),
        candidate_upper_bound_kind=_kind(bounds.candidate),
        baseline_gap_percent=bounds.baseline_gap_percent,
        candidate_gap_percent=bounds.candidate_gap_percent,
        gap_delta_pp=bounds.gap_delta_pp,
    )


def facts_of(dataset: ComparisonDataset) -> VersionFacts:
    """The record stored in *dataset*'s metadata.

    Raises:
        TypeError: the metadata holds no dict under :data:`METADATA_KEY`.
    """
    stored = dataset.metadata.get(METADATA_KEY)
    if not isinstance(stored, dict):
        raise TypeError(
            f"dataset metadata '{METADATA_KEY}' must be a dict, "
            f"got {type(stored).__name__}"
        )
    return VersionFacts.from_dict(stored)

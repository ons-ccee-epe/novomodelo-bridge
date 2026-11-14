"""Leaf ANALYZE-layer module deriving the headline compare verdict.

This module turns a canonical :class:`ComparisonDataset` into a
:class:`CompareVerdict`: how many compared variables fall in each agreement
band of :mod:`novomodelo_bridge.core.agreement`, and which ones are far. Every
number is read straight from ``dataset.summary`` rows, whose gaps and bands
were computed with the compare tolerance upstream; nothing is recomputed
here. Policy rows (:func:`~novomodelo_bridge.core.agreement.is_policy_row`)
are left out, so the headline judges the simulated operation only.

It is a leaf: it imports nothing from the other comparators at runtime (the
:class:`ComparisonDataset` type is referenced under ``TYPE_CHECKING`` only),
so the verdict can be wired into the console without creating import cycles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from novomodelo_bridge.core.agreement import Band, is_policy_row
from novomodelo_bridge.core.summary_counts import summary_tolerance

if TYPE_CHECKING:
    from novomodelo_bridge.comparators.dataset import ComparisonDataset


@dataclass(frozen=True)
class FarVariable:
    """A compared variable whose gap is far."""

    entity_type: str
    variable: str
    #: ``None`` when the gap is undefined.
    gap: float | None


@dataclass(frozen=True)
class CompareVerdict:
    """The compare headline: the operation variables counted by band.

    Attributes:
        tolerance: The close band the summary was judged with.
        close: Variables whose gap is within the tolerance.
        moderate: Variables between the close and far bands.
        far: Variables beyond the far band, or with an undefined gap.
        far_variables: The far variables, largest gap first.
    """

    tolerance: float
    close: int
    moderate: int
    far: int
    far_variables: tuple[FarVariable, ...]

    @property
    def total(self) -> int:
        """Variables compared."""
        return self.close + self.moderate + self.far

    @property
    def all_close(self) -> bool:
        """Whether at least one variable was compared and every one is close."""
        return self.total > 0 and self.close == self.total


def build_compare_verdict(dataset: ComparisonDataset) -> CompareVerdict:
    """Count the dataset's summary rows by band, leaving out policy rows.

    Far variables are ordered by descending gap, an undefined gap first, and
    then by ``(entity_type, variable)`` so the order is deterministic.
    """
    counts = dict.fromkeys(Band, 0)
    far: list[FarVariable] = []
    for row in dataset.summary.to_dicts():
        entity_type, variable = str(row["entity_type"]), str(row["variable"])
        if is_policy_row(entity_type, variable):
            continue
        row_band = Band(row["band"])
        counts[row_band] += 1
        if row_band is Band.FAR:
            gap = row["gap"]
            far.append(
                FarVariable(entity_type, variable, None if gap is None else float(gap))
            )
    far.sort(
        key=lambda v: (
            -(math.inf if v.gap is None else v.gap),
            v.entity_type,
            v.variable,
        )
    )
    return CompareVerdict(
        tolerance=summary_tolerance(dataset.metadata),
        close=counts[Band.CLOSE],
        moderate=counts[Band.MODERATE],
        far=counts[Band.FAR],
        far_variables=tuple(far),
    )


def compare_status(dataset: ComparisonDataset) -> str:
    """The compare ``--json`` status: one vocabulary for both tracks.

    ``"no-comparable-rows"`` when no operation variable was compared, ``"ok"``
    when every one is close, else ``"mismatch"``. Judged on ``summary``, not
    ``tidy``, which also carries single-source percentile rows.
    """
    verdict = build_compare_verdict(dataset)
    if verdict.total == 0:
        return "no-comparable-rows"
    return "ok" if verdict.all_close else "mismatch"

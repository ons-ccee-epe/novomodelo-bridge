"""How both tracks treat a constraint term on an entity outside the case.

The treatment follows the term's cause, not the track or file it comes from:
a term that is zero in the source model too is left out of its sum with a
note, and a term whose value is unknown leaves the whole constraint out with
a warning, rather than convert a sum the source model does not bound.
"""

from __future__ import annotations

from enum import Enum

from novomodelo_bridge.core.diagnostics import Severity


class OutsideTerm(Enum):
    """Why a constraint term names an entity the case does not hold."""

    #: An interchange term on a subsystem pair no line joins: no flow runs.
    NO_LINE = "no-line"
    #: A plant the source model holds but that generates nothing.
    NOT_GENERATING = "not-generating"
    #: A code absent from the source model's configuration.
    ABSENT = "absent"

    @property
    def drops_constraint(self) -> bool:
        """Whether the term leaves its whole constraint out, not only itself."""
        return self is OutsideTerm.ABSENT

    @property
    def severity(self) -> Severity:
        """WARNING when the case loses the constraint, INFO when only a term
        that is zero in the source model too."""
        return Severity.WARNING if self.drops_constraint else Severity.INFO

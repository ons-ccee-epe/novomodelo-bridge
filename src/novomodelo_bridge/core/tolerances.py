"""The shared relative-tolerance idiom for comparing a float against a
declared/reference value, mirroring cobre-io's semantic-validation
``ENVELOPE_TOLERANCE``.

Also carries :func:`is_effectively_infinite`, the shared "is this bound
unbounded?" predicate for the source model's big-M sentinel family.
"""

from __future__ import annotations

import math

#: Mirrors cobre-io's semantic-validation ``ENVELOPE_TOLERANCE``. A relative
#: tolerance, not an absolute one — an absolute epsilon would false-fire on a
#: plant declared at, say, 1e6 m^3/s and float-noise-pass a plant near zero.
RELATIVE_TOLERANCE: float = 1e-9


def relative_tolerance(reference: float) -> float:
    """The absolute magnitude ``RELATIVE_TOLERANCE * max(|reference|, 1.0)``
    for a comparison against *reference*.
    """
    return RELATIVE_TOLERANCE * max(abs(reference), 1.0)


def floats_differ(value: float, reference: float) -> bool:
    """Whether *value* differs from *reference* past relative float noise."""
    return abs(value - reference) > relative_tolerance(reference)


# The source model bound records use 99999 as a "big-M" sentinel meaning "no limit".
# Compare with >= this threshold to catch the family of 9999x sentinels.
BIG_M = 99990.0


def is_effectively_infinite(value: float) -> bool:
    """Return True if *value* represents an unbounded bound.

    Catches both IEEE infinity and the source model's big-M sentinel (``abs(value) >=
    BIG_M`` — the 99999 family meaning "no limit"). Shared with the chart layer so
    the "is this bound unbounded?" test has one definition.
    """
    return math.isinf(value) or abs(value) >= BIG_M

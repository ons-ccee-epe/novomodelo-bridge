"""The shared relative-tolerance idiom for comparing a float against a
declared/reference value, mirroring novomodelo-io's semantic-validation
``ENVELOPE_TOLERANCE``, and the wider written-precision gap below which a
report treats two values as the same number.

Also carries :func:`is_effectively_infinite`, the shared "is this bound
unbounded?" predicate for the source model's big-M sentinel family.
"""

from __future__ import annotations

import math

#: Mirrors novomodelo-io's semantic-validation ``ENVELOPE_TOLERANCE``. A relative
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


#: Relative gap within which a deck limit and a declared capacity are the same
#: number written at different precision. ``hidr.dat`` stores unit power as a
#: float32 of a value typed to a few decimals: BELO MONTE's 18 units of 611.111
#: MW (11000 / 18 cut to three decimals) sum to 10999.9984 MW, 1.4e-7 below
#: the 11000 MW its RE ceiling writes.
WRITTEN_PRECISION_TOLERANCE: float = 1e-6


def beyond_written_precision(value: float, reference: float) -> bool:
    """Whether *value* differs from *reference* by more than the deck's written
    precision (:data:`WRITTEN_PRECISION_TOLERANCE`, relative to
    ``max(|reference|, 1.0)``)."""
    return abs(value - reference) > WRITTEN_PRECISION_TOLERANCE * max(
        abs(reference), 1.0
    )


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

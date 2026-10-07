"""Small pandas/NumPy helpers shared across converters."""

from __future__ import annotations

import math

import pandas as pd


def is_na(value: object) -> bool:
    """Return True if *value* is a pandas/NumPy NA or a float NaN sentinel.

    Covers the scalar cases the converters hit: a Python ``float`` NaN, and the
    ``pd.NA`` / ``NaT`` / ``numpy.nan`` values that :func:`pandas.isna`
    recognises. Inputs that :func:`pandas.isna` can't classify to a single bool
    (e.g. an array) are treated as "not NA" rather than raising.

    This replaces the two slightly different ``_is_na`` definitions that used to
    live in ``hydro.py`` and ``network.py``.
    """
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False

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


def as_frame(accessor_result: object) -> pd.DataFrame | None:
    """Narrow a ``df=True`` accessor result to the frame it actually is.

    ``inewave``/``idecomp`` annotate those accessors as ``Record | list[Record] |
    DataFrame | None`` for every ``df`` value, so a bare result fails type
    narrowing; with ``df=True`` they return a frame, or ``None`` when the
    register is absent.
    """
    return accessor_result if isinstance(accessor_result, pd.DataFrame) else None


def as_frame_or_empty(accessor_result: object) -> pd.DataFrame:
    """:func:`as_frame` for a ``libs`` reader, whose ``df=True`` accessors yield an
    empty frame (never ``None``) when the register is absent."""
    frame = as_frame(accessor_result)
    return pd.DataFrame() if frame is None else frame


def as_int(value: object) -> int:
    """Convert a pandas/polars cell to ``int``.

    Cells and group keys are typed as a scalar union (or ``Hashable``) that
    ``int()`` rejects statically; an unconvertible value still raises as ``int()`` does.
    """
    return int(value)  # ty: ignore[invalid-argument-type] cell typed as a scalar union


def as_float(value: object) -> float:
    """Convert a pandas/polars cell to ``float`` (see :func:`as_int`)."""
    return float(value)  # ty: ignore[invalid-argument-type] cell typed as a scalar union

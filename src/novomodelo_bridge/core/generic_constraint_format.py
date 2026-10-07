"""Shared sense <-> interval mapping for cobre's F3 generic-constraint bounds.

cobre's generic-constraint model (the ``generic-constraint-authoring`` F3
clean break) is **sense-free**: a ``generic_constraints.json`` object carries
no ``sense`` key, and its companion bounds table carries two nullable
endpoints (``bound_lower``/``bound_upper``) instead of a single ``bound``
column. Direction is encoded entirely by which endpoint(s) are populated:
``>=`` -> lower-only, ``<=`` -> upper-only, ``==`` -> both endpoints equal to
the same value, and a genuine band -> both endpoints independently populated.

This module is model-agnostic — no coupling to the source model or DECOMP —
and is the single place every bridge writer (and, from there, every reader)
maps between the pre-F3 ``(sense, value)`` shape and the F3 interval shape.
"""

from __future__ import annotations

#: Canonical column list/order for a generic-constraint bounds table (the F3
#: shape): the per-``(constraint, stage)`` key, a nullable per-block axis,
#: and two nullable interval endpoints in place of the pre-F3 single
#: ``bound`` column.
GENERIC_BOUNDS_COLUMNS = [
    "constraint_id",
    "stage_id",
    "block_id",
    "bound_lower",
    "bound_upper",
]


def sense_to_interval(sense: str, value: float) -> tuple[float | None, float | None]:
    """Map a pre-F3 ``(sense, value)`` pair to an F3 ``(bound_lower, bound_upper)``.

    ``">="`` -> ``(value, None)``; ``"<="`` -> ``(None, value)``; ``"=="`` ->
    ``(value, value)``.

    Raises
    ------
    ValueError
        If *sense* is not one of ``">="``, ``"<="``, ``"=="``.
    """
    if sense == ">=":
        return value, None
    if sense == "<=":
        return None, value
    if sense == "==":
        return value, value
    raise ValueError(f"Unknown constraint sense {sense!r}.")


def shape_from_bounds(lower: float | None, upper: float | None) -> str:
    """Infer a constraint's direction/shape from its F3 bound endpoints.

    The inverse of :func:`sense_to_interval`, for readers: a lower-only
    endpoint pair means ``">="``; upper-only means ``"<="``; both endpoints
    populated and equal means ``"=="``; both populated and distinct means
    ``"range"`` (a genuine two-sided band, which has no pre-F3 equivalent).

    Raises
    ------
    ValueError
        If both *lower* and *upper* are ``None`` — a bound row must carry at
        least one endpoint.
    """
    if lower is None and upper is None:
        raise ValueError(
            "A constraint bound row must have at least one endpoint "
            "(bound_lower and bound_upper are both None)."
        )
    if lower is None:
        return "<="
    if upper is None:
        return ">="
    return "==" if lower == upper else "range"

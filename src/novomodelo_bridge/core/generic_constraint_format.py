"""Shared wire format of novomodelo's generic constraints: the expression text
and the sense <-> interval mapping of their bounds.

novomodelo's generic-constraint model is **sense-free**: a
``generic_constraints.json`` object carries no ``sense`` key, and its
companion bounds table carries two nullable
endpoints (``bound_lower``/``bound_upper``) instead of a single ``bound``
column. Direction is encoded entirely by which endpoint(s) are populated:
``>=`` -> lower-only, ``<=`` -> upper-only, ``==`` -> both endpoints equal to
the same value, and a genuine band -> both endpoints independently populated.

This module is model-agnostic — no coupling to the source model or DECOMP —
and is the single place every bridge writer renders a constraint's
``expression`` (:func:`render_expression`) and maps a ``(sense, value)``
pair onto the interval shape, the mapping every reader inverts
(:func:`shape_from_bounds`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Canonical column list/order for a generic-constraint bounds table: the
#: per-``(constraint, stage)`` key, a nullable per-block axis, and two
#: nullable interval endpoints.
GENERIC_BOUNDS_COLUMNS = [
    "constraint_id",
    "stage_id",
    "block_id",
    "bound_lower",
    "bound_upper",
]

#: A term's weight in an expression: a literal coefficient, or the ``@name``
#: of the per-stage parameter carrying a stage-varying one.
Weight = float | str


def render_expression(terms: Sequence[tuple[Weight, str]]) -> str | None:
    """Join ``(weight, token)`` pairs into a novomodelo expression string.

    A unit (``|coeff| == 1.0``) term is the bare token (``- token`` when
    negative); any other coefficient is ``{abs_coeff} * {token}`` (``-
    {abs_coeff} * {token}`` when negative). A parameter weight is
    ``@{name} * {token}``, its sign carried by the parameter's values. The
    first term omits the leading ``+``; every later term uses ``+``/``-`` as
    a binary operator. Returns ``None`` for an empty term list.
    """
    if not terms:
        return None
    parts: list[str] = []
    for i, (coeff, token) in enumerate(terms):
        if isinstance(coeff, str):
            body = f"@{coeff} * {token}"
            parts.append(body if i == 0 else f"+ {body}")
            continue
        abs_coeff = abs(coeff)
        is_negative = coeff < 0.0
        body = token if abs_coeff == 1.0 else f"{abs_coeff} * {token}"
        if i == 0:
            parts.append(f"- {body}" if is_negative else body)
        else:
            parts.append(f"- {body}" if is_negative else f"+ {body}")
    return " ".join(parts)


def sense_to_interval(sense: str, value: float) -> tuple[float | None, float | None]:
    """Map a ``(sense, value)`` pair to a ``(bound_lower, bound_upper)`` interval.

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
    """Infer a constraint's direction/shape from its interval bound endpoints.

    The inverse of :func:`sense_to_interval`, for readers: a lower-only
    endpoint pair means ``">="``; upper-only means ``"<="``; both endpoints
    populated and equal means ``"=="``; both populated and distinct means
    ``"range"`` (a genuine two-sided band, which no single ``sense`` can express).

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

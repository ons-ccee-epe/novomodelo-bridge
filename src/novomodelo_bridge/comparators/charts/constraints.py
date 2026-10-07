"""Generic-constraint (RE, AGRINT, VminOP) chart: source-model vs Cobre LHS."""

from __future__ import annotations

import polars as pl

from cobre_bridge.comparators.constraints import ResolvedBound
from cobre_bridge.comparators.html_report import (
    COLOR_COBRE,
    COLOR_NEWAVE,
)
from cobre_bridge.ui.html.plotly import facet_grid
from cobre_bridge.ui.html.plotly import plotly_div as _plotly_div


def constraints_comparison_chart(
    constraints: list[dict],
    lhs_newave: pl.DataFrame,
    lhs_cobre: pl.DataFrame,
    bound_by_constraint: dict[int, dict[int, ResolvedBound]],
    reference_label: str = "NEWAVE",
) -> str:
    """Per-constraint small-multiples comparing the source model vs Cobre LHS vs bound.

    One panel per constraint. Each panel shows the per-stage the source model LHS (mean
    evaluated from MEDIAS-USIH / int*.out outputs), the Cobre LHS (mean across scenarios
    and blocks from simulation parquet), and the constraint bound (dashed) overlaid as a
    horizontal-step series for every stage where the bound is defined. Constraints with
    no LHS data on either side are skipped silently.

    Parameters
    ----------
    constraints:
        Constraint dicts loaded from ``generic_constraints.json``.
    lhs_newave, lhs_cobre:
        DataFrames with columns ``constraint_id``, ``stage_id``,
        ``lhs_value``.
    bound_by_constraint:
        Output of
        :func:`cobre_bridge.comparators.constraints.per_stage_bounds`
        — maps ``constraint_id`` to ``{stage_id: ResolvedBound}`` (the
        resolved limit value plus its derived shape label).

    Returns
    -------
    str
        HTML ``<div>`` with an embedded Plotly figure.  Returns a short
        fallback ``<p>`` when no constraints have any data.
    """
    if not constraints:
        return "<p>No generic constraints defined.</p>"

    # Index by id for fast lookup.
    nw_by_cid: dict[int, dict[int, float]] = {}
    cb_by_cid: dict[int, dict[int, float]] = {}
    if not lhs_newave.is_empty():
        for r in lhs_newave.iter_rows(named=True):
            nw_by_cid.setdefault(int(r["constraint_id"]), {})[int(r["stage_id"])] = (
                float(r["lhs_value"])
            )
    if not lhs_cobre.is_empty():
        for r in lhs_cobre.iter_rows(named=True):
            cb_by_cid.setdefault(int(r["constraint_id"]), {})[int(r["stage_id"])] = (
                float(r["lhs_value"])
            )

    # A source-model↔Cobre comparison needs the reference (source-model) LHS:
    # facet only the constraints the reference side actually evaluated an LHS
    # for. Constraints with a bound (or a Cobre-only LHS) but NO reference LHS
    # — the cobre-only FI/QBOM terms, 52 of 76 on the mar-26 deck — have nothing
    # to compare against, and faceting them wallpapered the tab with dozens of
    # reference-less panels (a 38-row grid). Dropping them keeps the grid a
    # readable size and every panel a genuine comparison.
    renderable: list[dict] = []
    for c in constraints:
        cid = int(c["id"])
        if cid in nw_by_cid:
            renderable.append(c)
    if not renderable:
        return "<p>No constraint data available to compare.</p>"

    ncols = 2
    nrows = (len(renderable) + ncols - 1) // ncols
    panels = facet_grid(len(renderable), ncols=2)

    traces: list[dict] = []
    layout: dict = {"title": "Generic Constraints — LHS vs Bound"}
    first = True

    for idx, c in enumerate(renderable):
        cid = int(c["id"])
        name = c["name"]
        cid_bounds = bound_by_constraint.get(cid, {})
        # F3 constraints are sense-free; the shape (">="/"<="/"=="/"range") is
        # whatever per_stage_bounds derived from the resolved bound endpoints,
        # not a removed `sense` field.
        shape = next((rb.shape for rb in cid_bounds.values()), "<=")
        panel = panels[idx]
        row_i = panel.row
        ax_idx = idx + 1
        xa = f"x{ax_idx}" if ax_idx > 1 else "x"
        ya = f"y{ax_idx}" if ax_idx > 1 else "y"

        xa_key = f"xaxis{ax_idx}" if ax_idx > 1 else "xaxis"
        ya_key = f"yaxis{ax_idx}" if ax_idx > 1 else "yaxis"
        layout[xa_key] = {
            "domain": panel.x_domain,
            "title": "Stage" if row_i == nrows - 1 else "",
            "anchor": ya,
        }
        layout[ya_key] = {
            "domain": panel.y_domain,
            "title": f"{name} ({shape})",
            "anchor": xa,
        }

        # Union of stages with any data (the source model LHS, Cobre LHS, or bound).
        stages = sorted(
            set(nw_by_cid.get(cid, {}).keys())
            | set(cb_by_cid.get(cid, {}).keys())
            | set(cid_bounds.keys())
        )
        if not stages:
            continue

        nw_y = [nw_by_cid.get(cid, {}).get(s) for s in stages]
        cb_y = [cb_by_cid.get(cid, {}).get(s) for s in stages]
        bound_y = [cid_bounds[s].value if s in cid_bounds else None for s in stages]

        # The source model LHS line (only where defined).
        nw_x_present = [s for s, v in zip(stages, nw_y) if v is not None]
        nw_v_present = [v for v in nw_y if v is not None]
        if nw_x_present:
            traces.append(
                {
                    "x": nw_x_present,
                    "y": nw_v_present,
                    "name": f"{reference_label} LHS",
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"color": COLOR_NEWAVE, "width": 2},
                    "xaxis": xa,
                    "yaxis": ya,
                    "legendgroup": "nw",
                    "showlegend": first,
                }
            )

        # Cobre LHS line.
        cb_x_present = [s for s, v in zip(stages, cb_y) if v is not None]
        cb_v_present = [v for v in cb_y if v is not None]
        if cb_x_present:
            traces.append(
                {
                    "x": cb_x_present,
                    "y": cb_v_present,
                    "name": "Cobre LHS (mean)",
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"color": COLOR_COBRE, "width": 2},
                    "xaxis": xa,
                    "yaxis": ya,
                    "legendgroup": "cb",
                    "showlegend": first,
                }
            )

        # Bound line (dashed red, only where defined). Use markers so
        # gaps where the bound is None remain visually obvious.
        bd_x_present = [s for s, v in zip(stages, bound_y) if v is not None]
        bd_v_present = [v for v in bound_y if v is not None]
        if bd_x_present:
            traces.append(
                {
                    "x": bd_x_present,
                    "y": bd_v_present,
                    "name": "Bound",
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"color": "#DC4C4C", "width": 1.5, "dash": "dash"},
                    "xaxis": xa,
                    "yaxis": ya,
                    "legendgroup": "bound",
                    "showlegend": first,
                }
            )

        first = False

    return _plotly_div(traces, layout, height=max(nrows * 300 + 80, 360))

"""Presentation helpers for the constraints tab.

The constraint-expression *domain* logic (parser, parameter resolver, LHS
evaluator) now lives in :mod:`cobre_bridge.cobre.constraint_expr` so the dashboard and
the comparator share one definition. This module keeps only the HTML rendering
plus the small F3 bound-shape derivation used to display it.
"""

from __future__ import annotations

import pandas as pd

from cobre_bridge.core.generic_constraint_format import shape_from_bounds
from cobre_bridge.ui.html.document import escape_text


def derive_constraint_shape(bounds_rows: pd.DataFrame) -> str:
    """Derive a constraint's *displayed* direction label from its F3 bound endpoints.

    cobre's F3 ``generic_constraints.json`` objects are sense-free; direction
    is encoded entirely by which endpoint(s) of ``generic_constraint_bounds
    .parquet`` are populated (see :mod:`cobre_bridge.core.generic_constraint_format`).
    DECOMP's ``RE``/``HQ``/``HV`` families genuinely emit two-sided
    ``"range"`` rows (a single id carrying both ``bound_lower`` and
    ``bound_upper``), and a constraint's per-row direction can vary across
    its per-(stage, block) bound rows (e.g. an HQ/HV mix that floors some
    stages and ceilings others). This function nonetheless returns a single
    label taken from the *first* row with at least one populated endpoint —
    it feeds only the compact per-constraint "Sense" summary column
    (:func:`build_constraints_summary_table`). Numeric consumers that need
    the real per-(stage, block) bound — the LHS-vs-Bound chart
    (:func:`cobre_bridge.dashboard.tabs.constraints._build_constraint_lhs_data`)
    — read ``bound_lower``/``bound_upper`` directly per row instead of
    relying on this single derived label. Falls back to ``"<="`` — the
    pre-F3 default sense for a constraint with no bound rows — when
    *bounds_rows* has no such row (including when it is empty).
    """
    for _, row in bounds_rows.iterrows():
        lower = row.get("bound_lower")
        upper = row.get("bound_upper")
        lower = None if pd.isna(lower) else float(lower)
        upper = None if pd.isna(upper) else float(upper)
        if lower is None and upper is None:
            continue
        return shape_from_bounds(lower, upper)
    return "<="


def bound_value_column(shape: str) -> str:
    """Return which F3 endpoint column holds the summary display value for *shape*.

    ``"<="``/``"range"`` read ``bound_upper``; ``">="``/``"=="`` read
    ``bound_lower`` (both endpoints are equal for ``"=="``, so either would
    do). For ``"range"`` this deliberately surfaces only the ceiling: it
    feeds the per-constraint "Bound Range" summary column
    (:func:`build_constraints_summary_table`), which shows one endpoint's
    variation across stages, not the floor-to-ceiling span of a band.
    DECOMP's ``RE``/``HQ``/``HV`` families genuinely emit ``"range"`` rows —
    a live path, not a hypothetical one. The LHS-vs-Bound chart does **not**
    use this function: it resolves both endpoints directly per
    (stage, block) row so a two-sided band renders, and is violation-tested
    on, both its floor and its ceiling (see
    :func:`cobre_bridge.dashboard.tabs.constraints._build_constraint_lhs_data`).
    """
    return "bound_upper" if shape in ("<=", "range") else "bound_lower"


def build_constraints_summary_table(
    constraints: list[dict],
    gc_bounds: pd.DataFrame,
    violations_df: pd.DataFrame,
) -> str:
    """HTML summary table of all generic constraints.

    Columns: Name, Type, Sense, Active Stages, Bound Range, Slack, Penalty, Has
    Violations
    Rows are colour-coded by constraint type.
    """
    type_colors = {
        "VminOP": "#EEF4FB",
        "RE": "#F0FAF4",
        "AGRINT": "#FFF8EE",
    }
    rows_html: list[str] = []
    for c in constraints:
        cid = c["id"]
        name = c["name"]
        ctype = name.split("_")[0]
        slack = c["slack"]
        slack_enabled = "Yes" if slack.get("enabled") else "No"
        penalty = (
            f"{slack['penalty']:,.0f}"
            if slack.get("enabled") and "penalty" in slack
            else "—"
        )
        bounds_rows = gc_bounds[gc_bounds["constraint_id"] == cid]
        sense = derive_constraint_shape(bounds_rows)
        active_stages = (
            int(bounds_rows["stage_id"].nunique()) if not bounds_rows.empty else 0
        )
        # Coarse per-constraint summary: for a "range" row this is the
        # ceiling's variation across stages, not the floor-to-ceiling span —
        # the LHS-vs-Bound chart is where a band's floor AND ceiling are
        # both rendered (see `_build_constraint_lhs_data`).
        value_col = bound_value_column(sense)
        bmin = bounds_rows[value_col].min() if not bounds_rows.empty else 0.0
        bmax = bounds_rows[value_col].max() if not bounds_rows.empty else 0.0
        if abs(bmax - bmin) < 1e-6:
            bound_range = f"{bmin:,.1f}"
        else:
            bound_range = f"{bmin:,.1f} – {bmax:,.1f}"
        has_viol = "No"
        if not violations_df.empty and "constraint_id" in violations_df.columns:
            viol_sub = violations_df[violations_df["constraint_id"] == cid]
            if not viol_sub.empty and viol_sub["slack_value"].abs().sum() > 1e-6:
                has_viol = "Yes"
        bg = type_colors.get(ctype, "#FFFFFF")
        viol_style = (
            ' style="color:#DC4C4C;font-weight:600;"' if has_viol == "Yes" else ""
        )
        rows_html.append(
            f'<tr style="background:{bg};">'
            f"<td>{escape_text(name)}</td>"
            f"<td>{escape_text(ctype)}</td>"
            f"<td><code>{escape_text(sense)}</code></td>"
            f"<td style='text-align:center;'>{active_stages}</td>"
            f"<td style='text-align:right;'>{bound_range}</td>"
            f"<td style='text-align:center;'>{slack_enabled}</td>"
            f"<td style='text-align:right;'>{penalty}</td>"
            f"<td style='text-align:center;'{viol_style}>{has_viol}</td>"
            "</tr>"
        )

    legend_html = (
        '<div style="margin-top:8px;font-size:0.8rem;color:#666;">'
        '<span style="background:#EEF4FB;padding:2px 8px;margin-right:8px;">VminOP — Minimum stored energy</span>'
        '<span style="background:#F0FAF4;padding:2px 8px;margin-right:8px;">RE — Electric constraint</span>'
        '<span style="background:#FFF8EE;padding:2px 8px;">AGRINT — Exchange group constraint</span>'
        "</div>"
    )

    return (
        '<table class="data-table">'
        "<thead><tr>"
        "<th>Name</th><th>Type</th><th>Sense</th>"
        "<th>Active Stages</th><th>Bound Range</th>"
        "<th>Slack</th><th>Penalty (R$/unit)</th><th>Has Violations</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>" + legend_html
    )

"""v2 Constraints tab module for the Cobre dashboard.

Displays generic constraint summary, LHS vs bound charts, bounds timeline,
violation cost timeline, and violation summary/heatmap. Only rendered when
generic constraint definitions are present in the case.

Reuses chart functions from constraints.py and wraps them in the v2 UI
framework (collapsible sections, make_chart_card, chart_grid, metric_card,
metrics_grid).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from cobre_bridge.cobre.constraint_expr import (
    evaluate_constraint_expressions,
    load_rho_acum_overrides,
)
from cobre_bridge.dashboard.tabs.constraints_utils import (
    build_constraints_summary_table,
    derive_constraint_shape,
)
from cobre_bridge.ui.html.document import (
    escape_attr,
    json_for_script,
    metric_card,
    metrics_grid,
    section_title,
)
from cobre_bridge.ui.html.plotly import stage_x_dates, stage_x_labels

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-constraints"
TAB_LABEL = "Constraints"
TAB_ORDER = 80

# ---------------------------------------------------------------------------
# Metric colors
# ---------------------------------------------------------------------------

_COLOR_TOTAL = "#4A90B8"
_COLOR_VIOLATED = "#DC4C4C"
_COLOR_COST = "#F5A623"
_COLOR_TYPES = "#4A8B6F"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_metrics_row(data: DashboardData) -> str:
    """Build the 4-card metrics row for the Constraints tab.

    Cards:
    1. Total constraints count
    2. Constraints with violations count
    3. Total violation cost (sum of generic_violation_cost across all scenarios/stages)
    4. Active constraint types (comma-joined unique type prefixes)
    """
    total_constraints = len(data.gc_constraints)

    # Count constraints that have at least one nonzero slack_value in gc_violations
    violated_count = 0
    violations_df: pd.DataFrame = data.gc_violations
    if not violations_df.empty and "constraint_id" in violations_df.columns:
        viol_ids = set(
            violations_df[violations_df["slack_value"].abs() > 1e-6][
                "constraint_id"
            ].unique()
        )
        violated_count = len(viol_ids)

    # Total violation cost
    costs_df: pd.DataFrame = data.costs
    if "generic_violation_cost" in costs_df.columns:
        total_viol_cost = float(costs_df["generic_violation_cost"].sum())
    else:
        total_viol_cost = 0.0
    cost_str = f"R$ {total_viol_cost:,.0f}"

    # Active constraint types: unique first element after splitting name on "_"
    type_prefixes: list[str] = []
    seen: set[str] = set()
    for c in data.gc_constraints:
        prefix = c["name"].split("_")[0]
        if prefix not in seen:
            seen.add(prefix)
            type_prefixes.append(prefix)
    active_types_str = ", ".join(type_prefixes) if type_prefixes else "—"

    cards = [
        metric_card(
            value=str(total_constraints),
            label="Total Constraints",
            color=_COLOR_TOTAL,
        ),
        metric_card(
            value=str(violated_count),
            label="Constraints with Violations",
            color=_COLOR_VIOLATED,
        ),
        metric_card(
            value=cost_str,
            label="Total Violation Cost",
            color=_COLOR_COST,
        ),
        metric_card(
            value=active_types_str,
            label="Active Types",
            color=_COLOR_TYPES,
        ),
    ]
    return metrics_grid(cards)


def _compute_violation_zones(
    p10: list[float],
    p90: list[float],
    bound_lower: list[float | None],
    bound_upper: list[float | None],
) -> list[dict[str, int]]:
    """Compute contiguous stage index intervals where the band crosses either bound.

    Two-sided per-(stage, block) test: a stage is violated when its floor is
    breached (``bound_lower[i] is not None and p10[i] < bound_lower[i]``) OR
    its ceiling is breached (``bound_upper[i] is not None and p90[i] >
    bound_upper[i]``) — either condition alone flags the stage, and both may
    fire simultaneously. This one test correctly covers every F3 shape a
    per-row bound can take: ``">="`` (lower-only — only the floor term can
    fire), ``"<="`` (upper-only — only the ceiling term can fire), ``"=="``
    (both populated and equal — either deviation direction is a violation),
    and a genuine two-sided ``"range"`` (both populated and distinct — floor
    and ceiling are independent breaches). A stage with neither endpoint
    populated (both ``None``) carries no bound at that stage and is never a
    violation; it closes any open zone the same way a gap used to.

    Args:
        p10: List of p10 LHS values per stage.
        p90: List of p90 LHS values per stage.
        bound_lower: Per-stage floor (``bound_lower``), ``None`` where unbounded.
        bound_upper: Per-stage ceiling (``bound_upper``), ``None`` where unbounded.

    Returns:
        List of ``{"start": start_idx, "end": end_idx}`` dicts, one per
        contiguous violated interval.  Empty list when no violations occur.
    """
    n = len(p10)
    if n == 0:
        return []

    violations: list[dict[str, int]] = []
    in_zone = False
    zone_start = 0

    for i in range(n):
        lo = bound_lower[i] if i < len(bound_lower) else None
        hi = bound_upper[i] if i < len(bound_upper) else None
        if lo is None and hi is None:
            # No bound at this stage: close any open zone before this gap.
            if in_zone:
                violations.append({"start": zone_start, "end": i - 1})
                in_zone = False
            continue

        violated = (lo is not None and p10[i] < lo) or (hi is not None and p90[i] > hi)

        if violated and not in_zone:
            in_zone = True
            zone_start = i
        elif not violated and in_zone:
            violations.append({"start": zone_start, "end": i - 1})
            in_zone = False

    if in_zone:
        violations.append({"start": zone_start, "end": n - 1})

    return violations


def _build_constraint_lhs_data(
    constraints: list[dict],
    lhs_df: pd.DataFrame,
    gc_bounds: pd.DataFrame,
    stage_labels: dict[int, str],
) -> dict:
    """Build the JSON-serialisable LHS percentile data for all constraints.

    Precomputes p10/p50/p90 of LHS across scenarios per stage for every
    constraint, resolves each stage's bound endpoint(s) directly from its
    own ``gc_bounds`` row, and computes violation zone intervals.

    Args:
        constraints: List of constraint dicts with keys ``id``, ``name``. F3
            constraints are sense-free; the displayed ``sense`` label is
            derived from *gc_bounds* via :func:`derive_constraint_shape` for
            the dropdown/legend only — it does not drive which endpoint(s)
            are plotted or violation-tested (see ``bound_lower``/
            ``bound_upper`` below).
        lhs_df: DataFrame with columns ``constraint_id``, ``scenario_id``,
            ``stage_id``, ``block_id``, ``lhs_value``.
        gc_bounds: DataFrame with columns ``constraint_id``, ``stage_id``,
            ``block_id``, ``bound_lower``, ``bound_upper`` (the F3 sense-free
            interval; see :mod:`cobre_bridge.core.generic_constraint_format`).
        stage_labels: Stage id to human-readable label mapping.

    Returns:
        Dict with keys ``stages`` (list[int]), ``xlabels`` (list[str]), and
        ``constraints`` (dict mapping str(constraint_id) to per-constraint
        entry dicts).  Each entry contains ``name``, ``sense``, ``lhs_p10``,
        ``lhs_p50``, ``lhs_p90``, ``bound``, ``bound_lower``, ``bound_upper``,
        ``violations``. ``bound`` is the legacy single-value series (the
        ceiling when a stage has one, else the floor — unchanged for every
        single-sided constraint); ``bound_lower``/``bound_upper`` are the raw
        per-stage endpoints, both populated wherever a stage carries a
        genuine two-sided ``"range"`` bound.
    """
    # Derive a common sorted stage list
    if not lhs_df.empty:
        all_stages: list[int] = sorted(int(s) for s in lhs_df["stage_id"].unique())
    elif not gc_bounds.empty:
        all_stages = sorted(int(s) for s in gc_bounds["stage_id"].unique())
    else:
        all_stages = []

    xlabels = stage_x_labels(all_stages, stage_labels)
    n_stages = len(all_stages)

    constraints_data: dict[str, dict] = {}
    for c in constraints:
        cid = c["id"]
        name = c["name"]

        # --- LHS percentiles ---
        sub = (
            lhs_df[lhs_df["constraint_id"] == cid].copy()
            if not lhs_df.empty
            else pd.DataFrame()
        )

        if not sub.empty:
            pcts = (
                sub.groupby(["scenario_id", "stage_id"])["lhs_value"]
                .mean()
                .reset_index()
                .groupby("stage_id")["lhs_value"]
                .quantile([0.1, 0.5, 0.9])
                .unstack(level=-1)
                .rename(columns={0.1: "p10", 0.5: "p50", 0.9: "p90"})
            )
            p10 = [round(float(pcts["p10"].get(s, 0.0)), 4) for s in all_stages]
            p50 = [round(float(pcts["p50"].get(s, 0.0)), 4) for s in all_stages]
            p90 = [round(float(pcts["p90"].get(s, 0.0)), 4) for s in all_stages]
        else:
            p10 = [0.0] * n_stages
            p50 = [0.0] * n_stages
            p90 = [0.0] * n_stages

        # --- Bound endpoints, resolved per (stage, block) row ---
        bounds_c = (
            gc_bounds[gc_bounds["constraint_id"] == cid]
            if not gc_bounds.empty
            else pd.DataFrame()
        )
        # `sense` is a per-constraint display label only (see
        # `derive_constraint_shape`'s docstring) — it does not choose which
        # endpoint(s) are read below. Each stage's own bound_lower/bound_upper
        # are read directly, so a genuine two-sided "range" row (a live
        # DECOMP RE/HQ/HV path) keeps both endpoints, and a stage whose own
        # direction differs from the constraint's dominant one is never
        # dropped.
        sense = derive_constraint_shape(bounds_c)
        bound_lower_vals: list[float | None] = []
        bound_upper_vals: list[float | None] = []
        if not bounds_c.empty:
            if bounds_c["block_id"].isna().all():
                by_stage = bounds_c.set_index("stage_id")
            else:
                by_stage = bounds_c[bounds_c["block_id"] == 0.0].set_index("stage_id")
            lower_by_stage = by_stage["bound_lower"]
            upper_by_stage = by_stage["bound_upper"]
            for s in all_stages:
                # `.get` on an all-null column (an unbounded side across
                # every stage) can hand back a bare Python `None` rather
                # than a float NaN, so the null check is `pd.isna` (which
                # handles both) rather than `math.isnan(float(...))`.
                lo_raw = lower_by_stage.get(s, float("nan"))
                hi_raw = upper_by_stage.get(s, float("nan"))
                lo = None if pd.isna(lo_raw) else round(float(lo_raw), 4)
                hi = None if pd.isna(hi_raw) else round(float(hi_raw), 4)
                bound_lower_vals.append(lo)
                bound_upper_vals.append(hi)
        else:
            bound_lower_vals = [None] * n_stages
            bound_upper_vals = [None] * n_stages

        # Legacy single-value series: the ceiling when a stage has one, else
        # the floor. Identical to today's values for every single-sided
        # constraint (only one side is ever populated, so this always picks
        # it); for a two-sided "range" stage it keeps the ceiling, matching
        # the pre-fix value exactly — the floor is now additionally exposed
        # via `bound_lower` rather than dropped.
        bound_vals: list[float | None] = [
            hi if hi is not None else lo
            for lo, hi in zip(bound_lower_vals, bound_upper_vals, strict=True)
        ]

        # --- Violation zones: two-sided per-stage test (see
        # `_compute_violation_zones`) — flags a floor breach, a ceiling
        # breach, or both, independently at each stage.
        violations = _compute_violation_zones(
            p10, p90, bound_lower_vals, bound_upper_vals
        )

        constraints_data[str(cid)] = {
            "name": name,
            "sense": sense,
            "lhs_p10": p10,
            "lhs_p50": p50,
            "lhs_p90": p90,
            "bound": bound_vals,
            "bound_lower": bound_lower_vals,
            "bound_upper": bound_upper_vals,
            "violations": violations,
        }

    return {
        "stages": all_stages,
        "xlabels": xlabels,
        "constraints": constraints_data,
    }


def _build_lhs_section(data: DashboardData, lhs_df: pd.DataFrame) -> str:
    """Build the interactive LHS vs Bound section HTML.

    Embeds precomputed percentile data as a JSON blob and renders a
    ``<select id="gc-constraint-sel">`` dropdown driving a single
    ``<div id="gc-lhs-chart">`` chart via ``Plotly.react()``.

    Args:
        data: Dashboard data object.
        lhs_df: Evaluated LHS DataFrame (may be empty).

    Returns:
        HTML string for the LHS section.
    """
    lhs_data = _build_constraint_lhs_data(
        data.gc_constraints, lhs_df, data.gc_bounds, data.stage_labels
    )
    data_json = json_for_script(lhs_data)
    # ISO date x-positions (proportional date axis); xlabels stay the tick text.
    dates_json = json_for_script(stage_x_dates(lhs_data["stages"], data.stage_dates))

    options_html = "\n".join(
        f'<option value="{c["id"]}">{c["name"]}</option>' for c in data.gc_constraints
    )

    return (
        '<div style="margin-bottom:16px;">'
        '<label for="gc-constraint-sel"'
        ' style="font-weight:600;margin-right:8px;">Select Constraint:</label>'
        '<select id="gc-constraint-sel" onchange="updateConstraintChart()"'
        ' style="padding:8px 12px;font-size:0.9rem;border-radius:4px;'
        'border:1px solid #ccc;min-width:320px;">'
        + options_html
        + "</select>"
        + "</div>"
        + '<div class="chart-grid-single">'
        + '<div class="chart-card">'
        + '<div id="gc-lhs-chart" style="width:100%;height:380px;"></div>'
        + "</div>"
        + "</div>"
        + "<script>\n"
        + "const GC_LHS_DATA = "
        + data_json
        + ";\n"
        + "const GC_LHS_DATES = "
        + dates_json
        + ";\n"
        + r"""
function updateConstraintChart() {
  var cid = document.getElementById('gc-constraint-sel').value;
  var cdata = GC_LHS_DATA.constraints[cid];
  if (!cdata) return;
  var xlabels = GC_LHS_DATA.xlabels;
  var xdates = GC_LHS_DATES;

  var traces = [
    {
      x: xdates.concat(xdates.slice().reverse()),
      y: cdata.lhs_p90.concat(cdata.lhs_p10.slice().reverse()),
      fill: 'toself',
      fillcolor: 'rgba(74,144,184,0.15)',
      line: {color: 'rgba(0,0,0,0)'},
      name: 'P10\u2013P90',
      showlegend: true,
      hoverinfo: 'skip'
    },
    {
      x: xdates,
      y: cdata.lhs_p50,
      name: 'LHS Median (P50)',
      line: {color: '#4A90B8', width: 2}
    },
    {
      x: xdates,
      y: cdata.bound,
      name: 'Bound (RHS)',
      line: {color: '#DC4C4C', width: 1.5, dash: 'dash'}
    }
  ];

  // `bound` already carries the ceiling wherever a stage has one, else the
  // floor (see `_build_constraint_lhs_data`) -- identical to a single-sided
  // constraint's only line. A genuine two-sided "range" stage additionally
  // populates `bound_lower` with a value distinct from `bound`; only then
  // is a second dashed floor line drawn, so single-sided constraints render
  // exactly one bound line, unchanged.
  var hasDistinctFloor = (cdata.bound_lower || []).some(function(lo, i) {
    return lo !== null && lo !== cdata.bound[i];
  });
  if (hasDistinctFloor) {
    traces.push({
      x: xdates,
      y: cdata.bound_lower,
      name: 'Bound Floor',
      line: {color: '#B8860B', width: 1.5, dash: 'dot'}
    });
  }

  var shapes = (cdata.violations || []).map(function(v) {
    return {
      type: 'rect',
      xref: 'x',
      yref: 'paper',
      x0: xdates[v.start],
      x1: xdates[v.end],
      y0: 0,
      y1: 1,
      fillcolor: 'rgba(220,76,76,0.12)',
      line: {width: 0}
    };
  });

  var layout = {
    title: cdata.name + ' \u2014 LHS vs Bound',
    hovermode: 'x unified',
    xaxis: {type: 'date', tickmode: 'array', tickvals: xdates, ticktext: xlabels},
    shapes: shapes,
    margin: {l: 60, r: 20, t: 60, b: 10},
    legend: {
      orientation: 'h', yanchor: 'top', y: -0.15,
      xanchor: 'center', x: 0.5, font: {size: 11}
    }
  };

  Plotly.react('gc-lhs-chart', traces, layout, {responsive: true});
}

document.addEventListener('DOMContentLoaded', function() {
  setTimeout(updateConstraintChart, 100);
});
"""
        + "</script>"
    )


def _add_type_filter_and_row_attrs(
    constraints: list[dict],
    summary_html: str,
) -> str:
    """Wrap summary table with a type-filter dropdown and add data-type/data-cid attrs.

    Post-processes the HTML string returned by ``build_constraints_summary_table()``
    to add ``data-type`` and ``data-cid`` attributes to each ``<tr>`` in the
    tbody, without modifying the legacy ``constraints.py`` module.

    Args:
        constraints: Constraint dicts in the same order as the table rows.
        summary_html: Raw HTML from ``build_constraints_summary_table()``.

    Returns:
        HTML string with the type-filter dropdown prepended and ``<tr>``
        tags in the tbody augmented with ``data-type`` and ``data-cid``
        attributes.
    """
    # Augment each <tr style="background:…;"> in the tbody with data-cid, data-type,
    # and onclick attributes.  We replace the first un-augmented occurrence per
    # constraint so that sequential rows are handled correctly even when multiple
    # constraints share the same type colour.
    modified = summary_html
    for c in constraints:
        cid = c["id"]
        ctype = c["name"].split("_")[0]
        modified = modified.replace(
            '<tr style="background:',
            f'<tr data-cid="{cid}" data-type="{escape_attr(ctype)}"'
            f' onclick="selectConstraint({cid})" style="cursor:pointer;background:',
            1,
        )

    type_filter_html = (
        '<div style="margin-bottom:12px;">'
        '<label for="gc-type-filter"'
        ' style="font-weight:600;margin-right:8px;">Filter by Type:</label>'
        '<select id="gc-type-filter" onchange="filterConstraintTable()"'
        ' style="padding:6px 10px;font-size:0.9rem;border-radius:4px;'
        'border:1px solid #ccc;">'
        '<option value="All">All</option>'
        '<option value="VminOP">VminOP</option>'
        '<option value="RE">RE</option>'
        '<option value="AGRINT">AGRINT</option>'
        "</select>"
        "</div>"
        "<script>\n"
        "function filterConstraintTable() {\n"
        "  var val = document.getElementById('gc-type-filter').value;\n"
        "  var rows = document.querySelectorAll('.data-table tbody tr[data-type]');\n"
        "  rows.forEach(function(row) {\n"
        "    var show = val === 'All' || row.dataset.type === val;\n"
        "    row.style.display = show ? '' : 'none';\n"
        "  });\n"
        "}\n"
        "function selectConstraint(cid) {\n"
        "  var sel = document.getElementById('gc-constraint-sel');\n"
        "  if (sel) { sel.value = String(cid); updateConstraintChart(); }\n"
        "  var chart = document.getElementById('gc-lhs-chart');\n"
        "  if (chart) { chart.scrollIntoView({behavior:'smooth', block:'start'}); }\n"
        "}\n"
        "</script>\n"
    )

    return type_filter_html + modified


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:
    """Render when generic constraints exist and simulation data is available."""
    return data.simulation_available and len(data.gc_constraints) > 0


def render(data: DashboardData) -> str:
    """Return the full HTML string for the v2 Constraints tab content area."""
    # Evaluate LHS expressions once; passed to LHS vs Bound chart. A
    # @rho_acum_h{id}-scaled term (VminOP, RHE) resolves against the LP's
    # per-stage override, not the simulation's default point-productivity
    # column -- otherwise the LHS renders on the wrong scale relative to its
    # own (unmodified) bound.
    rho_acum_overrides = load_rho_acum_overrides(data.case_dir)
    lhs_df = evaluate_constraint_expressions(
        data.gc_constraints, data.hydros_lf, data.exchanges_lf, rho_acum_overrides
    )

    # --- Section A: metrics row ---
    section_a = _build_metrics_row(data)

    # --- Section B: Constraint Summary Table with type filter ---
    summary_table = build_constraints_summary_table(
        data.gc_constraints, data.gc_bounds, data.gc_violations
    )
    summary_with_filter = _add_type_filter_and_row_attrs(
        data.gc_constraints, summary_table
    )
    section_b = section_title("Constraint Summary") + summary_with_filter

    # --- Section C: Interactive LHS vs Bound chart (single, JS-driven) ---
    section_c = section_title("LHS vs Bound") + _build_lhs_section(data, lhs_df)

    return section_a + section_b + section_c

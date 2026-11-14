"""v2 Constraints tab module for the Novomodelo dashboard.

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
import polars as pl

from novomodelo_bridge.dashboard.chart_helpers import (
    compute_cost_summary,
    cost_color,
)
from novomodelo_bridge.dashboard.tabs.constraints_utils import (
    build_constraints_summary_table,
    derive_constraint_shape,
)
from novomodelo_bridge.novomodelo.constraint_expr import (
    BLOCK_HOURS_SCHEMA,
    BOUNDS_SCHEMA,
    evaluate_lhs,
    stage_weighted_bounds,
    stage_weighted_lhs,
)
from novomodelo_bridge.ui.html.document import (
    escape_attr,
    escape_text,
    json_for_script,
    metric_card,
    metrics_grid,
    section_title,
)
from novomodelo_bridge.ui.html.plotly import (
    LEGEND_DEFAULTS,
    stage_date_xaxis,
    stage_x_dates,
    stage_x_labels,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from novomodelo_bridge.dashboard.data import DashboardData

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
_COLOR_TYPES = "#4A8B6F"

#: A slack above this magnitude is a violation.
_SLACK_TOL = 1e-6

_LHS_QUANTILES = {"lhs_p10": 0.1, "lhs_p50": 0.5, "lhs_p90": 0.9}
_LHS_SCHEMA = pl.Schema(
    {
        "constraint_id": pl.Int32,
        "scenario_id": pl.Int64,
        "stage_id": pl.Int32,
        "block_id": pl.Int32,
        "lhs_value": pl.Float64,
    }
)
_LHS_COLUMNS = list(_LHS_SCHEMA)
_VIOLATION_SCHEMA = pl.Schema(
    {
        "constraint_id": pl.Int32,
        "scenario_id": pl.Int64,
        "stage_id": pl.Int32,
        "block_id": pl.Int32,
        "slack_value": pl.Float64,
    }
)
_VIOLATION_COLUMNS = list(_VIOLATION_SCHEMA)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_metrics_row(data: DashboardData) -> str:
    """Build the 4-card metrics row for the Constraints tab.

    Cards:
    1. Total constraints count
    2. Constraints with violations count
    3. NPV of the Generic Constraints cost component, mean across scenarios
    4. Active constraint types (comma-joined unique type prefixes)
    """
    total_constraints = len(data.gc_constraints)

    # Count constraints that have at least one nonzero slack_value in gc_violations
    violated_count = 0
    violations_df: pd.DataFrame = data.gc_violations
    if not violations_df.empty and "constraint_id" in violations_df.columns:
        viol_ids = set(
            violations_df[violations_df["slack_value"].abs() > _SLACK_TOL][
                "constraint_id"
            ].unique()
        )
        violated_count = len(viol_ids)

    summary = compute_cost_summary(data.costs, "component")
    generic = summary.loc[summary["group"] == "Generic Constraints", "mean"]
    cost_str = f"R$ {float(generic.sum()):,.0f}"

    type_prefixes = _constraint_types(data.gc_constraints)
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
            label="Generic Constraints (NPV)",
            color=cost_color("component", "Generic Constraints"),
        ),
        metric_card(
            value=active_types_str,
            label="Active Types",
            color=_COLOR_TYPES,
        ),
    ]
    return metrics_grid(cards)


def _constraint_type(constraint: dict) -> str:
    """The family prefix of *constraint*'s name (``RE_12`` is ``RE``)."""
    return constraint["name"].split("_")[0]


def _constraint_types(constraints: list[dict]) -> list[str]:
    """Distinct family prefixes of *constraints*, in first-seen order."""
    return list(dict.fromkeys(_constraint_type(c) for c in constraints))


def _violation_zones(violated: Sequence[int]) -> list[dict[str, int]]:
    """Contiguous stage-index intervals where some scenario violates the row."""
    zones: list[dict[str, int]] = []
    start: int | None = None
    for i, count in enumerate(violated):
        if count and start is None:
            start = i
        elif not count and start is not None:
            zones.append({"start": start, "end": i - 1})
            start = None
    if start is not None:
        zones.append({"start": start, "end": len(violated) - 1})
    return zones


def _stage_series(
    frame: pl.DataFrame, column: str, stages: list[int]
) -> list[float | None]:
    """*frame*'s *column* per stage of *stages*, ``None`` where it has none."""
    by_stage = dict(zip(frame["stage_id"].to_list(), frame[column].to_list()))
    return [
        None if by_stage.get(s) is None else round(float(by_stage[s]), 4)
        for s in stages
    ]


def _chart_view(
    lhs: pl.DataFrame,
    bounds: pl.DataFrame,
    violated: pl.DataFrame,
    stages: list[int],
) -> dict:
    """One LHS-vs-bound view: across-scenario LHS percentiles, bounds and
    violation counts per stage.

    *lhs* holds ``scenario_id``/``stage_id``/``lhs_value``; *bounds*
    ``stage_id``/``bound_lower``/``bound_upper``; *violated* the
    ``scenario_id``/``stage_id`` of every nonzero slack the view covers. A
    stage with no evaluated LHS reads ``None``, never zero.
    """
    pcts = lhs.group_by("stage_id").agg(
        pl.col("lhs_value").quantile(q, interpolation="linear").alias(key)
        for key, q in _LHS_QUANTILES.items()
    )
    counts = dict(
        violated.group_by("stage_id")
        .agg(pl.col("scenario_id").n_unique().alias("n"))
        .iter_rows()
    )
    n_violated = [int(counts.get(s, 0)) for s in stages]
    view: dict = {key: _stage_series(pcts, key, stages) for key in _LHS_QUANTILES}
    view["bound_lower"] = _stage_series(bounds, "bound_lower", stages)
    view["bound_upper"] = _stage_series(bounds, "bound_upper", stages)
    view["violated"] = n_violated
    view["violations"] = _violation_zones(n_violated)
    return view


def _build_constraint_lhs_data(
    constraints: list[dict],
    lhs_df: pd.DataFrame,
    gc_bounds: pd.DataFrame,
    gc_violations: pd.DataFrame,
    block_hours: pl.DataFrame,
    block_names: dict[int, str],
    stage_labels: dict[int, str],
) -> dict:
    """Build the JSON-serialisable LHS-vs-bound views of every constraint.

    Each constraint carries a ``stage`` view (the hours-weighted stage LHS
    against the hours-weighted stage bound, see
    :func:`~novomodelo_bridge.novomodelo.constraint_expr.stage_weighted_bounds`)
    and one ``blocks`` view per block it is bounded at (each block's LHS
    against its own bound; empty for a stage-level row). Violation counts
    and zones come from the run's own nonzero slacks, not from the
    percentile band.

    Args:
        constraints: Constraint dicts with keys ``id``, ``name``.
        lhs_df: :func:`evaluate_lhs` output.
        gc_bounds: Sense-free ``bound_lower``/``bound_upper`` per
            ``constraint_id``/``stage_id``/``block_id`` (``block_id`` null
            for a stage-level row).
        gc_violations: The run's ``violations/generic`` slacks.
        block_hours: ``stage_id``/``block_id``/``hours``.
        block_names: Block id to block name.
        stage_labels: Stage id to human-readable label.

    Returns:
        Dict with ``stages``, ``xlabels``, ``n_scenarios`` and
        ``constraints`` (str id to an entry with ``name``, ``sense``,
        ``evaluated``, ``stage`` and ``blocks``).
    """
    lhs = (
        pl.from_pandas(lhs_df[_LHS_COLUMNS])
        if not lhs_df.empty
        else pl.DataFrame(schema=_LHS_SCHEMA)
    ).cast(_LHS_SCHEMA)
    bounds = (
        pl.from_pandas(gc_bounds[list(BOUNDS_SCHEMA)])
        if not gc_bounds.empty
        else pl.DataFrame(schema=BOUNDS_SCHEMA)
    ).cast(BOUNDS_SCHEMA)
    violated = (
        pl.from_pandas(gc_violations[_VIOLATION_COLUMNS])
        if not gc_violations.empty
        else pl.DataFrame(schema=_VIOLATION_SCHEMA)
    ).cast(_VIOLATION_SCHEMA)
    n_scenarios = (
        pl.concat([lhs["scenario_id"], violated["scenario_id"]]).n_unique()
        if not (lhs.is_empty() and violated.is_empty())
        else 0
    )
    violated = violated.filter(pl.col("slack_value").abs() > _SLACK_TOL)

    all_stages = sorted(
        set(lhs["stage_id"].to_list()) | set(bounds["stage_id"].to_list())
    )
    stage_lhs = stage_weighted_lhs(lhs_df, block_hours)
    stage_bounds = stage_weighted_bounds(bounds, block_hours)

    constraints_data: dict[str, dict] = {}
    for c in constraints:
        cid = int(c["id"])
        bounds_c = bounds.filter(pl.col("constraint_id") == cid)
        lhs_c = lhs.filter(pl.col("constraint_id") == cid)
        violated_c = violated.filter(pl.col("constraint_id") == cid)
        blocks = sorted(
            bounds_c.filter(pl.col("block_id").is_not_null())["block_id"]
            .unique()
            .to_list()
        )
        constraints_data[str(cid)] = {
            "name": c["name"],
            "sense": derive_constraint_shape(bounds_c.to_pandas()),
            "evaluated": not lhs_c.is_empty(),
            "stage": _chart_view(
                stage_lhs.filter(pl.col("constraint_id") == cid),
                stage_bounds.filter(pl.col("constraint_id") == cid),
                violated_c,
                all_stages,
            ),
            "blocks": [
                {
                    "label": _block_label(b, block_names),
                    **_chart_view(
                        lhs_c.filter(pl.col("block_id") == b),
                        bounds_c.filter(pl.col("block_id") == b),
                        violated_c.filter(pl.col("block_id") == b),
                        all_stages,
                    ),
                }
                for b in blocks
            ],
        }

    return {
        "stages": all_stages,
        "xlabels": stage_x_labels(all_stages, stage_labels),
        "n_scenarios": n_scenarios,
        "constraints": constraints_data,
    }


def _block_label(block_id: int, block_names: dict[int, str]) -> str:
    name = block_names.get(block_id)
    return f"Block {block_id} · {name}" if name else f"Block {block_id}"


def _block_names(stages_data: dict) -> dict[int, str]:
    """Block id to name, from the first stage that names it."""
    names: dict[int, str] = {}
    for stage in stages_data.get("stages", []):
        for block in stage.get("blocks", []):
            if block.get("name"):
                names.setdefault(int(block["id"]), str(block["name"]))
    return names


def _build_lhs_section(
    data: DashboardData, lhs_df: pd.DataFrame, block_hours: pl.DataFrame
) -> str:
    """Build the interactive LHS vs Bound section HTML.

    Embeds the per-constraint views as a JSON blob; ``gc-constraint-sel``
    picks the constraint and ``gc-view-sel`` the stage or per-block view
    drawn into ``gc-lhs-chart`` via ``Plotly.react()``.
    """
    lhs_data = _build_constraint_lhs_data(
        data.gc_constraints,
        lhs_df,
        data.gc_bounds,
        data.gc_violations,
        block_hours,
        _block_names(data.stages_data),
        data.stage_labels,
    )
    data_json = json_for_script(lhs_data)
    xdates = stage_x_dates(lhs_data["stages"], data.stage_dates)
    dates_json = json_for_script(xdates)
    xaxis_json = json_for_script(stage_date_xaxis(xdates, lhs_data["xlabels"]))

    options_html = "\n".join(
        f'<option value="{c["id"]}">{escape_text(c["name"])}</option>'
        for c in data.gc_constraints
    )
    select_style = (
        ' style="padding:8px 12px;font-size:0.9rem;border-radius:4px;'
        "border:1px solid #ccc;"
    )

    return (
        '<div style="margin-bottom:16px;display:flex;flex-wrap:wrap;gap:16px;'
        'align-items:center;">'
        "<div>"
        '<label for="gc-constraint-sel"'
        ' style="font-weight:600;margin-right:8px;">Select Constraint:</label>'
        '<select id="gc-constraint-sel" onchange="updateConstraintChart()"'
        + select_style
        + 'min-width:320px;">'
        + options_html
        + "</select>"
        + "</div>"
        + "<div>"
        + '<label for="gc-view-sel"'
        ' style="font-weight:600;margin-right:8px;">View:</label>'
        + '<select id="gc-view-sel" onchange="updateConstraintChart()"'
        + select_style
        + '">'
        + '<option value="stage">Stage (hours-weighted)</option>'
        + '<option value="block">By block</option>'
        + "</select>"
        + "</div>"
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
        + ";\nconst GC_LHS_XAXIS = "
        + xaxis_json
        + ";\nconst GC_LHS_LEGEND = "
        + json_for_script(LEGEND_DEFAULTS)
        + ";\n"
        + _LHS_CHART_JS
        + "</script>"
    )


#: Draws the selected constraint's view. Zones span half a stage either side
#: of their stages, so a one-stage zone is visible on the date axis.
_LHS_CHART_JS = r"""
function gcZoneEdge(xdates, i, side) {
  var t = Date.parse(xdates[i]);
  var j = i + side;
  if (j >= 0 && j < xdates.length) {
    return new Date((t + Date.parse(xdates[j])) / 2).toISOString();
  }
  var k = i - side;
  if (k < 0 || k >= xdates.length) return xdates[i];
  return new Date(t + (t - Date.parse(xdates[k])) / 2).toISOString();
}

function gcPanelTraces(view, xdates, yaxis, first) {
  var n = GC_LHS_DATA.n_scenarios;
  var traces = [
    {
      x: xdates.concat(xdates.slice().reverse()),
      y: view.lhs_p90.concat(view.lhs_p10.slice().reverse()),
      fill: 'toself', fillcolor: 'rgba(74,144,184,0.15)',
      line: {color: 'rgba(0,0,0,0)'}, name: 'LHS P10–P90',
      legendgroup: 'band', showlegend: first, hoverinfo: 'skip', yaxis: yaxis
    },
    {
      x: xdates, y: view.lhs_p50, name: 'LHS Median (P50)',
      line: {color: '#4A90B8', width: 2}, legendgroup: 'p50',
      showlegend: first, yaxis: yaxis, customdata: view.violated,
      hovertemplate: '%{y:,.1f}<extra>LHS P50 · violated in '
        + '%{customdata} of ' + n + ' scenarios</extra>'
    }
  ];
  var hasUpper = view.bound_upper.some(function(v) { return v !== null; });
  var distinctLower = view.bound_lower.some(function(lo, i) {
    return lo !== null && lo !== view.bound_upper[i];
  });
  if (hasUpper) {
    traces.push({
      x: xdates, y: view.bound_upper,
      name: distinctLower ? 'Upper Bound' : 'Bound',
      line: {color: '#DC4C4C', width: 1.5, dash: 'dash'},
      legendgroup: 'upper', showlegend: first, yaxis: yaxis
    });
  }
  if (distinctLower) {
    traces.push({
      x: xdates, y: view.bound_lower,
      name: hasUpper ? 'Lower Bound' : 'Bound',
      line: {color: hasUpper ? '#B8860B' : '#DC4C4C', width: 1.5, dash: 'dot'},
      legendgroup: 'lower', showlegend: first, yaxis: yaxis
    });
  }
  return traces;
}

function gcPanelShapes(view, xdates, yref) {
  return view.violations.map(function(v) {
    return {
      type: 'rect', xref: 'x', yref: yref,
      x0: gcZoneEdge(xdates, v.start, -1), x1: gcZoneEdge(xdates, v.end, 1),
      y0: 0, y1: 1, fillcolor: 'rgba(220,76,76,0.12)', line: {width: 0},
      layer: 'below'
    };
  });
}

function updateConstraintChart() {
  var cid = document.getElementById('gc-constraint-sel').value;
  var cdata = GC_LHS_DATA.constraints[cid];
  if (!cdata) return;
  var xdates = GC_LHS_DATES;
  var byBlock = document.getElementById('gc-view-sel').value === 'block'
    && cdata.blocks.length > 0;
  var panels = byBlock
    ? cdata.blocks
    : [Object.assign({label: ''}, cdata.stage)];

  var traces = [];
  var shapes = [];
  var annotations = [];
  var layout = {
    title: cdata.name + ' — LHS vs Bound'
      + (byBlock ? ' by block' : ' (hours-weighted stage mean)'),
    hovermode: 'x unified',
    xaxis: GC_LHS_XAXIS,
    margin: {l: 60, r: 20, t: 60, b: 10},
    legend: GC_LHS_LEGEND
  };
  if (byBlock) {
    layout.grid = {rows: panels.length, columns: 1, pattern: 'coupled',
                   roworder: 'top to bottom', ygap: 0.3};
  }
  panels.forEach(function(view, i) {
    var suffix = i === 0 ? '' : String(i + 1);
    var yaxis = 'y' + suffix;
    traces = traces.concat(gcPanelTraces(view, xdates, yaxis, i === 0));
    shapes = shapes.concat(gcPanelShapes(view, xdates, yaxis + ' domain'));
    if (i > 0) layout['yaxis' + suffix] = {matches: 'y'};
    if (view.label) {
      annotations.push({
        text: view.label, showarrow: false, xref: 'paper',
        yref: yaxis + ' domain', x: 0, y: 1, xanchor: 'left',
        yanchor: 'bottom', font: {size: 12}
      });
    }
  });
  if (!cdata.evaluated) {
    annotations.push({
      text: 'LHS not evaluated: a term has no simulation output to value it',
      showarrow: false, xref: 'paper', yref: 'paper', x: 0.5, y: 0.5,
      font: {size: 12, color: '#6B7280'}
    });
  }
  layout.shapes = shapes;
  layout.annotations = annotations;
  var height = byBlock ? 120 + 200 * panels.length : 380;
  var el = document.getElementById('gc-lhs-chart');
  el.style.height = height + 'px';
  layout.height = height;
  Plotly.react(el, traces, layout, {responsive: true});
}

document.addEventListener('DOMContentLoaded', function() {
  setTimeout(updateConstraintChart, 100);
});
"""


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
        ctype = _constraint_type(c)
        modified = modified.replace(
            '<tr style="background:',
            f'<tr data-cid="{cid}" data-type="{escape_attr(ctype)}"'
            f' onclick="selectConstraint({cid})" style="cursor:pointer;background:',
            1,
        )

    type_options = "".join(
        f'<option value="{escape_attr(t)}">{escape_text(t)}</option>'
        for t in _constraint_types(constraints)
    )
    type_filter_html = (
        '<div style="margin-bottom:12px;">'
        '<label for="gc-type-filter"'
        ' style="font-weight:600;margin-right:8px;">Filter by Type:</label>'
        '<select id="gc-type-filter" onchange="filterConstraintTable()"'
        ' style="padding:6px 10px;font-size:0.9rem;border-radius:4px;'
        'border:1px solid #ccc;">'
        '<option value="All">All</option>' + type_options + "</select>"
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
    block_hours = pl.DataFrame(
        {
            "stage_id": [s for s, _ in data.block_hours],
            "block_id": [b for _, b in data.block_hours],
            "hours": list(data.block_hours.values()),
        },
        schema=BLOCK_HOURS_SCHEMA,
    )
    lhs_df = evaluate_lhs(data.gc_resolved, data.lhs_frames, block_hours)

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
    section_c = section_title("LHS vs Bound") + _build_lhs_section(
        data, lhs_df, block_hours
    )

    return section_a + section_b + section_c

"""Cost-breakdown chart/table and per-stage cost subplots.

Self-contained: no dependency on ``charts._shared``. External deps only --
``analyze`` for the per-category Δ/Δ% arithmetic and ``cost_categories`` for
the aggregate-column exclusion set.
"""

from __future__ import annotations

import polars as pl

from cobre_bridge.cobre.cost_categories import AGGREGATE_COST_COLUMNS
from cobre_bridge.comparators import analyze
from cobre_bridge.comparators.html_report import (
    COLOR_COBRE,
    COLOR_NEWAVE,
)
from cobre_bridge.ui.html.plotly import plotly_div as _plotly_div

# Per-category mapping between the source model pmo.dat
# `custo_operacao_series_simuladas` `parcela` labels and cobre simulation cost-record
# columns.
#
# Tuple layout: (display_label, [newave_keys], [cobre_columns], hex_color).
#
# Ordering reflects logical grouping and drives the legend order in the
# stacked bar.
_COST_MAP: list[tuple[str, list[str], list[str], str]] = [
    # Operational / generation costs. Cobre's anticipated_thermal_cost (GNL
    # forward-committed fuel, booked on the decision column) is folded in here so the
    # thermal category matches the source model GERACAO TERMICA / CTERM, which books GNL
    # fuel at delivery. Absent in pre-anticipation Cobre runs (summed as 0).
    (
        "Thermal Generation",
        ["GERACAO TERMICA"],
        ["thermal_cost", "anticipated_thermal_cost"],
        "#D97706",
    ),
    ("Deficit", ["DEFICIT"], ["deficit_cost"], "#DC2626"),
    ("Energy Excess", ["EXCESSO ENERGIA"], ["excess_cost"], "#F59E0B"),
    ("Exchange", ["INTERCAMBIO"], ["exchange_cost"], "#7C3AED"),
    ("Pumping", [], ["pumping_cost"], "#0891B2"),
    # Energy-contract cost — cobre-only column (no source-model parcela analogue).
    ("Contract", [], ["contract_cost"], "#0D9488"),
    # Regularisation costs (per-unit-flow charges, not violations)
    (
        "Spillage",
        ["VERTIMENTO", "VERTIMENTO UHE", "VERT. FIO N. TURB."],
        ["spillage_cost"],
        "#2563EB",
    ),
    ("Turbined Reg.", ["TURBINAMENTO UHE"], ["turbined_cost"], "#0EA5E9"),
    ("NCS Curtailment", ["CORTE GER. EOLICA"], ["curtailment_cost"], "#059669"),
    # Hydro operational-bound violations (slacks priced above regularisation)
    (
        "Outflow Min Viol.",
        ["VIOLACAO VZMIN"],
        ["outflow_violation_below_cost"],
        "#E11D48",
    ),
    (
        "Outflow Max Viol.",
        ["VIOL. DEFL. MAXIMA"],
        ["outflow_violation_above_cost"],
        "#F43F5E",
    ),
    (
        "Turbining Bounds Viol.",
        ["VIOL. TURB. MINIMO", "VIOL. TURB. MAXIMO"],
        ["turbined_violation_cost"],
        "#FB923C",
    ),
    (
        "Generation Bounds Viol.",
        ["VIOLACAO GHMIN", "VIOLACAO GHMINU"],
        ["generation_violation_cost"],
        "#F97316",
    ),
    (
        "Storage Bounds Viol.",
        [],
        ["storage_violation_cost"],
        "#C2410C",
    ),
    ("Filling Target Viol.", ["VIOLACAO EVMIN"], ["filling_target_cost"], "#A16207"),
    (
        "Water Withdrawal Viol.",
        ["VIOLACAO RETIRADA"],
        ["withdrawal_violation_cost"],
        "#0284C7",
    ),
    (
        "Evaporation Viol.",
        ["VIOL. EVAP. UHE"],
        ["evaporation_violation_cost"],
        "#9333EA",
    ),
    # FPHA folga slack (the source model) — cobre has no direct analogue; left as The
    # source-model-only column so it shows up in the report rather than being hidden.
    ("FPHA Slack", ["VIOLACAO FPHA"], [], "#DB2777"),
    ("Inflow Non-Negativity", [], ["inflow_penalty_cost"], "#EA580C"),
    # Generic constraint violations: The source model reports the risk-aversion curve
    # and surface (CAR/SAR), electric (RESTELETRICA), interchange-group (INTERC. MIN.),
    # hydraulic (RHQ/RHV) and piecewise-linear (RLPP) restriction violations as separate
    # parcelas, but cobre-bridge converts them all into Cobre generic constraints, so
    # Cobre aggregates their slacks into a single `generic_violation_cost`. Sum the
    # source model parcelas to compare like-for-like.
    (
        "Generic Constr. Viol.",
        [
            "VIOLACAO CAR",
            "VIOLACAO SAR",
            "VIOL. RESTELETRICA",
            "VIOL. INTERC. MIN.",
            "VIOLACAO RHQ",
            "VIOLACAO RHV",
            "VIOL.RLPP DEFLMAX",
            "VIOL.RLPP DEFLMAXU",
            "VIOL.RLPP TURBMAX",
            "VIOL.RLPP TURBMAXU",
        ],
        ["generic_violation_cost"],
        "#6D28D9",
    ),
]

_COST_COLOR_DEFAULT = "#6B7280"  # for any unmapped residual category

# Cobre cost-record fields that are aggregates / metadata and must not appear as
# their own category in the breakdown (they would double-count or pollute the
# chart). Single-sourced with the dashboard via cost_categories.
_COBRE_NON_COST_KEYS: frozenset[str] = AGGREGATE_COST_COLUMNS


def _resolve_cost_categories(
    nw_costs: dict[str, float],
    cobre_costs: dict[str, float],
) -> list[tuple[str, float, float, str]]:
    """Resolve the source model/Cobre cost dicts into a sorted list of categories.

    Each entry is ``(display_label, newave_sum, cobre_sum, color)``. Categories with
    both sides ≤ 0.01 R$ are filtered out. Mapped entries from :data:`_COST_MAP` come
    first (preserving its logical ordering); unmapped the source model/Cobre keys are
    appended at the end.
    """
    categories: list[tuple[str, float, float, str]] = []
    for display_label, nw_keys, cb_keys, color in _COST_MAP:
        nw_sum = sum(nw_costs.get(k, 0.0) for k in nw_keys)
        cb_sum = sum(cobre_costs.get(k, 0.0) for k in cb_keys)
        if abs(nw_sum) < 0.01 and abs(cb_sum) < 0.01:
            continue
        categories.append((display_label, nw_sum, cb_sum, color))

    mapped_nw = {k for _, nw_keys, _, _ in _COST_MAP for k in nw_keys}
    for k, v in sorted(nw_costs.items()):
        if k not in mapped_nw and abs(v) > 0.01:
            categories.append((k.title(), v, 0.0, _COST_COLOR_DEFAULT))

    mapped_cb = {k for _, _, cb_keys, _ in _COST_MAP for k in cb_keys}
    for k, v in sorted(cobre_costs.items()):
        if k in _COBRE_NON_COST_KEYS:
            continue
        if k not in mapped_cb and abs(v) > 0.01:
            categories.append(
                (k.replace("_", " ").title(), 0.0, v, _COST_COLOR_DEFAULT)
            )

    return categories


def cost_breakdown_chart(
    nw_costs: dict[str, float],
    cobre_costs: dict[str, float],
    reference_label: str = "NEWAVE",
) -> str:
    """Stacked vertical bar comparing the source model vs Cobre per cost category in
    NPV.

    Bars are stacked with the largest-magnitude category at the bottom for
    readability; each category has its own distinct color drawn from
    :data:`_COST_MAP`. Y-axis is 10⁹ R$ for legibility on Brazilian-scale
    cases.
    """
    if not nw_costs and not cobre_costs:
        return "<p>No cost data available.</p>"

    categories = _resolve_cost_categories(nw_costs, cobre_costs)
    if not categories:
        return "<p>No cost data available.</p>"

    # Sort so largest total cost is at the bottom of the stack (drawn first).
    categories = sorted(categories, key=lambda t: -(t[1] + t[2]))

    x_labels = [reference_label, "Cobre"]
    traces: list[dict] = []

    for label, nw_v, cb_v, color in categories:
        traces.append(
            {
                "x": x_labels,
                "y": [round(nw_v / 1e9, 3), round(cb_v / 1e9, 3)],
                "name": label,
                "type": "bar",
                "marker": {"color": color},
                "hovertemplate": (
                    f"%{{x}}<br>{label}: %{{y:.3f}} 10⁹ R$<extra></extra>"
                ),
            }
        )

    layout = {
        "title": "Cost Breakdown (Present Value)",
        "yaxis": {"title": "Cost (10⁹ R$)"},
        "barmode": "stack",
        "bargap": 0.4,
        # Vertical legend on the right — the cost map has 15+ categories
        # which wrap into 3+ rows when laid out horizontally above the
        # plot and collide with the chart title.
        "legend": {
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "font": {"size": 11},
        },
        "margin": {"l": 60, "r": 220, "t": 60, "b": 50},
    }

    return _plotly_div(traces, layout, height=600)


def cost_breakdown_table(
    nw_costs: dict[str, float],
    cobre_costs: dict[str, float],
    reference_label: str = "NEWAVE",
) -> str:
    """Per-category NPV diff table — the source model, Cobre, Δ, Δ% — sorted by |Δ|.

    Returns an HTML ``<table>`` styled to fit alongside
    :func:`cost_breakdown_chart` inside a 2-column ``chart_grid``. Color swatches
    in the first column mirror the colors used in the bar chart so the reader
    can cross-reference at a glance.
    """
    if not nw_costs and not cobre_costs:
        return "<p>No cost data available.</p>"

    categories = _resolve_cost_categories(nw_costs, cobre_costs)
    if not categories:
        return "<p>No cost data available.</p>"

    rows, total_nw, total_cb, total_diff, total_pct = analyze.cost_percent_deltas(
        categories
    )

    def _fmt_money(v: float) -> str:
        return f"{v / 1e9:.3f}"

    def _fmt_pct(p: float | None) -> str:
        if p is None:
            return "—"
        return f"{p:+.1f}%"

    def _diff_class(diff: float) -> str:
        if diff > 0.01 * 1e9:
            return "cb-num cb-diff-pos"
        if diff < -0.01 * 1e9:
            return "cb-num cb-diff-neg"
        return "cb-num"

    head = (
        "<thead><tr>"
        '<th class="cb-cat">Category</th>'
        f'<th class="cb-num">{reference_label}</th>'
        '<th class="cb-num">Cobre</th>'
        '<th class="cb-num">Δ</th>'
        '<th class="cb-num">Δ%</th>'
        "</tr></thead>"
    )

    body_rows: list[str] = []
    for label, nw_v, cb_v, diff, pct, color in rows:
        swatch = f'<span class="cb-swatch" style="background:{color}"></span>'
        diff_cls = _diff_class(diff)
        body_rows.append(
            "<tr>"
            f'<td class="cb-cat">{swatch}{label}</td>'
            f'<td class="cb-num">{_fmt_money(nw_v)}</td>'
            f'<td class="cb-num">{_fmt_money(cb_v)}</td>'
            f'<td class="{diff_cls}">{_fmt_money(diff)}</td>'
            f'<td class="{diff_cls}">{_fmt_pct(pct)}</td>'
            "</tr>"
        )
    body_rows.append(
        "<tr>"
        '<td class="cb-cat">Total</td>'
        f'<td class="cb-num">{_fmt_money(total_nw)}</td>'
        f'<td class="cb-num">{_fmt_money(total_cb)}</td>'
        f'<td class="cb-num">{_fmt_money(total_diff)}</td>'
        f'<td class="cb-num">{_fmt_pct(total_pct)}</td>'
        "</tr>"
    )
    body = "<tbody>" + "".join(body_rows) + "</tbody>"
    caption = "<caption>NPV by Category (10⁹ R$)</caption>"
    return '<table class="cost-breakdown-table">' + caption + head + body + "</table>"


def _extract_stage_cost_series(
    nw_sin: pl.DataFrame,
    cobre_stage_costs: pl.DataFrame,
    nw_offset: int,
    nw_variable: str,
    cb_column: str,
) -> tuple[list[int], dict[int, float], dict[int, float]]:
    """Pull aligned the source model/Cobre per-stage cost series.

    Returns the sorted list of 0-based stages present on either side, plus ``{stage:
    R$}`` dicts for the source model (from MEDIAS-SIN, converted from 10⁶ R$) and Cobre
    (from the simulation costs parquet).
    """
    nw_by_stage: dict[int, float] = {}
    if nw_sin is not None and not nw_sin.is_empty():
        sub = nw_sin.filter(pl.col("variable") == nw_variable)
        for row in sub.iter_rows(named=True):
            nw_by_stage[int(row["stage"]) - nw_offset] = float(row["value"]) * 1e6

    cb_by_stage: dict[int, float] = {}
    if cobre_stage_costs is not None and not cobre_stage_costs.is_empty():
        for row in cobre_stage_costs.iter_rows(named=True):
            v = row.get(cb_column)
            if v is None:
                continue
            cb_by_stage[int(row["stage_id"])] = float(v)

    stages = sorted(set(nw_by_stage) | set(cb_by_stage))
    return stages, nw_by_stage, cb_by_stage


def _stage_cost_subplot(
    nw_sin: pl.DataFrame,
    cobre_stage_costs: pl.DataFrame,
    nw_offset: int,
    *,
    nw_variable: str,
    cb_column: str,
    title: str,
    nw_label: str,
    cb_label: str,
) -> str:
    """Render a single stage-cost line chart (one variable, both sides)."""
    stages, nw_by_stage, cb_by_stage = _extract_stage_cost_series(
        nw_sin, cobre_stage_costs, nw_offset, nw_variable, cb_column
    )
    if not stages:
        return f"<p>No {nw_variable} data available.</p>"

    def _series(by_stage: dict[int, float]) -> list[float | None]:
        return [round(by_stage[s] / 1e6, 4) if s in by_stage else None for s in stages]

    traces: list[dict] = []
    if nw_by_stage:
        traces.append(
            {
                "x": stages,
                "y": _series(nw_by_stage),
                "name": nw_label,
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"color": COLOR_NEWAVE},
                "hovertemplate": (
                    f"stage %{{x}}<br>{nw_label}: %{{y:.2f}} 10⁶ R$<extra></extra>"
                ),
            }
        )
    if cb_by_stage:
        traces.append(
            {
                "x": stages,
                "y": _series(cb_by_stage),
                "name": cb_label,
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"color": COLOR_COBRE},
                "hovertemplate": (
                    f"stage %{{x}}<br>{cb_label}: %{{y:.2f}} 10⁶ R$<extra></extra>"
                ),
            }
        )

    layout = {
        "title": title,
        "xaxis": {"title": "Stage (0-based)"},
        "yaxis": {"title": "Cost (10⁶ R$)"},
        "legend": {"orientation": "h", "y": -0.2},
    }
    return _plotly_div(traces, layout)


def immediate_cost_chart(
    nw_sin: pl.DataFrame,
    cobre_stage_costs: pl.DataFrame,
    nw_offset: int = 0,
    reference_label: str = "NEWAVE",
) -> str:
    """Per-stage *immediate* cost: The source model ``COPER`` vs Cobre
    ``immediate_cost``.

    The source model values come from MEDIAS-SIN in 10⁶ R$ (converted to R$ here by
    multiplying by 1e6).  Stage numbering on the source model side starts at the study's
    first calendar month — *nw_offset* (the minimum stage in MEDIAS-SIN) is subtracted
    to align with Cobre's 0-based ``stage_id``.
    """
    return _stage_cost_subplot(
        nw_sin,
        cobre_stage_costs,
        nw_offset,
        nw_variable="COPER",
        cb_column="immediate_cost",
        title=f"Immediate Cost — {reference_label} COPER vs Cobre",
        nw_label=f"{reference_label} COPER",
        cb_label="Cobre immediate_cost",
    )


def future_cost_chart(
    nw_sin: pl.DataFrame,
    cobre_stage_costs: pl.DataFrame,
    nw_offset: int = 0,
    reference_label: str = "NEWAVE",
) -> str:
    """Per-stage *future* cost: The source model ``CUSTO_FUTURO`` vs Cobre
    ``future_cost``."""
    return _stage_cost_subplot(
        nw_sin,
        cobre_stage_costs,
        nw_offset,
        nw_variable="CUSTO_FUTURO",
        cb_column="future_cost",
        title=f"Future Cost — {reference_label} CUSTO_FUTURO vs Cobre",
        nw_label=f"{reference_label} CUSTO_FUTURO",
        cb_label="Cobre future_cost",
    )


def thermal_cost_chart(
    nw_sin: pl.DataFrame,
    cobre_stage_costs: pl.DataFrame,
    nw_offset: int = 0,
    reference_label: str = "NEWAVE",
) -> str:
    """Per-stage thermal cost: The source model ``CTERM`` vs Cobre thermal (incl.
    anticip.).

    The Cobre side is ``thermal_cost_total`` = ``thermal_cost`` +
    ``anticipated_thermal_cost`` (the GNL forward-committed fuel Cobre books on the
    decision column), so it lines up with the source model ``CTERM``, which carries GNL
    fuel at delivery. Both are the live thermal generation cost, so this is an
    apples-to-apples comparison even in the post-study (where COPER is frozen — see
    :func:`other_costs_chart`).
    """
    return _stage_cost_subplot(
        nw_sin,
        cobre_stage_costs,
        nw_offset,
        nw_variable="CTERM",
        cb_column="thermal_cost_total",
        title=f"Thermal Cost — {reference_label} CTERM vs Cobre",
        nw_label=f"{reference_label} CTERM",
        cb_label="Cobre thermal (incl. anticipated)",
    )


def other_costs_chart(
    nw_sin: pl.DataFrame,
    cobre_stage_costs: pl.DataFrame,
    nw_offset: int = 0,
    reference_label: str = "NEWAVE",
) -> str:
    """Per-stage non-thermal operation cost: ``COPER − CTERM`` per stage.

    The source model: ``COPER − CTERM``. Cobre: ``immediate_cost − thermal_cost_total``
    (``thermal_cost_total`` = live + anticipated GNL fuel, matching the thermal
    category). This isolates everything in the immediate cost that is *not* thermal
    generation (deficit, penalties, slacks). On the source model side it goes
    **negative** in the post-study because COPER is frozen at the last study value while
    CTERM tracks the live post-study thermal cost — so this chart surfaces that
    frozen-COPER gap explicitly.
    """
    _, nw_coper, cb_imm = _extract_stage_cost_series(
        nw_sin, cobre_stage_costs, nw_offset, "COPER", "immediate_cost"
    )
    _, nw_cterm, cb_therm = _extract_stage_cost_series(
        nw_sin, cobre_stage_costs, nw_offset, "CTERM", "thermal_cost_total"
    )

    nw_other = {s: nw_coper[s] - nw_cterm[s] for s in nw_coper if s in nw_cterm}
    cb_other = {s: cb_imm[s] - cb_therm[s] for s in cb_imm if s in cb_therm}

    stages = sorted(set(nw_other) | set(cb_other))
    if not stages:
        return "<p>No COPER/CTERM data available.</p>"

    def _series(by_stage: dict[int, float]) -> list[float | None]:
        return [round(by_stage[s] / 1e6, 4) if s in by_stage else None for s in stages]

    traces: list[dict] = []
    if nw_other:
        traces.append(
            {
                "x": stages,
                "y": _series(nw_other),
                "name": f"{reference_label} COPER − CTERM",
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"color": COLOR_NEWAVE},
                "hovertemplate": (
                    f"stage %{{x}}<br>{reference_label} COPER − CTERM: "
                    "%{y:.2f} 10⁶ R$<extra></extra>"
                ),
            }
        )
    if cb_other:
        traces.append(
            {
                "x": stages,
                "y": _series(cb_other),
                "name": "Cobre immediate − thermal",
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"color": COLOR_COBRE},
                "hovertemplate": (
                    "stage %{x}<br>Cobre immediate − thermal: "
                    "%{y:.2f} 10⁶ R$<extra></extra>"
                ),
            }
        )

    layout = {
        "title": "Other Costs — COPER − CTERM (non-thermal operation)",
        "xaxis": {"title": "Stage (0-based)"},
        "yaxis": {"title": "Cost (10⁶ R$)"},
        "legend": {"orientation": "h", "y": -0.2},
    }
    return _plotly_div(traces, layout)

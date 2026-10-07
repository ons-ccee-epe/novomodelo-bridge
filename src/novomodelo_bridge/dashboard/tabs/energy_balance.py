"""v2 Energy Balance tab — system-wide generation mix and per-bus breakdown.

Implements six sections of the Energy Balance tab (Tab 4):
  A. Metrics row — 6 cards (hydro, thermal, NCS GWh; deficit GWh;
     spillage m3/s; curtailment GWh).
  B. System generation mix hero chart — stacked area of hydro/thermal/NCS
     vs LP load (dashed overlay), stage-average MW.
  C. Generation by bus — collapsible facet with one subplot per
     non-fictitious bus.
  D. Deficit & Excess — two side-by-side charts, one line per bus,
     with mean+p50+p10-p90 band.
  E. Reservoir Storage — system aggregate trajectory with vol_min/vol_max
     reference lines, plus per-bus storage as % useful volume.
  F. NCS & Curtailment — NCS generation vs available capacity (left),
     curtailment by source horizontal bar chart (right).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from cobre_bridge.dashboard.chart_helpers import (
    add_mean_p50_band,
    compute_percentiles,
    make_chart_card,
)
from cobre_bridge.dashboard.data import _compute_lp_load, _stage_avg_mw, entity_name
from cobre_bridge.ui.html.document import (
    chart_grid,
    collapsible_section,
    json_for_script,
    metric_card,
    metrics_grid,
    section_title,
)
from cobre_bridge.ui.html.plotly import (
    LEGEND_DEFAULTS as _LEGEND,
)
from cobre_bridge.ui.html.plotly import (
    MARGIN_DEFAULTS as _MARGIN,
)
from cobre_bridge.ui.html.plotly import (
    apply_stage_date_axis,
    apply_standard_layout,
    stage_x_dates,
    stage_x_labels,
)
from cobre_bridge.ui.theme import BUS_COLORS, COLORS, GENERATION_COLORS

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-energy-balance"
TAB_LABEL = "Energy Balance"
TAB_ORDER = 30

# ---------------------------------------------------------------------------
# Private metric helpers
# ---------------------------------------------------------------------------


def _compute_total_gwh(lf: pl.LazyFrame, mwh_col: str) -> float:
    """Return mean total GWh across scenarios for *mwh_col*.

    Sums *mwh_col* per scenario, averages across scenarios, then converts
    MWh to GWh (divide by 1000).  Returns ``0.0`` when the LazyFrame is
    empty, *mwh_col* is absent, or any computation error occurs.

    Args:
        lf: Simulation LazyFrame containing ``scenario_id`` and *mwh_col*.
        mwh_col: Column name holding energy values in MWh.

    Returns:
        Mean total GWh as a float, or ``0.0`` on any failure.
    """
    try:
        result = (
            lf.group_by("scenario_id")
            .agg(pl.col(mwh_col).sum())
            .select(pl.col(mwh_col).mean())
            .collect(engine="streaming")
        )
        if result.height == 0:
            return 0.0
        value = result[mwh_col][0]
        return float(value) / 1_000.0 if value is not None else 0.0
    except (ValueError, TypeError, KeyError, pl.exceptions.ColumnNotFoundError):
        return 0.0


def _block_weighted_avg_rate(
    lf: pl.LazyFrame,
    col: str,
    bh_df: pl.DataFrame,
) -> float:
    """Return the block-hours-weighted horizon mean of a system-total rate.

    Sums *col* across entities per (scenario, stage, block), joins block
    hours, reduces to one Σ(col·hours)/Σ(hours) scalar per scenario over the
    whole horizon, then averages across scenarios — mirrors the
    ``available_mw`` weighting in ``_render_ncs_curtailment``.  Returns
    ``0.0`` on any failure.

    Args:
        lf: Simulation LazyFrame containing ``scenario_id``, ``stage_id``,
            ``block_id``, and *col*.
        col: Rate column name to aggregate (e.g. m3/s), never an MWh
            accumulation.
        bh_df: Block-hours frame with ``stage_id``, ``block_id``, ``_bh``
            (``data.bh_df`` — the single shared block-hours source).

    Returns:
        Mean block-hours-weighted rate as a float, or ``0.0`` on any failure.
    """
    try:
        # col is a rate (m3/s) — weight by block hours (Σ(rate·hours)/Σ(hours)),
        # never a bare row sum (inflates the value by the row count).
        result = (
            lf.group_by(["scenario_id", "stage_id", "block_id"])
            .agg(pl.col(col).sum())
            .join(bh_df.lazy(), on=["stage_id", "block_id"])
            .group_by("scenario_id")
            .agg(
                (pl.col(col) * pl.col("_bh")).sum().alias("_w"),
                pl.col("_bh").sum().alias("_h"),
            )
            .with_columns((pl.col("_w") / pl.col("_h")).alias("_avg"))
            .select(pl.col("_avg").mean())
            .collect(engine="streaming")
        )
        if result.height == 0:
            return 0.0
        value = result["_avg"][0]
        return float(value) if value is not None else 0.0
    # degrade to 0.0 rather than raise — an absent/malformed rate column must
    # not crash the metrics row.
    except (ValueError, TypeError, KeyError, pl.exceptions.ColumnNotFoundError):
        return 0.0


# ---------------------------------------------------------------------------
# Section A — Metrics row
# ---------------------------------------------------------------------------


def _build_metrics_row(data: DashboardData) -> str:
    """Build a 6-card metrics grid for the Energy Balance tab.

    Cards: Hydro GWh, Thermal GWh, NCS GWh, Deficit GWh, Avg Spillage
    (m3/s), Curtailment GWh.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        An HTML string containing the section title and metrics grid.
    """
    hydro_gwh = _compute_total_gwh(data.hydros_lf, "generation_mwh")
    thermal_gwh = _compute_total_gwh(data.thermals_lf, "generation_mwh")
    ncs_gwh = _compute_total_gwh(data.ncs_lf, "generation_mwh")

    # Deficit: filter to non-fictitious buses only
    try:
        deficit_gwh = _compute_total_gwh(
            data.buses_lf.filter(pl.col("bus_id").is_in(data.non_fictitious_bus_ids)),
            "deficit_mwh",
        )
    except (ValueError, TypeError, KeyError, pl.exceptions.ColumnNotFoundError):
        deficit_gwh = 0.0

    # Spillage: expressed as average m3/s (not GWh — no productivity data here)
    spillage_avg = _block_weighted_avg_rate(data.hydros_lf, "spillage_m3s", data.bh_df)

    # Curtailment: expressed in GWh
    curtailment_gwh = _compute_total_gwh(data.ncs_lf, "curtailment_mwh")

    cards = [
        metric_card(
            f"{hydro_gwh:,.0f} GWh",
            "Total Hydro GWh",
            color=GENERATION_COLORS["hydro"],
        ),
        metric_card(
            f"{thermal_gwh:,.0f} GWh",
            "Total Thermal GWh",
            color=GENERATION_COLORS["thermal"],
        ),
        metric_card(
            f"{ncs_gwh:,.0f} GWh",
            "Total NCS GWh",
            color=GENERATION_COLORS["ncs"],
        ),
        metric_card(
            f"{deficit_gwh:,.0f} GWh",
            "Total Deficit GWh",
            color=COLORS["deficit"],
        ),
        metric_card(
            f"{spillage_avg:,.1f} m\u00b3/s",
            "Avg Spillage (m\u00b3/s)",
            color=COLORS["spillage"],
        ),
        metric_card(
            f"{curtailment_gwh:,.0f} GWh",
            "Total Curtailment GWh",
            color=COLORS["curtailment"],
        ),
    ]
    return section_title("Energy Metrics") + metrics_grid(cards)


# ---------------------------------------------------------------------------
# Section B — System generation mix hero chart
# ---------------------------------------------------------------------------


def _chart_gen_mix_hero(data: DashboardData) -> go.Figure:
    """Build the system-wide generation mix stacked area hero chart.

    Shows stage-average MW for hydro, thermal, and NCS (stacked areas) plus
    LP load as a dashed overlay.  Uses ``data.non_fictitious_bus_ids`` for
    the LP load filter instead of a hardcoded bus list.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    h_gen = _stage_avg_mw(data.hydros_lf, "generation_mwh", data.stage_hours)
    t_gen = _stage_avg_mw(data.thermals_lf, "generation_mwh", data.stage_hours)
    n_gen = _stage_avg_mw(data.ncs_lf, "generation_mwh", data.stage_hours)

    load_ser = _compute_lp_load(
        data.load_stats,
        data.load_factors_list,
        data.stage_hours,
        data.block_hours,
        bus_filter=data.non_fictitious_bus_ids,
    )

    stages = sorted(h_gen.keys())
    xlabels = stage_x_labels(stages, data.stage_labels)
    xdates = stage_x_dates(stages, data.stage_dates)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xdates,
            y=[h_gen.get(s, 0) for s in stages],
            name="Hydro",
            stackgroup="gen",
            mode="lines",
            line={"color": GENERATION_COLORS["hydro"]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xdates,
            y=[t_gen.get(s, 0) for s in stages],
            name="Thermal",
            stackgroup="gen",
            mode="lines",
            line={"color": GENERATION_COLORS["thermal"]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xdates,
            y=[n_gen.get(s, 0) for s in stages],
            name="NCS",
            stackgroup="gen",
            mode="lines",
            line={"color": GENERATION_COLORS["ncs"]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xdates,
            y=[load_ser.get(s, 0) for s in stages],
            name="LP Load",
            line={"color": COLORS["load"], "width": 2.5, "dash": "dash"},
            mode="lines",
        )
    )
    fig.update_layout(
        xaxis_title="Stage",
        yaxis_title="MW",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=_MARGIN,
    )
    apply_stage_date_axis(fig, xdates, xlabels)
    return fig


# ---------------------------------------------------------------------------
# Section C — Generation by bus (collapsible facet)
# ---------------------------------------------------------------------------


def _chart_gen_by_bus(data: DashboardData) -> go.Figure:
    """Build subplots per non-fictitious bus: hydro+thermal+NCS+net import vs LP load.

    One row per bus in ``data.non_fictitious_bus_ids``.  Each subplot shows
    stacked area traces for hydro, thermal, and NCS generation plus a net
    import area trace and a dashed LP load overlay.  Y-axis scales are
    independent across buses.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        A :class:`plotly.graph_objects.Figure` with one subplot row per bus.
    """
    bus_ids = sorted(data.non_fictitious_bus_ids)
    n_buses = len(bus_ids)

    if n_buses == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No non-fictitious buses found",
            legend=_LEGEND,
            margin=_MARGIN,
        )
        return fig

    # Build mapping DataFrames for LazyFrame joins
    hbus_df = pl.DataFrame(
        {
            "hydro_id": list(data.hydro_bus_map.keys()),
            "bus_id": list(data.hydro_bus_map.values()),
        }
    )
    tbus_map = {k: v["bus_id"] for k, v in data.thermal_meta.items()}
    tbus_df = pl.DataFrame(
        {
            "thermal_id": list(tbus_map.keys()),
            "bus_id": list(tbus_map.values()),
        }
    )
    nbus_df = pl.DataFrame(
        {
            "non_controllable_id": list(data.ncs_bus_map.keys()),
            "bus_id": list(data.ncs_bus_map.values()),
        }
    )
    hours_df = pl.DataFrame(
        {
            "stage_id": list(data.stage_hours.keys()),
            "_hours": list(data.stage_hours.values()),
        }
    )

    def _bus_stage_avg(
        lf: pl.LazyFrame,
        mwh_col: str,
        id_col: str,
        mapping_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Join lf to mapping_df, compute per-bus stage-average MW."""
        return (
            lf.join(mapping_df.lazy(), on=id_col)
            .group_by(["scenario_id", "stage_id", "bus_id"])
            .agg(pl.col(mwh_col).sum())
            .join(hours_df.lazy(), on="stage_id")
            .with_columns((pl.col(mwh_col) / pl.col("_hours")).alias("_avg_mw"))
            .group_by(["stage_id", "bus_id"])
            .agg(pl.col("_avg_mw").mean())
            .sort("stage_id")
            .collect(engine="streaming")
        )

    h_by_bus = _bus_stage_avg(data.hydros_lf, "generation_mwh", "hydro_id", hbus_df)
    t_by_bus = _bus_stage_avg(data.thermals_lf, "generation_mwh", "thermal_id", tbus_df)
    n_by_bus = _bus_stage_avg(
        data.ncs_lf, "generation_mwh", "non_controllable_id", nbus_df
    )

    # Collect the union of stage IDs across all entity types
    stages_all = sorted(
        set(h_by_bus["stage_id"].to_list())
        | set(t_by_bus["stage_id"].to_list())
        | set(n_by_bus["stage_id"].to_list())
    )
    if not stages_all:
        stages_all = sorted(data.stage_hours.keys())
    xlabels = stage_x_labels(stages_all, data.stage_labels)
    xdates = stage_x_dates(stages_all, data.stage_dates)

    def _to_map(df: pl.DataFrame, bus_id: int) -> dict[int, float]:
        sub = df.filter(pl.col("bus_id") == bus_id)
        return dict(zip(sub["stage_id"].to_list(), sub["_avg_mw"].to_list()))

    # Compute net exchange import per bus
    ex_import: dict[int, dict[int, float]] = {}
    for bus_id in bus_ids:
        parts: list[pl.DataFrame] = []
        for ln in data.line_meta:
            lid, src, tgt = ln["id"], ln["source_bus_id"], ln["target_bus_id"]
            if src != bus_id and tgt != bus_id:
                continue
            sign = 1.0 if tgt == bus_id else -1.0
            part = (
                data.exchanges_lf.filter(pl.col("line_id") == lid)
                .group_by(["scenario_id", "stage_id"])
                .agg((pl.col("net_flow_mwh") * sign).sum().alias("_imp_mwh"))
                .join(hours_df.lazy(), on="stage_id")
                .with_columns((pl.col("_imp_mwh") / pl.col("_hours")).alias("_mw"))
                .group_by("stage_id")
                .agg(pl.col("_mw").mean())
                .collect(engine="streaming")
            )
            parts.append(part)
        if parts:
            combined = pl.concat(parts)
            agg = (
                combined.group_by("stage_id").agg(pl.col("_mw").sum()).sort("stage_id")
            )
            ex_import[bus_id] = dict(
                zip(agg["stage_id"].to_list(), agg["_mw"].to_list())
            )
        else:
            ex_import[bus_id] = {}

    import math

    n_cols = 2
    n_rows = math.ceil(n_buses / n_cols)
    subplot_titles = [data.bus_names.get(b, str(b)) for b in bus_ids]
    # Pad to fill the grid
    while len(subplot_titles) < n_rows * n_cols:
        subplot_titles.append("")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=False,
        subplot_titles=subplot_titles,
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )

    for idx, bus_id in enumerate(bus_ids):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        show_legend = idx == 0
        h_gen = _to_map(h_by_bus, bus_id)
        t_gen = _to_map(t_by_bus, bus_id)
        n_gen = _to_map(n_by_bus, bus_id)
        load_s = _compute_lp_load(
            data.load_stats,
            data.load_factors_list,
            data.stage_hours,
            data.block_hours,
            bus_filter=[bus_id],
        )
        net_imp = ex_import.get(bus_id, {})

        for name, vals, color, lgroup in [
            ("Hydro", h_gen, GENERATION_COLORS["hydro"], "hydro"),
            ("Thermal", t_gen, GENERATION_COLORS["thermal"], "thermal"),
            ("NCS", n_gen, GENERATION_COLORS["ncs"], "ncs"),
            ("Net Import", net_imp, COLORS["exchange"], "import"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=xdates,
                    y=[vals.get(s, 0) for s in stages_all],
                    name=name,
                    stackgroup=f"g{bus_id}",
                    mode="lines",
                    line={"color": color},
                    legendgroup=lgroup,
                    showlegend=show_legend,
                ),
                row=row,
                col=col,
            )
        fig.add_trace(
            go.Scatter(
                x=xdates,
                y=[load_s.get(s, 0) for s in stages_all],
                name="LP Load",
                mode="lines",
                line={"color": COLORS["load"], "width": 2, "dash": "dash"},
                legendgroup="load",
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=350 * n_rows,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=80, b=50),
    )
    apply_stage_date_axis(fig, xdates, xlabels)
    return fig


# ---------------------------------------------------------------------------
# Section D — Deficit & Excess by bus (collapsible)
# ---------------------------------------------------------------------------


def _render_deficit_excess(data: DashboardData) -> str:
    """Build two side-by-side charts: deficit by bus (left) and excess by bus (right).

    Each chart shows one mean+p50+band line per non-fictitious bus.  Data
    source is ``buses_lf`` with columns ``deficit_mwh`` and ``excess_mwh``,
    converted to stage-average MW by dividing by stage hours.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        An HTML string: a ``collapsible_section`` wrapping a ``chart_grid``
        with two chart cards.
    """
    bus_ids = sorted(data.non_fictitious_bus_ids)
    hours_df = pl.DataFrame(
        {
            "stage_id": list(data.stage_hours.keys()),
            "_hours": list(data.stage_hours.values()),
        }
    )

    def _bus_mw_collected(col: str) -> pl.DataFrame:
        """Return per-(scenario, stage, bus) MW for *col* from buses_lf."""
        try:
            return (
                data.buses_lf.filter(pl.col("bus_id").is_in(bus_ids))
                .group_by(["scenario_id", "stage_id", "bus_id"])
                .agg(pl.col(col).sum())
                .join(hours_df.lazy(), on="stage_id")
                .with_columns((pl.col(col) / pl.col("_hours")).alias("_mw"))
                .select(["scenario_id", "stage_id", "bus_id", "_mw"])
                .sort("stage_id")
                .collect(engine="streaming")
            )
        except (pl.exceptions.ColumnNotFoundError, KeyError):
            return pl.DataFrame()

    def _bus_chart(col: str, title: str, y_label: str) -> go.Figure:
        """Build a go.Figure with one mean+p50+band line per bus."""
        raw = _bus_mw_collected(col)
        fig = go.Figure()
        if raw.is_empty():
            fig.add_annotation(
                text="No data.",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )
            fig.update_layout(
                xaxis_title="Stage", yaxis_title=y_label, legend=_LEGEND, margin=_MARGIN
            )
            return fig

        stages = sorted(raw["stage_id"].unique().to_list())
        xlabels = stage_x_labels(stages, data.stage_labels)
        xdates = stage_x_dates(stages, data.stage_dates)

        for i, bus_id in enumerate(bus_ids):
            sub = (
                raw.filter(pl.col("bus_id") == bus_id)
                .select(["scenario_id", "stage_id", "_mw"])
                .to_pandas()
            )
            if sub.empty:
                continue
            pct = compute_percentiles(sub, ["stage_id"], "_mw")
            # Map stage_id -> ISO date x-position (proportional date axis)
            pct["_x"] = pct["stage_id"].map(dict(zip(stages, xdates)))
            color = BUS_COLORS[i % len(BUS_COLORS)]
            bus_name = data.bus_names.get(bus_id, str(bus_id))
            add_mean_p50_band(fig, pct, "_x", bus_name, color)

        fig.update_layout(
            xaxis_title="Stage", yaxis_title=y_label, legend=_LEGEND, margin=_MARGIN
        )
        apply_stage_date_axis(fig, xdates, xlabels)
        return fig

    deficit_fig = _bus_chart("deficit_mwh", "Deficit by Bus", "Deficit (MW)")
    excess_fig = _bus_chart("excess_mwh", "Excess by Bus", "Excess (MW)")

    deficit_card = make_chart_card(
        deficit_fig,
        "Deficit by Bus (stage-avg MW)",
        "v2-energy-deficit-by-bus",
        height=420,
    )
    excess_card = make_chart_card(
        excess_fig,
        "Excess by Bus (stage-avg MW)",
        "v2-energy-excess-by-bus",
        height=420,
    )

    return collapsible_section(
        "Deficit & Excess",
        chart_grid([deficit_card, excess_card]),
        section_id="v2-energy-deficit-excess-section",
        default_collapsed=False,
    )


# ---------------------------------------------------------------------------
# Section E — Reservoir Storage (collapsible, collapsed by default)
# ---------------------------------------------------------------------------


def _render_reservoir_storage(data: DashboardData) -> str:
    """Build system aggregate storage trajectory and per-bus storage % charts.

    Left chart: sum of ``storage_final_hm3`` across all hydros per stage,
    with mean+p50+p10-p90 band and horizontal reference lines for total
    system ``vol_min`` / ``vol_max`` from ``data.hydro_meta``.

    Right chart: per-bus storage as percentage of useful volume
    (``vol_max - vol_min`` for hydros in that bus), one subplot per
    non-fictitious bus that has hydros.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        An HTML string: a ``collapsible_section`` wrapping a ``chart_grid``
        with two chart cards.
    """
    bus_ids = sorted(data.non_fictitious_bus_ids)

    # ------------------------------------------------------------------
    # System aggregate storage as % of total useful volume
    # ------------------------------------------------------------------
    total_vol_max = sum(float(m.get("vol_max", 0)) for m in data.hydro_meta.values())
    total_vol_min = sum(float(m.get("vol_min", 0)) for m in data.hydro_meta.values())
    total_useful = total_vol_max - total_vol_min

    try:
        sys_stor = (
            data.hydros_lf.filter(pl.col("block_id") == 0)
            .group_by(["scenario_id", "stage_id"])
            .agg(pl.col("storage_final_hm3").sum())
            .sort("stage_id")
            .collect(engine="streaming")
        )
    except (pl.exceptions.ColumnNotFoundError, KeyError):
        sys_stor = pl.DataFrame()

    sys_fig = go.Figure()
    if not sys_stor.is_empty() and total_useful > 0:
        df_sys = sys_stor.to_pandas()
        # Convert to % of useful volume
        df_sys["_pct"] = (
            (df_sys["storage_final_hm3"] - total_vol_min) / total_useful * 100.0
        )
        pct = compute_percentiles(df_sys, ["stage_id"], "_pct")
        stages = sorted(pct["stage_id"].tolist())
        xlabels_sys = stage_x_labels(stages, data.stage_labels)
        xdates_sys = stage_x_dates(stages, data.stage_dates)

        for q_col, q_name, dash in [
            ("p10", "P10", "dot"),
            ("p50", "P50", "solid"),
            ("p90", "P90", "dot"),
        ]:
            sys_fig.add_trace(
                go.Scatter(
                    x=xdates_sys,
                    y=pct[q_col].tolist(),
                    name=q_name,
                    mode="lines",
                    line={
                        "color": COLORS["hydro"],
                        "width": 2 if q_col == "p50" else 1.5,
                        "dash": dash,
                    },
                )
            )
        apply_stage_date_axis(sys_fig, xdates_sys, xlabels_sys)

    else:
        sys_fig.add_annotation(
            text="No data.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )

    sys_fig.update_layout(
        xaxis_title="Stage",
        yaxis_title="% Useful Volume",
        yaxis=dict(range=[0, 105]),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=_MARGIN,
    )

    # ------------------------------------------------------------------
    # By-bus storage as % of useful volume
    # ------------------------------------------------------------------
    # Build hydro-to-bus mapping and per-bus useful volume / vol_min
    hydro_to_bus: dict[int, int] = data.hydro_bus_map
    bus_useful_vol: dict[int, float] = {}
    bus_vol_min: dict[int, float] = {}
    for hid, bus_id in hydro_to_bus.items():
        if bus_id not in bus_useful_vol:
            bus_useful_vol[bus_id] = 0.0
            bus_vol_min[bus_id] = 0.0
        meta = data.hydro_meta.get(hid, {})
        vol_max = float(meta.get("vol_max", 0))
        vol_min = float(meta.get("vol_min", 0))
        bus_useful_vol[bus_id] += vol_max - vol_min
        bus_vol_min[bus_id] += vol_min

    # Filter to buses that have hydros and are non-fictitious
    active_bus_ids = [b for b in bus_ids if b in bus_useful_vol]

    try:
        hbus_df = pl.DataFrame(
            {
                "hydro_id": list(hydro_to_bus.keys()),
                "bus_id": list(hydro_to_bus.values()),
            }
        )
        bus_stor = (
            data.hydros_lf.filter(pl.col("block_id") == 0)
            .join(hbus_df.lazy(), on="hydro_id")
            .filter(pl.col("bus_id").is_in(active_bus_ids))
            .group_by(["scenario_id", "stage_id", "bus_id"])
            .agg(pl.col("storage_final_hm3").sum())
            .sort("stage_id")
            .collect(engine="streaming")
        )
    except (pl.exceptions.ColumnNotFoundError, KeyError):
        bus_stor = pl.DataFrame()

    import math as _math

    n_buses = len(active_bus_ids)
    if n_buses > 0 and not bus_stor.is_empty():
        n_cols_bus = 2
        n_rows_bus = _math.ceil(n_buses / n_cols_bus)
        bus_titles = [data.bus_names.get(b, str(b)) for b in active_bus_ids]
        while len(bus_titles) < n_rows_bus * n_cols_bus:
            bus_titles.append("")

        bus_fig = make_subplots(
            rows=n_rows_bus,
            cols=n_cols_bus,
            shared_xaxes=False,
            subplot_titles=bus_titles,
            vertical_spacing=0.18,
            horizontal_spacing=0.08,
        )
        all_stages = sorted(bus_stor["stage_id"].unique().to_list())
        xlabels = stage_x_labels(all_stages, data.stage_labels)
        xdates = stage_x_dates(all_stages, data.stage_dates)

        for idx, bus_id in enumerate(active_bus_ids):
            row = idx // n_cols_bus + 1
            col = idx % n_cols_bus + 1
            sub = bus_stor.filter(pl.col("bus_id") == bus_id).to_pandas()
            useful_vol = bus_useful_vol.get(bus_id, 0.0)
            vmin = bus_vol_min.get(bus_id, 0.0)
            sub = sub.copy()
            if useful_vol > 0.0:
                sub["_pct"] = (sub["storage_final_hm3"] - vmin) / useful_vol * 100.0
            else:
                sub["_pct"] = 0.0

            pct = compute_percentiles(sub, ["stage_id"], "_pct")
            pct["_x"] = pct["stage_id"].map(dict(zip(all_stages, xdates)))
            color = BUS_COLORS[idx % len(BUS_COLORS)]
            bus_name = data.bus_names.get(bus_id, str(bus_id))
            add_mean_p50_band(bus_fig, pct, "_x", bus_name, color, row=row, col=col)
            bus_fig.update_yaxes(title_text="%", row=row, col=col)

        bus_fig.update_layout(
            height=320 * n_rows_bus + 60,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
            ),
            margin=dict(l=60, r=30, t=80, b=50),
        )
        apply_stage_date_axis(bus_fig, xdates, xlabels)
    else:
        bus_fig = go.Figure()
        bus_fig.add_annotation(
            text="No data.", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5
        )
        bus_fig.update_layout(legend=_LEGEND, margin=_MARGIN)

    sys_card = make_chart_card(
        sys_fig,
        "Aggregate Storage Trajectory (% useful volume)",
        "v2-energy-storage-system",
        height=420,
    )
    n_rows_bus_h = _math.ceil(n_buses / 2) if n_buses > 0 else 1
    bus_card = make_chart_card(
        bus_fig,
        "Storage by Bus (% useful volume)",
        "v2-energy-storage-by-bus",
        height=max(320 * n_rows_bus_h + 60, 420),
    )

    content = chart_grid([sys_card], single=True) + chart_grid([bus_card], single=True)
    return collapsible_section(
        "Reservoir Storage",
        content,
        section_id="v2-energy-storage-section",
        default_collapsed=False,
    )


# ---------------------------------------------------------------------------
# Section G — Stored Energy (EARM) and Section H — Natural Inflow Energy (ENA)
# ---------------------------------------------------------------------------


def _aggregate_hydro_column(
    data: DashboardData,
    column: str,
    *,
    block_filter: int | None = 0,
) -> pl.DataFrame:
    """Sum *column* across hydros per (scenario_id, stage_id).

    When *block_filter* is given (default 0), filters the LazyFrame to that
    block — appropriate for stage-level columns like ``stored_energy_*_mwh``.
    Returns an empty DataFrame when the column is absent (older simulation
    outputs without the cobre HEAD energy columns).
    """
    try:
        lf = data.hydros_lf
        if block_filter is not None:
            lf = lf.filter(pl.col("block_id") == block_filter)
        return (
            lf.group_by(["scenario_id", "stage_id"])
            .agg(pl.col(column).sum())
            .sort("stage_id")
            .collect(engine="streaming")
        )
    except (pl.exceptions.ColumnNotFoundError, KeyError):
        return pl.DataFrame()


def _aggregate_hydro_column_by_bus(
    data: DashboardData,
    column: str,
    *,
    block_filter: int | None = 0,
) -> pl.DataFrame:
    """Sum *column* across hydros per (scenario_id, stage_id, bus_id)."""
    try:
        hbus_df = pl.DataFrame(
            {
                "hydro_id": list(data.hydro_bus_map.keys()),
                "bus_id": list(data.hydro_bus_map.values()),
            }
        )
        lf = data.hydros_lf
        if block_filter is not None:
            lf = lf.filter(pl.col("block_id") == block_filter)
        return (
            lf.join(hbus_df.lazy(), on="hydro_id")
            .group_by(["scenario_id", "stage_id", "bus_id"])
            .agg(pl.col(column).sum())
            .sort("stage_id")
            .collect(engine="streaming")
        )
    except (pl.exceptions.ColumnNotFoundError, KeyError):
        return pl.DataFrame()


def _energy_quantiles_fig(
    df: pl.DataFrame,
    value_col: str,
    *,
    stage_labels: dict[int, str],
    stage_dates: dict[int, str],
    y_title: str,
    color: str,
    scale: float = 1.0,
) -> go.Figure:
    """Build a stage-axis p10/p50/p90 figure from a (scenario, stage, value) frame.

    *scale* is applied to the value (e.g. 1/1000 to convert MWh→GWh).
    """
    fig = go.Figure()
    if df.is_empty():
        fig.add_annotation(
            text="No data.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        fig.update_layout(margin=_MARGIN, legend=_LEGEND)
        return fig

    pdf = df.to_pandas().copy()
    pdf["_scaled"] = pdf[value_col] * scale
    pct = compute_percentiles(pdf, ["stage_id"], "_scaled")
    stages = sorted(pct["stage_id"].tolist())
    xlabels = stage_x_labels(stages, stage_labels)
    xdates = stage_x_dates(stages, stage_dates)

    for q_col, q_name, dash in [
        ("p10", "P10", "dot"),
        ("p50", "P50", "solid"),
        ("p90", "P90", "dot"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=xdates,
                y=pct[q_col].tolist(),
                name=q_name,
                mode="lines",
                line={
                    "color": color,
                    "width": 2 if q_col == "p50" else 1.5,
                    "dash": dash,
                },
            )
        )

    fig.update_layout(
        xaxis_title="Stage",
        yaxis_title=y_title,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=_MARGIN,
    )
    apply_stage_date_axis(fig, xdates, xlabels)
    return fig


def _render_stored_energy(data: DashboardData) -> str:
    """Stored energy (EARM) section.

    Uses cobre HEAD's ``stored_energy_final_mwh`` column (per (stage, block,
    hydro)). For each (scenario, stage) sums across all hydros at block_id=0,
    converts MWh→GWh, and renders a p10/p50/p90 band. Optionally a per-bus
    breakdown chart alongside.

    Degrades gracefully when the column is absent in the simulation output.
    """
    sys_df = _aggregate_hydro_column(data, "stored_energy_final_mwh")
    by_bus_df = _aggregate_hydro_column_by_bus(data, "stored_energy_final_mwh")

    if sys_df.is_empty():
        # Column absent — don't render the section at all to avoid an empty card.
        return ""

    sys_fig = _energy_quantiles_fig(
        sys_df,
        "stored_energy_final_mwh",
        stage_labels=data.stage_labels,
        stage_dates=data.stage_dates,
        y_title="System EARM (GWh)",
        color=COLORS["hydro"],
        scale=1.0 / 1000.0,
    )

    sys_card = make_chart_card(
        sys_fig,
        "Stored Energy (EARM) — System Aggregate (GWh)",
        "v2-energy-earm-system",
        height=420,
    )

    if not by_bus_df.is_empty():
        import math as _math

        bus_ids = sorted(by_bus_df["bus_id"].unique().to_list())
        bus_ids = [b for b in bus_ids if b in data.non_fictitious_bus_ids]
        n_cols = 2
        n_rows = _math.ceil(max(len(bus_ids), 1) / n_cols)
        titles = [data.bus_names.get(b, str(b)) for b in bus_ids]
        while len(titles) < n_rows * n_cols:
            titles.append("")

        bus_fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            shared_xaxes=False,
            subplot_titles=titles,
            vertical_spacing=0.18,
            horizontal_spacing=0.08,
        )
        all_stages = sorted(by_bus_df["stage_id"].unique().to_list())
        xlabels = stage_x_labels(all_stages, data.stage_labels)
        xdates = stage_x_dates(all_stages, data.stage_dates)
        for idx, bus_id in enumerate(bus_ids):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            sub = by_bus_df.filter(pl.col("bus_id") == bus_id).to_pandas().copy()
            sub["_gwh"] = sub["stored_energy_final_mwh"] / 1000.0
            pct = compute_percentiles(sub, ["stage_id"], "_gwh")
            pct["_x"] = pct["stage_id"].map(dict(zip(all_stages, xdates)))
            color = BUS_COLORS[idx % len(BUS_COLORS)]
            add_mean_p50_band(
                bus_fig,
                pct,
                "_x",
                data.bus_names.get(bus_id, str(bus_id)),
                color,
                row=row,
                col=col,
            )
            bus_fig.update_yaxes(title_text="GWh", row=row, col=col)

        bus_fig.update_layout(
            height=320 * n_rows + 60,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
            ),
            margin=dict(l=60, r=30, t=80, b=50),
        )
        apply_stage_date_axis(bus_fig, xdates, xlabels)
        bus_card = make_chart_card(
            bus_fig,
            "EARM by Bus (GWh)",
            "v2-energy-earm-by-bus",
            height=max(320 * n_rows + 60, 420),
        )
        content = chart_grid([sys_card], single=True) + chart_grid(
            [bus_card], single=True
        )
    else:
        content = chart_grid([sys_card], single=True)

    return collapsible_section(
        "Stored Energy (EARM)",
        content,
        section_id="v2-energy-earm-section",
        default_collapsed=False,
    )


def _render_inflow_energy(data: DashboardData) -> str:
    """Natural inflow energy (ENA) section.

    Uses ``incremental_inflow_energy_mw`` (a power, MW). System aggregate sums
    across hydros per (scenario, stage) at block_id=0 and shows the
    p10/p50/p90 band. Returns an empty string when the column is absent.
    """
    sys_df = _aggregate_hydro_column(data, "incremental_inflow_energy_mw")
    if sys_df.is_empty():
        return ""

    sys_fig = _energy_quantiles_fig(
        sys_df,
        "incremental_inflow_energy_mw",
        stage_labels=data.stage_labels,
        stage_dates=data.stage_dates,
        y_title="System ENA (MW)",
        color=GENERATION_COLORS.get("hydro", COLORS["hydro"]),
    )

    sys_card = make_chart_card(
        sys_fig,
        "Natural Inflow Energy (ENA) — System Aggregate (MW)",
        "v2-energy-ena-system",
        height=420,
    )

    return collapsible_section(
        "Natural Inflow Energy (ENA)",
        chart_grid([sys_card], single=True),
        section_id="v2-energy-ena-section",
        default_collapsed=False,
    )


# ---------------------------------------------------------------------------
# Section F — NCS & Curtailment (collapsible, collapsed by default)
# ---------------------------------------------------------------------------


def _chart_curtailment_by_source(data: DashboardData) -> go.Figure | None:
    """Build a horizontal bar chart of total curtailment GWh per NCS source.

    Aggregates ``curtailment_mwh`` by summing over stages and blocks per
    scenario, then takes the mean across scenarios.  Converts MWh to GWh.
    Bars are sorted descending by curtailment volume.  Hover text shows
    source name, GWh value, and percentage of total curtailment.

    Returns ``None`` when ``curtailment_mwh`` is missing, all values are
    zero, or the result DataFrame is empty.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData`
            instance.  Uses ``data.ncs_lf`` and ``data.names``.

    Returns:
        A :class:`plotly.graph_objects.Figure` with horizontal bars, or
        ``None`` when no curtailment data is available.
    """
    try:
        curt_df = (
            data.ncs_lf.group_by(["scenario_id", "non_controllable_id"])
            .agg(pl.col("curtailment_mwh").sum())
            .group_by("non_controllable_id")
            .agg(pl.col("curtailment_mwh").mean())
            .sort("curtailment_mwh", descending=True)
            .collect(engine="streaming")
        )
    except (pl.exceptions.ColumnNotFoundError, KeyError):
        return None

    if curt_df.is_empty() or curt_df["curtailment_mwh"].sum() == 0:
        return None

    ncs_ids = curt_df["non_controllable_id"].to_list()
    gwh_values = [v / 1_000.0 for v in curt_df["curtailment_mwh"].to_list()]
    total_gwh = sum(gwh_values)
    names_list = [
        entity_name(data.names, "non_controllable_sources", nid) for nid in ncs_ids
    ]

    # Filter out zero-curtailment sources
    filtered = [(n, g) for n, g in zip(names_list, gwh_values) if g > 0]
    if not filtered:
        return None
    names_f, gwh_f = zip(*filtered)

    fig = go.Figure(
        go.Bar(
            x=list(gwh_f),
            y=list(names_f),
            orientation="h",
            marker_color=COLORS["curtailment"],
            text=[f"{g:.1f} GWh ({g / total_gwh * 100:.1f}%)" for g in gwh_f],
            textposition="auto",
        )
    )
    apply_standard_layout(
        fig,
        xaxis_title="Curtailment (GWh)",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _render_ncs_curtailment(data: DashboardData) -> str:
    """Build NCS generation vs available capacity and curtailment by source charts.

    Left chart: stacked area of NCS generation (mean per stage, stage-avg MW)
    with an overlay line for available capacity (block-hours-weighted mean
    ``available_mw`` per stage).

    Right chart: horizontal bar chart of total curtailment GWh per NCS source,
    sorted descending by volume (via :func:`_chart_curtailment_by_source`).
    Falls back to a "No curtailment recorded" annotation when all values are
    zero or the ``curtailment_mwh`` column is absent.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        An HTML string: a ``collapsible_section`` wrapping a ``chart_grid``
        with two chart cards.
    """
    hours_df = pl.DataFrame(
        {
            "stage_id": list(data.stage_hours.keys()),
            "_hours": list(data.stage_hours.values()),
        }
    )
    # Build block-hours lookup LazyFrame for available_mw weighting
    bh_lazy = data.bh_df.lazy()

    # ------------------------------------------------------------------
    # NCS generation (stage-avg MW, mean across scenarios)
    # ------------------------------------------------------------------
    try:
        ncs_gen_raw = (
            data.ncs_lf.group_by(["scenario_id", "stage_id"])
            .agg(pl.col("generation_mwh").sum())
            .join(hours_df.lazy(), on="stage_id")
            .with_columns(
                (pl.col("generation_mwh") / pl.col("_hours")).alias("_gen_mw")
            )
            .group_by("stage_id")
            .agg(pl.col("_gen_mw").mean())
            .sort("stage_id")
            .collect(engine="streaming")
        )
    except (pl.exceptions.ColumnNotFoundError, KeyError):
        ncs_gen_raw = pl.DataFrame()

    # ------------------------------------------------------------------
    # NCS available capacity (block-hours-weighted mean per stage,
    # then mean across scenarios)
    # ------------------------------------------------------------------
    try:
        # Sum available_mw across entities per (scenario, stage, block),
        # then weight by block hours for stage-average MW.
        ncs_avail_raw = (
            data.ncs_lf.group_by(["scenario_id", "stage_id", "block_id"])
            .agg(pl.col("available_mw").sum())
            .join(bh_lazy, on=["stage_id", "block_id"])
            .group_by(["scenario_id", "stage_id"])
            .agg(
                (pl.col("available_mw") * pl.col("_bh")).sum().alias("_wmw"),
                pl.col("_bh").sum().alias("_bh_sum"),
            )
            .with_columns((pl.col("_wmw") / pl.col("_bh_sum")).alias("_avail_mw"))
            .group_by("stage_id")
            .agg(pl.col("_avail_mw").mean())
            .sort("stage_id")
            .collect(engine="streaming")
        )
    except (pl.exceptions.ColumnNotFoundError, KeyError):
        ncs_avail_raw = pl.DataFrame()

    # ------------------------------------------------------------------
    # Build left chart: generation stacked area + available capacity overlay
    # ------------------------------------------------------------------
    gen_fig = go.Figure()

    if not ncs_gen_raw.is_empty():
        stages = sorted(ncs_gen_raw["stage_id"].to_list())
        xlabels = stage_x_labels(stages, data.stage_labels)
        xdates = stage_x_dates(stages, data.stage_dates)
        gen_map = dict(
            zip(ncs_gen_raw["stage_id"].to_list(), ncs_gen_raw["_gen_mw"].to_list())
        )

        # Available capacity area (blue, behind generation)
        if not ncs_avail_raw.is_empty():
            avail_map = dict(
                zip(
                    ncs_avail_raw["stage_id"].to_list(),
                    ncs_avail_raw["_avail_mw"].to_list(),
                )
            )
            gen_fig.add_trace(
                go.Scatter(
                    x=xdates,
                    y=[avail_map.get(s, 0) for s in stages],
                    name="Available Capacity",
                    fill="tozeroy",
                    fillcolor="rgba(59,130,246,0.2)",
                    mode="lines",
                    line={"color": "#3B82F6", "width": 1.5},
                )
            )

        # Generation area (green, on top)
        gen_fig.add_trace(
            go.Scatter(
                x=xdates,
                y=[gen_map.get(s, 0) for s in stages],
                name="NCS Generation",
                fill="tozeroy",
                fillcolor="rgba(16,185,129,0.4)",
                mode="lines",
                line={"color": GENERATION_COLORS["ncs"], "width": 2},
            )
        )
        apply_stage_date_axis(gen_fig, xdates, xlabels)
    else:
        gen_fig.add_annotation(
            text="No data.", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5
        )

    gen_fig.update_layout(
        xaxis_title="Stage",
        yaxis_title="MW",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=_MARGIN,
    )

    # ------------------------------------------------------------------
    # Build right chart: curtailment by source (horizontal bar)
    # ------------------------------------------------------------------
    curtail_by_source_fig = _chart_curtailment_by_source(data)

    if curtail_by_source_fig is None:
        curtail_by_source_fig = go.Figure()
        curtail_by_source_fig.add_annotation(
            text="No curtailment recorded.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )

    gen_card = make_chart_card(
        gen_fig,
        "NCS Generation vs Available Capacity (stage-avg MW)",
        "v2-energy-ncs-gen-avail",
        height=420,
    )
    curtail_card = make_chart_card(
        curtail_by_source_fig,
        "Curtailment by Source (mean GWh, sorted by volume)",
        "v2-energy-ncs-curtailment",
        height=420,
    )

    return collapsible_section(
        "NCS & Curtailment",
        chart_grid([gen_card, curtail_card]),
        section_id="v2-energy-ncs-section",
        default_collapsed=False,
    )


# ---------------------------------------------------------------------------
# Section B — Interactive hero section with scenario selector
# ---------------------------------------------------------------------------


def _build_hero_data(
    data: DashboardData,
) -> tuple[dict, list[str]]:
    """Precompute per-view hero chart data and embed as a JSON-serialisable dict.

    Computes per-(scenario, stage) total generation MW for hydro, thermal, and
    NCS by summing ``generation_mwh`` across entities and dividing by stage
    hours.  Then derives p10/p50/p90 quantiles across scenarios per stage, and
    collects individual scenario arrays for the "All" view.  LP Load is
    deterministic (not scenario-dependent) and stored as a single array.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        A 2-tuple of:
          - dict with keys ``stages``, ``load``, ``p10``, ``p50``, ``p90``,
            ``all``.  Each percentile entry maps source keys ``hydro``,
            ``thermal``, ``ncs`` to ``list[float]``.  ``all`` is a list of
            per-scenario dicts with the same source keys.
          - list of human-readable x-axis labels aligned with ``stages``.
    """
    hours_df = pl.DataFrame(
        {
            "stage_id": list(data.stage_hours.keys()),
            "_hours": list(data.stage_hours.values()),
        }
    )

    def _per_scenario_stage_mw(lf: pl.LazyFrame, mwh_col: str) -> pl.DataFrame:
        """Return (scenario_id, stage_id, _avg_mw) summed across entities."""
        schema = lf.collect_schema()
        if mwh_col not in schema.names():
            return pl.DataFrame(
                schema={
                    "scenario_id": pl.Int64,
                    "stage_id": pl.Int64,
                    "_avg_mw": pl.Float64,
                }
            )
        try:
            result = (
                lf.group_by(["scenario_id", "stage_id"])
                .agg(pl.col(mwh_col).sum())
                .join(hours_df.lazy(), on="stage_id")
                .with_columns((pl.col(mwh_col) / pl.col("_hours")).alias("_avg_mw"))
                .select(["scenario_id", "stage_id", "_avg_mw"])
                .collect(engine="streaming")
            )
        except (
            pl.exceptions.ColumnNotFoundError,
            pl.exceptions.SchemaError,
            ValueError,
        ):
            return pl.DataFrame(
                schema={
                    "scenario_id": pl.Int64,
                    "stage_id": pl.Int64,
                    "_avg_mw": pl.Float64,
                }
            )
        return result

    h_df = _per_scenario_stage_mw(data.hydros_lf, "generation_mwh")
    t_df = _per_scenario_stage_mw(data.thermals_lf, "generation_mwh")
    n_df = _per_scenario_stage_mw(data.ncs_lf, "generation_mwh")

    # Determine canonical stage list
    all_stage_ids: set[int] = set()
    for df in (h_df, t_df, n_df):
        if df.height > 0:
            all_stage_ids.update(df["stage_id"].to_list())
    if not all_stage_ids:
        # Fall back: use stage_hours keys
        all_stage_ids = set(data.stage_hours.keys())

    stages = sorted(all_stage_ids)
    if not stages:
        return {
            "stages": [],
            "load": [],
            "p10": {"hydro": [], "thermal": [], "ncs": []},
            "p50": {"hydro": [], "thermal": [], "ncs": []},
            "p90": {"hydro": [], "thermal": [], "ncs": []},
            "all": [],
        }, []

    xlabels = stage_x_labels(stages, data.stage_labels)

    # Compute LP Load (deterministic)
    load_ser = _compute_lp_load(
        data.load_stats,
        data.load_factors_list,
        data.stage_hours,
        data.block_hours,
        bus_filter=data.non_fictitious_bus_ids,
    )
    load_vals = [round(load_ser.get(s, 0.0), 2) for s in stages]

    def _quantiles(df: pl.DataFrame, stage_id: int, q: float) -> float:
        """Return quantile *q* of _avg_mw for *stage_id* across scenarios."""
        if df.height == 0:
            return 0.0
        sub = df.filter(pl.col("stage_id") == stage_id)
        if sub.height == 0:
            return 0.0
        val = sub["_avg_mw"].quantile(q, interpolation="linear")
        return round(float(val) if val is not None else 0.0, 2)

    def _scenario_vals(df: pl.DataFrame, stage_id: int) -> dict[int, float]:
        """Return {scenario_id: avg_mw} for *stage_id*."""
        if df.height == 0:
            return {}
        sub = df.filter(pl.col("stage_id") == stage_id)
        return {
            int(r["scenario_id"]): round(float(r["_avg_mw"]), 2)
            for r in sub.iter_rows(named=True)
        }

    # Build percentile views
    p10: dict[str, list[float]] = {"hydro": [], "thermal": [], "ncs": []}
    p50: dict[str, list[float]] = {"hydro": [], "thermal": [], "ncs": []}
    p90: dict[str, list[float]] = {"hydro": [], "thermal": [], "ncs": []}

    for sid in stages:
        for key, df in [("hydro", h_df), ("thermal", t_df), ("ncs", n_df)]:
            p10[key].append(_quantiles(df, sid, 0.1))
            p50[key].append(_quantiles(df, sid, 0.5))
            p90[key].append(_quantiles(df, sid, 0.9))

    # Build "All" view: total generation per scenario (hydro + thermal + NCS)
    scenario_ids: set[int] = set()
    for df in (h_df, t_df, n_df):
        if df.height > 0:
            scenario_ids.update(df["scenario_id"].to_list())
    all_scenarios = sorted(scenario_ids)

    all_view: list[list[float]] = []
    for scen in all_scenarios:
        total: list[float] = []
        for sid in stages:
            s = 0.0
            for df in (h_df, t_df, n_df):
                vals = _scenario_vals(df, sid)
                s += vals.get(scen, 0.0)
            total.append(round(s, 2))
        all_view.append(total)

    result: dict = {
        "stages": stages,
        "load": load_vals,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "all": all_view,
    }
    return result, xlabels


def _build_hero_section(data: DashboardData) -> str:
    """Build the Section B HTML: scenario selector + hero chart div + JS.

    Emits a ``<select id="eb-scenario-sel">`` dropdown with options p10,
    p50, p90, and All.  Selecting an option calls ``updateEBHero()`` which
    uses ``Plotly.react()`` to redraw ``<div id="eb-hero">`` without a full
    page reload.  The LP Load overlay remains constant across all views.

    Falls back to the static :func:`_chart_gen_mix_hero` when no stage data
    is available (i.e. the ``stages`` array in the computed hero data is
    empty).

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        HTML string with the selector, chart div, and inline JS.
    """
    from cobre_bridge.dashboard.chart_helpers import make_chart_card
    from cobre_bridge.ui.html.document import chart_grid

    hero_data, xlabels = _build_hero_data(data)

    if not hero_data["stages"]:
        # Fall back to static chart when no stage data
        hero_fig = _chart_gen_mix_hero(data)
        return chart_grid(
            [
                make_chart_card(
                    hero_fig,
                    "System-Wide Generation Mix vs LP Load (stage-avg MW)",
                    "v2-energy-gen-mix-hero",
                    height=420,
                )
            ],
            single=True,
        )

    data_json = json_for_script(hero_data)
    labels_json = json_for_script(xlabels)
    xdates = stage_x_dates(hero_data["stages"], data.stage_dates)
    dates_json = json_for_script(xdates)

    hydro_color = GENERATION_COLORS["hydro"]
    thermal_color = GENERATION_COLORS["thermal"]
    ncs_color = GENERATION_COLORS["ncs"]
    load_color = COLORS["load"]

    selector_html = (
        '<div style="margin-bottom:16px;">'
        '<label for="eb-scenario-sel" '
        'style="font-weight:600;margin-right:8px;">Scenario View:</label>'
        '<select id="eb-scenario-sel" onchange="updateEBHero()" '
        'style="padding:8px 12px;font-size:0.9rem;border-radius:4px;'
        'border:1px solid #ccc;min-width:160px;">'
        '<option value="p50" selected>P50</option>'
        '<option value="p10">P10</option>'
        '<option value="p90">P90</option>'
        '<option value="all">All</option>'
        "</select>"
        "</div>"
    )
    chart_html = (
        '<div class="chart-grid-single">'
        '<div class="chart-card">'
        '<div class="chart-card-title">'
        "System-Wide Generation Mix vs LP Load (stage-avg MW)"
        "</div>"
        '<div id="eb-hero" style="width:100%;height:420px;"></div>'
        "</div>"
        "</div>"
    )

    script = (
        "<script>\n"
        "const EB_DATA = " + data_json + ";\n"
        "const EB_LABELS = " + labels_json + ";\n"
        "const EB_DATES = " + dates_json + ";\n"
        f"const _EB_HYDRO_COLOR = '{hydro_color}';\n"
        f"const _EB_THERMAL_COLOR = '{thermal_color}';\n"
        f"const _EB_NCS_COLOR = '{ncs_color}';\n"
        f"const _EB_LOAD_COLOR = '{load_color}';\n"
        + r"""
var _EB_L = {hovermode:'x unified',
             xaxis:{title:'Stage', type:'date', tickmode:'array', tickvals:EB_DATES, ticktext:EB_LABELS},
             yaxis:{title:'MW'},
             margin:{l:60,r:20,t:60,b:10},
             legend:{orientation:'h',yanchor:'bottom',y:1.02,
                     xanchor:'center',x:0.5,font:{size:11}}};
var _EB_C = {responsive:true};

function _eb_area(nm, y, color, stack) {
  var t = {x:EB_DATES, y:y, name:nm, mode:'lines',
           line:{color:color}};
  if(stack) { t.stackgroup = 'gen'; }
  return t;
}
function _eb_line(nm, y, color, dash) {
  return {x:EB_DATES, y:y, name:nm, mode:'lines',
          line:{color:color, width:2.5, dash:dash||'solid'}};
}
function _eb_thin_line(nm, y, color) {
  return {x:EB_DATES, y:y, name:nm, mode:'lines',
          line:{color:color, width:1}, opacity:0.15,
          showlegend:false, hoverinfo:'skip'};
}

function updateEBHero() {
  var sel = document.getElementById('eb-scenario-sel').value;
  var d = EB_DATA;
  var traces = [];
  var loadTrace = _eb_line('LP Load', d.load, _EB_LOAD_COLOR, 'dash');

  if(sel === 'all') {
    for(var i = 0; i < d.all.length; i++) {
      traces.push(_eb_thin_line('Scenario '+i, d.all[i], '#3B82F6'));
    }
    // Add p50 stacked area as reference
    traces.push(_eb_area('Hydro (p50)', d.p50.hydro, _EB_HYDRO_COLOR, true));
    traces.push(_eb_area('Thermal (p50)', d.p50.thermal, _EB_THERMAL_COLOR, true));
    traces.push(_eb_area('NCS (p50)', d.p50.ncs, _EB_NCS_COLOR, true));
    traces.push(loadTrace);
  } else {
    var view = d[sel];
    traces.push(_eb_area('Hydro', view.hydro, _EB_HYDRO_COLOR, true));
    traces.push(_eb_area('Thermal', view.thermal, _EB_THERMAL_COLOR, true));
    traces.push(_eb_area('NCS', view.ncs, _EB_NCS_COLOR, true));
    traces.push(loadTrace);
  }
  Plotly.react('eb-hero', traces, _EB_L, _EB_C);
}
document.addEventListener('DOMContentLoaded',
  function(){setTimeout(updateEBHero, 100);});
"""
        + "</script>"
    )

    return selector_html + chart_html + script


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:
    """Energy Balance tab renders only when simulation data is available."""
    return data.simulation_available


def render(data: DashboardData) -> str:
    """Return the full HTML string for the v2 Energy Balance tab content area.

    Sections:
      A. 6-card metrics row.
      B. System generation mix hero chart (stacked area + LP load).
      C. Generation by bus — collapsible facet, one subplot per
         non-fictitious bus.
      D. Deficit & Excess — two side-by-side per-bus charts with
         mean+p50+band, expanded by default.
      E. Reservoir Storage — system aggregate trajectory + per-bus %,
         collapsed by default.
      F. NCS & Curtailment — generation vs available + curtailment bar,
         collapsed by default.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        An HTML string ready to be inserted into the tab content area.
    """
    # Section A — metrics row
    metrics_html = _build_metrics_row(data)

    # Section B — static hero chart (mean stacked area + load)
    hero_fig = _chart_gen_mix_hero(data)
    hero_html = chart_grid(
        [
            make_chart_card(
                hero_fig,
                "System Generation Mix vs Load (mean, stage-avg MW)",
                "v2-energy-gen-mix-hero",
                height=420,
            )
        ],
        single=True,
    )

    # Section C — generation by bus (collapsible, default expanded)
    by_bus_fig = _chart_gen_by_bus(data)
    n_buses = len(data.non_fictitious_bus_ids)
    by_bus_html = collapsible_section(
        "Generation by Bus",
        chart_grid(
            [
                make_chart_card(
                    by_bus_fig,
                    "Generation + Net Import vs LP Load by Bus (stage-avg MW)",
                    "v2-energy-gen-by-bus",
                    height=max(350 * ((n_buses + 1) // 2), 400),
                )
            ],
            single=True,
        ),
        section_id="v2-energy-gen-by-bus-section",
        default_collapsed=False,
    )

    # Section D — deficit & excess (collapsible, default expanded)
    deficit_excess_html = _render_deficit_excess(data)

    # Section E — reservoir storage (collapsible, default collapsed)
    storage_html = _render_reservoir_storage(data)

    # Section G — stored energy (EARM), new cobre HEAD columns; renders empty
    # string when the simulation output lacks the energy columns.
    earm_html = _render_stored_energy(data)

    # Section H — natural inflow energy (ENA), new cobre HEAD column.
    ena_html = _render_inflow_energy(data)

    # Section F — NCS & curtailment (collapsible, default collapsed)
    ncs_html = _render_ncs_curtailment(data)

    return (
        metrics_html
        + hero_html
        + by_bus_html
        + deficit_excess_html
        + storage_html
        + earm_html
        + ena_html
        + ncs_html
    )

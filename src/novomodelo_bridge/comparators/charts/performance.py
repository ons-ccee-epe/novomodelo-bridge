"""Timing KPI cards and per-iteration training charts."""

from __future__ import annotations

import polars as pl

from cobre_bridge.comparators.html_report import (
    COLOR_COBRE,
    COLOR_NEWAVE,
)
from cobre_bridge.ui.html.plotly import facet_grid
from cobre_bridge.ui.html.plotly import plotly_div as _plotly_div


def performance_metric_cards(
    nw_tim_stages: dict[str, float],
    cobre_training_seconds: float,
    reference_label: str = "NEWAVE",
) -> str:
    """Headline timing KPIs: The source model total / training, Cobre total, speedup."""
    from cobre_bridge.comparators.html_report import metric_card, metrics_grid
    from cobre_bridge.ui.theme import COMPARISON_COLORS

    nw_total = float(nw_tim_stages.get("Tempo Total", 0.0))
    nw_policy = float(nw_tim_stages.get("Calculo da Politica", 0.0))
    cb_total = float(cobre_training_seconds)
    speedup = (nw_policy / cb_total) if cb_total > 1e-6 else float("nan")

    def _fmt_dur(seconds: float) -> str:
        if seconds <= 0:
            return "—"
        if seconds < 60:
            return f"{seconds:.1f} s"
        if seconds < 3600:
            return f"{seconds / 60:.1f} min"
        return f"{seconds / 3600:.2f} h"

    def _fmt_x(v: float) -> str:
        return "—" if v != v else f"{v:.1f}×"

    cards = [
        metric_card(
            _fmt_dur(nw_total),
            f"{reference_label} Total Wall-Clock",
            color=COMPARISON_COLORS.get("newave"),
        ),
        metric_card(
            _fmt_dur(nw_policy),
            f"{reference_label} Policy Training",
            color=COMPARISON_COLORS.get("newave"),
        ),
        metric_card(
            _fmt_dur(cb_total),
            "Cobre Training",
            color=COMPARISON_COLORS.get("cobre"),
        ),
        metric_card(
            _fmt_x(speedup),
            f"Speedup ({reference_label} policy ÷ Cobre training)",
            color=COMPARISON_COLORS.get("match"),
        ),
    ]
    return metrics_grid(cards)


def performance_iteration_chart(
    nw_tim_iterations: pl.DataFrame,
    cobre_convergence: pl.DataFrame,
    reference_label: str = "NEWAVE",
) -> str:
    """Line chart of total seconds per training iteration.

    The source model total times come from ``newave.tim`` (already in seconds); Cobre
    comes from ``training/convergence.parquet:time_total_ms`` converted to seconds.
    Iteration 1 carries clock-init garbage on the source model side; we clip to a
    sensible max for the chart but show the raw value in the tooltip via a textual
    hover.
    """
    has_nw = not nw_tim_iterations.is_empty()
    has_cb = (
        not cobre_convergence.is_empty()
        and "iteration" in cobre_convergence.columns
        and "time_total_ms" in cobre_convergence.columns
    )
    if not has_nw and not has_cb:
        return "<p>No timing data available.</p>"

    traces: list[dict] = []
    all_seconds: list[float] = []

    if has_nw:
        it_col = nw_tim_iterations["iteration"].to_list()
        tot = nw_tim_iterations["total_seconds"].to_list()
        all_seconds.extend([v for v in tot if v < 1e5])  # exclude clock-init garbage
        traces.append(
            {
                "x": it_col,
                "y": tot,
                "name": reference_label,
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"color": COLOR_NEWAVE, "width": 2},
                "marker": {"size": 5},
            }
        )

    if has_cb:
        df = (
            cobre_convergence.sort("iteration")
            if isinstance(cobre_convergence, pl.DataFrame)
            else cobre_convergence.sort_values("iteration")
        )
        it_col = (
            df["iteration"].to_list()
            if isinstance(df, pl.DataFrame)
            else df["iteration"].tolist()
        )
        ms = (
            df["time_total_ms"].to_list()
            if isinstance(df, pl.DataFrame)
            else df["time_total_ms"].tolist()
        )
        secs = [float(v) / 1000.0 for v in ms]
        all_seconds.extend(secs)
        traces.append(
            {
                "x": it_col,
                "y": secs,
                "name": "Cobre",
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"color": COLOR_COBRE, "width": 2},
                "marker": {"size": 5},
            }
        )

    layout: dict = {
        "title": "Iteration Wall-Clock (seconds)",
        "xaxis": {"title": "Iteration"},
        "yaxis": {"title": "Total seconds"},
    }
    # Clip the y-axis to actual data so the source model iter-1's clock-init garbage
    # doesn't compress every other iteration into a flat line at zero.
    if all_seconds:
        ymax = max(all_seconds) * 1.15
        layout["yaxis"]["range"] = [0, ymax]

    return _plotly_div(traces, layout, height=420)


def performance_fwd_bwd_split_chart(
    nw_tim_iterations: pl.DataFrame,
    cobre_convergence: pl.DataFrame,
    reference_label: str = "NEWAVE",
) -> str:
    """Stacked forward / backward split per iteration, the source model vs Cobre.

    Two panels stacked vertically: top panel = the source model (backward +
    forward stacked bars in seconds), bottom panel = Cobre (same but
    converted from ms).

    ``nw_tim_iterations`` renders its panel only when it actually carries the
    ``forward_seconds``/``backward_seconds`` columns this chart stacks -- a
    source with no forward/backward pass structure (e.g. DECOMP's nested
    Benders over an explicit tree) fills only ``iteration``/``total_seconds``
    and must never fabricate a split, so a non-empty-but-columnless frame
    degrades to "no source panel" here rather than raising.
    """
    has_nw = not nw_tim_iterations.is_empty() and {
        "forward_seconds",
        "backward_seconds",
    }.issubset(nw_tim_iterations.columns)
    has_cb = (
        not cobre_convergence.is_empty()
        and "iteration" in cobre_convergence.columns
        and {"time_forward_ms", "time_backward_ms"}.issubset(cobre_convergence.columns)
    )
    if not has_nw and not has_cb:
        return "<p>No forward/backward split available.</p>"

    panels: list[tuple[str, list[int], list[float], list[float]]] = []
    all_secs: list[float] = []
    if has_nw:
        it = nw_tim_iterations["iteration"].to_list()
        bw = nw_tim_iterations["backward_seconds"].to_list()
        fw = nw_tim_iterations["forward_seconds"].to_list()
        # Clip clock-init garbage on iter 1 so the chart auto-scales to the
        # real range. The raw value remains in the parsed DataFrame.
        bw_clipped = [v if v < 1e5 else 0.0 for v in bw]
        fw_clipped = [v if v < 1e5 else 0.0 for v in fw]
        panels.append((reference_label, it, bw_clipped, fw_clipped))
        all_secs.extend([v for v in bw_clipped + fw_clipped if v > 0])
    if has_cb:
        df = cobre_convergence.sort("iteration")
        it = df["iteration"].to_list()
        bw = [float(v) / 1000.0 for v in df["time_backward_ms"].to_list()]
        fw = [float(v) / 1000.0 for v in df["time_forward_ms"].to_list()]
        panels.append(("Cobre", it, bw, fw))
        all_secs.extend(bw + fw)

    nrows = len(panels)
    facets = facet_grid(nrows, ncols=1, row_gap=0.10, min_row_h=0.0)
    traces: list[dict] = []
    layout: dict = {"title": "Forward / Backward Split per Iteration (seconds)"}
    for idx, (label, it, bw, fw) in enumerate(panels):
        ax_idx = idx + 1
        xa = f"x{ax_idx}" if ax_idx > 1 else "x"
        ya = f"y{ax_idx}" if ax_idx > 1 else "y"
        facet = facets[idx]
        xa_key = f"xaxis{ax_idx}" if ax_idx > 1 else "xaxis"
        ya_key = f"yaxis{ax_idx}" if ax_idx > 1 else "yaxis"
        layout[xa_key] = {
            "domain": facet.x_domain,
            "title": "Iteration" if idx == nrows - 1 else "",
            "anchor": ya,
        }
        layout[ya_key] = {
            "domain": facet.y_domain,
            "title": f"{label} (s)",
            "anchor": xa,
        }
        first = idx == 0
        traces.append(
            {
                "x": it,
                "y": bw,
                "name": "Backward",
                "type": "bar",
                "marker": {"color": "#7C3AED"},
                "xaxis": xa,
                "yaxis": ya,
                "legendgroup": "bw",
                "showlegend": first,
            }
        )
        traces.append(
            {
                "x": it,
                "y": fw,
                "name": "Forward",
                "type": "bar",
                "marker": {"color": "#0EA5E9"},
                "xaxis": xa,
                "yaxis": ya,
                "legendgroup": "fw",
                "showlegend": first,
            }
        )
    layout["barmode"] = "stack"
    return _plotly_div(traces, layout, height=max(nrows * 320 + 80, 360))

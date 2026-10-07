"""Productivity tab charts: fidelity scatter, per-stage detail, blocks table."""

from __future__ import annotations

import polars as pl

from cobre_bridge.comparators import analyze
from cobre_bridge.comparators.charts._shared import _build_interactive_detail_html
from cobre_bridge.comparators.html_report import COLOR_COBRE
from cobre_bridge.ui.html.document import escape_text
from cobre_bridge.ui.html.plotly import plotly_div as _plotly_div

# kind -> (pmo column, cobre-bridge column, pmo label, cobre-bridge label). Each
# productivity-comparison scatter is a *static* conversion-fidelity check: The source
# model pmo.dat productivity against the value cobre-bridge computes from the same HIDR
# cadastro inputs. Both sides live in the ``productivity_detail`` frame built in
# results.py and should land on y = x.
_PRODUCTIVITY_KINDS: dict[str, tuple[str, str, str, str]] = {
    "point": (
        "nw_altura_65",
        "cb_point",
        "produtibilidade_altura_65",
        "compute_productivity",
    ),
    "equivalent": (
        "nw_equivalent",
        "cb_equivalent",
        "produtibilidade_equivalente_volmin_volmax",
        "stored_energy_productivity",
    ),
    "accumulated": (
        "nw_accumulated_earm",
        "cb_accumulated",
        "produtibilidade_acumulada_calculo_earm",
        "accumulated_integrated_productivity",
    ),
}


def productivity_comparison_scatter(
    df: pl.DataFrame,
    kind: str,
    title: str | None = None,
    reference_label: str = "NEWAVE",
) -> str:
    """Static conversion-fidelity scatter for one productivity *kind*.

    *kind* selects the (pmo, cobre-bridge) column pair from :data:`_PRODUCTIVITY_KINDS`:
    ``"point"`` (pmo ``produtibilidade_altura_65`` vs ``compute_productivity``),
    ``"equivalent"`` (pmo ``produtibilidade_equivalente_volmin_volmax`` vs
    ``stored_energy_productivity``), ``"accumulated"`` (pmo
    ``produtibilidade_acumulada_calculo_earm`` vs the cascade accumulated-integrated
    value). Both sides are derived from the same the source model inputs, so the points
    should land on the ``y = x`` reference line — this validates the conversion rather
    than comparing against the per-stage simulation output. The source model pmo is on
    x, cobre-bridge on y; rows where either side is null are skipped. Annotated with
    mean & max relative error ``|cobre-bridge − pmo| / pmo`` and the number of plants
    compared.
    """
    if kind not in _PRODUCTIVITY_KINDS:
        raise ValueError(f"Unknown productivity kind: {kind!r}")
    nw_col, cb_col, nw_label, cb_label = _PRODUCTIVITY_KINDS[kind]
    if df.is_empty() or nw_col not in df.columns or cb_col not in df.columns:
        return "<p>No productivity data available.</p>"

    sub = df.select("plant_name", nw_col, cb_col).drop_nulls([nw_col, cb_col])
    if sub.is_empty():
        return "<p>No productivity data available.</p>"

    nw_vals = [float(v) for v in sub[nw_col].to_list()]
    cb_vals = [float(v) for v in sub[cb_col].to_list()]
    names = [escape_text(n) for n in sub["plant_name"].to_list()]

    mean_rel, max_rel = analyze.productivity_scatter_errors(nw_vals, cb_vals)

    min_val = min(min(nw_vals), min(cb_vals))
    max_val = max(max(nw_vals), max(cb_vals))

    traces = [
        {
            "x": nw_vals,
            "y": cb_vals,
            "text": names,
            "name": "Plants",
            "type": "scatter",
            "mode": "markers",
            "marker": {"color": COLOR_COBRE, "size": 8},
            "hovertemplate": (
                "%{text}<br>pmo: %{x:.4f}<br>cobre-bridge: %{y:.4f}<extra></extra>"
            ),
        },
        {
            "x": [min_val, max_val],
            "y": [min_val, max_val],
            "name": "y = x",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": "#8B9298", "dash": "dash"},
            "showlegend": False,
            "hoverinfo": "skip",
        },
    ]

    layout = {
        "title": title or f"Static productivity: {nw_label} vs {cb_label}",
        "xaxis": {"title": f"{reference_label} pmo {nw_label}"},
        "yaxis": {"title": f"cobre-bridge {cb_label}"},
        "annotations": [
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.02,
                "y": 0.98,
                "xanchor": "left",
                "yanchor": "top",
                "showarrow": False,
                "align": "left",
                "bgcolor": "rgba(255,255,255,0.75)",
                "bordercolor": "#D1D5DB",
                "borderwidth": 1,
                "borderpad": 4,
                "font": {"size": 11},
                "text": (
                    f"N = {len(nw_vals)} plants<br>"
                    f"mean rel. err = {mean_rel * 100:.2f}%<br>"
                    f"max rel. err = {max_rel * 100:.2f}%"
                ),
            }
        ],
    }

    return _plotly_div(traces, layout)


def productivity_per_stage_chart(
    per_stage: pl.DataFrame, reference_label: str = "NEWAVE"
) -> str:
    """Per-plant realized productivity (generation / turbined) across stages.

    Productivity is **constant within a stage but varies across stages** in both models,
    tracking the reservoir head reached each stage. Reuses the shared interactive
    per-plant widget (:func:`_build_interactive_detail_html` — the same JS ``<select>``
    dropdown the hydro/thermal detail tabs use), so every reservoir is selectable one at
    a time (the source model vs Cobre) rather than a hand-picked subset. Consumes the
    per-(plant, stage) frame from
    :func:`cobre_bridge.comparators.analyze.productivity_per_stage_frame`.
    """
    var_key = "productivity_mw_per_m3s"
    if per_stage.is_empty():
        return "<p>No per-stage productivity data available.</p>"

    plants: dict[tuple[str, int], dict[int, tuple[float, float]]] = {}
    cobre_ids: dict[tuple[str, int], int] = {}
    for row in per_stage.iter_rows(named=True):
        key = (row["plant_name"], row["newave_code"])
        plants.setdefault(key, {})[row["stage"]] = (
            row["newave_value"],
            row["cobre_value"],
        )
        cobre_ids[key] = row["cobre_id"]

    if not plants:
        return "<p>No per-stage productivity data available.</p>"

    js_plants: dict[str, dict] = {}
    for (name, code), stage_data in sorted(plants.items()):
        stages = sorted(stage_data)
        js_plants[f"{code}_{name}"] = {
            "name": name,
            "code": code,
            "cobre_id": cobre_ids[(name, code)],
            f"{var_key}_stages": stages,
            f"{var_key}_nw": [stage_data[s][0] for s in stages],
            f"{var_key}_cb": [stage_data[s][1] for s in stages],
        }

    variables = [(var_key, "Realized productivity — Gen / Turbined (MW per m³/s)")]
    return _build_interactive_detail_html(
        js_plants, variables, "prodstage", "Reservoir", reference_label
    )


def _prod_blocks_pct(nw: float | None, cb: float | None) -> float | None:
    """Relative diff (Cobre − the source model)/the source model in %, or None when the
    source model ≈ 0."""
    if nw is None or cb is None or abs(nw) <= 1e-12:
        return None
    return (cb - nw) / nw * 100.0


def productivity_blocks_table(df: pl.DataFrame, reference_label: str = "NEWAVE") -> str:
    """Grouped building-blocks table — per metric: The source model | Cobre | Δ%.

    One row per aligned hydro. The columns are organised into metric groups (ρ_esp,
    tailwater, losses, vmin, vmax), each spanning three sub-columns — the source model,
    Cobre, Δ% — via a two-level header (``colspan`` on the top row). Alternate metric
    groups get a subtle background tint across both header and body cells and a stronger
    left border, so the 2-by-2 (3-by-3 with Δ%) pairing is visually unmistakable. Δ% =
    (Cobre − the source model)/the source model (blank when the source model ≈ 0); cells
    with ``|Δ%| > 1%`` are highlighted. Reuses the ``cost-breakdown-table`` styling.
    """
    if df.is_empty():
        return "<p>No productivity data available.</p>"

    # Column groups: (label, nw_col, cb_col, fmt_decimals).
    groups: list[tuple[str, str, str, int]] = [
        ("ρ_esp", "nw_specific_productivity", "cb_specific_productivity", 5),
        ("Tailwater (m)", "nw_tailwater_m", "cb_tailwater_m", 2),
        ("Losses (m)", "nw_losses_m", "cb_losses_m", 3),
        ("Vmin (hm³)", "nw_vmin_hm3", "cb_vmin_hm3", 1),
        ("Vmax (hm³)", "nw_vmax_hm3", "cb_vmax_hm3", 1),
    ]
    groups = [g for g in groups if g[1] in df.columns and g[2] in df.columns]
    if not groups:
        return "<p>No productivity data available.</p>"

    def _fmt(v: float | None, decimals: int) -> str:
        return "—" if v is None else f"{v:.{decimals}f}"

    def _fmt_pct(p: float | None) -> str:
        return "" if p is None else f"{p:+.1f}%"

    def _cls(idx: int, *, sub_index: int, extra: str = "") -> str:
        """Class string for a numeric cell in metric-group *idx*.

        Even groups (0-based) get ``cb-group-tint``; the first sub-column
        (``sub_index == 0``) of any group after the first gets
        ``cb-group-sep`` (the stronger vertical separator).
        """
        parts = ["cb-num"]
        if idx % 2 == 0:
            parts.append("cb-group-tint")
        if idx > 0 and sub_index == 0:
            parts.append("cb-group-sep")
        if extra:
            parts.append(extra)
        return " ".join(parts)

    # --- Two-level header ---
    top_cells = ['<th class="cb-cat" rowspan="2">Plant</th>']
    sub_cells: list[str] = []
    for idx, (label, _, _, _) in enumerate(groups):
        top_cells.append(
            f'<th class="{_cls(idx, sub_index=0)}" colspan="3">'
            f"{escape_text(label)}</th>"
        )
        for j, sub in enumerate((reference_label, "Cobre", "Δ%")):
            sub_cells.append(f'<th class="{_cls(idx, sub_index=j)}">{sub}</th>')
    head = f"<thead><tr>{''.join(top_cells)}</tr><tr>{''.join(sub_cells)}</tr></thead>"

    # --- Body ---
    body_rows: list[str] = []
    for row in df.iter_rows(named=True):
        cells = [f'<td class="cb-cat">{escape_text(row["plant_name"])}</td>']
        for idx, (_, nw_col, cb_col, decimals) in enumerate(groups):
            nw_v = row.get(nw_col)
            cb_v = row.get(cb_col)
            pct = _prod_blocks_pct(nw_v, cb_v)
            highlight = "cb-diff-pos" if (pct is not None and abs(pct) > 1.0) else ""
            cells.append(
                f'<td class="{_cls(idx, sub_index=0)}">{_fmt(nw_v, decimals)}</td>'
            )
            cells.append(
                f'<td class="{_cls(idx, sub_index=1)}">{_fmt(cb_v, decimals)}</td>'
            )
            cells.append(
                f'<td class="{_cls(idx, sub_index=2, extra=highlight)}">'
                f"{_fmt_pct(pct)}</td>"
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    body = "<tbody>" + "".join(body_rows) + "</tbody>"
    caption = (
        "<caption>Productivity Building Blocks "
        '<span class="cb-caption-note">— columns are grouped per metric: '
        f"{reference_label} vs Cobre vs Δ%</span></caption>"
    )
    return (
        '<table class="cost-breakdown-table prod-blocks-table">'
        + caption
        + head
        + body
        + "</table>"
    )

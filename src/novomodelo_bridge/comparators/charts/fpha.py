"""Production-function (FPHA) comparison charts.

Fitted-surface fidelity table and the interactive per-plant detail (heatmaps
+ spillage slice) rendered from precomputed surface/spill frames, and the
operating-point views: the realized-productivity split table, the per-stage
tie share, and the GH-vs-turbined-flow slice with each model's fit inputs.
"""

from __future__ import annotations

from typing import TypedDict, cast

import polars as pl

from novomodelo_bridge.comparators import analyze
from novomodelo_bridge.comparators.html_report import (
    COLOR_NEWAVE,
    COLOR_NOVOMODELO,
)
from novomodelo_bridge.ui.html.document import escape_text, json_for_script
from novomodelo_bridge.ui.html.plotly import MARGIN_DEFAULTS as _MARGIN


def fpha_metrics_table(metrics: pl.DataFrame, reference_label: str = "NEWAVE") -> str:
    """Per-plant fitted-production-function fidelity table.

    Aggregates the per-(plant, stage) metrics into one row per plant: the mean and
    worst surface NMAE (as a % of the plant's max generation), the mean bias, the
    GHmax-ratio range across stages, and the plane counts. Sorted worst-NMAE
    first so the most divergent plants surface at the top; rows whose worst NMAE
    exceeds 5% are tinted. Reuses the ``cost-breakdown-table`` styling.
    """
    if metrics.is_empty():
        return "<p>No production-function (FPHA) data available.</p>"

    agg = analyze.fpha_metric_summary(metrics)

    def _fmt(value: float | None, decimals: int = 2) -> str:
        return "—" if value is None else f"{value:.{decimals}f}"

    rows_html: list[str] = []
    for r in agg.iter_rows(named=True):
        worst = r["worst_nmae"]
        tint = (
            ' style="background:#FEF3C7"' if worst is not None and worst > 5.0 else ""
        )
        kind = "reservoir" if (r["n_v"] or 0) > 1 else "run-of-river"
        rows_html.append(
            f"<tr{tint}>"
            f'<td class="cb-cat">{escape_text(str(r["plant_name"]))}</td>'
            f"<td>{kind}</td>"
            f"<td>{r['planes_nw']}/{r['planes_cb']}</td>"
            f"<td>{_fmt(r['mean_nmae'])}%</td>"
            f"<td>{_fmt(worst)}%</td>"
            f"<td>{_fmt(r['mean_bias'])}%</td>"
            f"<td>{_fmt(r['ghr_min'], 3)}–{_fmt(r['ghr_max'], 3)}</td>"
            f"</tr>"
        )

    head = (
        "<thead><tr>"
        "<th>Plant</th><th>Type</th><th>Planes nw/cb</th>"
        "<th>Mean NMAE</th><th>Worst NMAE</th><th>Mean bias</th>"
        "<th>GHmax ratio (min–max)</th>"
        "</tr></thead>"
    )
    caption = (
        f"<caption>Fitted production surface — Novomodelo vs {reference_label} "
        '<span class="cb-caption-note">— NMAE / bias as % of each plant\'s '
        f"max generation; GHmax ratio = Novomodelo / {reference_label} at the max "
        "V/Q corner</span></caption>"
    )
    return (
        '<table class="cost-breakdown-table fpha-table">'
        + caption
        + head
        + "<tbody>"
        + "".join(rows_html)
        + "</tbody></table>"
    )


class _PlantSurfaces(TypedDict):
    name: str
    n_v: int
    by_stage: dict[str, dict]


def _fpha_widget_data(
    surface: pl.DataFrame, spill: pl.DataFrame
) -> dict[str, dict[str, object]]:
    """Pivot the FPHA surface/spill frames into the per-plant widget payload.

    Per plant, per stage, the dense ``(V, Q)`` grid is reshaped into the ``z``
    matrix the heatmap needs (volume-major, turbined-minor — the order the
    surface frame is already sorted in), plus the spillage slice. Coordinates are
    rounded to keep the embedded JSON compact.
    """

    def _round(values: list[float], decimals: int) -> list[float]:
        return [round(float(v), decimals) for v in values]

    plants: dict[int, _PlantSurfaces] = {}
    for (cid, stage, source), sub in surface.partition_by(
        "novomodelo_id", "stage", "source", as_dict=True
    ).items():
        v_axis = _round(sub["v_hm3"].unique(maintain_order=True).to_list(), 1)
        n_v = len(v_axis)
        gh = _round(sub["gh_mw"].to_list(), 1)
        n_q = len(gh) // n_v if n_v else 0
        if n_q == 0:
            continue
        q_axis = _round(sub["q_m3s"][:n_q].to_list(), 1)
        z = [gh[i * n_q : (i + 1) * n_q] for i in range(n_v)]
        plant = plants.setdefault(
            int(cid),
            {"name": str(sub["plant_name"][0]), "n_v": n_v, "by_stage": {}},
        )
        plant["n_v"] = max(plant["n_v"], n_v)
        entry = plant["by_stage"].setdefault(str(int(stage)), {})
        entry["v"] = v_axis
        entry["q"] = q_axis
        entry["znw" if source == "newave" else "zcb"] = z

    for (cid, stage, source), sub in spill.partition_by(
        "novomodelo_id", "stage", "source", as_dict=True
    ).items():
        plant = plants.get(int(cid))
        if plant is None:
            continue
        entry = plant["by_stage"].get(str(int(stage)))
        if entry is None:
            continue
        entry["ss"] = _round(sub["s_m3s"].to_list(), 1)
        entry["spnw" if source == "newave" else "spcb"] = _round(
            sub["gh_mw"].to_list(), 2
        )

    out: dict[str, dict[str, object]] = {}
    for cid, plant in plants.items():
        out[str(cid)] = {
            "name": plant["name"],
            "n_v": plant["n_v"],
            "stages": sorted(int(s) for s in plant["by_stage"]),
            "byStage": plant["by_stage"],
        }
    return out


def fpha_detail_chart(
    surface: pl.DataFrame, spill: pl.DataFrame, reference_label: str = "NEWAVE"
) -> str:
    """Interactive per-plant FPHA surface comparison (heatmaps + spillage slice).

    A plant ``<select>`` (every plant fitted on both sides) drives a stage
    ``<select>``. For a reservoir plant the selected (plant, stage) renders one
    full-width rotatable 3D view of the production surface ``GH(V, Q)`` (sampled
    at the fitting-grid nodes at ``S = 0``) with NEWAVE / Novomodelo / Both / Difference
    toggle buttons — the two surfaces nearly coincide at ``S = 0``, so toggling
    isolates each and the difference rather than reading a muddy overlay.
    Run-of-river plants (single volume) render an overlaid ``GH`` vs
    turbined-flow curve instead. A further panel shows ``GH`` vs spilled flow at
    the max V/Q corner, exposing the spillage-coefficient behaviour the ``(V, Q)``
    grid holds fixed. Consumes the
    :func:`novomodelo_bridge.comparators.analyze.build_fpha_comparison` surface/spill
    frames.
    """
    if surface.is_empty():
        return "<p>No production-function (FPHA) data available.</p>"

    data = _fpha_widget_data(surface, spill)
    if not data:
        return "<p>No production-function (FPHA) data available.</p>"

    data_json = json_for_script(data)
    options = "".join(
        f'<option value="{cid}">{escape_text(str(entry["name"]))}</option>'
        for cid, entry in sorted(data.items(), key=lambda kv: str(kv[1]["name"]))
    )

    js = f"""
    var fphaData = {data_json};
    var fphaNw = '{COLOR_NEWAVE}';
    var fphaCb = '{COLOR_NOVOMODELO}';
    function fphaShow(id, on) {{
        var el = document.getElementById(id);
        if (el) el.style.display = on ? 'block' : 'none';
    }}
    function fphaPopulateStages() {{
        var p = fphaData[document.getElementById('fpha-plant').value];
        var ssel = document.getElementById('fpha-stage');
        var prev = ssel.value;
        ssel.innerHTML = '';
        (p.stages || []).forEach(function(s) {{
            var o = document.createElement('option');
            o.value = s; o.text = 'Stage ' + s; ssel.appendChild(o);
        }});
        if (prev && p.byStage[prev]) ssel.value = prev;
    }}
    function fphaUpdate() {{
        var p = fphaData[document.getElementById('fpha-plant').value];
        if (!p) return;
        var d = p.byStage[document.getElementById('fpha-stage').value];
        if (!d) return;
        var lay = {{margin: {json_for_script(_MARGIN)}, template: 'plotly_white',
            height: 360}};
        var reservoir = p.n_v > 1;
        fphaShow('fpha-surf-card', reservoir);
        fphaShow('fpha-line-card', !reservoir);
        if (reservoir) {{
            // One full-width 3D view; NEWAVE/Novomodelo/Both/Difference toggle buttons
            // switch which surface(s) show (the two nearly coincide at S=0, so an
            // always-on overlay reads as a blob — toggling isolates the signal).
            var zdiff = d.zcb.map(function(row, i) {{
                return row.map(function(val, j) {{ return val - d.znw[i][j]; }});
            }});
            var traces = [
                {{z: d.znw, x: d.q, y: d.v, type: 'surface', name: '{reference_label}',
                    visible: true, colorscale: 'Viridis', colorbar: {{title: 'MW'}},
                    hovertemplate: '{reference_label}<br>Q=%{{x}}<br>V=%{{y}}' +
                        '<br>GH=%{{z}} MW<extra></extra>'}},
                {{z: d.zcb, x: d.q, y: d.v, type: 'surface', name: 'Novomodelo',
                    visible: true, showscale: false, opacity: 0.9,
                    colorscale: [[0, fphaCb], [1, fphaCb]],
                    hovertemplate: 'Novomodelo<br>Q=%{{x}}<br>V=%{{y}}' +
                        '<br>GH=%{{z}} MW<extra></extra>'}},
                {{z: zdiff, x: d.q, y: d.v, type: 'surface', name: 'Difference',
                    visible: false, colorscale: 'RdBu', reversescale: true,
                    cmid: 0, colorbar: {{title: 'ΔMW'}},
                    hovertemplate: 'Q=%{{x}}<br>V=%{{y}}<br>' +
                        'Δ=%{{z}} MW<extra></extra>'}}];
            function fphaBtn(label, vis, ztitle, ttl) {{
                return {{label: label, method: 'update', args: [{{visible: vis}},
                    {{'title.text': ttl, 'scene.zaxis.title.text': ztitle,
                        'scene.zaxis.autorange': true}}]}};
            }}
            Plotly.react('fpha-surf', traces, {{
                title: {{text: 'GH(V,Q): {reference_label} (color) + Novomodelo (orange)'}},
                height: 600, margin: {{l: 0, r: 0, t: 80, b: 0}},
                template: 'plotly_white',
                scene: {{xaxis: {{title: 'Turbined (m³/s)'}},
                    yaxis: {{title: 'Volume (hm³)'}},
                    zaxis: {{title: 'GH (MW)', autorange: true}},
                    aspectmode: 'cube', camera: {{eye: {{x: 1.7, y: 1.7, z: 0.9}}}}}},
                updatemenus: [{{type: 'buttons', direction: 'right',
                    showactive: true, active: 2, x: 0, xanchor: 'left',
                    y: 1.06, yanchor: 'bottom', buttons: [
                    fphaBtn('{reference_label}', [true, false, false], 'GH (MW)',
                        '{reference_label} GH(V,Q)'),
                    fphaBtn('Novomodelo', [false, true, false], 'GH (MW)',
                        'Novomodelo GH(V,Q)'),
                    fphaBtn('Both', [true, true, false], 'GH (MW)',
                        'GH(V,Q): {reference_label} (color) + Novomodelo (orange)'),
                    fphaBtn('Difference', [false, false, true], 'Δ MW',
                        'Novomodelo − {reference_label} (MW)')]}}]
            }}, {{responsive: true}});
        }} else {{
            Plotly.react('fpha-line', [
                {{x: d.q, y: d.znw[0], name: '{reference_label}', type: 'scatter',
                    mode: 'lines', line: {{color: fphaNw, width: 2}}}},
                {{x: d.q, y: d.zcb[0], name: 'Novomodelo', type: 'scatter',
                    mode: 'lines', line: {{color: fphaCb, width: 2}}}}],
                Object.assign({{title: 'GH vs turbined flow',
                    xaxis: {{title: 'Turbined (m³/s)'}},
                    yaxis: {{title: 'GH (MW)'}}, hovermode: 'x unified'}}, lay),
                {{responsive: true}});
        }}
        Plotly.react('fpha-spill', [
            {{x: d.ss, y: d.spnw, name: '{reference_label}', type: 'scatter',
                mode: 'lines', line: {{color: fphaNw, width: 2}}}},
            {{x: d.ss, y: d.spcb, name: 'Novomodelo', type: 'scatter',
                mode: 'lines', line: {{color: fphaCb, width: 2}}}}],
            Object.assign({{title: 'GH vs spilled flow (at max V/Q)',
                xaxis: {{title: 'Spilled (m³/s)'}},
                yaxis: {{title: 'GH (MW)'}}, hovermode: 'x unified'}}, lay),
            {{responsive: true}});
    }}
    document.addEventListener('DOMContentLoaded', function() {{
        var psel = document.getElementById('fpha-plant');
        if (psel && psel.options.length > 0) {{ fphaPopulateStages(); fphaUpdate(); }}
    }});
    """

    return f"""
    <div class="plant-selector">
        <label for="fpha-plant">Plant:</label>
        <select id="fpha-plant"
                onchange="fphaPopulateStages(); fphaUpdate()">{options}</select>
        <label for="fpha-stage" style="margin-left:12px">Stage:</label>
        <select id="fpha-stage" onchange="fphaUpdate()"></select>
    </div>
    <div id="fpha-surf-card" class="chart-card" style="display:none">
        <div id="fpha-surf" style="width:100%;height:600px;"></div>
    </div>
    <div id="fpha-line-card" class="chart-card" style="display:none">
        <div id="fpha-line" style="width:100%;height:360px;"></div>
    </div>
    <div class="chart-card">
        <div id="fpha-spill" style="width:100%;height:360px;"></div>
    </div>
    <script>{js}</script>
    """


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{round(value * 100.0, 2) or 0.0:+.2f}%"


def _num(value: float | None, decimals: int = 3) -> str:
    return "—" if value is None else f"{value:.{decimals}f}"


def fpha_split_table(split: pl.DataFrame, reference_label: str = "NEWAVE") -> str:
    """Per-(plant, stage) realized-productivity gap, split at the operating points.

    One row per plant and stage from
    :func:`novomodelo_bridge.comparators.fpha.productivity_split`: the source
    model's mean generation, both realized productivities, the total gap in
    percent and in MW at the source model's mean turbined flow, its model and
    operating-point parts, each surface's error against the exact production
    function at the source model's points, and the weight share of tie points
    left out. Largest gap in MW first.
    """
    if split.is_empty():
        return "<p>No operating-point comparison available.</p>"
    ordered = split.with_columns(
        (
            (pl.col("rho_novomodelo") - pl.col("rho_source")) * pl.col("q_source_m3s")
        ).alias("_gap_mw")
    ).sort(pl.col("_gap_mw").abs(), descending=True, nulls_last=True)
    rows_html = [
        "<tr>"
        f'<td class="cb-cat">{escape_text(str(r["plant_name"] or r["novomodelo_id"]))}</td>'
        f'<td class="cb-num">{r["stage"]}</td>'
        f'<td class="cb-num">{_num(r["gh_source_mw"], 1)}</td>'
        f'<td class="cb-num">{_num(r["rho_source"])}</td>'
        f'<td class="cb-num">{_num(r["rho_novomodelo"])}</td>'
        f'<td class="cb-num">{_pct(r["d_total"])}</td>'
        f'<td class="cb-num">{_num(r["_gap_mw"], 1)}</td>'
        f'<td class="cb-num">{_pct(r["d_model"])}</td>'
        f'<td class="cb-num">{_pct(r["d_operating_point"])}</td>'
        f'<td class="cb-num">{_pct(r["fit_error_source"])}</td>'
        f'<td class="cb-num">{_pct(r["fit_error_novomodelo"])}</td>'
        f'<td class="cb-num">{_num(r["tie_share"] * 100.0, 1)}%</td>'
        "</tr>"
        for r in ordered.iter_rows(named=True)
    ]
    head = (
        "<thead><tr>"
        '<th class="cb-cat">Plant</th><th class="cb-num">Stage</th>'
        f'<th class="cb-num">GH {reference_label} (MW)</th>'
        f'<th class="cb-num">ρ {reference_label}</th><th class="cb-num">ρ Novomodelo</th>'
        '<th class="cb-num">Total</th><th class="cb-num">Total (MW)</th>'
        '<th class="cb-num">Model</th><th class="cb-num">Operating point</th>'
        f'<th class="cb-num">{reference_label} fit vs exact</th>'
        '<th class="cb-num">Novomodelo fit vs exact</th><th class="cb-num">Ties</th>'
        "</tr></thead>"
    )
    caption = (
        f"<caption>Realized productivity (MW per m³/s): Novomodelo vs {reference_label}"
        ' <span class="cb-caption-note">— Total = ρ Novomodelo / ρ '
        f"{reference_label} − 1 = (1 + Model)(1 + Operating point) − 1; Total "
        f"(MW) is that gap at {reference_label}'s mean turbined flow. Model "
        f"evaluates Novomodelo's planes at {reference_label}'s operating points "
        f"against {reference_label}'s realized generation; Operating point is "
        "the rest, from where Novomodelo operates. The fit columns compare each "
        f"model's planes with the exact production function at {reference_label}'s"
        " points. The percentages of a plant that barely generates mostly "
        f"reflect the rounding of {reference_label}'s printed generation. Tie "
        "points are left out (Ties = their weight share).</span>"
        "</caption>"
    )
    return (
        '<div class="fpha-scroll"><table class="cost-breakdown-table fpha-table">'
        + caption
        + head
        + "<tbody>"
        + "".join(rows_html)
        + "</tbody></table></div>"
    )


def tie_share_table(tie_share: pl.DataFrame, reference_label: str = "NEWAVE") -> str:
    """Per-stage weight share of each model's points that are ties."""
    if tie_share.is_empty():
        return ""

    def _share(value: float | None) -> str:
        return "—" if value is None else f"{value * 100.0:.1f}%"

    rows_html = [
        "<tr>"
        f'<td class="cb-num">{r["stage"]}</td>'
        f'<td class="cb-num">{_share(r["source_tie_share"])}</td>'
        f'<td class="cb-num">{_share(r["novomodelo_tie_share"])}</td>'
        "</tr>"
        for r in tie_share.sort("stage").iter_rows(named=True)
    ]
    return (
        '<table class="cost-breakdown-table fpha-table">'
        "<caption>Tie points by stage "
        '<span class="cb-caption-note">— weight share (probability × hours) of '
        "the points where the plant's energy or stored water is worth nothing "
        f"in either model: zero {reference_label} CMO, zero Novomodelo spot price at "
        "every bus the plant feeds, or zero Novomodelo water value. They are left "
        "out of the split above and of the per-plant ranking on the Hydro Plant "
        "Details tab.</span></caption>"
        "<thead><tr>"
        f'<th class="cb-num">Stage</th><th class="cb-num">{reference_label} points</th>'
        '<th class="cb-num">Novomodelo points</th>'
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table>"
    )


def _slice_widget_data(
    slices: pl.DataFrame, fit_inputs: pl.DataFrame, split: pl.DataFrame
) -> dict[str, dict[str, object]]:
    """Per plant and stage: the slice curves on one turbined-flow axis, the fit
    inputs and mean-point heads, and the split values."""

    def _clean(row: dict[str, object]) -> dict[str, object]:
        return {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in row.items()
            if k not in ("novomodelo_id", "stage", "plant_name")
        }

    fits = {
        (r["novomodelo_id"], r["stage"]): r for r in fit_inputs.iter_rows(named=True)
    }
    splits = {(r["novomodelo_id"], r["stage"]): r for r in split.iter_rows(named=True)}
    plants: dict[str, dict[str, object]] = {}
    for (cid, stage), sub in slices.partition_by(
        "novomodelo_id", "stage", as_dict=True
    ).items():
        key = (int(cid), int(stage))
        fit = fits.get(key)
        if fit is None:
            continue
        curves = {
            str(curve): [round(float(g), 2) for g in group["gh_mw"].to_list()]
            for (curve,), group in sub.partition_by("curve", as_dict=True).items()
        }
        q_axis = sub.filter(pl.col("curve") == "exact")["q_m3s"].to_list()
        plant = plants.setdefault(
            str(key[0]), {"name": str(fit["plant_name"] or key[0]), "byStage": {}}
        )
        cast("dict[str, object]", plant["byStage"])[str(key[1])] = {
            "q": [round(float(q), 1) for q in q_axis],
            "curves": curves,
            "fit": _clean(fit),
            "split": _clean(splits.get(key, {})),
        }
    for plant in plants.values():
        plant["stages"] = sorted(
            int(s) for s in cast("dict[str, object]", plant["byStage"])
        )
    return plants


def fpha_slice_chart(
    slices: pl.DataFrame,
    fit_inputs: pl.DataFrame,
    split: pl.DataFrame,
    reference_label: str = "NEWAVE",
) -> str:
    """Interactive per-plant GH-vs-turbined-flow slice at the operating point.

    A plant ``<select>`` drives a stage ``<select>``. The chart sweeps turbined
    flow with storage, spill and lateral flow held at the source model's mean
    operating point, drawing the source model's planes, Novomodelo's planes and the
    exact production function there, plus Novomodelo's planes at Novomodelo's own mean
    storage and spill (dashed). Markers place both models' mean operating
    points, and a second panel draws each model's planes minus the exact
    function. A table beside it lists each model's fit inputs, the head
    components at its mean point, and the split values. Consumes the
    operating-point comparison's slice, fit-input and split frames.
    """
    if slices.is_empty() or fit_inputs.is_empty():
        return "<p>No operating-point comparison available.</p>"
    data = _slice_widget_data(slices, fit_inputs, split)
    if not data:
        return "<p>No operating-point comparison available.</p>"
    options = "".join(
        f'<option value="{cid}">{escape_text(str(entry["name"]))}</option>'
        for cid, entry in sorted(data.items(), key=lambda kv: str(kv[1]["name"]))
    )
    ref = escape_text(reference_label)
    js = f"""
    var opData = {json_for_script(data)};
    function opFmt(v, d) {{
        return (v === null || v === undefined) ? '—' : Number(v).toFixed(d);
    }}
    function opPct(v) {{
        if (v === null || v === undefined) return '—';
        var p = Math.round(1e4 * v) / 100 || 0;
        return (p >= 0 ? '+' : '') + p.toFixed(2) + '%';
    }}
    function opPopulateStages() {{
        var p = opData[document.getElementById('op-plant').value];
        var ssel = document.getElementById('op-stage');
        var prev = ssel.value;
        ssel.innerHTML = '';
        (p.stages || []).forEach(function(s) {{
            var o = document.createElement('option');
            o.value = s; o.text = 'Stage ' + s; ssel.appendChild(o);
        }});
        if (prev && p.byStage[prev]) ssel.value = prev;
    }}
    function opRow(label, a, b) {{
        return '<tr><td class="cb-cat">' + label + '</td><td class="cb-num">' + a +
            '</td><td class="cb-num">' + b + '</td></tr>';
    }}
    function opTable(f, s) {{
        var w = function(lo, hi) {{ return opFmt(lo, 1) + ' – ' + opFmt(hi, 1); }};
        var rows = [
            opRow('Storage fit window (hm³)', w(f.source_v_min_hm3, f.source_v_max_hm3),
                w(f.novomodelo_v_min_hm3, f.novomodelo_v_max_hm3)),
            opRow('Turbined fit maximum (m³/s)', opFmt(f.source_q_max_m3s, 1),
                opFmt(f.novomodelo_q_max_m3s, 1)),
            opRow('Turbined limit this stage (m³/s)', '—',
                opFmt(f.novomodelo_q_stage_max_m3s, 1)),
            opRow('Spill fit maximum (m³/s)', '—', opFmt(f.novomodelo_spill_fit_max_m3s, 1)),
            opRow('Fit grid (V × Q) / planes', (f.source_grid || '—') + ' / ' +
                f.n_planes_source, '— / ' + f.n_planes_novomodelo),
            opRow('Generation fit maximum (MW)', opFmt(f.source_gh_max_mw, 1), '—')];
        [['Mean storage (hm³)', 'v_hm3', 1], ['Mean turbined (m³/s)', 'q_m3s', 1],
         ['Mean spill (m³/s)', 's_m3s', 1], ['Mean generation (MW)', 'gh_mw', 1],
         ['Lateral flow into the tailrace (m³/s)', 'lateral_m3s', 1],
         ['Downstream forebay (m)', 'downstream_level_m', 2],
         ['Forebay (m)', 'forebay_m', 2], ['Tailrace (m)', 'tailrace_m', 2],
         ['Losses (m)', 'loss_m', 2], ['Net head (m)', 'net_head_m', 2],
         ['Exact ρ at the mean point', 'rho_exact', 4]].forEach(function(h) {{
            rows.push(opRow(h[0], opFmt(f['source_' + h[1]], h[2]),
                opFmt(f['novomodelo_' + h[1]], h[2])));
        }});
        [['Realized ρ (ties left out)', 'rho_source', 'rho_novomodelo'],
         ['ρ of the planes at {ref} points', 'rho_source_planes_at_source',
            'rho_novomodelo_planes_at_source'],
         ["Exact ρ at each model's points", 'rho_exact_at_source',
            'rho_exact_at_novomodelo']].forEach(function(h) {{
            rows.push(opRow(h[0], opFmt(s[h[1]], 4), opFmt(s[h[2]], 4)));
        }});
        rows.push(opRow('Gap: total / model / operating point', '',
            opPct(s.d_total) + ' / ' + opPct(s.d_model) + ' / ' +
            opPct(s.d_operating_point)));
        rows.push(opRow('Tie share', opFmt(100 * (s.tie_share || 0), 1) + '%', ''));
        rows.push(opRow('Exact vs {ref} printed FPH, max (MW)',
            opFmt(f.exact_check_max_mw, 2), ''));
        return '<table class="cost-breakdown-table fpha-table"><thead><tr>' +
            '<th class="cb-cat"></th><th class="cb-num">{ref}</th>' +
            '<th class="cb-num">Novomodelo</th></tr></thead><tbody>' + rows.join('') +
            '</tbody></table>';
    }}
    function opUpdate() {{
        var p = opData[document.getElementById('op-plant').value];
        if (!p) return;
        var d = p.byStage[document.getElementById('op-stage').value];
        if (!d) return;
        var f = d.fit, c = d.curves;
        var traces = [
            {{x: d.q, y: c.source_planes, name: '{ref} planes', mode: 'lines',
                line: {{color: '{COLOR_NEWAVE}', width: 2}}}},
            {{x: d.q, y: c.novomodelo_planes, name: 'Novomodelo planes', mode: 'lines',
                line: {{color: '{COLOR_NOVOMODELO}', width: 2}}}},
            {{x: d.q, y: c.exact, name: 'Exact', mode: 'lines',
                line: {{color: '#374151', width: 1.5, dash: 'dot'}}}}];
        if (c.novomodelo_planes_own) traces.push({{x: d.q, y: c.novomodelo_planes_own,
            name: "Novomodelo planes at Novomodelo's point", mode: 'lines',
            line: {{color: '{COLOR_NOVOMODELO}', width: 1.5, dash: 'dash'}}}});
        traces.push({{x: [f.source_q_m3s], y: [f.source_gh_mw],
            name: '{ref} mean point', mode: 'markers',
            marker: {{color: '{COLOR_NEWAVE}', size: 11, symbol: 'circle'}}}});
        if (f.novomodelo_q_m3s !== null) traces.push({{x: [f.novomodelo_q_m3s],
            y: [f.novomodelo_gh_mw], name: 'Novomodelo mean point', mode: 'markers',
            marker: {{color: '{COLOR_NOVOMODELO}', size: 11, symbol: 'diamond'}}}});
        var legend = {{orientation: 'h', y: -0.3, yanchor: 'top'}};
        var margin = Object.assign({{}}, {json_for_script(_MARGIN)}, {{b: 110}});
        Plotly.react('op-slice', traces, {{
            title: 'GH vs turbined flow at the {ref} mean storage and spill',
            xaxis: {{title: 'Turbined (m³/s)'}}, yaxis: {{title: 'GH (MW)'}},
            hovermode: 'x unified', template: 'plotly_white', height: 460,
            legend: legend, margin: margin}},
            {{responsive: true}});
        var minus = function(curve) {{
            return curve.map(function(g, i) {{ return g - c.exact[i]; }});
        }};
        var gaps = [
            {{x: d.q, y: minus(c.source_planes), name: '{ref} planes − exact',
                mode: 'lines', line: {{color: '{COLOR_NEWAVE}', width: 2}}}},
            {{x: d.q, y: minus(c.novomodelo_planes), name: 'Novomodelo planes − exact',
                mode: 'lines', line: {{color: '{COLOR_NOVOMODELO}', width: 2}}}}];
        Plotly.react('op-gap', gaps, {{
            title: 'Planes − exact at the {ref} mean storage and spill (MW)',
            xaxis: {{title: 'Turbined (m³/s)'}}, yaxis: {{title: 'ΔGH (MW)'}},
            hovermode: 'x unified', template: 'plotly_white', height: 340,
            legend: legend, margin: margin}},
            {{responsive: true}});
        document.getElementById('op-table').innerHTML = opTable(f, d.split);
    }}
    document.addEventListener('DOMContentLoaded', function() {{
        var psel = document.getElementById('op-plant');
        if (psel && psel.options.length > 0) {{ opPopulateStages(); opUpdate(); }}
    }});
    """
    return f"""
    <div class="plant-selector">
        <label for="op-plant">Plant:</label>
        <select id="op-plant"
                onchange="opPopulateStages(); opUpdate()">{options}</select>
        <label for="op-stage" style="margin-left:12px">Stage:</label>
        <select id="op-stage" onchange="opUpdate()"></select>
    </div>
    <div class="chart-grid">
        <div class="chart-card">
            <div id="op-slice" style="width:100%;height:460px;"></div>
            <div id="op-gap" style="width:100%;height:340px;"></div>
        </div>
        <div class="chart-card" id="op-table"></div>
    </div>
    <script>{js}</script>
    """

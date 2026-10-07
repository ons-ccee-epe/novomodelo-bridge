"""Production-function (FPHA) comparison charts.

Fitted-surface fidelity table and the interactive per-plant detail (heatmaps
+ spillage slice) rendered from precomputed surface/spill frames.
"""

from __future__ import annotations

from typing import cast

import polars as pl

from cobre_bridge.comparators import analyze
from cobre_bridge.comparators.html_report import (
    COLOR_COBRE,
    COLOR_NEWAVE,
)
from cobre_bridge.ui.html.document import escape_text, json_for_script
from cobre_bridge.ui.html.plotly import MARGIN_DEFAULTS as _MARGIN


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
        f"<caption>Fitted production surface — Cobre vs {reference_label} "
        '<span class="cb-caption-note">— NMAE / bias as % of each plant\'s '
        f"max generation; GHmax ratio = Cobre / {reference_label} at the max "
        "V/Q corner</span></caption>"
    )
    return (
        '<table class="cost-breakdown-table fpha-metrics-table">'
        + caption
        + head
        + "<tbody>"
        + "".join(rows_html)
        + "</tbody></table>"
    )


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

    plants: dict[int, dict[str, object]] = {}
    for (cid, stage, source), sub in surface.partition_by(
        "cobre_id", "stage", "source", as_dict=True
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
        plant["n_v"] = max(int(plant["n_v"]), n_v)  # type: ignore[arg-type]
        by_stage = cast("dict[str, dict]", plant["by_stage"])
        entry = by_stage.setdefault(str(int(stage)), {})
        entry["v"] = v_axis
        entry["q"] = q_axis
        entry["znw" if source == "newave" else "zcb"] = z

    for (cid, stage, source), sub in spill.partition_by(
        "cobre_id", "stage", "source", as_dict=True
    ).items():
        plant = plants.get(int(cid))
        if plant is None:
            continue
        by_stage = cast("dict[str, dict]", plant["by_stage"])
        entry = by_stage.get(str(int(stage)))
        if entry is None:
            continue
        entry["ss"] = _round(sub["s_m3s"].to_list(), 1)
        entry["spnw" if source == "newave" else "spcb"] = _round(
            sub["gh_mw"].to_list(), 2
        )

    out: dict[str, dict[str, object]] = {}
    for cid, plant in plants.items():
        by_stage = cast("dict[str, dict]", plant["by_stage"])
        out[str(cid)] = {
            "name": plant["name"],
            "n_v": plant["n_v"],
            "stages": sorted(int(s) for s in by_stage),
            "byStage": by_stage,
        }
    return out


def fpha_detail_chart(
    surface: pl.DataFrame, spill: pl.DataFrame, reference_label: str = "NEWAVE"
) -> str:
    """Interactive per-plant FPHA surface comparison (heatmaps + spillage slice).

    A plant ``<select>`` (every plant fitted on both sides) drives a stage
    ``<select>``. For a reservoir plant the selected (plant, stage) renders one
    full-width rotatable 3D view of the production surface ``GH(V, Q)`` (sampled
    at the fitting-grid nodes at ``S = 0``) with NEWAVE / Cobre / Both / Difference
    toggle buttons — the two surfaces nearly coincide at ``S = 0``, so toggling
    isolates each and the difference rather than reading a muddy overlay.
    Run-of-river plants (single volume) render an overlaid ``GH`` vs
    turbined-flow curve instead. A further panel shows ``GH`` vs spilled flow at
    the max V/Q corner, exposing the spillage-coefficient behaviour the ``(V, Q)``
    grid holds fixed. Consumes the
    :func:`cobre_bridge.comparators.analyze.build_fpha_comparison` surface/spill
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
    var fphaCb = '{COLOR_COBRE}';
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
            // One full-width 3D view; NEWAVE/Cobre/Both/Difference toggle buttons
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
                {{z: d.zcb, x: d.q, y: d.v, type: 'surface', name: 'Cobre',
                    visible: true, showscale: false, opacity: 0.9,
                    colorscale: [[0, fphaCb], [1, fphaCb]],
                    hovertemplate: 'Cobre<br>Q=%{{x}}<br>V=%{{y}}' +
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
                title: {{text: 'GH(V,Q): {reference_label} (color) + Cobre (orange)'}},
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
                    fphaBtn('Cobre', [false, true, false], 'GH (MW)',
                        'Cobre GH(V,Q)'),
                    fphaBtn('Both', [true, true, false], 'GH (MW)',
                        'GH(V,Q): {reference_label} (color) + Cobre (orange)'),
                    fphaBtn('Difference', [false, false, true], 'Δ MW',
                        'Cobre − {reference_label} (MW)')]}}]
            }}, {{responsive: true}});
        }} else {{
            Plotly.react('fpha-line', [
                {{x: d.q, y: d.znw[0], name: '{reference_label}', type: 'scatter',
                    mode: 'lines', line: {{color: fphaNw, width: 2}}}},
                {{x: d.q, y: d.zcb[0], name: 'Cobre', type: 'scatter',
                    mode: 'lines', line: {{color: fphaCb, width: 2}}}}],
                Object.assign({{title: 'GH vs turbined flow',
                    xaxis: {{title: 'Turbined (m³/s)'}},
                    yaxis: {{title: 'GH (MW)'}}, hovermode: 'x unified'}}, lay),
                {{responsive: true}});
        }}
        Plotly.react('fpha-spill', [
            {{x: d.ss, y: d.spnw, name: '{reference_label}', type: 'scatter',
                mode: 'lines', line: {{color: fphaNw, width: 2}}}},
            {{x: d.ss, y: d.spcb, name: 'Cobre', type: 'scatter',
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

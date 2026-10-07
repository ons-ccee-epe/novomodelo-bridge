"""Cross-subject chart render helpers: layer 0 of the ``charts`` package.

Imports only external modules (``analyze``, ``ui.html.document``, ``html_report``
colors, ``ui.html.plotly``) — never a sibling subject submodule, so every
subject can depend on this module without risking a cycle.
"""

from __future__ import annotations

import polars as pl

from cobre_bridge.comparators import analyze
from cobre_bridge.comparators.html_report import (
    COLOR_COBRE,
    COLOR_NEWAVE,
)
from cobre_bridge.comparators.model import ResultComparison
from cobre_bridge.ui.html.document import escape_text, json_for_script
from cobre_bridge.ui.html.plotly import LEGEND_DEFAULTS as _LEGEND
from cobre_bridge.ui.html.plotly import MARGIN_DEFAULTS as _MARGIN
from cobre_bridge.ui.theme import BAND_FILL, BAND_LINE

_BAND_FILL = BAND_FILL
_BAND_LINE = BAND_LINE

#: The four real Brazilian submarket buses, for a clean 2x2 facet grid. Shared
#: by ``compare newave`` and ``compare decomp``, whose decks name the same
#: submarkets differently: NEWAVE uses the full names (``SUDESTE``/``SUL``/
#: ``NORDESTE``/``NORTE``), DECOMP the short codes (``SE``/``NE``/``S``/``N``).
#: Both naming conventions are listed — a report only ever carries one — so
#: filtering ``[b for b in _REAL_SUBMARKET_ORDER if b in buses]`` keeps NEWAVE's
#: historical order for a NEWAVE report and yields SE, NE, S, N for a DECOMP one.
#: Any bus NOT listed here — the fictitious transhipment nodes (``FC``/``IV``,
#: NEWAVE ``NOFICT*``) and any unnamed phantom — is excluded, never faceted.
_REAL_SUBMARKET_ORDER: list[str] = [
    # NEWAVE full names, in NEWAVE's historical facet order (unchanged).
    "SUDESTE",
    "SUL",
    "NORDESTE",
    "NORTE",
    # DECOMP short codes, ordered SE, NE, S, N (Sudeste, Nordeste, Sul, Norte).
    "SE",
    "NE",
    "S",
    "N",
]


def _aggregate_percentile_traces(
    pct_df: pl.DataFrame | None,
    variable: str,
    stages: list[int],
    entity_ids: set[int] | None = None,
) -> list[dict]:
    """Build p10-p90 band + p10/p90 line traces from aggregate percentiles.

    Sums percentiles across matched entities per stage (aggregate view). When
    *entity_ids* is provided, only those entities are included — this keeps the band
    consistent with the mean line which only covers entities matched between the source
    model and Cobre.
    """
    # Guard mirrors the legacy early-return: no band when the percentile frame
    # is missing/empty or lacks the variable's p10/p90 columns. ``stages`` being
    # empty is NOT such a case — it still yields (empty) band traces, exactly as
    # before, so the check is on the band source, not the returned lists.
    if pct_df is None or pct_df.is_empty():
        return []
    if (
        f"{variable}_p10" not in pct_df.columns
        or f"{variable}_p90" not in pct_df.columns
    ):
        return []

    p10, p90 = analyze.aggregate_percentile_band(pct_df, variable, stages, entity_ids)

    return [
        {
            "x": stages + stages[::-1],
            "y": p90 + p10[::-1],
            "fill": "toself",
            "fillcolor": _BAND_FILL,
            "line": {"color": _BAND_LINE},
            "name": "Cobre P10–P90",
            "type": "scatter",
            "legendgroup": "band",
            "showlegend": True,
            # Wrap-around polygon — skip hover so the cursor's x-position
            # picks up the meaningful p10/p90 line traces instead of the
            # band's literal name.
            "hoverinfo": "skip",
        },
        {
            "x": stages,
            "y": p10,
            "name": "Cobre P10",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_COBRE, "width": 1, "dash": "dot"},
            "legendgroup": "band",
            "showlegend": False,
        },
        {
            "x": stages,
            "y": p90,
            "name": "Cobre P90",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_COBRE, "width": 1, "dash": "dot"},
            "legendgroup": "band",
            "showlegend": False,
        },
    ]


def _enrich_with_percentiles(
    js_plants: dict[str, dict],
    variables: list[tuple[str, str]],
    pct_df: pl.DataFrame | None,
    cobre_id_key: str = "cobre_id",
) -> None:
    """Add p10/p90 arrays to each plant entry from percentile data.

    The in-place ``js_plants`` mutation lives here; the per-plant numeric
    extraction (the rounded p10/p90 arrays, each aligned to its variable's own
    stage axis) is delegated to :func:`analyze.plant_percentile_arrays`, which
    filters ``pct_df`` ONCE per plant and reuses that subframe across every
    variable.
    """
    if pct_df is None or pct_df.is_empty():
        return

    for _pid, entry in js_plants.items():
        cid = entry.get(cobre_id_key)
        if cid is None:
            continue
        var_stages = [
            (var_key, var_label, entry.get(f"{var_key}_stages", []))
            for var_key, var_label in variables
        ]
        entry.update(analyze.plant_percentile_arrays(pct_df, var_stages, cid))


def _plant_max_reldiff_table(
    results: list[ResultComparison],
    entity_type: str,
    variables: list[tuple[str, str]],
    reference_label: str = "NEWAVE",
) -> str:
    """Per-plant max relative-difference summary table.

    Rows: plants (sorted by overall worst max-rel-diff across all
    variables, worst first). Columns: one per variable in *variables*
    (skipping variables for which no source-model row exists).

    Cell value = ``max_stages |cobre − newave| / |newave| × 100`` (the
    source-model-relative, per the report convention). Stages with ``|newave| ≈ 0`` are
    excluded — for any plant/variable that has no eligible stage the cell shows "—".

    Colour cues: ≤ 1 % green, ≤ 10 % amber, > 10 % red.
    """
    max_rd, plant_keys, medians = analyze.plant_max_reldiff_ranking(
        results, entity_type, variables
    )
    if not plant_keys:
        return ""

    def _cell(rd: float | None) -> str:
        if rd is None:
            return '<td class="cb-num">—</td>'
        pct = rd * 100.0
        if pct <= 1.0:
            cls = "cb-num cb-diff-neg"  # reuse green styling
        elif pct <= 10.0:
            cls = "cb-num"
        else:
            cls = "cb-num cb-diff-pos"  # reuse red styling
        return f'<td class="{cls}">{pct:.2f}%</td>'

    header_cells = '<th class="cb-cat">Plant</th>' + "".join(
        f'<th class="cb-num">{label}</th>' for _, label in variables
    )
    body_rows: list[str] = []
    for name, code in plant_keys:
        cells = [f'<td class="cb-cat">{escape_text(name)}</td>']
        for var_key, _ in variables:
            rd = max_rd.get((name, code, var_key))
            cells.append(_cell(rd))
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    # Footer: median max-rel-diff per variable across plants. The table
    # CSS bolds the last tbody row to call out a totals/summary line —
    # without this row the styling would land on whichever plant
    # happened to sort last, which is misleading.
    summary_cells = ['<td class="cb-cat">Median</td>']
    for var_key, _ in variables:
        summary_cells.append(_cell(medians[var_key]))
    body_rows.append("<tr>" + "".join(summary_cells) + "</tr>")

    caption_label = "Hydro" if entity_type == "hydro" else "Thermal"
    return (
        '<table class="cost-breakdown-table">'
        f"<caption>{caption_label} per-plant max relative difference "
        f"(|Cobre − {reference_label}| / |{reference_label}|, over stages)"
        "</caption>"
        f"<thead><tr>{header_cells}</tr></thead>"
        "<tbody>" + "".join(body_rows) + "</tbody>"
        "</table>"
    )


def _build_interactive_detail_html(
    js_plants: dict[str, dict],
    variables: list[tuple[str, str]],
    prefix: str,
    label: str,
    reference_label: str = "NEWAVE",
) -> str:
    """Build the HTML/JS for interactive per-plant detail charts."""
    data_json = json_for_script(js_plants)

    # Build chart divs.
    chart_divs: list[str] = []
    for var_key, var_label in variables:
        div_id = f"{prefix}-chart-{var_key.replace('_', '-')}"
        chart_divs.append(
            f'<div class="chart-card">'
            f'<div id="{div_id}" style="width:100%;height:350px;"></div>'
            f"</div>"
        )

    n_vars = len(variables)
    grid_class = "chart-grid" if n_vars > 1 else "chart-grid-single"
    charts_html = f'<div class="{grid_class}">{"".join(chart_divs)}</div>'

    # Build option list sorted by name.
    options: list[str] = []
    for pid, entry in sorted(js_plants.items(), key=lambda x: x[1]["name"]):
        name = entry["name"]
        code = entry["code"]
        options.append(f'<option value="{pid}">{name} ({code})</option>')

    # JS to update charts on selection (with optional p10/p90 bands).
    update_calls: list[str] = []
    for var_key, var_label in variables:
        div_id = f"{prefix}-chart-{var_key.replace('_', '-')}"
        update_calls.append(f"""
        (function() {{
            var s = d['{var_key}_stages'] || [];
            var nw = d['{var_key}_nw'] || [];
            var cb = d['{var_key}_cb'] || [];
            var p10 = d['{var_key}_p10'] || null;
            var p90 = d['{var_key}_p90'] || null;
            var traces = [];
            if (p10 && p90 && p10.length > 0) {{
                traces.push({{
                    x: s.concat(s.slice().reverse()),
                    y: p90.concat(p10.slice().reverse()),
                    fill: 'toself',
                    fillcolor: '{_BAND_FILL}',
                    line: {{color: '{_BAND_LINE}'}},
                    name: 'Cobre P10\u2013P90',
                    type: 'scatter',
                    legendgroup: 'band',
                    showlegend: true,
                    // Skip hover on the wrap-around polygon \u2014 its y values
                    // are the reversed-p10 closing edge and have no
                    // meaning at the cursor x.  Real values come from
                    // the visible p10 / p90 line traces below.
                    hoverinfo: 'skip'
                }});
                traces.push({{
                    x: s, y: p10,
                    name: 'Cobre P10', type: 'scatter', mode: 'lines',
                    line: {{color: '{COLOR_COBRE}', width: 1, dash: 'dot'}},
                    legendgroup: 'band', showlegend: false
                }});
                traces.push({{
                    x: s, y: p90,
                    name: 'Cobre P90', type: 'scatter', mode: 'lines',
                    line: {{color: '{COLOR_COBRE}', width: 1, dash: 'dot'}},
                    legendgroup: 'band', showlegend: false
                }});
            }}
            if (nw && nw.length > 0) {{
                traces.push({{x: s, y: nw, name: '{reference_label}', type: 'scatter',
                    mode: 'lines', line: {{color: '{COLOR_NEWAVE}', width: 2}}}});
            }}
            traces.push({{x: s, y: cb, name: 'Cobre Mean', type: 'scatter',
                mode: 'lines', line: {{color: '{COLOR_COBRE}', width: 2}}}});
            var maxCb = d['{var_key}_max_cb'];
            if (maxCb && maxCb.length > 0) {{
                traces.push({{x: s, y: maxCb, name: 'Cobre LP gen_max',
                    type: 'scatter', mode: 'lines',
                    line: {{color: '{COLOR_COBRE}', width: 1.5, dash: 'dash'}}}});
            }}
            // Bound overlays (static from hydros.json, with per-stage
            // overrides from hydro_bounds.parquet shadowing where present).
            // Rendered in a muted grey so they sit behind the comparison
            // traces without competing for attention.  ``connectgaps:false``
            // ensures stages where the bound is structurally undefined
            // produce a gap in the dashed line.
            var bMin = d['{var_key}_bound_min'];
            if (bMin && bMin.length > 0) {{
                traces.push({{x: s, y: bMin, name: 'Lower bound',
                    type: 'scatter', mode: 'lines', connectgaps: false,
                    line: {{color: '#888', width: 1.2, dash: 'dash'}}}});
            }}
            var bMax = d['{var_key}_bound_max'];
            if (bMax && bMax.length > 0) {{
                traces.push({{x: s, y: bMax, name: 'Upper bound',
                    type: 'scatter', mode: 'lines', connectgaps: false,
                    line: {{color: '#888', width: 1.2, dash: 'dash'}}}});
            }}
            Plotly.react('{div_id}', traces, {{
                title: d.name + ' \u2014 {var_label}',
                xaxis: {{title: 'Stage'}},
                yaxis: {{title: '{var_label}'}},
                legend: {json_for_script(_LEGEND)},
                margin: {json_for_script(_MARGIN)},
                template: 'plotly_white',
                hovermode: 'x unified',
                height: 350
            }}, {{responsive: true}});
        }})();""")

    js = f"""
    var {prefix}Data = {data_json};
    function update{prefix.title()}Charts() {{
        var sel = document.getElementById('{prefix}-select');
        var pid = sel.value;
        var d = {prefix}Data[pid];
        if (!d) return;
        document.getElementById('{prefix}-info').innerHTML =
            '<span>Code: ' + d.code + '</span>';
        {"".join(update_calls)}
    }}
    document.addEventListener('DOMContentLoaded', function() {{
        var sel = document.getElementById('{prefix}-select');
        if (sel && sel.options.length > 0) {{
            update{prefix.title()}Charts();
        }}
    }});
    """

    return f"""
    <div class="plant-selector">
        <label for="{prefix}-select">{label}:</label>
        <select id="{prefix}-select"
                onchange="update{prefix.title()}Charts()">
            {"".join(options)}
        </select>
        <div class="plant-info" id="{prefix}-info"></div>
    </div>
    {charts_html}
    <script>{js}</script>
    """

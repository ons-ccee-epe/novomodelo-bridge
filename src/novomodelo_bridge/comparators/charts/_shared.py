"""Cross-subject chart render helpers: layer 0 of the ``charts`` package.

Imports only external modules (``analyze``, ``core.agreement``,
``ui.html.document``, ``html_report`` colors, ``ui.html.plotly``) — never a
sibling subject submodule, so every
subject can depend on this module without risking a cycle.
"""

from __future__ import annotations

import polars as pl

from novomodelo_bridge.comparators import analyze
from novomodelo_bridge.comparators.html_report import (
    COLOR_NEWAVE,
    COLOR_NOVOMODELO,
)
from novomodelo_bridge.comparators.model import ResultComparison
from novomodelo_bridge.core import agreement
from novomodelo_bridge.ui.html.document import (
    escape_attr,
    escape_text,
    json_for_script,
)
from novomodelo_bridge.ui.html.plotly import LEGEND_DEFAULTS as _LEGEND
from novomodelo_bridge.ui.html.plotly import MARGIN_DEFAULTS as _MARGIN
from novomodelo_bridge.ui.theme import BAND_FILL, BAND_LINE

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
    model and Novomodelo.
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
            "name": "Novomodelo P10–P90",
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
            "name": "Novomodelo P10",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_NOVOMODELO, "width": 1, "dash": "dot"},
            "legendgroup": "band",
            "showlegend": False,
        },
        {
            "x": stages,
            "y": p90,
            "name": "Novomodelo P90",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_NOVOMODELO, "width": 1, "dash": "dot"},
            "legendgroup": "band",
            "showlegend": False,
        },
    ]


def _enrich_with_percentiles(
    js_plants: dict[str, dict],
    variables: list[tuple[str, str]],
    pct_df: pl.DataFrame | None,
    novomodelo_id_key: str = "novomodelo_id",
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
        cid = entry.get(novomodelo_id_key)
        if cid is None:
            continue
        var_stages = [
            (var_key, var_label, entry.get(f"{var_key}_stages", []))
            for var_key, var_label in variables
        ]
        entry.update(analyze.plant_percentile_arrays(pct_df, var_stages, cid))


#: Sorts a gap table by the clicked header and opens a plant's charts when its
#: row is clicked, by selecting it in the tab's ``{prefix}-select``.
_GAP_TABLE_JS = """
function gapTableSort(th) {
    var table = th.closest('table');
    var body = table.tBodies[table.tBodies.length - 1];
    var column = th.cellIndex;
    var descending = column > 0 && th.dataset.sort !== 'desc';
    if (column === 0) { descending = th.dataset.sort === 'asc'; }
    Array.prototype.forEach.call(th.parentNode.cells, function (cell) {
        delete cell.dataset.sort;
    });
    th.dataset.sort = descending ? 'desc' : 'asc';
    var rows = Array.prototype.slice.call(body.rows);
    rows.sort(function (a, b) {
        var x = a.cells[column], y = b.cells[column];
        var order = column === 0
            ? x.textContent.localeCompare(y.textContent)
            : parseFloat(x.dataset.gap) - parseFloat(y.dataset.gap);
        return descending ? -order : order;
    });
    rows.forEach(function (row) { body.appendChild(row); });
}
function gapTableOpen(row) {
    var select = document.getElementById(row.closest('table').dataset.prefix + '-select');
    if (!select) { return; }
    select.value = row.dataset.plant;
    select.dispatchEvent(new Event('change'));
    select.scrollIntoView({behavior: 'smooth', block: 'start'});
}
"""


def _gap_cell(value: float | None, present: bool, tolerance: float) -> str:
    if not present:
        return '<td class="cb-num" data-gap="-1">—</td>'
    band = agreement.band(value, tolerance)
    sort_key = "Infinity" if value is None else f"{value:.6g}"
    return (
        f'<td class="cb-num gap-{band.value}" data-gap="{sort_key}" '
        f'title="{band.value}">{agreement.format_gap(value)}</td>'
    )


def _plant_gap_table(
    results: list[ResultComparison],
    entity_type: str,
    variables: list[tuple[str, str]],
    prefix: str,
    tolerance: float,
    reference_label: str = "NEWAVE",
    note: str | None = None,
) -> str:
    """Per-plant gap table: one row per plant, one column per variable.

    Each cell is the plant's gap over stages (see
    :class:`~novomodelo_bridge.comparators.analyze.PlantGaps`), tinted by
    its band; an "All plants" row pools every plant. Plants come largest gap
    first, a header click sorts by that column, and a row click opens the
    plant in the ``{prefix}-select`` charts below. *note* is appended to the
    caption.
    """
    gaps = analyze.plant_gaps(results, entity_type, variables)
    if not gaps.plants:
        return ""
    shown = [(key, label) for key, label in variables if key in gaps.totals]

    header = '<th class="cb-cat" onclick="gapTableSort(this)">Plant</th>' + "".join(
        f'<th class="cb-num" onclick="gapTableSort(this)">{escape_text(label)}</th>'
        for _, label in shown
    )
    total = '<tr class="cb-total"><td class="cb-cat">All plants</td>' + "".join(
        _gap_cell(gaps.totals[key], True, tolerance) for key, _ in shown
    )
    rows: list[str] = []
    for name, code in gaps.plants:
        cells = "".join(
            _gap_cell(
                gaps.by_plant.get((name, code, key)),
                (name, code, key) in gaps.by_plant,
                tolerance,
            )
            for key, _ in shown
        )
        rows.append(
            f'<tr class="gap-row" data-plant="{escape_attr(f"{code}_{name}")}" '
            'tabindex="0" onclick="gapTableOpen(this)" '
            "onkeydown=\"if (event.key === 'Enter') gapTableOpen(this)\">"
            f'<td class="cb-cat">{escape_text(name)}</td>{cells}</tr>'
        )

    group = "Hydro" if entity_type == "hydro" else "Thermal"
    bands = (
        f"close up to {agreement.format_gap(tolerance)}, far beyond "
        f"{agreement.format_gap(agreement.FAR_FACTOR * tolerance)}; click a plant "
        "to open its charts"
    )
    caption_note = bands if note is None else f"{note}. {bands[0].upper()}{bands[1:]}"
    return (
        '<div class="gap-scroll">'
        f'<table class="cost-breakdown-table gap-table" data-prefix="{prefix}">'
        f"<caption>{group} plants: gap over stages, "
        f"Σ|Novomodelo − {reference_label}| / Σ|{reference_label}|, "
        f"Σ|{reference_label}| no smaller than the mean plant's"
        f' <span class="cb-caption-note">— {escape_text(caption_note)}</span>'
        "</caption>"
        f"<thead><tr>{header}</tr></thead>"
        f'<tbody class="gap-total">{total}</tr></tbody>'
        "<tbody>" + "".join(rows) + "</tbody>"
        f"</table></div><script>{_GAP_TABLE_JS}</script>"
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
                    name: 'Novomodelo P10\u2013P90',
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
                    name: 'Novomodelo P10', type: 'scatter', mode: 'lines',
                    line: {{color: '{COLOR_NOVOMODELO}', width: 1, dash: 'dot'}},
                    legendgroup: 'band', showlegend: false
                }});
                traces.push({{
                    x: s, y: p90,
                    name: 'Novomodelo P90', type: 'scatter', mode: 'lines',
                    line: {{color: '{COLOR_NOVOMODELO}', width: 1, dash: 'dot'}},
                    legendgroup: 'band', showlegend: false
                }});
            }}
            if (nw && nw.length > 0) {{
                traces.push({{x: s, y: nw, name: '{reference_label}', type: 'scatter',
                    mode: 'lines', line: {{color: '{COLOR_NEWAVE}', width: 2}}}});
            }}
            traces.push({{x: s, y: cb, name: 'Novomodelo Mean', type: 'scatter',
                mode: 'lines', line: {{color: '{COLOR_NOVOMODELO}', width: 2}}}});
            var maxCb = d['{var_key}_max_cb'];
            if (maxCb && maxCb.length > 0) {{
                traces.push({{x: s, y: maxCb, name: 'Novomodelo LP gen_max',
                    type: 'scatter', mode: 'lines',
                    line: {{color: '{COLOR_NOVOMODELO}', width: 1.5, dash: 'dash'}}}});
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
                xaxis: {{title: 'Stage', automargin: true}},
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

"""HTML builder functions for UI fragments and complete tabbed documents."""

from __future__ import annotations

import html as _html
import json

from plotly.offline import get_plotlyjs_version

_PLOTLYJS_VERSION = get_plotlyjs_version()


def escape_text(value: object) -> str:
    """Escape *value* for use as HTML **text** content (e.g. inside ``<td>``).

    Neutralises ``&``, ``<`` and ``>`` so the source-model-derived names (plant, line,
    case names) cannot inject markup. Use :func:`escape_attr` for values placed inside
    quoted attributes.
    """
    return _html.escape(str(value), quote=False)


def escape_attr(value: object) -> str:
    """Escape *value* for use inside a quoted HTML **attribute**.

    Like :func:`escape_text` but also escapes ``"`` and ``'`` so the value
    cannot terminate the attribute (safe for both single- and double-quoted).
    """
    return _html.escape(str(value), quote=True)


def json_for_script(obj: object) -> str:
    """Serialise *obj* to JSON safe to embed inside an HTML ``<script>`` block.

    ``json.dumps`` alone does not neutralise ``</script>`` or the JavaScript line
    separators U+2028/U+2029, so a source-model name containing ``</script>`` could
    break out of the script context. This escapes ``<``, ``>``, ``&`` and the line
    separators as ``\\uXXXX`` sequences — valid JSON (round-trips via ``json.loads``)
    but inert in HTML.  Output uses compact separators.
    """
    return (
        json.dumps(obj, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def wrap_chart(html: str) -> str:
    """Wrap an HTML fragment in a chart-card container div with an expand button."""
    return (
        '<div class="chart-card">'
        '<button type="button" class="expand-btn" title="Expand">'
        '<svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.5">'
        '<polyline points="1,5 1,1 5,1"/><polyline points="11,1 15,1 15,5"/>'
        '<polyline points="15,11 15,15 11,15"/><polyline points="5,15 1,15 1,11"/>'
        "</svg></button>"
        f"{html}</div>"
    )


def section_title(text: str) -> str:
    """Create a section title element using the .section-title class."""
    return f'<div class="section-title">{text}</div>'


def collapsible_section(
    title: str,
    content: str,
    section_id: str = "",
    default_collapsed: bool = False,
) -> str:
    """Wrap a section title and content in a collapsible container.

    Args:
        title: Section heading text.
        content: HTML string for the section body.
        section_id: Optional ``id`` attribute for the outer section div.
        default_collapsed: When ``True``, starts collapsed with right-pointing
            chevron; when ``False``, starts expanded with down-pointing chevron.

    Returns:
        A ``<div class="collapsible-section">`` fragment with clickable title
        and content wrapped in ``<div class="collapsible-content">``.
    """
    chevron = (
        '<svg class="chevron" width="10" height="10" viewBox="0 0 10 10">'
        '<polyline points="2,3 5,7 8,3" fill="none" stroke="currentColor" stroke-width="1.5"/>'
        "</svg>"
    )

    section_class = "collapsible-section"
    if default_collapsed:
        title_class = "section-title collapsed-title"
        content_class = "collapsible-content collapsed"
    else:
        title_class = "section-title"
        content_class = "collapsible-content"

    id_attr = f' id="{section_id}"' if section_id else ""
    return (
        f'<div class="{section_class}"{id_attr}>'
        f'<div class="{title_class}" data-collapsible="true">'
        f"{title}"
        f"{chevron}"
        "</div>"
        f'<div class="{content_class}">{content}</div>'
        "</div>"
    )


def sparkline_svg(
    values: list[float],
    color: str,
    width: int = 60,
    height: int = 20,
) -> str:
    """Render an inline SVG sparkline from a list of float values."""
    n = len(values)
    min_v = min(values)
    max_v = max(values)
    value_range = max_v - min_v
    scale = height / value_range if value_range > 0 else 0.0
    pts_list: list[str] = []
    for i, v in enumerate(values):
        x = i / (n - 1) * width
        # SVG y-axis is top-down, so invert
        y = height - (v - min_v) * scale
        pts_list.append(f"{x:.1f},{y:.1f}")
    pts = " ".join(pts_list)
    return (
        f'<svg width="{width}" height="{height}" class="metric-sparkline">'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>'
        f"</svg>"
    )


def metric_card(
    value: str,
    label: str,
    *,
    color: str | None = None,
    delta: str | None = None,
    delta_direction: str | None = None,
    sparkline_values: list[float] | None = None,
) -> str:
    """Create a metric card with value, label, optional delta, and sparkline."""
    style = f' style="border-top: 4px solid {color};"' if color is not None else ""

    delta_html = ""
    if delta is not None:
        if delta_direction == "up":
            delta_class = "metric-delta metric-delta-up"
            arrow = "&#9650;"
        elif delta_direction == "down":
            delta_class = "metric-delta metric-delta-down"
            arrow = "&#9660;"
        else:
            delta_class = "metric-delta"
            arrow = ""
        arrow_span = f"<span>{arrow}</span>" if arrow else ""
        delta_html = f'<div class="{delta_class}">{arrow_span}{delta}</div>'

    sparkline_html = ""
    if sparkline_values is not None and len(sparkline_values) >= 2:
        sparkline_color = color if color is not None else "#8B9298"
        sparkline_html = (
            f'<div class="metric-sparkline">'
            f"{sparkline_svg(sparkline_values, sparkline_color)}"
            f"</div>"
        )

    return (
        f'<div class="metric-card"{style}>'
        f'<div class="metric-value">{value}</div>'
        f"{delta_html}"
        f'<div class="metric-label">{label}</div>'
        f"{sparkline_html}"
        "</div>"
    )


def metrics_grid(cards: list[str]) -> str:
    """Wrap a list of metric card HTML fragments in a metrics-grid container."""
    return f'<div class="metrics-grid">{"".join(cards)}</div>'


def chart_grid(charts: list[str], single: bool = False) -> str:
    """Wrap chart HTML fragments in a grid container.

    Args:
        charts: List of HTML fragments, one per chart.
        single: When True, uses the single-column grid class.
    """
    cls = "chart-grid-single" if single else "chart-grid"
    return f'<div class="{cls}">{"".join(charts)}</div>'


def plant_explorer_table(
    table_id: str,
    search_id: str,
    columns: list[tuple[str, str]],
    rows_html: str,
) -> str:
    """Generate a plant explorer table with search and sortable columns.

    Args:
        table_id: ``id`` for the ``<tbody>`` element.
        search_id: ``id`` for the search ``<input>`` element.
        columns: List of ``(header_text, sort_type)`` pairs.
            ``sort_type`` must be ``"string"``, ``"number"``, or ``"none"``.
        rows_html: Pre-rendered ``<tr>`` elements for the ``<tbody>``.
    """
    header_cells: list[str] = []
    for col_index, (header_text, sort_type) in enumerate(columns):
        if sort_type != "none":
            header_cells.append(
                f'<th class="sortable" data-sort-asc="false"'
                f" onclick=\"sortTable('{table_id}', {col_index}, '{sort_type}')\">"
                f'{header_text}<span class="sort-arrow"></span></th>'
            )
        else:
            header_cells.append(f"<th>{header_text}</th>")

    return (
        f'<input type="search" id="{search_id}" class="explorer-search"'
        f' placeholder="Search...">'
        f'<table class="explorer-table">'
        f"<thead><tr>{''.join(header_cells)}</tr></thead>"
        f'<tbody id="{table_id}">{rows_html}</tbody>'
        f"</table>"
    )


def build_html(
    title: str,
    tab_defs: list[tuple[str, str]],
    tab_contents: dict[str, str],
    css: str,
    js: str,
    required_js: str = "",
) -> str:
    """Assemble a complete HTML document with tabbed navigation.

    Args:
        title: Page title for ``<title>`` and ``<header>``.
        tab_defs: Ordered list of ``(tab_id, tab_label)`` pairs.
        tab_contents: Mapping from tab_id to HTML content; missing tabs
            render as ``"<p>No data</p>"``.
        css: CSS string injected into ``<style>`` in ``<head>``.
        js: JavaScript string injected into ``<script>`` at end of ``<body>``.
        required_js: Shared JS function definitions needed by tab-body
            scripts. Emitted in its own ``<script>`` in ``<head>``,
            immediately after the plotly ``<script>``, so the definitions
            exist before any tab's inline ``DOMContentLoaded`` handler runs.
            Empty (the default) emits nothing.

    Returns:
        A complete ``<!DOCTYPE html>`` document string.
    """
    # ``title`` carries the case-directory name (the source-model-derived) and tab
    # labels are interpolated as text; escape both so a crafted name cannot inject HTML.
    title = escape_text(title)

    nav_buttons: list[str] = []
    for i, (tab_id, tab_label) in enumerate(tab_defs):
        active_cls = ' class="active"' if i == 0 else ""
        nav_buttons.append(
            f"<button{active_cls} onclick=\"showTab('{tab_id}', this)\">"
            f"{escape_text(tab_label)}</button>"
        )

    tab_sections: list[str] = []
    for i, (tab_id, _) in enumerate(tab_defs):
        active_cls = " active" if i == 0 else ""
        content = tab_contents.get(tab_id, "<p>No data</p>")
        tab_sections.append(
            f'<section id="{tab_id}" class="tab-content{active_cls}">\n{content}\n</section>'
        )

    # Empty when required_js is "" so the comparison-report path (no shared JS
    # needed) renders byte-identical <head> markup to before this parameter existed.
    shared_js_block = f"\n    <script>\n{required_js}\n</script>" if required_js else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-{_PLOTLYJS_VERSION}.min.js"></script>{shared_js_block}
    <style>
{css}
    </style>
</head>
<body>
    <header>{title}</header>
    <nav>
        {"".join(nav_buttons)}
    </nav>
    <main>
        {"".join(tab_sections)}
    </main>
    <script>
{js}
    </script>
</body>
</html>"""

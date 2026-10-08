"""Unit tests for the novomodelo_bridge.ui package.

Covers all 5 modules: theme.py, plotly_helpers.py, css.py, js.py, html.py.
Also verifies backward-compatible re-exports from comparators.html_report.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest
from plotly.offline import get_plotlyjs_version

from novomodelo_bridge.ui.html.css import (
    PLANT_EXPLORER_CSS,
    comparison_css,
    dashboard_css,
)
from novomodelo_bridge.ui.html.document import (
    build_html,
    chart_grid,
    collapsible_section,
    metric_card,
    metrics_grid,
    plant_explorer_table,
    section_title,
    wrap_chart,
)
from novomodelo_bridge.ui.html.js import (
    PLANT_EXPLORER_JS,
    PLOTLY_TITLE_SHIM_JS,
    TAB_SWITCH_JS,
)
from novomodelo_bridge.ui.html.plotly import (
    LEGEND_DEFAULTS,
    MARGIN_DEFAULTS,
    fig_to_html,
    plotly_div,
    stage_x_labels,
)
from novomodelo_bridge.ui.theme import (
    BUS_COLORS,
    COLORS,
    COMPARISON_COLORS,
    COPPER_ACCENT,
)


@pytest.fixture
def simple_figure() -> go.Figure:
    """Minimal Plotly figure with one scatter trace."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="test"))
    return fig


@pytest.fixture
def simple_tab_defs() -> list[tuple[str, str]]:
    return [("tab-a", "Tab A"), ("tab-b", "Tab B")]


@pytest.fixture
def simple_tab_contents() -> dict[str, str]:
    return {"tab-a": "<p>Content A</p>", "tab-b": "<p>Content B</p>"}


def test_theme_colors_complete() -> None:
    """COLORS must have exactly 11 keys covering every semantic role."""
    expected_keys = {
        "hydro",
        "thermal",
        "ncs",
        "load",
        "deficit",
        "spillage",
        "curtailment",
        "exchange",
        "lower_bound",
        "upper_bound",
        "future_cost",
    }
    assert set(COLORS.keys()) == expected_keys


def test_theme_colors_are_hex_strings() -> None:
    """Every value in COLORS must be a non-empty hex color string."""
    for key, value in COLORS.items():
        assert isinstance(value, str), f"COLORS[{key!r}] is not a string"
        assert value.startswith("#"), f"COLORS[{key!r}] does not start with '#'"


def test_theme_bus_colors_count() -> None:
    """BUS_COLORS must contain exactly 5 elements."""
    assert len(BUS_COLORS) == 5


def test_theme_bus_colors_are_hex_strings() -> None:
    """Every entry in BUS_COLORS must be a hex color string."""
    for i, color in enumerate(BUS_COLORS):
        assert isinstance(color, str), f"BUS_COLORS[{i}] is not a string"
        assert color.startswith("#"), f"BUS_COLORS[{i}] does not start with '#'"


def test_theme_copper_accent_value() -> None:
    """COPPER_ACCENT must be the documented copper tone."""
    assert COPPER_ACCENT == "#B87333"


def test_theme_comparison_colors_keys() -> None:
    """COMPARISON_COLORS must contain 'novomodelo', 'newave', 'diff', 'match'."""
    assert set(COMPARISON_COLORS.keys()) == {"novomodelo", "newave", "diff", "match"}


def test_legend_defaults_orientation() -> None:
    """LEGEND_DEFAULTS must specify horizontal orientation."""
    assert LEGEND_DEFAULTS["orientation"] == "h"


def test_margin_defaults_keys() -> None:
    """MARGIN_DEFAULTS must provide l, r, t, b keys."""
    assert {"l", "r", "t", "b"} <= set(MARGIN_DEFAULTS.keys())


def test_fig_to_html_output(simple_figure: go.Figure) -> None:
    """fig_to_html must return a fragment with plotly-graph-div, no <html> tag."""
    result = fig_to_html(simple_figure)
    assert "plotly-graph-div" in result
    assert "<html" not in result


def test_fig_to_html_no_plotlyjs_script(simple_figure: go.Figure) -> None:
    """fig_to_html must not embed plotly.js — the caller loads it separately."""
    result = fig_to_html(simple_figure)
    assert "cdn.plot.ly" not in result
    assert "plotly.min.js" not in result


def test_plotly_div_output() -> None:
    """plotly_div must produce a div/script pair calling Plotly.newPlot."""
    traces = [{"type": "scatter", "x": [1, 2], "y": [3, 4]}]
    result = plotly_div(traces, layout={"title": "Test"})
    assert "Plotly.newPlot" in result
    assert "<div" in result
    assert "<script>" in result


def test_plotly_div_contains_chart_id_prefix() -> None:
    """plotly_div div id must match the 'chart-' prefix."""
    traces: list[dict] = []
    result = plotly_div(traces, layout={})
    assert 'id="chart-' in result


def test_stage_x_labels_mapped() -> None:
    """stage_x_labels must return mapped label when id is present."""
    labels = {1: "Jan/2024", 2: "Feb/2024"}
    result = stage_x_labels([1, 2], labels)
    assert result == ["Jan/2024", "Feb/2024"]


def test_stage_x_labels_fallback_to_string() -> None:
    """stage_x_labels must fall back to str(id) for unmapped stage ids."""
    labels: dict[int, str] = {}
    result = stage_x_labels([5, 10], labels)
    assert result == ["5", "10"]


def test_stage_x_labels_mixed() -> None:
    """stage_x_labels must map known ids and fall back for unknown ones."""
    labels = {1: "Jan"}
    result = stage_x_labels([1, 2], labels)
    assert result == ["Jan", "2"]


def test_dashboard_css_selectors() -> None:
    """dashboard_css() must contain all required class selectors."""
    output = dashboard_css()
    required = [".chart-card", ".metric-card", ".section-title", ".data-table"]
    for selector in required:
        assert selector in output, f"Missing selector: {selector!r}"


def test_comparison_css_selectors() -> None:
    """comparison_css() must contain all required class selectors."""
    output = comparison_css()
    required = [".chart-card", ".metric-card", ".section-title", ".plant-selector"]
    for selector in required:
        assert selector in output, f"Missing selector: {selector!r}"


def test_dashboard_css_excludes_plant_selector() -> None:
    """dashboard_css() must not include .plant-selector (comparator-only)."""
    assert ".plant-selector" not in dashboard_css()


def test_comparison_css_excludes_data_table() -> None:
    """comparison_css() must not include .data-table (dashboard-only)."""
    assert ".data-table" not in comparison_css()


def test_dashboard_css_returns_string() -> None:
    """dashboard_css() must return a non-empty string."""
    result = dashboard_css()
    assert isinstance(result, str)
    assert len(result) > 0


def test_tab_switch_js_contains_show_tab_function() -> None:
    """TAB_SWITCH_JS must define a showTab function."""
    assert "function showTab" in TAB_SWITCH_JS


def test_tab_switch_js_contains_resize_dispatch() -> None:
    """TAB_SWITCH_JS must dispatch a resize event."""
    assert "dispatchEvent" in TAB_SWITCH_JS
    assert "resize" in TAB_SWITCH_JS


def test_tab_switch_js_contains_tab_id_param() -> None:
    """showTab signature must accept tabId and btn parameters."""
    assert "tabId" in TAB_SWITCH_JS
    assert "btn" in TAB_SWITCH_JS


def test_tab_switch_js_is_non_empty_string() -> None:
    """TAB_SWITCH_JS must be a non-empty string."""
    assert isinstance(TAB_SWITCH_JS, str)
    assert len(TAB_SWITCH_JS.strip()) > 0


def test_plotly_title_shim_js_wraps_both_render_entry_points() -> None:
    """PLOTLY_TITLE_SHIM_JS must override both Plotly.newPlot and Plotly.react."""
    assert "Plotly.newPlot = function" in PLOTLY_TITLE_SHIM_JS
    assert "Plotly.react = function" in PLOTLY_TITLE_SHIM_JS
    assert "origNewPlot.call" in PLOTLY_TITLE_SHIM_JS
    assert "origReact.call" in PLOTLY_TITLE_SHIM_JS


def test_plotly_title_shim_js_normalizes_data_and_layout() -> None:
    """The shim must normalize both the data (traces, e.g. per-trace colorbar
    titles) and the layout argument, not layout alone — a bare colorbar title
    lives on a trace object, not on layout."""
    assert "normalizePlotlyTitles(data)" in PLOTLY_TITLE_SHIM_JS
    assert "normalizePlotlyTitles(layout)" in PLOTLY_TITLE_SHIM_JS


def test_wrap_chart_produces_chart_card() -> None:
    """wrap_chart must wrap content in a chart-card div (now includes expand button)."""
    result = wrap_chart("<p>Chart</p>")
    assert result.startswith('<div class="chart-card">')
    assert result.endswith("</div>")
    assert "<p>Chart</p>" in result


def test_section_title_produces_section_title_div() -> None:
    """section_title must produce a div with class section-title."""
    result = section_title("Overview")
    assert 'class="section-title"' in result
    assert "Overview" in result


def test_metric_card_contains_value_and_label() -> None:
    """metric_card must contain the metric-value and metric-label divs."""
    result = metric_card("42 MW", "Total Load")
    assert "42 MW" in result
    assert "Total Load" in result
    assert 'class="metric-value"' in result
    assert 'class="metric-label"' in result


def test_metrics_grid_wraps_cards() -> None:
    """metrics_grid must wrap cards in a metrics-grid container."""
    cards = [metric_card("1", "A"), metric_card("2", "B")]
    result = metrics_grid(cards)
    assert 'class="metrics-grid"' in result
    assert "1" in result
    assert "2" in result


def test_chart_grid_default_uses_chart_grid_class() -> None:
    """chart_grid without single=True must use the chart-grid class."""
    result = chart_grid(["<div>A</div>"])
    assert 'class="chart-grid"' in result


def test_chart_grid_single_uses_single_class() -> None:
    """chart_grid with single=True must use the chart-grid-single class."""
    result = chart_grid(["<div>A</div>"], single=True)
    assert 'class="chart-grid-single"' in result


def test_build_html_structure(
    simple_tab_defs: list[tuple[str, str]],
    simple_tab_contents: dict[str, str],
) -> None:
    """build_html must produce a complete HTML document with required elements."""
    result = build_html(
        title="Test Report",
        tab_defs=simple_tab_defs,
        tab_contents=simple_tab_contents,
        css=dashboard_css(),
        js=TAB_SWITCH_JS,
    )
    assert "<!DOCTYPE html>" in result
    assert f"plotly-{get_plotlyjs_version()}.min.js" in result
    assert "<header>" in result
    assert "<nav>" in result
    assert "<main>" in result
    assert "<script>" in result


def test_build_html_first_tab_active(
    simple_tab_defs: list[tuple[str, str]],
    simple_tab_contents: dict[str, str],
) -> None:
    """build_html must mark only the first tab as active."""
    result = build_html(
        title="Test",
        tab_defs=simple_tab_defs,
        tab_contents=simple_tab_contents,
        css="",
        js="",
    )
    # First nav button should be class="active"
    assert 'class="active"' in result
    # First section should contain 'active' in its class
    assert "tab-content active" in result


def test_build_html_missing_tab_content_renders_no_data(
    simple_tab_defs: list[tuple[str, str]],
) -> None:
    """build_html must render '<p>No data</p>' for tabs missing from tab_contents."""
    result = build_html(
        title="Test",
        tab_defs=simple_tab_defs,
        tab_contents={},  # Empty — both tabs missing
        css="",
        js="",
    )
    assert "<p>No data</p>" in result


def test_build_html_title_in_header(
    simple_tab_defs: list[tuple[str, str]],
    simple_tab_contents: dict[str, str],
) -> None:
    """build_html must include the title in both <title> and <header>."""
    result = build_html(
        title="My Report",
        tab_defs=simple_tab_defs,
        tab_contents=simple_tab_contents,
        css="",
        js="",
    )
    assert "<title>My Report</title>" in result
    assert "<header>My Report</header>" in result


def test_dashboard_css_contains_transitions() -> None:
    """dashboard_css() must include .chart-card:hover with translateY lift."""
    output = dashboard_css()
    assert ".chart-card:hover" in output
    assert "translateY" in output


def test_dashboard_css_contains_responsive_breakpoints() -> None:
    """dashboard_css() must include mobile and tablet media query breakpoints."""
    output = dashboard_css()
    assert "@media (max-width: 767px)" in output
    assert "@media (min-width: 768px)" in output


def test_dashboard_css_contains_tab_fade() -> None:
    """dashboard_css() must include .tab-content-fade for opacity transitions."""
    output = dashboard_css()
    assert ".tab-content-fade" in output


def test_dashboard_css_contains_collapsible() -> None:
    """dashboard_css() must include .collapsible-content for collapsible sections."""
    output = dashboard_css()
    assert ".collapsible-content" in output


def test_comparison_css_includes_enhanced_styles() -> None:
    """comparison_css() must include enhanced UX styles."""
    output = comparison_css()
    assert "translateY" in output
    assert "@media (max-width" in output
    assert ".tab-underline" in output
    # Must NOT include dashboard-only styles
    assert ".collapsible-content" not in output
    assert ".explorer-container" not in output


def test_comparators_backward_compat() -> None:
    """All symbols re-exported from comparators.html_report must be importable."""
    from novomodelo_bridge.comparators.html_report import (  # noqa: PLC0415
        COLOR_DIFF,
        COLOR_MATCH,
        COLOR_NEWAVE,
        COLOR_NOVOMODELO,
        CSS,
        JS,
        build_comparison_html,
        chart_grid,
        metric_card,
        metrics_grid,
        section_title,
        wrap_chart,
    )

    assert isinstance(CSS, str) and len(CSS) > 0
    assert isinstance(JS, str) and len(JS) > 0
    assert COLOR_NOVOMODELO == COMPARISON_COLORS["novomodelo"]
    assert COLOR_NEWAVE == COMPARISON_COLORS["newave"]
    assert COLOR_DIFF == COMPARISON_COLORS["diff"]
    assert COLOR_MATCH == COMPARISON_COLORS["match"]
    assert callable(wrap_chart)
    assert callable(section_title)
    assert callable(chart_grid)
    assert callable(metrics_grid)
    assert callable(metric_card)
    assert callable(build_comparison_html)


def test_build_comparison_html_head_includes_plotly_title_shim() -> None:
    """The compare (newave + decomp, shared shell) report <head> must carry the
    client-side title-normalizer shim.

    plotly.js 3.x drops a bare-string ``title`` at render; the shim wraps
    ``Plotly.newPlot``/``Plotly.react`` before any tab's chart script runs, so
    it must land in ``<head>`` (via ``required_js``), not the end-of-body
    ``<script>`` (``JS``/``TAB_SWITCH_JS``), which executes too late.
    """
    from novomodelo_bridge.comparators.html_report import build_comparison_html

    html = build_comparison_html("Test", {"tab-overview": "<p>x</p>"})

    head = html.split("<head>", 1)[1].split("</head>", 1)[0]
    assert "normalizePlotlyTitles" in head


# ---------------------------------------------------------------------------
# animated tab underline + expand-to-full-width
# ---------------------------------------------------------------------------


def test_tab_switch_js_contains_underline_logic() -> None:
    """TAB_SWITCH_JS must reference the tab-underline element."""
    assert "tab-underline" in TAB_SWITCH_JS


def test_tab_switch_js_contains_expand_toggle() -> None:
    """TAB_SWITCH_JS must toggle the chart-card-expanded class."""
    assert "chart-card-expanded" in TAB_SWITCH_JS


def test_tab_switch_js_preserves_show_tab_signature() -> None:
    """showTab must keep its original (tabId, btn) signature unchanged."""
    assert "function showTab(tabId, btn)" in TAB_SWITCH_JS


def test_wrap_chart_contains_expand_button() -> None:
    """wrap_chart output must include a button with class expand-btn and an SVG."""
    result = wrap_chart("<p>X</p>")
    assert 'class="expand-btn"' in result
    assert "<svg" in result


def test_dashboard_css_contains_underline_styles() -> None:
    """dashboard_css() must include the .tab-underline selector."""
    assert ".tab-underline" in dashboard_css()


def test_dashboard_css_contains_expand_btn_styles() -> None:
    """dashboard_css() must include the .expand-btn selector."""
    assert ".expand-btn" in dashboard_css()


# ---------------------------------------------------------------------------
# Stripe-style metric cards with sparklines
# ---------------------------------------------------------------------------


def test_metric_card_backward_compatible() -> None:
    """metric_card positional-only call must match the original structure."""
    result = metric_card("42", "Test")
    assert 'class="metric-value"' in result
    assert 'class="metric-label"' in result
    assert "42" in result
    assert "Test" in result
    assert "metric-delta" not in result
    assert "metric-sparkline" not in result
    assert "border-top" not in result


def test_metric_card_with_color() -> None:
    """metric_card with color= must add a border-top inline style."""
    result = metric_card("42", "Test", color="#B87333")
    assert "border-top: 4px solid #B87333" in result


def test_metric_card_with_delta_up() -> None:
    """delta_direction='up' must render metric-delta-up class and the delta text."""
    result = metric_card("42", "Test", delta="+5%", delta_direction="up")
    assert 'class="metric-delta metric-delta-up"' in result
    assert "+5%" in result


def test_metric_card_with_delta_down() -> None:
    """metric_card with delta_direction='down' must render metric-delta-down class."""
    result = metric_card("42", "Test", delta="-3%", delta_direction="down")
    assert 'class="metric-delta metric-delta-down"' in result
    assert "-3%" in result


def test_metric_card_with_sparkline() -> None:
    """metric_card with sparkline_values must include an <svg> and a <polyline>."""
    result = metric_card("42", "Test", sparkline_values=[1.0, 2.0, 3.0, 2.5])
    assert "<svg" in result
    assert "<polyline" in result


def test_metric_card_sparkline_omitted_for_empty_values() -> None:
    """metric_card with sparkline_values=[] must not render any SVG."""
    result = metric_card("42", "Test", sparkline_values=[])
    assert "<svg" not in result


def test_metric_card_delta_without_direction() -> None:
    """delta alone (no direction) must render the text without a directional class."""
    result = metric_card("42", "Test", delta="+5%")
    assert "+5%" in result
    assert "metric-delta-up" not in result
    assert "metric-delta-down" not in result
    assert 'class="metric-delta"' in result


def test_dashboard_css_contains_metric_delta_styles() -> None:
    """dashboard_css() must include all four new metric-delta CSS selectors."""
    output = dashboard_css()
    assert ".metric-delta" in output
    assert ".metric-delta-up" in output
    assert ".metric-delta-down" in output
    assert ".metric-sparkline" in output


# ---------------------------------------------------------------------------
# collapsible sections and staggered card entry animations
# ---------------------------------------------------------------------------


def test_collapsible_section_structure() -> None:
    """Verify data-collapsible attr, collapsible-content wrapper, and chevron SVG."""
    result = collapsible_section("Title", "<p>Body</p>")
    assert 'data-collapsible="true"' in result
    assert 'class="collapsible-content"' in result
    assert 'class="chevron"' in result
    assert "<svg" in result
    assert "<polyline" in result


def test_collapsible_section_contains_title_text() -> None:
    """collapsible_section must render the title text inside the section-title div."""
    result = collapsible_section("My Section", "<p>Body</p>")
    assert "My Section" in result


def test_collapsible_section_contains_content() -> None:
    """collapsible_section must render the content string inside collapsible-content."""
    result = collapsible_section("Title", "<p>Custom body content</p>")
    assert "<p>Custom body content</p>" in result
    # Content must appear after the section-title div (inside collapsible-content).
    content_idx = result.index("collapsible-content")
    body_idx = result.index("Custom body content")
    assert body_idx > content_idx


def test_section_title_unchanged() -> None:
    """section_title must return exactly its established output, no regression."""
    result = section_title("Test")
    assert result == '<div class="section-title">Test</div>'


def test_tab_switch_js_contains_collapsible_handler() -> None:
    """TAB_SWITCH_JS must contain a delegated click handler for data-collapsible."""
    assert "data-collapsible" in TAB_SWITCH_JS
    assert "collapsed" in TAB_SWITCH_JS


def test_tab_switch_js_contains_stagger_animation() -> None:
    """TAB_SWITCH_JS must contain stagger animation logic with card-enter and delay."""
    assert "card-enter" in TAB_SWITCH_JS
    assert "animationDelay" in TAB_SWITCH_JS


# ---------------------------------------------------------------------------
# plant explorer JS infrastructure
# ---------------------------------------------------------------------------


def test_plant_explorer_js_contains_all_functions() -> None:
    """PLANT_EXPLORER_JS must define all 8 required function names."""
    required = [
        "initPlantExplorer",
        "filterTable",
        "sortTable",
        "selectRow",
        "plotlyBand",
        "plotlyLine",
        "plotlyRef",
        "plotlyLayout",
    ]
    for name in required:
        assert name in PLANT_EXPLORER_JS, f"Missing function: {name!r}"


def test_plant_explorer_css_contains_expected_classes() -> None:
    """PLANT_EXPLORER_CSS must define all required class selectors."""
    required = [
        ".explorer-container",
        ".explorer-table-pane",
        ".explorer-detail-pane",
        ".explorer-search",
        ".explorer-row-selected",
    ]
    for selector in required:
        assert selector in PLANT_EXPLORER_CSS, f"Missing selector: {selector!r}"


def test_plant_explorer_table_structure() -> None:
    """plant_explorer_table returns HTML with expected structure."""
    result = plant_explorer_table(
        "tbody-hydro",
        "search-hydro",
        [("Name", "string"), ("MW", "number")],
        "<tr><td>A</td><td>1</td></tr>",
    )
    assert "<input" in result
    assert 'id="search-hydro"' in result
    assert "<table" in result
    assert "<thead>" in result or "<thead" in result
    assert "<tbody" in result
    assert 'id="tbody-hydro"' in result
    assert "<th" in result
    assert "<tr><td>A</td><td>1</td></tr>" in result


def test_dashboard_css_includes_explorer_container() -> None:
    """dashboard_css() must include .explorer-container from PLANT_EXPLORER_CSS."""
    assert ".explorer-container" in dashboard_css()


# ---------------------------------------------------------------------------
# synchronized hover and comparison mode
# ---------------------------------------------------------------------------


def test_plant_explorer_js_contains_sync_hover_and_comparison_mode() -> None:
    """PLANT_EXPLORER_JS must define syncHover and initComparisonMode functions."""
    assert "syncHover" in PLANT_EXPLORER_JS
    assert "initComparisonMode" in PLANT_EXPLORER_JS


def test_plant_explorer_css_contains_comparison_active_classes() -> None:
    """PLANT_EXPLORER_CSS must define compare-active-1, -2, and -3 selectors."""
    assert "compare-active-1" in PLANT_EXPLORER_CSS
    assert "compare-active-2" in PLANT_EXPLORER_CSS
    assert "compare-active-3" in PLANT_EXPLORER_CSS

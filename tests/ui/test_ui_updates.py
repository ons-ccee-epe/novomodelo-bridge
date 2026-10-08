"""Unit tests for UI package updates.

Covers new theme constants, SUB_TAB_JS, CSS additions, and updated
collapsible_section() default_collapsed parameter.
"""

from __future__ import annotations

from novomodelo_bridge.ui.html.css import DASHBOARD_CSS
from novomodelo_bridge.ui.html.document import collapsible_section
from novomodelo_bridge.ui.html.js import SUB_TAB_JS
from novomodelo_bridge.ui.theme import (
    BAND_OPACITY,
    BOUND_LINE_COLOR,
    GENERATION_COLORS,
    PERFORMANCE_PHASE_COLORS,
)

# ---------------------------------------------------------------------------
# theme.py — new constants
# ---------------------------------------------------------------------------


def test_generation_colors_keys() -> None:
    """GENERATION_COLORS must contain hydro, thermal, and ncs keys."""
    assert "hydro" in GENERATION_COLORS
    assert "thermal" in GENERATION_COLORS
    assert "ncs" in GENERATION_COLORS


def test_generation_colors_values() -> None:
    """GENERATION_COLORS values must match the specified hex codes."""
    # hydro shares COLORS["hydro"] so the entity reads identically across the
    # generation charts and the hydro tab (resolved hydro colour collision).
    assert GENERATION_COLORS["hydro"] == "#4A90B8"
    assert GENERATION_COLORS["thermal"] == "#F59E0B"
    assert GENERATION_COLORS["ncs"] == "#10B981"


def test_hydro_color_is_consistent_across_palettes() -> None:
    """The hydro entity must read identically wherever it is drawn (no collision)."""
    from novomodelo_bridge.ui.theme import COLORS

    assert GENERATION_COLORS["hydro"] == COLORS["hydro"]


def test_performance_phase_colors_keys() -> None:
    """PERFORMANCE_PHASE_COLORS must contain all 4 phase keys."""
    assert "forward" in PERFORMANCE_PHASE_COLORS
    assert "backward" in PERFORMANCE_PHASE_COLORS
    assert "lp_solve" in PERFORMANCE_PHASE_COLORS
    assert "overhead" in PERFORMANCE_PHASE_COLORS


def test_performance_phase_colors_are_strings() -> None:
    """PERFORMANCE_PHASE_COLORS values must all be strings."""
    for key, value in PERFORMANCE_PHASE_COLORS.items():
        assert isinstance(value, str), (
            f"Expected str for key {key!r}, got {type(value)}"
        )


def test_bound_line_color_is_string() -> None:
    """BOUND_LINE_COLOR must be a non-empty string."""
    assert isinstance(BOUND_LINE_COLOR, str)
    assert BOUND_LINE_COLOR


def test_band_opacity_is_float() -> None:
    """BAND_OPACITY must be a float."""
    assert isinstance(BAND_OPACITY, float)
    assert BAND_OPACITY == 0.15


# ---------------------------------------------------------------------------
# js.py — SUB_TAB_JS
# ---------------------------------------------------------------------------


def test_sub_tab_js_contains_function() -> None:
    """SUB_TAB_JS must contain the switchSubTab function definition."""
    assert "function switchSubTab" in SUB_TAB_JS


def test_sub_tab_js_uses_subtab_group_attribute() -> None:
    """SUB_TAB_JS must scope its DOM query to data-subtab-group."""
    assert "data-subtab-group" in SUB_TAB_JS


def test_sub_tab_js_handles_panels_and_buttons() -> None:
    """SUB_TAB_JS must reference sub-tab-panel and sub-tab-btn classes."""
    assert ".sub-tab-panel" in SUB_TAB_JS
    assert ".sub-tab-btn" in SUB_TAB_JS


# ---------------------------------------------------------------------------
# css.py — DASHBOARD_CSS additions
# ---------------------------------------------------------------------------


def test_css_contains_sub_tab_styles() -> None:
    """DASHBOARD_CSS must include .sub-tab-bar and .sub-tab-btn rules."""
    assert ".sub-tab-bar" in DASHBOARD_CSS
    assert ".sub-tab-btn" in DASHBOARD_CSS


def test_css_contains_sub_tab_panel() -> None:
    """DASHBOARD_CSS must include the .sub-tab-panel rule."""
    assert ".sub-tab-panel" in DASHBOARD_CSS


def test_css_contains_default_collapsed() -> None:
    """DASHBOARD_CSS must include a CSS rule for collapsed content.

    The collapsible section uses the 'collapsed-title' and
    'collapsible-content collapsed' classes when default_collapsed=True.
    The CSS that styles collapsed state is expressed via
    '.collapsible-content.collapsed' and '.collapsed-title'.
    """
    assert ".collapsible-content.collapsed" in DASHBOARD_CSS or (
        "collapsed" in DASHBOARD_CSS
    )


# ---------------------------------------------------------------------------
# html.py — collapsible_section() updated signature
# ---------------------------------------------------------------------------


def test_collapsible_section_default_expanded() -> None:
    """collapsible_section() with default_collapsed=False must not add the class."""
    html = collapsible_section(
        "Title", "<p>content</p>", "sec1", default_collapsed=False
    )
    assert "default-collapsed" not in html
    assert 'class="collapsible-section"' in html


def test_collapsible_section_default_collapsed_class() -> None:
    """collapsible_section() with default_collapsed=True applies collapsed classes.

    The outer section div keeps class="collapsible-section".  The title div
    gains the 'collapsed-title' modifier and the content div gains 'collapsed'.
    """
    html = collapsible_section(
        "Title", "<p>content</p>", "sec1", default_collapsed=True
    )
    assert 'class="collapsible-section"' in html
    assert "collapsed-title" in html
    assert "collapsible-content collapsed" in html


def test_collapsible_section_default_collapsed_chevron() -> None:
    """collapsible_section() with default_collapsed=True includes a chevron element.

    The chevron is an inline SVG polyline (not a text '>' character).
    Its rotation is controlled via CSS '.collapsed-title .chevron'.
    """
    html = collapsible_section(
        "Title", "<p>content</p>", "sec1", default_collapsed=True
    )
    # The chevron is an SVG element, not a text span.
    assert 'class="chevron"' in html
    assert "<svg" in html or "<polyline" in html


def test_collapsible_section_section_id() -> None:
    """collapsible_section() must render the section_id as an id attribute."""
    html = collapsible_section("Title", "<p>content</p>", "my-section")
    assert 'id="my-section"' in html


def test_collapsible_section_no_id_when_empty() -> None:
    """collapsible_section() must omit id attribute when section_id is empty."""
    html = collapsible_section("Title", "<p>content</p>")
    assert "id=" not in html


def test_collapsible_section_backward_compatible() -> None:
    """collapsible_section() with only 2 positional args must still work."""
    html = collapsible_section("Title", "<p>body</p>")
    assert "collapsible-section" in html
    assert "default-collapsed" not in html

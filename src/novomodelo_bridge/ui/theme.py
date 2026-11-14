"""Shared visual design tokens for the novomodelo-bridge UI.

Canonical source for the **semantic** colours used across the dashboard and
comparator Plotly charts: the per-entity generation palette (hydro/thermal/ncs),
bounds/comparison colours, the cost category and component colours, and chart
accents. Reference these constants from chart code instead of repeating the hex
values.

Two kinds of colour legitimately live elsewhere and are *not* governed here: the
page **CSS** (``ui/css.py``) and **local categorical palettes** that map a fixed
set of labels to distinct hues for a single chart (e.g. the constraint-type
map, the plant-explorer scenario colours). Those are intentionally independent
of the semantic tokens.
"""

from __future__ import annotations

COLORS: dict[str, str] = {
    "hydro": "#4A90B8",
    "thermal": "#F5A623",
    "ncs": "#4A8B6F",
    "load": "#374151",
    "deficit": "#DC4C4C",
    "spillage": "#B87333",
    "curtailment": "#8B5E3C",
    "exchange": "#4A90B8",
    "lower_bound": "#4A8B6F",
    "upper_bound": "#DC4C4C",
    "future_cost": "#8B9298",
}

#: One colour per cost category and per cost component, keyed by the names in
#: :data:`novomodelo_bridge.novomodelo.cost_categories.COST_CATEGORIES`, so a
#: cost keeps its colour in every dashboard chart and in the compare report.
#: Each category's components are an OKLCH lightness ramp of the category's
#: hue, light to dark in taxonomy order, so a stack groups them by category.
COST_CATEGORY_COLORS: dict[str, str] = {
    "Thermal & Contracts": "#eda100",
    "Deficit & Excess": "#e34948",
    "Violations": "#6250d6",
    "Regularization": "#1baf7a",
    "Pumping": "#2a78d6",
}

COST_COMPONENT_COLORS: dict[str, str] = {
    "Thermal": "#eda100",
    "Contract": "#ab7305",
    "Deficit": "#e34948",
    "Energy Excess": "#ac031b",
    "Outflow Min": "#aaaaff",
    "Outflow Max": "#9591fe",
    "Turbining Bounds": "#8277fe",
    "Generation Bounds": "#7162ea",
    "Storage Bounds": "#614ed4",
    "Filling Target": "#513abf",
    "Evaporation": "#4223aa",
    "Water Withdrawal": "#350296",
    "Inflow Penalty": "#260171",
    "Generic Constraints": "#18004e",
    "Spillage": "#1baf7a",
    "Turbined Reg.": "#098d61",
    "NCS Curtailment": "#006c49",
    "Exchange": "#004d32",
    "Pumping": "#2a78d6",
}

#: Colour of a cost the taxonomy does not classify.
COST_OTHER_COLOR: str = "#6B7280"

#: Colour of the expected total cost.
COST_TOTAL_COLOR: str = "#374151"

BUS_COLORS: list[str] = ["#4A90B8", "#F5A623", "#4A8B6F", "#DC4C4C", "#B87333"]

COMPARISON_COLORS: dict[str, str] = {
    "novomodelo": "#4A90B8",
    "newave": "#F5A623",
    "diff": "#DC4C4C",
    "match": "#4A8B6F",
}

COPPER_ACCENT: str = "#B87333"

# ``hydro`` is intentionally the same #4A90B8 as ``COLORS["hydro"]`` so the
# entity reads identically across the generation charts and the hydro tab
# (resolves the prior #4A90B8/#3B82F6 hydro collision).
GENERATION_COLORS: dict[str, str] = {
    "hydro": "#4A90B8",
    "thermal": "#F59E0B",
    "ncs": "#10B981",
}

PERFORMANCE_PHASE_COLORS: dict[str, str] = {
    "forward": "#3B82F6",
    "backward": "#14B8A6",
    "lp_solve": "#B87333",
    "overhead": "#6B7280",
}

# ``SEVERITY`` coincides with COLORS["thermal"]/["deficit"]/["ncs"], but a
# severity is not an entity domain — kept as its own palette so a WARNING/ERROR
# panel is never coupled to a domain token that happens to share its hex.
SEVERITY: dict[str, str] = {
    "warning": "#F5A623",
    "error": "#DC4C4C",
    "ok": "#4A8B6F",
}

BOUND_LINE_COLOR: str = "#6B7280"

BAND_OPACITY: float = 0.15


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex colour string to an ``rgba(...)`` CSS value.

    Args:
        hex_color: A hex colour string such as ``"#3B82F6"`` (the leading
            ``#`` is optional).
        alpha: Opacity in the range 0.0–1.0.

    Returns:
        An ``rgba(r, g, b, alpha)`` string suitable for Plotly's
        ``fillcolor`` parameter.
    """
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


#: Canonical p10-p90 band colours, shared by the dashboard's go.Figure band
#: helper and the comparators' raw-trace band. ``BAND_FILL`` is derived from
#: the same hydro hue as the rest of the semantic palette rather than a bare
#: literal, so the two consumers stay pinned to one source.
BAND_FILL: str = hex_to_rgba(COLORS["hydro"], BAND_OPACITY)
BAND_LINE: str = "rgba(255,255,255,0)"

"""Shared CSS stylesheets for dashboard and comparator HTML reports."""

from __future__ import annotations

BASE_CSS: str = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'IBM Plex Sans', 'Inter', 'Segoe UI', system-ui, sans-serif; background: #F0EDE8; color: #374151; }

header {
    background: linear-gradient(135deg, #0F1419 0%, #1A2028 100%);
    border-top: 3px solid #B87333;
    color: white;
    padding: 16px 32px;
    font-size: 1.3rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

nav {
    background: #1A2028;
    padding: 0 24px;
    display: flex;
    gap: 4px;
    overflow-x: auto;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

nav button {
    background: none;
    border: none;
    color: #8B9298;
    padding: 14px 20px;
    font-size: 0.88rem;
    font-weight: 500;
    cursor: pointer;
    border-bottom: 3px solid transparent;
    white-space: nowrap;
    transition: color 0.2s, border-color 0.2s;
    letter-spacing: 0.3px;
}

nav button:hover { color: #E8E6E3; border-bottom-color: #B87333; }
nav button.active { color: #B87333; border-bottom-color: #B87333; }

main { padding: 24px 32px; max-width: 1800px; margin: 0 auto; }

.tab-content { display: none; }
.tab-content.active { display: block; }

.chart-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
}

.chart-grid-single {
    display: grid;
    grid-template-columns: 1fr;
    gap: 20px;
    margin-bottom: 20px;
}

.chart-card {
    background: #FAFAF8;
    border-radius: 8px;
    padding: 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    overflow: hidden;
    min-width: 0;
}

.chart-card .plotly-graph-div { width: 100% !important; }
.chart-card .js-plotly-plot { width: 100% !important; }
.chart-card .plot-container { width: 100% !important; }

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
}

.metric-card {
    background: #FAFAF8;
    border-radius: 8px;
    padding: 20px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    text-align: center;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #374151;
    margin-bottom: 6px;
}

.metric-label {
    font-size: 0.8rem;
    color: #8B9298;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.section-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #374151;
    margin: 24px 0 12px;
    padding-left: 10px;
    border-left: 4px solid #B87333;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
"""

COST_BREAKDOWN_TABLE_CSS: str = """
.cost-breakdown-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.85rem;
    background: #FFFFFF;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08), 0 1px 2px rgba(15, 23, 42, 0.04);
    font-variant-numeric: tabular-nums;
}
.cost-breakdown-table caption {
    caption-side: top;
    text-align: left;
    padding: 12px 14px 8px 14px;
    font-weight: 600;
    font-size: 0.95rem;
    color: #1F2937;
    letter-spacing: 0.01em;
}
.cost-breakdown-table thead th {
    background: #F3F4F6;
    color: #4B5563;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    padding: 10px 14px;
    border-bottom: 1px solid #E5E7EB;
}
.cost-breakdown-table tbody td {
    padding: 8px 14px;
    color: #1F2937;
    border-bottom: 1px solid #F1F1EE;
    transition: background-color 120ms ease;
}
.cost-breakdown-table tbody tr:nth-child(even) td {
    background: #FBFBFA;
}
.cost-breakdown-table tbody tr:hover td {
    background: #EFF6FF;
}
.cost-breakdown-table tbody tr:last-child td {
    border-bottom: none;
    border-top: 2px solid #374151;
    background: #F9FAFB;
    font-weight: 700;
}
.cost-breakdown-table tbody tr:last-child:hover td {
    background: #F9FAFB;
}
.cost-breakdown-table .cb-swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 3px;
    margin-right: 0.6em;
    vertical-align: middle;
    box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08);
}
.cost-breakdown-table .cb-num { text-align: right; }
.cost-breakdown-table .cb-cat { text-align: left; }
.cost-breakdown-table .cb-diff-pos { color: #DC2626; font-weight: 600; }
.cost-breakdown-table .cb-diff-neg { color: #059669; font-weight: 600; }
/* Productivity building-blocks table: per-metric column grouping cues. */
.prod-blocks-table thead th { text-align: center; }
.prod-blocks-table .cb-group-tint { background: #F1F5F9; }
.prod-blocks-table thead .cb-group-tint { background: #E7EEF5; }
.prod-blocks-table .cb-group-sep { border-left: 2px solid #94A3B8; }
.prod-blocks-table tbody tr:nth-child(even) td.cb-group-tint { background: #E9EFF5; }
.prod-blocks-table tbody tr:hover td.cb-group-tint { background: #E0EAF6; }
.prod-blocks-table .cb-cat {
    position: sticky;
    left: 0;
    background: #FFFFFF;
    z-index: 1;
}
.prod-blocks-table .cb-caption-note {
    font-weight: 400;
    font-size: 0.8rem;
    color: #6B7280;
}
"""

DATA_TABLE_CSS: str = """
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    background: #FAFAF8;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.data-table th, .data-table td {
    padding: 10px 12px;
    text-align: right;
    border-bottom: 1px solid #E0E0E0;
}
.data-table th {
    background: #374151;
    color: white;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
}
.data-table td:first-child, .data-table th:first-child { text-align: left; }
.data-table td:nth-child(2), .data-table th:nth-child(2) { text-align: left; }
.data-table tr:hover td { background: #F0EDE8; }
"""

PLANT_SELECTOR_CSS: str = """
.plant-selector {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}

.plant-selector label {
    font-weight: 600;
    font-size: 0.9rem;
    color: #374151;
}

.plant-selector select {
    padding: 8px 12px;
    border: 1px solid #ccc;
    border-radius: 6px;
    font-size: 0.88rem;
    min-width: 280px;
    background: #FAFAF8;
}

.plant-info {
    display: flex;
    gap: 20px;
    font-size: 0.82rem;
    color: #8B9298;
    flex-wrap: wrap;
}

.plant-info span { white-space: nowrap; }
"""


TRANSITIONS_CSS: str = """
.chart-card {
    transition: transform 0.2s ease-out, box-shadow 0.2s ease-out;
}

.chart-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.18);
}
"""

RESPONSIVE_CSS: str = """
@media (max-width: 767px) {
    .chart-grid {
        grid-template-columns: 1fr;
    }
    .metrics-grid {
        grid-template-columns: 1fr;
    }
    main {
        padding: 12px;
    }
}

@media (min-width: 768px) and (max-width: 1199px) {
    .chart-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
"""

TAB_FADE_CSS: str = """
.tab-content-fade {
    opacity: 0;
    transition: opacity 0.15s ease-in;
}

.tab-content-fade.active {
    opacity: 1;
}

.chart-card-expanded {
    grid-column: 1 / -1;
}
"""

COLLAPSIBLE_CSS: str = """
.section-title[data-collapsible] {
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.chevron {
    transition: transform 0.2s ease-out;
    flex-shrink: 0;
}

.section-title[data-collapsible].collapsed-title .chevron {
    transform: rotate(-90deg);
}

.collapsible-content {
    max-height: 4000px;
    overflow: hidden;
    transition: max-height 0.25s ease-out;
}

.collapsible-content.collapsed {
    max-height: 0;
    overflow: hidden;
}

@keyframes cardEnter {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.card-enter {
    animation: cardEnter 0.2s ease-out both;
}
"""

PLANT_EXPLORER_CSS: str = """
.explorer-container {
    display: flex;
    gap: 20px;
}

.explorer-table-pane {
    width: 480px;
    flex-shrink: 0;
    max-height: 80vh;
    overflow-y: auto;
}

.explorer-detail-pane {
    flex: 1;
    min-width: 0;
}

.explorer-search {
    width: 100%;
    padding: 8px 12px;
    border: 1px solid #ccc;
    border-radius: 6px;
    font-size: 0.88rem;
    background: #FAFAF8;
    margin-bottom: 8px;
    box-sizing: border-box;
}

.explorer-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    background: #FAFAF8;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
}

.explorer-table th, .explorer-table td {
    padding: 7px 10px;
    text-align: right;
    border-bottom: 1px solid #E0E0E0;
}

.explorer-table th {
    background: #374151;
    color: white;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.5px;
}

.explorer-table td:first-child, .explorer-table th:first-child { text-align: left; }

.explorer-table tbody tr {
    cursor: pointer;
}

.explorer-table tbody tr:hover td { background: #F0EDE8; }

.explorer-row-selected td {
    background: rgba(184, 115, 51, 0.12) !important;
}

.explorer-table th.sortable {
    cursor: pointer;
    user-select: none;
}

.explorer-table th .sort-arrow {
    display: inline-block;
    width: 1em;
}

@media (max-width: 1023px) {
    .explorer-container {
        flex-direction: column;
    }
    .explorer-table-pane {
        width: 100%;
        max-height: 40vh;
    }
}

.compare-checkbox {
    width: 14px;
    height: 14px;
    cursor: pointer;
    display: block;
    margin: 0 auto;
}

.compare-active-1 {
    border-left: 3px solid #2196F3;
}

.compare-active-2 {
    border-left: 3px solid #FF9800;
}

.compare-active-3 {
    border-left: 3px solid #4CAF50;
}

.compare-legend {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    font-size: 0.8rem;
    margin-bottom: 8px;
    padding: 6px 10px;
    background: #F0EDE8;
    border-radius: 6px;
}

.compare-legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
}

.compare-legend-swatch {
    width: 12px;
    height: 12px;
    border-radius: 2px;
    flex-shrink: 0;
}
"""

METRIC_CARD_CSS: str = """
.metric-delta {
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    margin-bottom: 4px;
}

.metric-delta-up {
    color: #4A8B6F;
}

.metric-delta-down {
    color: #DC4C4C;
}

.metric-sparkline {
    display: flex;
    justify-content: center;
    margin-top: 8px;
}
"""

UNDERLINE_EXPAND_CSS: str = """
nav { position: relative; }

.tab-underline {
    position: absolute;
    bottom: 0;
    height: 3px;
    background: #B87333;
    transition: transform 0.3s ease-out, width 0.3s ease-out;
    pointer-events: none;
}

.chart-card { position: relative; }

.expand-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    opacity: 0;
    transition: opacity 0.2s ease-out;
    cursor: pointer;
    background: none;
    border: none;
    color: #8B9298;
    padding: 4px;
    z-index: 10;
    line-height: 0;
}

.chart-card:hover .expand-btn { opacity: 1; }
"""


SUB_TAB_CSS: str = """
.sub-tab-bar {
    display: flex;
    gap: 0;
    margin-bottom: 16px;
    border-bottom: 1px solid #2D3748;
}

.sub-tab-btn {
    background: none;
    border: none;
    color: #8B9298;
    padding: 8px 16px;
    font-size: 0.82rem;
    font-weight: 500;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    white-space: nowrap;
    transition: color 0.2s, border-color 0.2s;
    letter-spacing: 0.3px;
}

.sub-tab-btn:hover { color: #E8E6E3; border-bottom-color: #B87333; }
.sub-tab-btn.active { color: #B87333; border-bottom-color: #B87333; }

.sub-tab-panel { display: none; }
.sub-tab-panel:first-of-type { display: block; }

"""


def dashboard_css() -> str:
    """Compose CSS for the dashboard."""
    return (
        BASE_CSS
        + DATA_TABLE_CSS
        + TRANSITIONS_CSS
        + RESPONSIVE_CSS
        + TAB_FADE_CSS
        + COLLAPSIBLE_CSS
        + UNDERLINE_EXPAND_CSS
        + METRIC_CARD_CSS
        + PLANT_EXPLORER_CSS
        + SUB_TAB_CSS
    )


def comparison_css() -> str:
    """Compose CSS for comparison reports."""
    return (
        BASE_CSS
        + PLANT_SELECTOR_CSS
        + TRANSITIONS_CSS
        + RESPONSIVE_CSS
        + TAB_FADE_CSS
        + UNDERLINE_EXPAND_CSS
        + METRIC_CARD_CSS
        + COST_BREAKDOWN_TABLE_CSS
    )


DASHBOARD_CSS: str = dashboard_css()

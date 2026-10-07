"""Modular dashboard package for Cobre simulation results.

Entry point: ``build_dashboard(case_dir, output_path)`` loads data,
discovers renderable tabs, and writes the assembled HTML file (it loads
plotly.js from a CDN, so it is not self-contained offline).
"""

from __future__ import annotations

import logging
from pathlib import Path

from cobre_bridge.dashboard.data import DashboardData
from cobre_bridge.dashboard.tabs import collect_required_js, get_renderable_tabs
from cobre_bridge.ui.html.css import dashboard_css
from cobre_bridge.ui.html.document import build_html
from cobre_bridge.ui.html.js import PLOTLY_TITLE_SHIM_JS, TAB_SWITCH_JS

logger = logging.getLogger(__name__)


def build_dashboard(case_dir: Path, output_path: Path) -> None:
    """Build an interactive HTML dashboard from Cobre simulation results."""
    data = DashboardData.load(case_dir)

    renderable = get_renderable_tabs(data)
    tab_defs = [(tab_id, tab_label) for tab_id, tab_label, _ in renderable]
    tab_contents = {tab_id: html for tab_id, _, html in renderable}

    case_name = case_dir.resolve().name
    html = build_html(
        title=f"Cobre Simulation Dashboard \u2014 {case_name}",
        tab_defs=tab_defs,
        tab_contents=tab_contents,
        css=dashboard_css(),
        js=TAB_SWITCH_JS,
        required_js=PLOTLY_TITLE_SHIM_JS + collect_required_js(data),
    )

    logger.info("Writing dashboard to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    size_kb = output_path.stat().st_size / 1024
    logger.info("Dashboard written (%.0f KB)", size_kb)

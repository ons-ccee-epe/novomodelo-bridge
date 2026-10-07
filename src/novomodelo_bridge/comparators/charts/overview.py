"""Overview-tab headline KPI cards."""

from __future__ import annotations

from cobre_bridge.comparators.model import ResultsSummary


def overview_metrics(
    summary: ResultsSummary,
    nw_costs: dict[str, float] | None = None,
    cobre_costs: dict[str, float] | None = None,
    reference_label: str = "NEWAVE",
) -> str:
    """Headline KPI cards for the overview tab.

    Four cards: total NPV cost on each side, the absolute Δ (10⁹ R$)
    with red/green coloring, and the relative Δ%. The previous
    Total Comparisons / Entity Types / Variables triple was meta about
    the report itself and offered no operational insight.
    """
    from cobre_bridge.comparators.html_report import (
        metric_card,
        metrics_grid,
    )
    from cobre_bridge.ui.theme import COMPARISON_COLORS

    # Pull thermal-generation cost only (single-category NPV). The source model parcela
    # "GERACAO TERMICA" vs Cobre ``thermal_cost`` + the GNL ``anticipated_thermal_cost``
    # (matching the "Thermal Generation" ``_COST_MAP`` category; anticipated is 0 /
    # absent on non-GNL runs).
    nw_thermal = (nw_costs or {}).get("GERACAO TERMICA", 0.0)
    _cb = cobre_costs or {}
    cb_thermal = _cb.get("thermal_cost", 0.0) + _cb.get("anticipated_thermal_cost", 0.0)
    diff = cb_thermal - nw_thermal
    pct = (diff / nw_thermal * 100.0) if abs(nw_thermal) > 1e-6 else float("nan")

    def _bn(v: float) -> str:
        return f"{v / 1e9:.3f}"

    def _pct(v: float) -> str:
        if v != v:  # NaN
            return "—"
        return f"{v:+.1f}%"

    # Color the Δ cards based on sign (Cobre overshoot = red, undershoot = green).
    diff_color = (
        COMPARISON_COLORS.get("diff", "#DC4C4C")
        if diff > 0
        else COMPARISON_COLORS.get("match", "#3F8E5F")
    )

    cards = [
        metric_card(
            _bn(nw_thermal),
            f"{reference_label} Thermal Cost (10⁹ R$, NPV)",
            color=COMPARISON_COLORS.get("newave"),
        ),
        metric_card(
            _bn(cb_thermal),
            "Cobre Thermal Cost (10⁹ R$, NPV)",
            color=COMPARISON_COLORS.get("cobre"),
        ),
        metric_card(
            f"{diff / 1e9:+.3f}",
            f"Δ Thermal Cost (Cobre − {reference_label}, 10⁹ R$)",
            color=diff_color,
        ),
        metric_card(
            _pct(pct),
            "Δ Thermal Cost (%)",
            color=diff_color,
        ),
    ]
    # Suppress the cards entirely when neither side reports cost data —
    # keeps the overview tidy on training-only outputs.
    if nw_thermal == 0 and cb_thermal == 0:
        if summary.total == 0:
            return ""
        cards = [
            metric_card(str(summary.total), "Comparisons"),
            metric_card(str(len(summary.by_entity_type)), "Entity Types"),
            metric_card(str(len(summary.by_variable)), "Variables"),
        ]
    return metrics_grid(cards)

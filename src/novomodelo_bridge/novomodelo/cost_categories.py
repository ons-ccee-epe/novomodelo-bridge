"""Canonical Novomodelo cost taxonomy — one source for every report.

Novomodelo's per-stage ``costs`` parquet carries one column per cost component, a
few aggregate columns and the row keys. This module says which columns are
components, how they group into categories, and the name each component and
category carries in the dashboard and in the compare report, so no report can
miss a column, count one twice, or name it its own way. Their colours live
beside the other chart tokens in :mod:`novomodelo_bridge.ui.theme`.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Aggregate columns in the ``costs`` parquet that are NOT cost components.
#: ``hydro_violation_cost`` is the sum of six hydro-violation components and
#: ``total/immediate/future_cost`` are roll-ups — summing any of these alongside
#: the components would double-count. ``discount_factor`` is the
#: per-(scenario, stage) NPV weight, not a cost.
AGGREGATE_COST_COLUMNS: frozenset[str] = frozenset(
    {
        "total_cost",
        "immediate_cost",
        "future_cost",
        "discount_factor",
        "hydro_violation_cost",
    }
)

#: Row-key columns of the per-row costs frame (the dashboard reads the frame
#: wide and must also exclude these; the comparator works from an
#: already-aggregated dict and never sees them).
COST_KEY_COLUMNS: frozenset[str] = frozenset(
    {"scenario_id", "stage_id", "node_id", "block_id"}
)


@dataclass(frozen=True)
class CostComponent:
    """One entry of a cost breakdown: its name and the columns it sums."""

    label: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class CostCategory:
    """A named group of cost components."""

    label: str
    components: tuple[CostComponent, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        """Every cost column of the category, in component order."""
        return tuple(col for comp in self.components for col in comp.columns)


#: The categories of Novomodelo's own per-scenario cost breakdown (resource,
#: recourse, violation, regularization and imputed costs). The violation
#: category lists the six components of ``hydro_violation_cost`` instead of
#: that sum.
COST_CATEGORIES: tuple[CostCategory, ...] = (
    CostCategory(
        "Thermal & Contracts",
        (
            # Anticipated (forward-committed, GNL) thermal fuel is booked on the
            # decision-stage commitment column; grouped with thermal generation so
            # the component matches the source model's CTERM, which books GNL at
            # delivery.
            CostComponent("Thermal", ("thermal_cost", "anticipated_thermal_cost")),
            CostComponent("Contract", ("contract_cost",)),
        ),
    ),
    CostCategory(
        "Deficit & Excess",
        (
            CostComponent("Deficit", ("deficit_cost",)),
            CostComponent("Energy Excess", ("excess_cost",)),
        ),
    ),
    CostCategory(
        "Violations",
        (
            CostComponent("Outflow Min", ("outflow_violation_below_cost",)),
            CostComponent("Outflow Max", ("outflow_violation_above_cost",)),
            CostComponent("Turbining Bounds", ("turbined_violation_cost",)),
            CostComponent("Generation Bounds", ("generation_violation_cost",)),
            CostComponent("Storage Bounds", ("storage_violation_cost",)),
            CostComponent("Filling Target", ("filling_target_cost",)),
            CostComponent("Evaporation", ("evaporation_violation_cost",)),
            CostComponent("Water Withdrawal", ("withdrawal_violation_cost",)),
            CostComponent("Inflow Penalty", ("inflow_penalty_cost",)),
            CostComponent("Generic Constraints", ("generic_violation_cost",)),
        ),
    ),
    CostCategory(
        "Regularization",
        (
            CostComponent("Spillage", ("spillage_cost",)),
            CostComponent("Turbined Reg.", ("turbined_cost",)),
            CostComponent("NCS Curtailment", ("curtailment_cost",)),
            CostComponent("Exchange", ("exchange_cost",)),
        ),
    ),
    CostCategory("Pumping", (CostComponent("Pumping", ("pumping_cost",)),)),
)

#: Every category by name.
COST_CATEGORY_BY_LABEL: dict[str, CostCategory] = {
    cat.label: cat for cat in COST_CATEGORIES
}

#: The category that prices constraint slacks.
VIOLATIONS: CostCategory = COST_CATEGORY_BY_LABEL["Violations"]

#: Every component by name, in category order.
COST_COMPONENT_BY_LABEL: dict[str, CostComponent] = {
    comp.label: comp for cat in COST_CATEGORIES for comp in cat.components
}

#: The category each component belongs to, by name.
COST_CATEGORY_OF_COMPONENT: dict[str, str] = {
    comp.label: cat.label for cat in COST_CATEGORIES for comp in cat.components
}

#: Every individual Novomodelo cost-component column: the single definition of
#: "which columns are summable cost components".
NOVOMODELO_COST_COMPONENT_COLUMNS: tuple[str, ...] = tuple(
    col for cat in COST_CATEGORIES for col in cat.columns
)

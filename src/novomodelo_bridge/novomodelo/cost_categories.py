"""Canonical Cobre cost-component taxonomy — one source for both stacks.

Cobre's per-stage ``costs`` parquet carries one column per cost component plus a
few derived/aggregate columns. Two places classify those columns:

- the dashboard cost chart (``dashboard/chart_helpers.py::COST_GROUPS``), and
- the source model↔Cobre cost-breakdown comparison
  (``comparators.charts._COST_MAP`` + ``cobre_readers.read_cobre_cost_breakdown``).

They used to each hard-code their own column lists and drifted: the comparator's
sum list omitted ``contract_cost`` entirely (silently dropping it from the
breakdown), and the dashboard had no ``excess_cost`` group (lumping it into
"Other"). This module owns the canonical sets so neither stack can silently miss
a column; a drift-guard test asserts both classify exactly
:data:`COBRE_COST_COMPONENT_COLUMNS`.

This is presentation-free: labels, colours, grouping and the source-model-side alignment
stay in the two consumers.
"""

from __future__ import annotations

# : Derived/aggregate columns in the ``costs`` parquet that are NOT individual
# : cost components. ``hydro_violation_cost`` is the sum of the six hydro-violation
# : components, and ``total/immediate/future_cost`` are roll-ups — summing any of
# : these alongside the components would double-count. ``discount_factor`` is the
# : per-(scenario, stage) NPV weight, not a cost.
AGGREGATE_COST_COLUMNS: frozenset[str] = frozenset(
    {
        "total_cost",
        "immediate_cost",
        "future_cost",
        "discount_factor",
        "hydro_violation_cost",
    }
)

# : Hive-partition / time-index columns present in the per-row costs frame (the
# : dashboard reads the frame wide and must also exclude these; the comparator
# : works from an already-aggregated dict and never sees them).
COST_PARTITION_COLUMNS: frozenset[str] = frozenset(
    {"scenario_id", "stage_id", "block_id"}
)

# : Every individual Cobre cost-component column, in a stable display-ish order.
# : This is the single definition of "which columns are summable cost components".
# : Both consumers must classify exactly these (enforced by
# : ``tests/cobre/test_cost_categories.py``); a new Cobre cost column added here that a
# : consumer fails to map is a drift bug, caught by that test.
COBRE_COST_COMPONENT_COLUMNS: tuple[str, ...] = (
    # Generation / operational
    "thermal_cost",
    # Anticipated (forward-committed, GNL) thermal fuel, booked on the decision-stage
    # commitment column. Added to Cobre's costs schema after 0.8.0; absent in older runs
    # (read as 0). Grouped with thermal generation so the thermal category matches the
    # source model CTERM (which books GNL at delivery).
    "anticipated_thermal_cost",
    "deficit_cost",
    "excess_cost",
    "contract_cost",
    "exchange_cost",
    "pumping_cost",
    # Regularisation (per-unit-flow charges, not violations)
    "spillage_cost",
    "turbined_cost",
    "curtailment_cost",
    "inflow_penalty_cost",
    # Hydro operational-bound violations
    "outflow_violation_below_cost",
    "outflow_violation_above_cost",
    "turbined_violation_cost",
    "generation_violation_cost",
    "storage_violation_cost",
    "filling_target_cost",
    "evaporation_violation_cost",
    "withdrawal_violation_cost",
    # Generic constraints
    "generic_violation_cost",
)

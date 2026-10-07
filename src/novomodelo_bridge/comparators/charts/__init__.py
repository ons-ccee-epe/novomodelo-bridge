"""Chart implementations for the HTML comparison report.

Thin facade — defines nothing itself. Re-exports the package's public API from
:mod:`.constraints`, :mod:`.convergence`, :mod:`.costs`, :mod:`.fpha`,
:mod:`.hydro`, :mod:`.network`, :mod:`.overview`, :mod:`.performance`,
:mod:`.productivity`, :mod:`.spillage`, :mod:`.system`, and :mod:`.thermal`.
"""

from __future__ import annotations

from cobre_bridge.comparators.charts.constraints import (
    constraints_comparison_chart as constraints_comparison_chart,
)
from cobre_bridge.comparators.charts.convergence import (
    convergence_chart as convergence_chart,
)
from cobre_bridge.comparators.charts.costs import (
    _COBRE_NON_COST_KEYS as _COBRE_NON_COST_KEYS,
)
from cobre_bridge.comparators.charts.costs import (
    _COST_MAP as _COST_MAP,
)
from cobre_bridge.comparators.charts.costs import (
    _resolve_cost_categories as _resolve_cost_categories,
)
from cobre_bridge.comparators.charts.costs import (
    cost_breakdown_chart as cost_breakdown_chart,
)
from cobre_bridge.comparators.charts.costs import (
    cost_breakdown_table as cost_breakdown_table,
)
from cobre_bridge.comparators.charts.costs import (
    future_cost_chart as future_cost_chart,
)
from cobre_bridge.comparators.charts.costs import (
    immediate_cost_chart as immediate_cost_chart,
)
from cobre_bridge.comparators.charts.costs import (
    other_costs_chart as other_costs_chart,
)
from cobre_bridge.comparators.charts.costs import (
    thermal_cost_chart as thermal_cost_chart,
)
from cobre_bridge.comparators.charts.fpha import (
    fpha_detail_chart as fpha_detail_chart,
)
from cobre_bridge.comparators.charts.fpha import (
    fpha_metrics_table as fpha_metrics_table,
)
from cobre_bridge.comparators.charts.hydro import (
    cobre_aggregate_chart as cobre_aggregate_chart,
)
from cobre_bridge.comparators.charts.hydro import (
    hydro_aggregate_chart as hydro_aggregate_chart,
)
from cobre_bridge.comparators.charts.hydro import (
    hydro_per_bus_chart as hydro_per_bus_chart,
)
from cobre_bridge.comparators.charts.hydro import (
    hydro_slack_aggregate_chart as hydro_slack_aggregate_chart,
)
from cobre_bridge.comparators.charts.hydro import (
    hydro_slack_per_bus_chart as hydro_slack_per_bus_chart,
)
from cobre_bridge.comparators.charts.network import (
    line_summary_chart as line_summary_chart,
)
from cobre_bridge.comparators.charts.overview import (
    overview_metrics as overview_metrics,
)
from cobre_bridge.comparators.charts.performance import (
    performance_fwd_bwd_split_chart as performance_fwd_bwd_split_chart,
)
from cobre_bridge.comparators.charts.performance import (
    performance_iteration_chart as performance_iteration_chart,
)
from cobre_bridge.comparators.charts.performance import (
    performance_metric_cards as performance_metric_cards,
)
from cobre_bridge.comparators.charts.productivity import (
    productivity_blocks_table as productivity_blocks_table,
)
from cobre_bridge.comparators.charts.productivity import (
    productivity_comparison_scatter as productivity_comparison_scatter,
)
from cobre_bridge.comparators.charts.productivity import (
    productivity_per_stage_chart as productivity_per_stage_chart,
)
from cobre_bridge.comparators.charts.spillage import (
    system_spillage_energy_chart as system_spillage_energy_chart,
)
from cobre_bridge.comparators.charts.system import (
    _BALANCE_VARS as _BALANCE_VARS,
)
from cobre_bridge.comparators.charts.system import (
    ree_energy_chart as ree_energy_chart,
)
from cobre_bridge.comparators.charts.system import (
    system_comparison_chart as system_comparison_chart,
)
from cobre_bridge.comparators.charts.system import (
    system_per_bus_chart as system_per_bus_chart,
)
from cobre_bridge.comparators.charts.thermal import (
    thermal_generation_chart as thermal_generation_chart,
)

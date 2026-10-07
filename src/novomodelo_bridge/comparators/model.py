"""Result-comparison dataclasses and summary statistics shared across comparators."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import polars as pl


@dataclass
class PercentileData:
    """Cobre simulation percentile statistics for the report.

    Each DataFrame has columns: entity_id, stage_id, and for each
    variable: ``{var}_p10``, ``{var}_p50``, ``{var}_p90``.
    """

    hydro: pl.DataFrame = field(default_factory=pl.DataFrame)
    thermal: pl.DataFrame = field(default_factory=pl.DataFrame)
    bus: pl.DataFrame = field(default_factory=pl.DataFrame)
    bus_aggregates: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_market: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_net_load: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_sin: pl.DataFrame = field(default_factory=pl.DataFrame)
    cobre_hydro_means: pl.DataFrame = field(default_factory=pl.DataFrame)
    cobre_bus_meta: dict[int, dict] = field(default_factory=dict)
    cobre_hydro_meta: dict[int, dict] = field(default_factory=dict)
    nw_bus_names: dict[int, str] = field(default_factory=dict)
    nw_hydro_names: dict[int, str] = field(default_factory=dict)
    nw_convergence: pl.DataFrame = field(default_factory=pl.DataFrame)
    cobre_convergence: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_costs: dict[str, float] = field(default_factory=dict)
    cobre_costs: dict[str, float] = field(default_factory=dict)
    # Per-stage immediate/future cost on the Cobre side — mean across
    # scenarios, in R$.  Joined with MEDIAS-SIN's COPER/CUSTO_FUTURO
    # (after the 10⁶ R$ unit conversion) by the overview-tab chart.
    cobre_stage_costs: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_offset: int = 0
    nw_max_stage: int | None = None

    # --- line interchange --- Cobre per-(line_id, stage_id) p10/p50/p90 of
    # net_flow_mw across scenarios; bound table from constraints/line_bounds.parquet;
    # metadata list straight from system/lines.json["lines"]. The source model side:
    # mean interchange per (line_id, stage_0based) read from int*.out NWLISTOP files and
    # aligned via EntityAlignment.lines.
    line: pl.DataFrame = field(default_factory=pl.DataFrame)
    line_bounds: pl.DataFrame = field(default_factory=pl.DataFrame)
    line_meta: list[dict] = field(default_factory=list)
    nw_line_means: pl.DataFrame = field(default_factory=pl.DataFrame)

    # --- per-hydro derived flow variables ---
    # Cobre per-(hydro_id, stage_id) total_inflow_m3s (incremental +
    # upstream turbined+spilled) and total_outflow_m3s (turbined + spilled).
    # Outflow comes directly from the parquet ``outflow_m3s`` column when
    # present.
    hydro_total_flows: pl.DataFrame = field(default_factory=pl.DataFrame)

    # --- Per-stage hydro operational bounds (dashboard overlay) ---
    # Per-(entity_id, stage_id) optional bound columns from
    # ``constraints/hydro_bounds.parquet`` — read by
    # ``read_cobre_hydro_per_stage_bounds``.  Columns are nullable per row:
    # NULL means "no per-stage override, fall back to the static value
    # from ``hydros.json``" (carried by ``cobre_hydro_meta``).
    cobre_hydro_per_stage_bounds: pl.DataFrame = field(default_factory=pl.DataFrame)

    # --- the source model hydro slacks (VIOL_POS/NEG_VRETIRUH and VIOL_POS/NEG_EVAP)
    # --- Per-(entity_id, stage_id) flow-domain values for the four the source model
    # hydro slack columns (water-withdrawal pos/neg and evaporation pos/neg), aligned to
    # Cobre IDs and stage 0-base.  Used by the plant-detail tab to overlay the source
    # model on Cobre-only withdrawal slack panels and by the Hydro Operation tab to
    # drive per-bus and SIN-total slack aggregates. Built by
    # :func:`_compute_nw_hydro_slacks`.
    nw_hydro_slacks: pl.DataFrame = field(default_factory=pl.DataFrame)

    # --- system spillage in MWmes ---
    # Per-stage stage-mean MW of system spillage ``spillage_m3s × ρ_eq``,
    # split into total / reservoir / run-of-river via the
    # ``max_storage_hm3 > 0`` discriminator.
    cobre_spillage_energy: pl.DataFrame = field(default_factory=pl.DataFrame)

    # --- Performance / wall-clock timings --- The source model per-iteration
    # ``backward_seconds`` / ``forward_seconds`` / ``total_seconds`` from
    # ``newave.tim``. The source model stage labels (e.g. ``"Tempo Total"``, ``"Calculo
    # da Politica"``) → seconds. Cobre total training duration in seconds from
    # ``output/training/metadata.json``.
    nw_tim_iterations: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_tim_stages: dict[str, float] = field(default_factory=dict)
    cobre_training_seconds: float = 0.0
    cobre_iteration_timing: pl.DataFrame = field(default_factory=pl.DataFrame)

    # --- Generic constraints (RE, AGRINT, VminOP) — LHS comparison --- Constraint
    # definitions (F3, sense-free) and per-(stage, block) bound table (F3 nullable
    # ``bound_lower``/``bound_upper`` endpoints) read straight from the converted
    # Cobre case. ``lhs_newave`` evaluates each constraint's LHS against the source
    # model outputs (MEDIAS-USIH GHIDUH + int*.out interchanges) at stage_0based
    # granularity; ``lhs_cobre`` does the same against Cobre simulation parquets,
    # collapsed to mean across scenarios and blocks.
    gc_constraints: list[dict] = field(default_factory=list)
    gc_bounds: pl.DataFrame = field(default_factory=pl.DataFrame)
    gc_lhs_newave: pl.DataFrame = field(default_factory=pl.DataFrame)
    gc_lhs_cobre: pl.DataFrame = field(default_factory=pl.DataFrame)

    # --- Productivity detail (Productivity tab) --- One row per aligned hydro carrying
    # the source model's head-dependent productivities (altura min/65/max, equivalent,
    # accumulated_earm) plus the HIDR cadastro building blocks (specific productivity,
    # tailwater = canal_fuga_medio, losses, vmin, vmax), alongside Cobre's point /
    # equivalent / accumulated productivities and building blocks.  Built by
    # :func:`cobre_bridge.comparators.analyze.build_productivity_detail`.
    productivity_detail: pl.DataFrame = field(default_factory=pl.DataFrame)

    # --- Production-function (FPHA) comparison (Productivity tab) --- The
    # per-(plant, stage) fidelity metrics plus the dense (V, Q) surfaces and
    # spillage slices both solvers' fitted hyperplanes evaluate to. ``None`` when
    # either side used constant productivity. Built by
    # :func:`cobre_bridge.comparators.analyze.build_fpha_comparison`; render-only
    # (excluded from the serialized artifact — the dense surface is large).
    fpha_metrics: pl.DataFrame | None = None
    fpha_surface: pl.DataFrame | None = None
    fpha_spill: pl.DataFrame | None = None


@dataclass(frozen=True)
class ResultComparison:
    """Single variable comparison result for results comparison."""

    entity_type: str  # "hydro", "thermal", "bus", "convergence", "productivity"
    entity_name: str
    newave_code: int
    cobre_id: int
    stage: int  # 0-indexed (Cobre convention)
    variable: str
    newave_value: float
    cobre_value: float
    abs_diff: float
    rel_diff: float | None  # None when newave_value == 0


@dataclass
class ResultVariableStats:
    """Per-variable comparison statistics."""

    count: int = 0
    mean_abs_diff: float = 0.0
    max_abs_diff: float = 0.0
    mean_rel_diff: float = 0.0
    max_rel_diff: float = 0.0
    within_tol_rate: float = 0.0  # fraction in [0, 1] within --tolerance
    mean_smape: float = 0.0  # symmetric MAPE in [0, 2]
    max_smape: float = 0.0
    correlation: float | None = None  # None when undefined (constant series)


@dataclass
class ResultsSummary:
    """Aggregate results comparison statistics."""

    total: int = 0
    by_entity_type: dict[str, int] = field(default_factory=dict)
    by_variable: dict[str, ResultVariableStats] = field(default_factory=dict)


def smape(nw_value: float, cobre_value: float) -> float:
    """Symmetric mean absolute percentage error for one pair, in [0, 2].

    Robust to a near-zero reference (unlike ``|d| / |nw|``): returns 0 when
    both values are effectively zero (perfect agreement).
    """
    denom = (abs(nw_value) + abs(cobre_value)) / 2.0
    if denom <= 1e-12:
        return 0.0
    return abs(nw_value - cobre_value) / denom


def _within_tolerance(nw_value: float, cobre_value: float, tolerance: float) -> bool:
    """True if Cobre is within ``tolerance`` (relative) of the source model reference.

    When the reference is ~0, counts as a match only if Cobre is also ~0 (both
    effectively zero); otherwise uses ``|nw - cobre| <= tolerance * |nw|``.
    The 1e-9 / 1e-6 floors are absolute and tuned for physical magnitudes
    (MW, m³/s, hm³); adjust if comparing very small-scale quantities.
    """
    denom = abs(nw_value)
    if denom <= 1e-9:
        return abs(cobre_value) <= 1e-6
    return abs(nw_value - cobre_value) <= tolerance * denom


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient; ``None`` when undefined.

    Returns ``None`` for fewer than two points or a constant (zero-variance)
    series, so callers can distinguish "uncomputable" from a genuine 0.0.
    """
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom < 1e-15:
        return None
    return cov / denom


def build_results_summary(
    results: list[ResultComparison], tolerance: float = 1e-2
) -> ResultsSummary:
    """Compute aggregate statistics from comparison results.

    ``tolerance`` is the relative tolerance used for the per-variable
    within-tolerance match rate.
    """
    summary = ResultsSummary(total=len(results))

    # Group by entity type.
    for r in results:
        summary.by_entity_type[r.entity_type] = (
            summary.by_entity_type.get(r.entity_type, 0) + 1
        )

    # Group by variable for stats.
    var_groups: dict[str, list[ResultComparison]] = {}
    for r in results:
        var_groups.setdefault(r.variable, []).append(r)

    for var, group in var_groups.items():
        stats = ResultVariableStats(count=len(group))

        abs_diffs = [r.abs_diff for r in group]
        rel_diffs = [r.rel_diff for r in group if r.rel_diff is not None]

        stats.mean_abs_diff = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0
        stats.max_abs_diff = max(abs_diffs) if abs_diffs else 0.0
        stats.mean_rel_diff = sum(rel_diffs) / len(rel_diffs) if rel_diffs else 0.0
        stats.max_rel_diff = max(rel_diffs) if rel_diffs else 0.0

        # Bounded symmetric error (robust to near-zero references) and the
        # within-tolerance match rate.
        smapes = [smape(r.newave_value, r.cobre_value) for r in group]
        stats.mean_smape = sum(smapes) / len(smapes) if smapes else 0.0
        stats.max_smape = max(smapes) if smapes else 0.0
        n_within = sum(
            1
            for r in group
            if _within_tolerance(r.newave_value, r.cobre_value, tolerance)
        )
        stats.within_tol_rate = n_within / len(group) if group else 0.0

        # Pearson correlation (None when undefined, e.g. a constant series).
        nw_vals = [r.newave_value for r in group]
        cb_vals = [r.cobre_value for r in group]
        if len(nw_vals) > 1:
            stats.correlation = _pearson(nw_vals, cb_vals)

        summary.by_variable[var] = stats

    return summary

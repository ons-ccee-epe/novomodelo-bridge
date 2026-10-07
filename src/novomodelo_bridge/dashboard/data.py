"""Dashboard data layer — loading helpers and DashboardData dataclass.

All data loading previously performed inside ``build_dashboard()`` is
consolidated here into ``DashboardData.load(case_dir)``.  Chart functions
receive a single ``DashboardData`` instance instead of a long list of
individual arguments.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from cobre_bridge.cobre.case_io import resolve_hydro_productivities
from cobre_bridge.cobre.readers import (
    read_cobre_line_bounds,
    read_cobre_lines,
    read_cobre_training_metadata,
)
from cobre_bridge.core.diagnostics import Diagnostic, Severity, emit
from cobre_bridge.core.errors import CobreOutputError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def scan_entity(case_dir: Path, entity: str) -> pl.LazyFrame:
    """Return a LazyFrame scanning all hive-partitioned parquet files for entity.

    Returns an empty LazyFrame when the entity directory is missing or
    contains no parquet files, so callers can consume simulation-less cases
    without triggering polars' "expanded paths were empty" error.
    """
    sim_dir = case_dir / "output" / "simulation" / entity
    if not sim_dir.exists() or not any(sim_dir.rglob("*.parquet")):
        return pl.LazyFrame()
    return pl.scan_parquet(str(sim_dir / "**" / "*.parquet"), hive_partitioning=True)


def load_names(case_dir: Path) -> dict[tuple[str, int], str]:
    """Load entity name mappings from system JSON files."""
    names: dict[tuple[str, int], str] = {}
    for entity in ("hydros", "buses", "thermals", "lines", "non_controllable_sources"):
        path = case_dir / "system" / f"{entity}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            for item in data.get(entity, []):
                names[(entity, item["id"])] = item.get("name", str(item["id"]))
    return names


def entity_name(names: dict[tuple[str, int], str], entity: str, eid: int) -> str:
    return names.get((entity, eid), str(eid))


def load_stage_labels(case_dir: Path) -> dict[int, str]:
    """Return stage_id -> date label from stages.json, keyed on each stage's date.

    Monthly horizons (the source model's typical monthly stages) get a compact
    ``"Mon YYYY"`` label. When a horizon packs several stages into one calendar
    month — e.g. the weekly stages of a DECOMP week/month deck — a month-only
    label would collapse them onto a single x-axis point, so every stage instead
    gets a day-resolution ``"DD Mon YYYY"`` label from its own ``start_date``.
    A stage with a missing or unparseable ``start_date`` falls back to its id.
    """
    path = case_dir / "stages.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)

    parsed: list[tuple[int, pd.Timestamp | None]] = []
    for stage in data.get("stages", []):
        sid = stage["id"]
        start = stage.get("start_date", "")
        stamp: pd.Timestamp | None = None
        if start:
            try:
                stamp = pd.to_datetime(start)
            except (ValueError, TypeError):
                stamp = None
        parsed.append((sid, stamp))

    # If two stages share a calendar month, a month-only label would map them to
    # the same x-axis point; drop to day resolution for the whole axis so each
    # stage stays a distinct, date-accurate point.
    year_months = [(s.year, s.month) for _, s in parsed if s is not None]
    sub_monthly = len(year_months) != len(set(year_months))
    fmt = "%d %b %Y" if sub_monthly else "%b %Y"

    return {
        sid: (stamp.strftime(fmt) if stamp is not None else str(sid))
        for sid, stamp in parsed
    }


def load_stage_dates(case_dir: Path) -> dict[int, str]:
    """Return stage_id -> ISO ``YYYY-MM-DD`` start date from stages.json.

    These are the x-axis *positions* (paired with :func:`load_stage_labels`'s
    display text as tick labels): feeding them to a Plotly ``type="date"`` axis
    spaces the stages by their true calendar distance, so weekly stages sit
    closer together than a following monthly stage. A stage with a missing or
    unparseable ``start_date`` is omitted (its chart point falls back to the
    stage-id label via the axis helper).
    """
    path = case_dir / "stages.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    dates: dict[int, str] = {}
    for stage in data.get("stages", []):
        start = stage.get("start_date", "")
        if not start:
            continue
        try:
            dates[stage["id"]] = pd.to_datetime(start).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return dates


def resolve_hydro_bus_id(hydro: dict, *, hydros_path: Path) -> int | None:
    """Resolve one hydro plant's bus id from its ``unit_groups`` array.

    Single implementation of the pre-0.13 top-level ``hydros.json``
    ``bus_id`` -> 0.13 ``unit_groups[].bus_id`` relocation, shared by
    ``load_hydro_bus_map`` and ``load_hydro_metadata`` so the two never
    disagree on a plant's bus.

    - Every plant either pipeline emits today carries exactly one distinct
      ``bus_id`` across its groups (cobre rule 41's mirror invariant) -- that
      value is returned.
    - A plant recorded at more than one distinct bus is **not** collapsed
      onto an arbitrary group's bus (that would silently mislabel it) and is
      **not** duplicated across every bus it touches either (that would
      double-count it wherever a caller sums per bus) -- the same dilemma the
      compare layer resolved the same way in
      ``comparators.analyze._bus_name_lookups``. Implementation note (a
      deliberate degraded-rendering choice): a ``WARNING``
      :class:`~cobre_bridge.core.diagnostics.Diagnostic` is emitted and ``None``
      is returned. Callers must omit the plant from any single-bus-keyed
      view: ``load_hydro_bus_map``'s id->bus map has no room for more than
      one bus per plant, and ``load_hydro_metadata`` drops the ``"bus_id"``
      key entirely so its existing ``.get("bus_id", <default>)`` call sites
      in the dashboard tabs degrade to "unknown bus" for it. The plant's
      non-bus metadata (name, volumes, productivity, ...) is unaffected and
      keeps rendering in the per-plant tables that are not bus-keyed.
    - Missing or empty ``unit_groups`` raises :class:`CobreOutputError`
      naming the plant. This cannot happen on a real 0.13 case (cobre rejects
      it at load), so it only guards a hand-edited or pre-0.13 file.
    """
    hid = hydro.get("id")
    name = hydro.get("name", str(hid))
    groups = hydro.get("unit_groups")
    if not groups:
        raise CobreOutputError(
            f"hydro {hid} ({name}) has no unit_groups; cannot resolve its bus",
            path=str(hydros_path),
        )
    bus_ids = {g["bus_id"] for g in groups}
    if len(bus_ids) == 1:
        return next(iter(bus_ids))
    emit(
        Diagnostic(
            code="hydro-unit-groups-multi-bus",
            severity=Severity.WARNING,
            category="Dashboard data",
            title="Hydro plant spans multiple buses",
            summary=(
                f"Hydro {hid} ({name}) has unit_groups recorded at buses "
                f"{sorted(bus_ids)}; a single plant-level bus cannot be "
                "chosen without collapsing onto one of them or "
                "double-counting the plant at each, so it is omitted from "
                "bus-keyed dashboard views."
            ),
            notes=[f"hydro_id: {hid}", f"bus_ids: {sorted(bus_ids)}"],
        ),
        logger=logger,
    )
    return None


def load_hydro_bus_map(case_dir: Path) -> dict[int, int]:
    path = case_dir / "system" / "hydros.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    result: dict[int, int] = {}
    for h in d["hydros"]:
        bus_id = resolve_hydro_bus_id(h, hydros_path=path)
        if bus_id is not None:
            result[h["id"]] = bus_id
    return result


def load_thermal_metadata(case_dir: Path) -> dict[int, dict]:
    """Return thermal_id -> {bus_id, max_mw, cost_per_mwh} metadata."""
    path = case_dir / "system" / "thermals.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    result = {}
    for t in d["thermals"]:
        # v0.4.3+: scalar cost_per_mwh; legacy: cost_segments array.
        if "cost_per_mwh" in t:
            cost = float(t["cost_per_mwh"])
        else:
            segments = t.get("cost_segments", [])
            cost = float(segments[0]["cost_per_mwh"]) if segments else 0.0
        cap = t.get("generation", {}).get("max_mw", 0.0)
        result[t["id"]] = {
            "bus_id": t["bus_id"],
            "max_mw": cap,
            "cost_per_mwh": cost,
            "name": t.get("name", str(t["id"])),
        }
    return result


def load_ncs_bus_map(case_dir: Path) -> dict[int, int]:
    path = case_dir / "system" / "non_controllable_sources.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    return {n["id"]: n["bus_id"] for n in d["non_controllable_sources"]}


def load_hydro_metadata(
    case_dir: Path, *, bus_map: dict[int, int] | None = None
) -> dict[int, dict]:
    """Return hydro_id -> {bus_id, name, vol_max, vol_min, max_gen_mw, max_turbined}.

    ``bus_id`` is resolved via :func:`resolve_hydro_bus_id` and is absent from
    a plant's dict when its ``unit_groups`` disagree on a bus (see that
    function's docstring for the degraded-rendering rationale).

    *bus_map*, when supplied, is used instead of re-resolving each plant's
    bus: :func:`load_entity_metadata` computes it once via
    :func:`load_hydro_bus_map` and passes it here, so an ambiguous plant's
    ``hydro-unit-groups-multi-bus`` warning is diagnosed once per case load
    rather than once per loader (``load_hydro_bus_map`` and
    ``load_hydro_metadata`` otherwise each resolve every plant's bus
    independently over the same ``hydros.json``). Called standalone (the
    default, *bus_map* omitted) this still resolves each plant's bus itself,
    exactly as before.

    Productivity is read from ``hydro_energy_productivity.parquet`` (the
    cobre productivity-resolution-rules contract). Falls back through
    ``hydro_production_models.json`` and then the deprecated
    ``hydros.json:generation.productivity_mw_per_m3s`` field so older
    converted cases still render.
    """
    path = case_dir / "system" / "hydros.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    # Share the single canonical productivity cascade with the comparator so
    # the dashboard and compare report never disagree on a plant's ρ.
    productivities = resolve_hydro_productivities(case_dir, d["hydros"])
    result = {}
    for h in d["hydros"]:
        gen = h.get("generation", {})
        res = h.get("reservoir", {})
        prod = productivities.get(h["id"]) or 0
        max_turbined = gen.get("max_turbined_m3s", 0)
        bus_id = (
            bus_map.get(h["id"])
            if bus_map is not None
            else resolve_hydro_bus_id(h, hydros_path=path)
        )
        entry: dict = {
            "name": h.get("name", str(h["id"])),
            "vol_max": res.get("max_storage_hm3", 0),
            "vol_min": res.get("min_storage_hm3", 0),
            "max_gen_mw": gen.get("max_generation_mw", 0),
            "max_gen_physical": prod * max_turbined,
            "max_turbined": max_turbined,
            "productivity": prod,
            "downstream_id": h.get("downstream_id"),
        }
        if bus_id is not None:
            entry["bus_id"] = bus_id
        result[h["id"]] = entry
    return result


# ---------------------------------------------------------------------------
# Polars helpers
# ---------------------------------------------------------------------------


def _stage_avg_mw(
    lf: pl.LazyFrame,
    mwh_col: str,
    stage_hours: dict[int, float],
) -> dict[int, float]:
    """Compute stage-average MW from MWh summed across all blocks via LazyFrame.

    Scans all blocks (no block_id filter), sums ``mwh_col`` per
    (scenario, stage), divides by total stage hours, then averages across
    scenarios.

    Returns ``{stage_id: avg_mw}``.
    """
    hours_df = pl.DataFrame(
        {"stage_id": list(stage_hours.keys()), "_hours": list(stage_hours.values())}
    )
    result = (
        lf.group_by(["scenario_id", "stage_id"])
        .agg(pl.col(mwh_col).sum())
        .join(hours_df.lazy(), on="stage_id")
        .with_columns((pl.col(mwh_col) / pl.col("_hours")).alias("_avg_mw"))
        .group_by("stage_id")
        .agg(pl.col("_avg_mw").mean())
        .sort("stage_id")
        .collect(engine="streaming")
    )
    return dict(zip(result["stage_id"].to_list(), result["_avg_mw"].to_list()))


def _compute_lp_load(
    load_stats: pd.DataFrame,
    load_factors: list[dict],
    stage_hours: dict[int, float],
    block_hours: dict[tuple[int, int], float],
    bus_filter: list[int] | None = None,
) -> dict[int, float]:
    """Compute LP load balance RHS from input files as stage-average MW."""
    factor_lk: dict[tuple[int, int, int], float] = {}
    for entry in load_factors:
        bid, sid = entry["bus_id"], entry["stage_id"]
        for bf in entry["block_factors"]:
            factor_lk[(bid, sid, bf["block_id"])] = bf["factor"]

    rows: list[dict] = []
    for _, r in load_stats.iterrows():
        bid, sid = int(r["bus_id"]), int(r["stage_id"])
        if bus_filter is not None and bid not in bus_filter:
            continue
        mean = float(r["mean_mw"])
        for blk_id in range(20):
            bh = block_hours.get((sid, blk_id))
            if bh is None:
                break
            factor = factor_lk.get((bid, sid, blk_id), 1.0)
            rows.append(
                {"scenario_id": 0, "stage_id": sid, "_lp_mwh": mean * factor * bh}
            )

    if not rows:
        return {}
    df = pd.DataFrame(rows)
    hours_map = {
        sid: stage_hours[sid] for sid in df["stage_id"].unique() if sid in stage_hours
    }
    scen_stage = df.groupby(["scenario_id", "stage_id"])["_lp_mwh"].sum().reset_index()
    scen_stage["_hours"] = scen_stage["stage_id"].map(hours_map)
    scen_stage["_avg_mw"] = scen_stage["_lp_mwh"] / scen_stage["_hours"]
    ser = scen_stage.groupby("stage_id")["_avg_mw"].mean()
    return dict(zip(ser.index.tolist(), ser.values.tolist()))


def _aggregate_timing_by_iteration(timing_raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse the multi-row-per-iteration timing shape.

    Newer cobre (``training/timing/iterations.parquet``) emits one
    rank-aggregated row plus N per-worker rows per iteration — per-worker
    rows only carry fwd/bwd wall + setup, rank rows carry everything else.
    ``SUM(col) GROUP BY iteration`` reconstructs the single-row totals that
    the single-row chart code expects. Legacy frames (no ``rank``/
    ``worker_id`` columns) pass through unchanged.

    The sum semantics are correct for the rank-only columns (one contributing
    row per iteration) and for the aggregate per-worker CPU columns
    (``fwd_setup_ms``, ``bwd_setup_ms`` — meant to be summed).  It is
    *incorrect* for ``forward_wall_ms`` / ``backward_wall_ms`` because the
    per-worker rows each carry a per-worker clock, not a wall-clock.  Pair
    this aggregation with :func:`_correct_wall_times_from_convergence` to
    overwrite those two columns with true wall-clock values.
    """
    if timing_raw.empty:
        return timing_raw
    if "rank" not in timing_raw.columns and "worker_id" not in timing_raw.columns:
        return timing_raw
    numeric_cols = [c for c in timing_raw.columns if c.endswith("_ms")]
    if not numeric_cols:
        return timing_raw
    return (
        timing_raw.groupby("iteration", as_index=False)[numeric_cols]
        .sum()
        .sort_values("iteration")
        .reset_index(drop=True)
    )


def _correct_wall_times_from_convergence(
    timing: pd.DataFrame, conv: pd.DataFrame
) -> pd.DataFrame:
    """Overwrite per-iteration ``forward_wall_ms`` / ``backward_wall_ms`` with
    true wall-clock values from ``convergence.parquet``.

    The naive ``SUM`` in :func:`_aggregate_timing_by_iteration` turns
    per-worker wall time into ``N × wall_clock`` for an N-thread run. The
    authoritative per-iteration wall-clock lives in ``convergence.parquet``
    as ``time_forward_ms`` / ``time_backward_ms`` (captured on rank 0 around
    the forward / backward entry points in cobre ``training.rs``).

    No-op when ``conv`` is empty, lacks the required columns, or ``timing``
    is empty — the legacy single-row-per-iteration format reaches this path
    with already-correct values, so overwriting with identical numbers is a
    safe default.
    """
    if timing.empty or conv.empty:
        return timing
    if "iteration" not in timing.columns:
        return timing
    required_conv = {"iteration", "time_forward_ms", "time_backward_ms"}
    if not required_conv.issubset(conv.columns):
        return timing

    corrected = timing.copy()
    conv_wall = conv.set_index("iteration")[["time_forward_ms", "time_backward_ms"]]

    for wall_col, conv_col in (
        ("forward_wall_ms", "time_forward_ms"),
        ("backward_wall_ms", "time_backward_ms"),
    ):
        if wall_col not in corrected.columns:
            continue
        mapped = corrected["iteration"].map(conv_wall[conv_col])
        merged = mapped.fillna(corrected[wall_col])
        corrected[wall_col] = merged.astype(corrected[wall_col].dtype)
    return corrected


def compute_non_fictitious_bus_ids(load_stats: pd.DataFrame) -> list[int]:
    """Return sorted bus IDs that have mean_mw > 0 in at least one stage.

    A fictitious bus is defined as one whose mean load is zero across all
    stages in the input load seasonal stats.  Tabs that show per-bus facets
    should filter to this list so that accounting-only buses (e.g. NOFICT1)
    are excluded.

    Args:
        load_stats: DataFrame with columns ``bus_id``, ``stage_id``, ``mean_mw``.

    Returns:
        Sorted list of integer bus IDs with at least one stage where
        ``mean_mw > 0``.  Returns an empty list when ``load_stats`` is empty
        or the ``mean_mw`` column is absent.
    """
    if load_stats.empty or "mean_mw" not in load_stats.columns:
        return []
    positive = load_stats[load_stats["mean_mw"] > 0]
    return sorted(positive["bus_id"].unique().tolist())


# ---------------------------------------------------------------------------
# Section loaders
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TemporalContext:
    """Run configuration and stage/block time resolution."""

    config: dict
    discount_rate: float
    stages_data: dict
    stage_hours: dict[int, float]
    block_hours: dict[tuple[int, int], float]

    # Block-hours join key for weighted-average chart computations.
    # Columns: ["stage_id", "block_id", "_bh"]
    bh_df: pl.DataFrame
    line_meta: list[dict]


def load_temporal_context(case_dir: Path) -> TemporalContext:
    """Load config.json / stages.json and the derived stage & block hour maps."""
    config_path = case_dir / "config.json"
    config: dict = {}
    if config_path.exists():
        try:
            with config_path.open() as f:
                config = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Failed to parse config.json; using empty dict")
    discount_rate = float(config.get("discount_rate", 0.0))

    stages_json_path = case_dir / "stages.json"
    with stages_json_path.open() as f:
        stages_data = json.load(f)

    # stages.json policy_graph is the authoritative discount rate when present.
    pg_rate = stages_data.get("policy_graph", {}).get("annual_discount_rate")
    if pg_rate is not None:
        discount_rate = float(pg_rate)

    # A non-empty node list is a scenario tree with per-branch probabilities;
    # every mean/quantile downstream still averages leaf paths uniformly (full
    # probability-weighting is a separate follow-up), so the reader needs a
    # signal that the aggregates are not weighted expectations.
    nodes = stages_data.get("policy_graph", {}).get("nodes")
    if nodes:
        emit(
            Diagnostic(
                code="dashboard-unweighted-tree-averages",
                severity=Severity.WARNING,
                category="Dashboard data",
                title="Scenario tree averages are not probability-weighted",
                summary=(
                    "This case is built from a scenario tree whose branches "
                    "occur with different probabilities, but every mean and "
                    "percentile band in this dashboard treats each scenario "
                    "path with equal weight. The reported statistics are "
                    "unweighted averages across scenario paths, not the "
                    "probability-weighted expectations the underlying "
                    "optimization targets."
                ),
                remediation=(
                    "Read the displayed means and percentile bands as "
                    "unweighted scenario-path averages, not probability-"
                    "weighted expectations."
                ),
            ),
            logger=logger,
        )

    stage_hours: dict[int, float] = {}
    for s in stages_data["stages"]:
        stage_hours[s["id"]] = sum(b["hours"] for b in s["blocks"])

    block_hours: dict[tuple[int, int], float] = {}
    for s in stages_data["stages"]:
        for b in s["blocks"]:
            block_hours[(s["id"], b["id"])] = b["hours"]

    bh_keys = list(block_hours.keys())
    bh_df = pl.DataFrame(
        {
            "stage_id": [k[0] for k in bh_keys],
            "block_id": [k[1] for k in bh_keys],
            "_bh": [block_hours[k] for k in bh_keys],
        }
    )

    line_meta: list[dict] = read_cobre_lines(case_dir / "output")

    return TemporalContext(
        config=config,
        discount_rate=discount_rate,
        stages_data=stages_data,
        stage_hours=stage_hours,
        block_hours=block_hours,
        bh_df=bh_df,
        line_meta=line_meta,
    )


@dataclasses.dataclass
class EntityMetadata:
    """Entity name/metadata lookups derived from the system JSON files."""

    names: dict[tuple[str, int], str]
    stage_labels: dict[int, str]
    stage_dates: dict[int, str]
    hydro_bus_map: dict[int, int]
    thermal_meta: dict[int, dict]
    ncs_bus_map: dict[int, int]
    hydro_meta: dict[int, dict]
    bus_names: dict[int, str]


def load_entity_metadata(case_dir: Path) -> EntityMetadata:
    """Load entity name maps and hydro/thermal metadata dictionaries.

    ``hydro_bus_map`` is resolved once via :func:`load_hydro_bus_map` and
    reused by :func:`load_hydro_metadata` (both loaders resolve
    every plant's bus over the same ``system/hydros.json``; without sharing
    the map, an ambiguous plant's ``hydro-unit-groups-multi-bus`` warning
    fires twice per dashboard build instead of once).
    """
    names = load_names(case_dir)
    bus_names = {eid: nm for (entity, eid), nm in names.items() if entity == "buses"}
    hydro_bus_map = load_hydro_bus_map(case_dir)
    return EntityMetadata(
        names=names,
        stage_labels=load_stage_labels(case_dir),
        stage_dates=load_stage_dates(case_dir),
        hydro_bus_map=hydro_bus_map,
        thermal_meta=load_thermal_metadata(case_dir),
        ncs_bus_map=load_ncs_bus_map(case_dir),
        hydro_meta=load_hydro_metadata(case_dir, bus_map=hydro_bus_map),
        bus_names=bus_names,
    )


@dataclasses.dataclass
class SimulationData:
    """Simulation outputs: per-entity LazyFrames plus collected cost/violation
    frames. Empty frames when the case is training-only."""

    hydros_lf: pl.LazyFrame
    thermals_lf: pl.LazyFrame
    ncs_lf: pl.LazyFrame
    buses_lf: pl.LazyFrame
    exchanges_lf: pl.LazyFrame
    inflow_lags_lf: pl.LazyFrame
    costs: pd.DataFrame
    gc_violations: pd.DataFrame


def load_simulation_data(case_dir: Path) -> SimulationData:
    """Scan the hive-partitioned simulation outputs (LazyFrames) and collect the
    small cost/violation frames consumed eagerly by chart functions."""
    hydros_lf = scan_entity(case_dir, "hydros")
    thermals_lf = scan_entity(case_dir, "thermals")
    ncs_lf = scan_entity(case_dir, "non_controllables")
    buses_lf = scan_entity(case_dir, "buses")
    exchanges_lf = scan_entity(case_dir, "exchanges")

    inflow_lags_dir = case_dir / "output" / "simulation" / "inflow_lags"
    if inflow_lags_dir.exists():
        inflow_lags_lf = pl.scan_parquet(
            str(inflow_lags_dir / "**" / "*.parquet"), hive_partitioning=True
        )
    else:
        inflow_lags_lf = pl.LazyFrame()

    # Costs is stage-level only (~236 K rows total), collect to pandas for
    # chart_cost_* functions. Empty when the case is training-only.
    costs_dir = case_dir / "output" / "simulation" / "costs"
    if costs_dir.exists() and any(costs_dir.rglob("*.parquet")):
        costs = (
            pl.scan_parquet(str(costs_dir / "**" / "*.parquet"), hive_partitioning=True)
            .collect(engine="streaming")
            .to_pandas()
        )
    else:
        costs = pd.DataFrame()

    # violations/generic: collect to pandas once (needed by
    # build_constraints_summary_table).
    gc_viol_dir = case_dir / "output" / "simulation" / "violations" / "generic"
    if gc_viol_dir.exists():
        gc_violations = (
            pl.scan_parquet(
                str(gc_viol_dir / "**" / "*.parquet"), hive_partitioning=True
            )
            .collect(engine="streaming")
            .to_pandas()
        )
    else:
        gc_violations = pd.DataFrame()

    return SimulationData(
        hydros_lf=hydros_lf,
        thermals_lf=thermals_lf,
        ncs_lf=ncs_lf,
        buses_lf=buses_lf,
        exchanges_lf=exchanges_lf,
        inflow_lags_lf=inflow_lags_lf,
        costs=costs,
        gc_violations=gc_violations,
    )


@dataclasses.dataclass
class ScenarioInputs:
    """Optional Cobre scenario/constraint input artifacts (empty when absent)."""

    load_stats: pd.DataFrame
    load_factors_list: list[dict]
    non_fictitious_bus_ids: list[int]
    ncs_stats: pd.DataFrame
    inflow_history: pd.DataFrame
    line_bounds: pd.DataFrame
    line_block_bounds: pd.DataFrame
    hydro_bounds: pd.DataFrame
    thermal_bounds: pd.DataFrame


def load_scenario_inputs(case_dir: Path) -> ScenarioInputs:
    """Load the optional scenario inputs and input constraint bounds.

    Both load files are optional Cobre inputs (see case-format.md); they fall
    back to empty structures so cases without an explicit load scenario still
    render — charts degrade to a zero-load series via ``_compute_lp_load``.
    """
    ls_path = case_dir / "scenarios" / "load_seasonal_stats.parquet"
    load_stats = (
        pq.read_table(ls_path).to_pandas() if ls_path.exists() else pd.DataFrame()
    )
    lf_path = case_dir / "scenarios" / "load_factors.json"
    load_factors_list: list[dict] = []
    if lf_path.exists():
        with lf_path.open() as f:
            load_factors_list = json.load(f)["load_factors"]

    non_fictitious_bus_ids = compute_non_fictitious_bus_ids(load_stats)

    ncs_stats_path = case_dir / "scenarios" / "non_controllable_stats.parquet"
    ncs_stats = (
        pq.read_table(ncs_stats_path).to_pandas()
        if ncs_stats_path.exists()
        else pd.DataFrame()
    )

    ih_path = case_dir / "scenarios" / "inflow_history.parquet"
    inflow_history = (
        pq.read_table(ih_path).to_pandas() if ih_path.exists() else pd.DataFrame()
    )

    line_bounds = read_cobre_line_bounds(case_dir / "output").to_pandas()
    # ``block_id`` is non-null only on the absolute-MW per-block override rows
    # (the exchange factor lives there now), never on the stage-level base row.
    # Do not reconstruct a factor by dividing back through the base — that
    # reintroduces the removed division and can divide by a zero base. A
    # uniform-block line-stage legitimately has no override row, so an empty
    # frame here is a correct steady state, not an error.
    line_block_bounds = (
        line_bounds[line_bounds["block_id"].notna()].reset_index(drop=True)
        if "block_id" in line_bounds.columns
        else pd.DataFrame()
    )
    hb_path = case_dir / "constraints" / "hydro_bounds.parquet"
    hydro_bounds = (
        pq.read_table(hb_path).to_pandas() if hb_path.exists() else pd.DataFrame()
    )
    tb_path = case_dir / "constraints" / "thermal_bounds.parquet"
    thermal_bounds = (
        pq.read_table(tb_path).to_pandas() if tb_path.exists() else pd.DataFrame()
    )

    return ScenarioInputs(
        load_stats=load_stats,
        load_factors_list=load_factors_list,
        non_fictitious_bus_ids=non_fictitious_bus_ids,
        ncs_stats=ncs_stats,
        inflow_history=inflow_history,
        line_bounds=line_bounds,
        line_block_bounds=line_block_bounds,
        hydro_bounds=hydro_bounds,
        thermal_bounds=thermal_bounds,
    )


@dataclasses.dataclass
class SolverPerformance:
    """Training/simulation solver timing and diagnostics (optional)."""

    # ``timing`` is aggregated to one row per iteration for backward
    # compatibility with older chart code. ``timing_raw`` retains the
    # multi-row-per-iteration view from newer cobre outputs, where each
    # iteration has one rank-aggregated row (``worker_id`` NULL) and N
    # per-worker rows (``worker_id`` populated) — see
    # ``build_worker_timing_records`` in cobre-sddp training_output.rs.
    timing: pd.DataFrame
    timing_raw: pd.DataFrame
    solver_train: pd.DataFrame
    solver_sim: pd.DataFrame
    scaling_report: dict
    cut_selection: pd.DataFrame
    retry_histogram: pd.DataFrame
    lp_bounds: pd.DataFrame


#: Renames cobre 0.14's canonical diagnostic-output axes back to the pre-0.14
#: spellings the dashboard's chart layer was written against, applied at this
#: single load choke point (mirroring
#: :func:`cobre_bridge.cobre.readers.read_cobre_convergence`).
_OUTPUT_COLUMN_ALIASES: dict[str, str] = {
    "stage_id": "stage",
    "opening_index": "opening",
    "upper_bound": "upper_bound_mean",
}


def _normalize_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename cobre 0.14 canonical diagnostic columns to the dashboard's names.

    A 0.14 column is renamed only when its legacy target is absent, so a
    pre-0.14 frame (legacy names already present) is returned unchanged, and a
    frame that shares a source name for a different concept it keeps canonical
    elsewhere (e.g. ``bounds.parquet``'s ``stage_id``) is deliberately NOT run
    through here. The ``-1`` -> ``NULL`` marker change needs no handling: a null
    stage/opening reaches pandas as ``NaN``, and the chart layer's
    ``stage >= 0`` filters drop those rows exactly as they dropped the old
    ``-1`` rows, while its ``opening.notna()`` checks rely on the null directly.
    """
    if df.empty:
        return df
    rename = {
        src: dst
        for src, dst in _OUTPUT_COLUMN_ALIASES.items()
        if src in df.columns and dst not in df.columns
    }
    return df.rename(columns=rename) if rename else df


def load_solver_performance(case_dir: Path, conv: pd.DataFrame) -> SolverPerformance:
    """Load solver/performance artifacts. ``conv`` (convergence) is needed to
    correct the timing wall-times. All frames fall back to empty when absent."""
    timing_path = case_dir / "output" / "training" / "timing" / "iterations.parquet"
    timing_raw = (
        pq.read_table(timing_path).to_pandas()
        if timing_path.exists()
        else pd.DataFrame()
    )
    timing = _aggregate_timing_by_iteration(timing_raw)
    timing = _correct_wall_times_from_convergence(timing, conv)

    solver_train_path = (
        case_dir / "output" / "training" / "solver" / "iterations.parquet"
    )
    solver_train = _normalize_output_columns(
        pq.read_table(solver_train_path).to_pandas()
        if solver_train_path.exists()
        else pd.DataFrame()
    )
    sim_solver_path = (
        case_dir / "output" / "simulation" / "solver" / "iterations.parquet"
    )
    solver_sim = _normalize_output_columns(
        pq.read_table(sim_solver_path).to_pandas()
        if sim_solver_path.exists()
        else pd.DataFrame()
    )
    scaling_path = case_dir / "output" / "training" / "scaling_report.json"
    if scaling_path.exists():
        with scaling_path.open() as f:
            scaling_report: dict = json.load(f)
    else:
        scaling_report = {}
    cs_path = case_dir / "output" / "training" / "cut_selection" / "iterations.parquet"
    cut_selection = _normalize_output_columns(
        pq.read_table(cs_path).to_pandas() if cs_path.exists() else pd.DataFrame()
    )
    retry_path = case_dir / "output" / "training" / "solver" / "retry_histogram.parquet"
    retry_histogram = _normalize_output_columns(
        pq.read_table(retry_path).to_pandas() if retry_path.exists() else pd.DataFrame()
    )
    bounds_path = case_dir / "output" / "training" / "dictionaries" / "bounds.parquet"
    lp_bounds = (
        pq.read_table(bounds_path).to_pandas()
        if bounds_path.exists()
        else pd.DataFrame()
    )
    return SolverPerformance(
        timing=timing,
        timing_raw=timing_raw,
        solver_train=solver_train,
        solver_sim=solver_sim,
        scaling_report=scaling_report,
        cut_selection=cut_selection,
        retry_histogram=retry_histogram,
        lp_bounds=lp_bounds,
    )


@dataclasses.dataclass
class StochasticData:
    """Stochastic model fitting outputs (empty/absent for training-less cases)."""

    available: bool
    inflow_stats_stoch: pd.DataFrame
    ar_coefficients: pd.DataFrame
    noise_openings: pd.DataFrame
    fitting_report: dict
    correlation: dict


def load_stochastic_data(case_dir: Path) -> StochasticData:
    """Load output/stochastic/* (inflow fit, AR coefficients, correlation).

    Every file below is optional: a directory that exists but is missing one
    of them (e.g. a fit with no AR-coefficient file) degrades that one field
    to an empty DataFrame/dict with a WARNING instead of raising.
    """
    stochastic_dir = case_dir / "output" / "stochastic"
    available = stochastic_dir.exists()
    if available:
        stats_path = stochastic_dir / "inflow_seasonal_stats.parquet"
        if stats_path.exists():
            inflow_stats_stoch = pq.read_table(stats_path).to_pandas()
        else:
            logger.warning(
                "output/stochastic/inflow_seasonal_stats.parquet missing; "
                "using empty DataFrame"
            )
            inflow_stats_stoch = pd.DataFrame()

        ar_path = stochastic_dir / "inflow_ar_coefficients.parquet"
        if ar_path.exists():
            ar_coefficients = pq.read_table(ar_path).to_pandas()
        else:
            logger.warning(
                "output/stochastic/inflow_ar_coefficients.parquet missing; "
                "using empty DataFrame"
            )
            ar_coefficients = pd.DataFrame()

        noise_path = stochastic_dir / "noise_openings.parquet"
        if noise_path.exists():
            noise_openings = pq.read_table(noise_path).to_pandas()
        else:
            logger.warning(
                "output/stochastic/noise_openings.parquet missing; using "
                "empty DataFrame"
            )
            noise_openings = pd.DataFrame()

        report_path = stochastic_dir / "fitting_report.json"
        if report_path.exists():
            fitting_report: dict = json.load(report_path.open())
        else:
            logger.warning(
                "output/stochastic/fitting_report.json missing; using empty dict"
            )
            fitting_report = {}

        corr_path = stochastic_dir / "correlation.json"
        if corr_path.exists():
            correlation: dict = json.load(corr_path.open())
        else:
            logger.warning(
                "output/stochastic/correlation.json missing; using empty dict"
            )
            correlation = {}
    else:
        inflow_stats_stoch = pd.DataFrame()
        ar_coefficients = pd.DataFrame()
        noise_openings = pd.DataFrame()
        fitting_report = {}
        correlation = {}
    return StochasticData(
        available=available,
        inflow_stats_stoch=inflow_stats_stoch,
        ar_coefficients=ar_coefficients,
        noise_openings=noise_openings,
        fitting_report=fitting_report,
        correlation=correlation,
    )


#: Every exception type an ``output/policy`` checkpoint read can fail with,
#: mirroring ``decomp.fcf.capability``'s CBVF round-trip probe: cobre absent
#: or missing the ``results`` binding, a malformed reloaded pool/dict, or any
#: ``cobre.errors.CobreError`` leaf (each subclasses one of ``ValueError``/
#: ``OSError``/``RuntimeError``). Never a bare ``except``.
_POLICY_READ_FAILURE_TYPES: tuple[type[Exception], ...] = (
    ModuleNotFoundError,
    AttributeError,
    KeyError,
    TypeError,
    ValueError,
    OSError,
    RuntimeError,
)


def _load_policy_metadata(case_dir: Path) -> dict:
    """Load ``output/policy``'s terminal ``state_dimension`` via cobre.

    Returns ``{}`` immediately when ``case_dir/output/policy`` does not
    exist, without importing cobre. Otherwise imports cobre lazily (kept out
    of module scope so this module stays importable in a cobre-free
    environment), reads the checkpoint via ``cobre.results.load_policy``,
    and selects the terminal pool (max ``stage_id``). Degrades to ``{}``
    with one ``logger.warning`` on any failure in
    :data:`_POLICY_READ_FAILURE_TYPES` — a training-only case with no (or an
    unreadable) policy output must still render.
    """
    if not (case_dir / "output" / "policy").exists():
        return {}
    try:
        import cobre

        policy = cobre.results.load_policy(case_dir / "output", policy_subdir="policy")
        terminal = max(policy["stage_cuts"], key=lambda stage: stage["stage_id"])
        return {"state_dimension": int(terminal["state_dimension"])}
    except _POLICY_READ_FAILURE_TYPES as exc:
        logger.warning("output/policy metadata could not be loaded: %s", exc)
        return {}


@dataclasses.dataclass
class OutputMetadata:
    """Per-``output/`` subdirectory metadata: ``training``/``simulation`` from
    each ``metadata.json``, ``policy`` from the self-describing ``manifest.bin``
    checkpoint (see :func:`load_output_metadata`)."""

    training: dict
    simulation: dict
    policy: dict


def load_output_metadata(case_dir: Path) -> OutputMetadata:
    """Load output/{training,simulation,policy} metadata (empty if absent).

    ``training``/``simulation`` still read the sibling ``metadata.json``
    file; ``policy`` reads the self-describing ``manifest.bin`` checkpoint
    instead (see :func:`_load_policy_metadata`) — that subdirectory carries
    no ``metadata.json`` of its own.
    """

    def _load_metadata(subdir: str) -> dict:
        meta_path = case_dir / "output" / subdir / "metadata.json"
        if not meta_path.exists():
            return {}
        try:
            with meta_path.open() as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Failed to parse output/%s/metadata.json", subdir)
            return {}

    return OutputMetadata(
        training=read_cobre_training_metadata(case_dir / "output"),
        simulation=_load_metadata("simulation"),
        policy=_load_policy_metadata(case_dir),
    )


@dataclasses.dataclass
class GenericConstraintData:
    """Generic-constraint definitions and their resolved bounds (optional).

    F3 shape: ``constraints`` dicts carry no ``sense`` key, and ``bounds``
    has nullable ``bound_lower``/``bound_upper`` endpoint columns instead of
    a single ``bound`` column (see
    :mod:`cobre_bridge.core.generic_constraint_format`). Readers derive the
    displayed direction from the endpoints via
    :func:`cobre_bridge.dashboard.tabs.constraints_utils.derive_constraint_shape`.
    """

    constraints: list[dict]
    bounds: pd.DataFrame


def load_generic_constraints(case_dir: Path) -> GenericConstraintData:
    """Load constraints/generic_constraints.json and the bounds parquet.

    The bounds parquet is read verbatim (whatever columns the converter
    wrote — the F3 ``bound_lower``/``bound_upper`` pair, not a single
    ``bound`` column); no column selection happens here.
    """
    gc_path = case_dir / "constraints" / "generic_constraints.json"
    constraints: list[dict] = []
    if gc_path.exists():
        with gc_path.open() as f:
            gc_data = json.load(f)
        constraints = gc_data.get("constraints", [])

    gc_bounds_path = case_dir / "constraints" / "generic_constraint_bounds.parquet"
    bounds = (
        pq.read_table(gc_bounds_path).to_pandas()
        if gc_bounds_path.exists()
        else pd.DataFrame()
    )
    return GenericConstraintData(constraints=constraints, bounds=bounds)


# ---------------------------------------------------------------------------
# DashboardData dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DashboardData:
    """All data sources required to render the Cobre dashboard.

    Holds the eight ``load_*``-produced sub-structs verbatim (``temporal``,
    ``entities``, ``simulation``, ``scenario``, ``performance``,
    ``stochastic``, ``metadata``, ``gc``) plus the handful of values derived
    at load time that belong to none of them. Every historical flat
    ``data.<field>`` name a tab module reads is exposed as a read-only
    ``@property`` forwarding to its sub-struct, so tab code keeps working
    unchanged. Populated via ``DashboardData.load(case_dir)``.
    """

    case_dir: Path
    case_name: str

    # Training — small single-file parquet loaded eagerly as pandas.
    conv: pd.DataFrame

    # True when ``output/simulation/costs/`` contains parquet files. When
    # ``False`` the simulation-dependent tabs (overview, energy balance,
    # costs, plants, network, constraints) are hidden.
    simulation_available: bool
    n_scenarios: int
    n_stages: int

    temporal: TemporalContext
    entities: EntityMetadata
    simulation: SimulationData
    scenario: ScenarioInputs
    performance: SolverPerformance
    stochastic: StochasticData
    metadata: OutputMetadata
    gc: GenericConstraintData

    # ---------------------------------------------------------------------------
    # Factory
    # ---------------------------------------------------------------------------

    @classmethod
    def load(cls, case_dir: Path) -> DashboardData:
        """Load all dashboard data from a Cobre case directory.

        Thin orchestrator: each cohesive section is read by a dedicated
        ``load_*`` loader returning a sub-struct, which :class:`DashboardData`
        holds directly (see the flat-field ``@property`` shims below). Raises
        ``FileNotFoundError`` if required files (convergence.parquet,
        stages.json) are missing; optional files fall back to empty
        DataFrames / empty dicts inside each loader.
        """
        logger.info("Loading dashboard data from %s", case_dir)

        # Training convergence is required and feeds the timing wall-time fix.
        conv = _normalize_output_columns(
            pq.read_table(
                case_dir / "output" / "training" / "convergence.parquet"
            ).to_pandas()
        )

        temporal = load_temporal_context(case_dir)
        entities = load_entity_metadata(case_dir)
        simulation = load_simulation_data(case_dir)
        scenario = load_scenario_inputs(case_dir)
        performance = load_solver_performance(case_dir, conv)
        stochastic = load_stochastic_data(case_dir)
        metadata = load_output_metadata(case_dir)
        gc = load_generic_constraints(case_dir)

        case_name = case_dir.resolve().name
        costs = simulation.costs
        simulation_available = not costs.empty
        if simulation_available:
            n_scenarios = int(costs["scenario_id"].nunique())
            n_stages = int(costs["stage_id"].nunique())
        else:
            n_scenarios = 0
            n_stages = len(temporal.stages_data.get("stages", []))
        logger.info(
            "Loaded %d scenarios, %d stages, %d stage labels%s",
            n_scenarios,
            n_stages,
            len(entities.stage_labels),
            "" if simulation_available else " (training-only; no simulation)",
        )

        # Filter simulation solver to actual scenario count from metadata,
        # folded into the held struct so data.solver_sim and
        # data.performance.solver_sim never diverge.
        actual_sim_scenarios = metadata.simulation.get("scenarios", {}).get(
            "completed", n_scenarios
        )
        if not performance.solver_sim.empty:
            performance = dataclasses.replace(
                performance,
                solver_sim=performance.solver_sim.head(actual_sim_scenarios),
            )

        return cls(
            case_dir=case_dir,
            case_name=case_name,
            conv=conv,
            simulation_available=simulation_available,
            n_scenarios=n_scenarios,
            n_stages=n_stages,
            temporal=temporal,
            entities=entities,
            simulation=simulation,
            scenario=scenario,
            performance=performance,
            stochastic=stochastic,
            metadata=metadata,
            gc=gc,
        )

    # ---------------------------------------------------------------------------
    # Flat-field shims — read-only pass-throughs onto the held sub-structs
    # above, preserved so every existing tab module keeps reading
    # ``data.<field>`` without change.
    # ---------------------------------------------------------------------------

    @property
    def hydros_lf(self) -> pl.LazyFrame:
        return self.simulation.hydros_lf

    @property
    def thermals_lf(self) -> pl.LazyFrame:
        return self.simulation.thermals_lf

    @property
    def ncs_lf(self) -> pl.LazyFrame:
        return self.simulation.ncs_lf

    @property
    def buses_lf(self) -> pl.LazyFrame:
        return self.simulation.buses_lf

    @property
    def exchanges_lf(self) -> pl.LazyFrame:
        return self.simulation.exchanges_lf

    @property
    def costs(self) -> pd.DataFrame:
        return self.simulation.costs

    @property
    def line_bounds(self) -> pd.DataFrame:
        return self.scenario.line_bounds

    @property
    def names(self) -> dict[tuple[str, int], str]:
        return self.entities.names

    @property
    def stage_labels(self) -> dict[int, str]:
        return self.entities.stage_labels

    @property
    def stage_dates(self) -> dict[int, str]:
        return self.entities.stage_dates

    @property
    def hydro_bus_map(self) -> dict[int, int]:
        return self.entities.hydro_bus_map

    @property
    def thermal_meta(self) -> dict[int, dict]:
        return self.entities.thermal_meta

    @property
    def ncs_bus_map(self) -> dict[int, int]:
        return self.entities.ncs_bus_map

    @property
    def hydro_meta(self) -> dict[int, dict]:
        return self.entities.hydro_meta

    @property
    def bus_names(self) -> dict[int, str]:
        return self.entities.bus_names

    @property
    def non_fictitious_bus_ids(self) -> list[int]:
        return self.scenario.non_fictitious_bus_ids

    @property
    def stage_hours(self) -> dict[int, float]:
        return self.temporal.stage_hours

    @property
    def block_hours(self) -> dict[tuple[int, int], float]:
        return self.temporal.block_hours

    @property
    def bh_df(self) -> pl.DataFrame:
        return self.temporal.bh_df

    @property
    def line_meta(self) -> list[dict]:
        return self.temporal.line_meta

    @property
    def load_stats(self) -> pd.DataFrame:
        return self.scenario.load_stats

    @property
    def load_factors_list(self) -> list[dict]:
        return self.scenario.load_factors_list

    @property
    def timing(self) -> pd.DataFrame:
        return self.performance.timing

    @property
    def timing_raw(self) -> pd.DataFrame:
        return self.performance.timing_raw

    @property
    def solver_train(self) -> pd.DataFrame:
        return self.performance.solver_train

    @property
    def solver_sim(self) -> pd.DataFrame:
        return self.performance.solver_sim

    @property
    def scaling_report(self) -> dict:
        return self.performance.scaling_report

    @property
    def cut_selection(self) -> pd.DataFrame:
        return self.performance.cut_selection

    @property
    def stochastic_available(self) -> bool:
        return self.stochastic.available

    @property
    def inflow_stats_stoch(self) -> pd.DataFrame:
        return self.stochastic.inflow_stats_stoch

    @property
    def ar_coefficients(self) -> pd.DataFrame:
        return self.stochastic.ar_coefficients

    @property
    def noise_openings(self) -> pd.DataFrame:
        return self.stochastic.noise_openings

    @property
    def fitting_report(self) -> dict:
        return self.stochastic.fitting_report

    @property
    def inflow_history(self) -> pd.DataFrame:
        return self.scenario.inflow_history

    @property
    def correlation(self) -> dict:
        return self.stochastic.correlation

    @property
    def inflow_lags_lf(self) -> pl.LazyFrame:
        return self.simulation.inflow_lags_lf

    @property
    def training_metadata(self) -> dict:
        return self.metadata.training

    @property
    def simulation_metadata(self) -> dict:
        return self.metadata.simulation

    @property
    def policy_metadata(self) -> dict:
        return self.metadata.policy

    @property
    def config(self) -> dict:
        return self.temporal.config

    @property
    def discount_rate(self) -> float:
        return self.temporal.discount_rate

    @property
    def stages_data(self) -> dict:
        return self.temporal.stages_data

    @property
    def lp_bounds(self) -> pd.DataFrame:
        return self.performance.lp_bounds

    @property
    def gc_constraints(self) -> list[dict]:
        return self.gc.constraints

    @property
    def gc_bounds(self) -> pd.DataFrame:
        return self.gc.bounds

    @property
    def gc_violations(self) -> pd.DataFrame:
        return self.simulation.gc_violations

    @property
    def hydro_bounds(self) -> pd.DataFrame:
        return self.scenario.hydro_bounds

    @property
    def thermal_bounds(self) -> pd.DataFrame:
        return self.scenario.thermal_bounds

    @property
    def ncs_stats(self) -> pd.DataFrame:
        return self.scenario.ncs_stats

    @property
    def line_block_bounds(self) -> pd.DataFrame:
        return self.scenario.line_block_bounds

    @property
    def retry_histogram(self) -> pd.DataFrame:
        return self.performance.retry_histogram

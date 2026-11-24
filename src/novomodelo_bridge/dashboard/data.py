"""Dashboard data layer — loading helpers and DashboardData dataclass.

``DashboardData.load(case_dir)`` loads every input the dashboard renders;
chart functions receive that single ``DashboardData`` instance.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
    exception_reason,
    report_file_missing_diagnostic,
    report_file_unreadable_diagnostic,
)
from novomodelo_bridge.core.errors import NovomodeloOutputError
from novomodelo_bridge.novomodelo.case_io import resolve_hydro_productivities
from novomodelo_bridge.novomodelo.constraint_expr import (
    ResolvedGenericConstraints,
)
from novomodelo_bridge.novomodelo.readers import (
    SPLIT_PLANT_COLUMNS,
    SPLIT_PLANT_SUMMARY,
    hydro_home_bus,
    read_novomodelo_buses,
    read_novomodelo_hydros,
    read_novomodelo_line_bounds,
    read_novomodelo_lines,
    read_novomodelo_load_bus_ids,
    read_novomodelo_non_controllable_sources,
    read_novomodelo_resolved_generic_constraints,
    read_novomodelo_scenario_weights,
    read_novomodelo_thermals,
    read_novomodelo_training_metadata,
    split_plant_row,
)
from novomodelo_bridge.novomodelo.scenario_weights import ScenarioWeights

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


_ENTITY_READERS: dict[str, Callable[[Path], list[dict]]] = {
    "hydros": read_novomodelo_hydros,
    "buses": read_novomodelo_buses,
    "thermals": read_novomodelo_thermals,
    "non_controllable_sources": read_novomodelo_non_controllable_sources,
    "lines": read_novomodelo_lines,
}


def load_names(case_dir: Path) -> dict[tuple[str, int], str]:
    """Load entity name mappings from system JSON files."""
    names: dict[tuple[str, int], str] = {}
    for entity, read in _ENTITY_READERS.items():
        for item in read(case_dir / "output"):
            names[(entity, item["id"])] = item["name"]
    return names


def entity_name(names: dict[tuple[str, int], str], entity: str, eid: int) -> str:
    return names.get((entity, eid), str(eid))


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _read_json_object(path: Path) -> tuple[dict, Exception | str | None]:
    """*path*'s JSON object, or ``{}`` and why the file holds none."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {}, exc
    if not isinstance(data, dict):
        return {}, "no JSON object"
    return data, None


def _read_report_json(
    path: Path,
    label: str,
    effect: str,
    shape_problem: Callable[[dict], str | None] | None = None,
) -> dict:
    """*path*'s JSON object, shown to the user as *label*; ``{}`` when absent.

    A file that does not parse, holds no JSON object, or lacks what
    *shape_problem* names yields ``{}`` with a ``report-file-unreadable``
    warning naming *effect*.
    """
    if not path.exists():
        return {}
    data, problem = _read_json_object(path)
    if problem is None and shape_problem is not None:
        problem = shape_problem(data)
    if problem is None:
        return data
    emit(report_file_unreadable_diagnostic(label, path, effect, problem), logger=logger)
    return {}


def _stages_problem(data: dict) -> str | None:
    """What ``stages.json`` lacks of what the dashboard reads, if anything."""
    stages = data.get("stages")
    if not isinstance(stages, list):
        return "no list of stages"
    for stage in stages:
        if not isinstance(stage, dict) or type(stage.get("id")) is not int:
            return "a stage without an id"
        blocks = stage.get("blocks")
        if not isinstance(blocks, list) or not all(
            isinstance(b, dict)
            and type(b.get("id")) is int
            and _is_number(b.get("hours"))
            for b in blocks
        ):
            return f"stage {stage['id']} without blocks that have an id and hours"
    policy_graph = data.get("policy_graph", {})
    if not isinstance(policy_graph, dict):
        return "a policy graph that is not an object"
    rate = policy_graph.get("annual_discount_rate")
    if rate is not None and not _is_number(rate):
        return "a discount rate that is not a number"
    return None


def _read_stages(case_dir: Path) -> dict:
    """``stages.json``, which the dashboard cannot build without: every chart's
    stage axis and every energy and hours-weighted average come from its
    stages and block hours.

    Raises:
        NovomodeloOutputError: the file is absent, does not parse, or lacks
            a stage's id or blocks, a block's id or hours, or holds a
            discount rate that is not a number.
    """
    path = case_dir / "stages.json"
    effect = "so the dashboard has no stages or block hours to build its charts on."
    if not path.exists():
        raise NovomodeloOutputError(
            f"stages.json was not found in {case_dir}, {effect}", path=str(path)
        )
    data, problem = _read_json_object(path)
    if problem is None:
        problem = _stages_problem(data)
    if problem is not None:
        reason = (
            exception_reason(problem) if isinstance(problem, Exception) else problem
        )
        raise NovomodeloOutputError(
            f"stages.json could not be read ({reason}), {effect}", path=str(path)
        )
    return data


def stage_labels(stages_data: dict) -> dict[int, str]:
    """Return stage_id -> date label from *stages_data*, keyed on each stage's date.

    Monthly horizons (the source model's typical monthly stages) get a compact
    ``"Mon YYYY"`` label. When a horizon packs several stages into one calendar
    month — e.g. the weekly stages of a DECOMP week/month deck — a month-only
    label would collapse them onto a single x-axis point, so every stage instead
    gets a day-resolution ``"DD Mon YYYY"`` label from its own ``start_date``.
    A stage with a missing or unparseable ``start_date`` falls back to its id.
    """
    parsed: list[tuple[int, pd.Timestamp | None]] = []
    for stage in stages_data.get("stages", []):
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


def stage_dates(stages_data: dict) -> dict[int, str]:
    """Return stage_id -> ISO ``YYYY-MM-DD`` start date from *stages_data*.

    These are the x-axis *positions* (paired with :func:`stage_labels`'s
    display text as tick labels): feeding them to a Plotly ``type="date"`` axis
    spaces the stages by their true calendar distance, so weekly stages sit
    closer together than a following monthly stage. A stage with a missing or
    unparseable ``start_date`` is omitted (its chart point falls back to the
    stage-id label via the axis helper).
    """
    dates: dict[int, str] = {}
    for stage in stages_data.get("stages", []):
        start = stage.get("start_date", "")
        if not start:
            continue
        try:
            dates[stage["id"]] = pd.to_datetime(start).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return dates


def _unit_group_bus_ids(hydro: dict) -> list[int] | None:
    """The buses of *hydro*'s unit groups; ``None`` unless it has unit
    groups and each names its bus."""
    groups = hydro.get("unit_groups")
    if not isinstance(groups, list) or not groups:
        return None
    if not all(isinstance(g, dict) and type(g.get("bus_id")) is int for g in groups):
        return None
    return sorted({g["bus_id"] for g in groups})


def load_hydro_bus_map(
    case_dir: Path, *, load_bus_ids: frozenset[int] | None = None
) -> dict[int, int]:
    """Return hydro_id -> the bus a plant's reservoir counts at, from its
    ``unit_groups``.

    Resolved by the compare layer's rule
    (``novomodelo_readers.hydro_home_bus``): a plant split across buses,
    such as Itaipu, resolves to the one of them with load. *load_bus_ids*
    defaults to the case's ``novomodelo_readers.read_novomodelo_load_bus_ids``.

    A plant without unit groups that name their buses, which novomodelo
    rejects at load, and a split plant with no such bus get no entry, so
    the views keyed by one bus per plant leave them out; each kind is
    reported once. A split plant's generation still counts at each bus,
    read per bus from ``simulation/hydro_bus_generation``.
    """
    hydros = read_novomodelo_hydros(case_dir / "output")
    if not hydros:
        return {}
    if load_bus_ids is None:
        load_bus_ids = read_novomodelo_load_bus_ids(case_dir / "output")
    result: dict[int, int] = {}
    ungrouped_rows: list[list[object]] = []
    split_rows: list[list[object]] = []
    for h in hydros:
        name = h["name"]
        bus_ids = _unit_group_bus_ids(h)
        if bus_ids is None:
            ungrouped_rows.append([h["id"], name])
            continue
        bus_id = hydro_home_bus(bus_ids, load_bus_ids)
        if bus_id is not None:
            result[h["id"]] = bus_id
            continue
        split_rows.append(split_plant_row(h["id"], name, bus_ids, load_bus_ids))
    if ungrouped_rows:
        emit(
            Diagnostic(
                code="hydro-unit-groups-missing",
                severity=Severity.WARNING,
                category="Dashboard data",
                title=f"Hydro plants without unit groups ({len(ungrouped_rows)})",
                summary=(
                    "The listed hydro plants have no unit groups that name their "
                    "bus in system/hydros.json, which novomodelo rejects when it "
                    "loads a case. The dashboard leaves them out of the per-bus "
                    "storage, stored-energy and inflow charts."
                ),
                table=DiagnosticTable(
                    columns=["Hydro ID", "Name"],
                    rows=ungrouped_rows,
                    justify=["right", "left"],
                ),
            ),
            logger=logger,
        )
    if split_rows:
        emit(
            Diagnostic(
                code="hydro-unit-groups-multi-bus",
                severity=Severity.WARNING,
                category="Dashboard data",
                title=f"Hydro plants spanning several buses ({len(split_rows)})",
                summary=(
                    f"{SPLIT_PLANT_SUMMARY} Their generation still counts at "
                    "each bus, but the dashboard leaves them out of the per-bus "
                    "storage, stored-energy and inflow charts."
                ),
                table=DiagnosticTable(
                    columns=SPLIT_PLANT_COLUMNS,
                    rows=split_rows,
                    justify=["right", "left", "left", "right"],
                ),
            ),
            logger=logger,
        )
    return result


def load_thermal_metadata(case_dir: Path) -> dict[int, dict]:
    """Return thermal_id -> {bus_id, max_mw, cost_per_mwh} metadata."""
    result = {}
    for t in read_novomodelo_thermals(case_dir / "output"):
        result[t["id"]] = {
            "bus_id": t["bus_id"],
            "max_mw": t["generation"]["max_mw"],
            "cost_per_mwh": float(t["cost_per_mwh"]),
            "name": t["name"],
        }
    return result


def load_ncs_bus_map(case_dir: Path) -> dict[int, int]:
    return {
        n["id"]: n["bus_id"]
        for n in read_novomodelo_non_controllable_sources(case_dir / "output")
    }


def load_hydro_metadata(
    case_dir: Path, *, bus_map: dict[int, int] | None = None
) -> dict[int, dict]:
    """Return hydro_id -> {bus_id, bus_ids, name, vol_max, vol_min, max_gen_mw,
    min_spillage_m3s, max_spillage_m3s, ...}.

    ``bus_ids`` lists every bus the plant's ``unit_groups`` sit at, and
    ``bus_id`` is the plant's *bus_map* entry, absent when it has none.
    *bus_map* defaults to :func:`load_hydro_bus_map`.

    ``min_spillage_m3s`` and ``max_spillage_m3s`` are the entity-level
    spillage limits from ``hydros.json``; each is ``None`` when the plant
    carries no value for it.

    Productivity comes from :func:`resolve_hydro_productivities`.
    """
    hydros = read_novomodelo_hydros(case_dir / "output")
    if not hydros:
        return {}
    if bus_map is None:
        bus_map = load_hydro_bus_map(case_dir)
    # Share the single canonical productivity cascade with the comparator so
    # the dashboard and compare report never disagree on a plant's ρ.
    productivities = resolve_hydro_productivities(case_dir, hydros)
    result = {}
    for h in hydros:
        gen = h["generation"]
        res = h["reservoir"]
        spill = h.get("spillage") or {}
        min_spill = spill.get("min_spillage_m3s")
        max_spill = spill.get("max_spillage_m3s")
        prod = productivities.get(h["id"]) or 0
        max_turbined = gen["max_turbined_m3s"]
        bus_id = bus_map.get(h["id"])
        entry: dict = {
            "name": h["name"],
            "bus_ids": _unit_group_bus_ids(h) or [],
            "vol_max": res["max_storage_hm3"],
            "vol_min": res["min_storage_hm3"],
            "max_gen_mw": gen["max_generation_mw"],
            "max_gen_physical": prod * max_turbined,
            "max_turbined": max_turbined,
            "productivity": prod,
            "downstream_id": h.get("downstream_id"),
            "min_spillage_m3s": None if min_spill is None else float(min_spill),
            "max_spillage_m3s": None if max_spill is None else float(max_spill),
        }
        if bus_id is not None:
            entry["bus_id"] = bus_id
        result[h["id"]] = entry
    return result


def hydro_bus_label(meta: dict, bus_names: dict[int, str]) -> str:
    """Name a hydro plant's buses, the one it resolves to first ("SE, IV")."""
    home = meta.get("bus_id")
    bus_ids = meta.get("bus_ids") or ([home] if home is not None else [])
    ordered = sorted(bus_ids, key=lambda b: (b != home, b))
    return ", ".join(bus_names.get(b, str(b)) for b in ordered)


# ---------------------------------------------------------------------------
# Polars helpers
# ---------------------------------------------------------------------------


def _stage_avg_mw(
    lf: pl.LazyFrame,
    mwh_col: str,
    stage_hours: dict[int, float],
    weights: ScenarioWeights,
) -> dict[int, float]:
    """Compute stage-average MW from MWh summed across all blocks via LazyFrame.

    Scans all blocks (no block_id filter), sums ``mwh_col`` per
    (scenario, stage), divides by total stage hours, then takes the
    expectation across scenarios with *weights*.

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
        .pipe(weights.attach)
        .group_by("stage_id")
        .agg(weights.mean("_avg_mw"))
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

    ``training/timing/iterations.parquet`` holds one
    rank-aggregated row plus N per-worker rows per iteration — per-worker
    rows only carry fwd/bwd wall + setup, rank rows carry everything else.
    ``SUM(col) GROUP BY iteration`` reconstructs the single-row totals that
    the single-row chart code expects.

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
    the forward / backward entry points in novomodelo ``training.rs``).

    No-op when ``conv`` is empty, lacks the required columns, or ``timing``
    is empty.
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


def _config_problem(data: dict) -> str | None:
    rate = data.get("discount_rate")
    if rate is not None and not _is_number(rate):
        return "a discount rate that is not a number"
    return None


def load_temporal_context(case_dir: Path) -> TemporalContext:
    """Load config.json / stages.json and the derived stage & block hour maps.

    Raises:
        NovomodeloOutputError: from :func:`_read_stages`.
    """
    stages_data = _read_stages(case_dir)

    config_path = case_dir / "config.json"
    config: dict = {}
    config_problem: Exception | str | None = None
    if config_path.exists():
        config, config_problem = _read_json_object(config_path)
        if config_problem is None:
            config_problem = _config_problem(config)
        if config_problem is not None:
            config = {}
    discount_rate = float(config.get("discount_rate", 0.0))

    # stages.json policy_graph is the authoritative discount rate when present.
    pg_rate = stages_data.get("policy_graph", {}).get("annual_discount_rate")
    if pg_rate is not None:
        discount_rate = float(pg_rate)
    elif config_problem is not None:
        emit(
            report_file_unreadable_diagnostic(
                "config.json",
                config_path,
                "the Overview shows a 0.0% discount rate.",
                config_problem,
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

    line_meta: list[dict] = read_novomodelo_lines(case_dir / "output")

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
    non_fictitious_bus_ids: list[int]


def load_entity_metadata(case_dir: Path, stages_data: dict) -> EntityMetadata:
    """Load entity name maps and hydro/thermal metadata dictionaries, with the
    stage labels and dates of *stages_data* (``TemporalContext.stages_data``).

    ``hydro_bus_map`` is resolved once and reused by
    :func:`load_hydro_metadata`, so a split plant's
    ``hydro-unit-groups-multi-bus`` warning fires once per dashboard build.
    """
    names = load_names(case_dir)
    bus_names = {eid: nm for (entity, eid), nm in names.items() if entity == "buses"}
    load_bus_ids = read_novomodelo_load_bus_ids(case_dir / "output")
    hydro_bus_map = load_hydro_bus_map(case_dir, load_bus_ids=load_bus_ids)
    return EntityMetadata(
        names=names,
        stage_labels=stage_labels(stages_data),
        stage_dates=stage_dates(stages_data),
        hydro_bus_map=hydro_bus_map,
        thermal_meta=load_thermal_metadata(case_dir),
        ncs_bus_map=load_ncs_bus_map(case_dir),
        hydro_meta=load_hydro_metadata(case_dir, bus_map=hydro_bus_map),
        bus_names=bus_names,
        non_fictitious_bus_ids=sorted(load_bus_ids),
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
    pumping_stations_lf: pl.LazyFrame
    hydro_bus_generation_lf: pl.LazyFrame
    inflow_lags_lf: pl.LazyFrame
    costs: pd.DataFrame
    gc_violations: pd.DataFrame
    scenario_weights: ScenarioWeights


def load_simulation_data(case_dir: Path) -> SimulationData:
    """Scan the hive-partitioned simulation outputs (LazyFrames) and collect the
    small cost/violation frames consumed eagerly by chart functions."""
    hydros_lf = scan_entity(case_dir, "hydros")
    thermals_lf = scan_entity(case_dir, "thermals")
    ncs_lf = scan_entity(case_dir, "non_controllables")
    buses_lf = scan_entity(case_dir, "buses")
    exchanges_lf = scan_entity(case_dir, "exchanges")
    pumping_stations_lf = scan_entity(case_dir, "pumping_stations")
    hydro_bus_generation_lf = scan_entity(case_dir, "hydro_bus_generation")

    inflow_lags_lf = scan_entity(case_dir, "inflow_lags")

    # Collected eagerly: stage-level only, and the chart_cost_* functions and
    # build_constraints_summary_table consume pandas.
    costs = scan_entity(case_dir, "costs").collect(engine="streaming").to_pandas()
    gc_violations = (
        scan_entity(case_dir, "violations/generic")
        .collect(engine="streaming")
        .to_pandas()
    )

    return SimulationData(
        hydros_lf=hydros_lf,
        thermals_lf=thermals_lf,
        ncs_lf=ncs_lf,
        buses_lf=buses_lf,
        exchanges_lf=exchanges_lf,
        pumping_stations_lf=pumping_stations_lf,
        hydro_bus_generation_lf=hydro_bus_generation_lf,
        inflow_lags_lf=inflow_lags_lf,
        costs=costs,
        gc_violations=gc_violations,
        scenario_weights=read_novomodelo_scenario_weights(case_dir / "output"),
    )


@dataclasses.dataclass
class ScenarioInputs:
    """Optional Novomodelo scenario/constraint input artifacts (empty when absent)."""

    load_stats: pd.DataFrame
    load_factors_list: list[dict]
    ncs_stats: pd.DataFrame
    inflow_history: pd.DataFrame
    line_bounds: pd.DataFrame
    line_block_bounds: pd.DataFrame
    hydro_bounds: pd.DataFrame
    thermal_bounds: pd.DataFrame


def _load_factors_problem(data: dict) -> str | None:
    entries = data.get("load_factors")
    if not isinstance(entries, list):
        return "no list of load factors"
    for e in entries:
        factors = e.get("block_factors") if isinstance(e, dict) else None
        if not (
            isinstance(factors, list)
            and type(e.get("bus_id")) is int
            and type(e.get("stage_id")) is int
            and all(
                isinstance(f, dict)
                and type(f.get("block_id")) is int
                and _is_number(f.get("factor"))
                for f in factors
            )
        ):
            return "a load factor without its bus, stage or block factors"
    return None


def load_scenario_inputs(case_dir: Path) -> ScenarioInputs:
    """Load the optional scenario inputs and input constraint bounds.

    Both load files are optional Novomodelo inputs (see case-format.md); they fall
    back to empty structures so cases without an explicit load scenario still
    render — charts degrade to a zero-load series via ``_compute_lp_load``.
    """
    ls_path = case_dir / "scenarios" / "load_seasonal_stats.parquet"
    load_stats = (
        pq.read_table(ls_path).to_pandas() if ls_path.exists() else pd.DataFrame()
    )
    load_factors_list: list[dict] = _read_report_json(
        case_dir / "scenarios" / "load_factors.json",
        "scenarios/load_factors.json",
        "the dashboard takes each bus's load in every block as its stage mean.",
        _load_factors_problem,
    ).get("load_factors", [])

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

    line_bounds = read_novomodelo_line_bounds(case_dir / "output").to_pandas()
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

    # ``timing`` is aggregated to one row per iteration for the charts.
    # ``timing_raw`` retains the multi-row-per-iteration view, where each
    # iteration has one rank-aggregated row (``worker_id`` NULL) and N
    # per-worker rows (``worker_id`` populated) — see
    # ``build_worker_timing_records`` in novomodelo-sddp training_output.rs.
    timing: pd.DataFrame
    timing_raw: pd.DataFrame
    solver_train: pd.DataFrame
    solver_sim: pd.DataFrame
    scaling_report: dict
    cut_selection: pd.DataFrame
    retry_histogram: pd.DataFrame
    lp_bounds: pd.DataFrame


def _cut_selection_per_stage(cut_selection: pd.DataFrame) -> pd.DataFrame:
    """One ``cut_selection`` row per ``(iteration, stage_id)``, summing its pools.

    novomodelo writes one row per row pool, and a stage of a scenario tree has one
    pool per node; the charts read one row per stage. A column novomodelo leaves
    null where no selection pass ran stays null when every pool is null.
    """
    if cut_selection.empty:
        return cut_selection
    counts = cut_selection.drop(columns=["pool_id", "node_id"], errors="ignore")
    return counts.groupby(["iteration", "stage_id"], as_index=False).sum(min_count=1)


def load_training_convergence(case_dir: Path) -> pd.DataFrame:
    """Load ``output/training/convergence.parquet``; empty when absent, as in a
    run that skipped training and simulated from a stored policy."""
    path = case_dir / "output" / "training" / "convergence.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pq.read_table(path).to_pandas()


def _scaling_report_problem(data: dict) -> str | None:
    stages = data.get("stages", [])
    if not isinstance(stages, list):
        return "no list of stages"
    for stage in stages:
        dims = stage.get("dimensions") if isinstance(stage, dict) else None
        if not (
            isinstance(dims, dict)
            and "stage_id" in stage
            and all(k in dims for k in ("num_cols", "num_rows", "num_nz"))
        ):
            return "a stage without its LP dimensions"
    return None


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
    solver_train = (
        pq.read_table(solver_train_path).to_pandas()
        if solver_train_path.exists()
        else pd.DataFrame()
    )
    sim_solver_path = (
        case_dir / "output" / "simulation" / "solver" / "iterations.parquet"
    )
    solver_sim = (
        pq.read_table(sim_solver_path).to_pandas()
        if sim_solver_path.exists()
        else pd.DataFrame()
    )
    scaling_report = _read_report_json(
        case_dir / "output" / "training" / "scaling_report.json",
        "output/training/scaling_report.json",
        "the Performance tab leaves out the LP dimensions and scaling quality.",
        _scaling_report_problem,
    )
    cs_path = case_dir / "output" / "training" / "cut_selection" / "iterations.parquet"
    cut_selection = _cut_selection_per_stage(
        pq.read_table(cs_path).to_pandas() if cs_path.exists() else pd.DataFrame()
    )
    retry_path = case_dir / "output" / "training" / "solver" / "retry_histogram.parquet"
    retry_histogram = (
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


def _stochastic_effect(content: str) -> str:
    return f"the dashboard's Stochastic Model tab leaves out {content}."


def _stochastic_file_present(path: Path, content: str) -> bool:
    """Whether *path* exists; when not, a note that the Stochastic Model tab
    leaves out its *content*."""
    if path.exists():
        return True
    emit(
        report_file_missing_diagnostic(
            f"output/stochastic/{path.name}", path, _stochastic_effect(content)
        ),
        logger=logger,
    )
    return False


def _read_stochastic_json(
    path: Path, content: str, shape_problem: Callable[[dict], str | None]
) -> dict:
    """:func:`_read_report_json` of ``output/stochastic/<name>``, with the
    effect its absence note gives."""
    return _read_report_json(
        path,
        f"output/stochastic/{path.name}",
        _stochastic_effect(content),
        shape_problem,
    )


def _fitting_report_problem(data: dict) -> str | None:
    hydros = data.get("hydros", {})
    if not isinstance(hydros, dict) or not all(
        isinstance(info, dict) and str(hid).isdigit() for hid, info in hydros.items()
    ):
        return "no hydro plants keyed by their id"
    return None


def _correlation_problem(data: dict) -> str | None:
    if not isinstance(data.get("profiles", {}), dict):
        return "no correlation profiles"
    return None


def load_stochastic_data(case_dir: Path) -> StochasticData:
    """Load output/stochastic/* (inflow fit, AR coefficients, correlation).

    Every file below is optional: a directory that exists but is missing one
    of them (e.g. a fit with no AR-coefficient file) degrades that one field
    to an empty DataFrame/dict with a ``report-file-missing`` note instead of
    raising.
    """
    stochastic_dir = case_dir / "output" / "stochastic"
    available = stochastic_dir.exists()
    inflow_stats_stoch = pd.DataFrame()
    ar_coefficients = pd.DataFrame()
    noise_openings = pd.DataFrame()
    fitting_report: dict = {}
    correlation: dict = {}
    if available:
        path = stochastic_dir / "inflow_seasonal_stats.parquet"
        if _stochastic_file_present(path, "the fitted inflow statistics"):
            inflow_stats_stoch = pq.read_table(path).to_pandas()
        path = stochastic_dir / "inflow_ar_coefficients.parquet"
        if _stochastic_file_present(path, "the fitted autoregressive coefficients"):
            ar_coefficients = pq.read_table(path).to_pandas()
        path = stochastic_dir / "noise_openings.parquet"
        if _stochastic_file_present(path, "the noise openings"):
            noise_openings = pq.read_table(path).to_pandas()
        path = stochastic_dir / "fitting_report.json"
        if _stochastic_file_present(path, "the fitting report"):
            fitting_report = _read_stochastic_json(
                path, "the fitting report", _fitting_report_problem
            )
        path = stochastic_dir / "correlation.json"
        if _stochastic_file_present(path, "the inflow correlation"):
            correlation = _read_stochastic_json(
                path, "the inflow correlation", _correlation_problem
            )
    return StochasticData(
        available=available,
        inflow_stats_stoch=inflow_stats_stoch,
        ar_coefficients=ar_coefficients,
        noise_openings=noise_openings,
        fitting_report=fitting_report,
        correlation=correlation,
    )


#: Every exception type an ``output/policy`` checkpoint read can fail with,
#: mirroring ``decomp.fcf.capability``'s CBVF round-trip probe: novomodelo absent
#: or missing the ``results`` binding, a malformed reloaded pool/dict, or any
#: ``novomodelo.errors.NovomodeloError`` leaf (each subclasses one of ``ValueError``/
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
    """Load ``output/policy``'s terminal ``state_dimension`` via novomodelo.

    Returns ``{}`` immediately when ``case_dir/output/policy`` does not
    exist, without importing novomodelo. Otherwise imports novomodelo lazily (kept out
    of module scope so this module stays importable in a novomodelo-free
    environment), reads the checkpoint via ``novomodelo.results.load_policy``,
    and selects the terminal pool (max ``stage_id``). Degrades to ``{}``
    with one ``report-file-unreadable`` warning on any failure in
    :data:`_POLICY_READ_FAILURE_TYPES` — a training-only case with no (or an
    unreadable) policy output must still render.
    """
    if not (case_dir / "output" / "policy").exists():
        return {}
    try:
        import novomodelo

        policy = novomodelo.results.load_policy(
            case_dir / "output", policy_subdir="policy"
        )
        terminal = max(policy["stage_cuts"], key=lambda stage: stage["stage_id"])
        return {"state_dimension": int(terminal["state_dimension"])}
    except _POLICY_READ_FAILURE_TYPES as exc:
        emit(
            report_file_unreadable_diagnostic(
                "output/policy",
                case_dir / "output" / "policy",
                "the dashboard leaves out the policy's state dimension.",
                exc,
            ),
            logger=logger,
        )
        return {}


@dataclasses.dataclass
class OutputMetadata:
    """Per-``output/`` subdirectory metadata: ``training``/``simulation`` from
    each ``metadata.json``, ``policy`` from the self-describing ``manifest.bin``
    checkpoint (see :func:`load_output_metadata`)."""

    training: dict
    simulation: dict
    policy: dict


def _simulation_metadata_problem(data: dict) -> str | None:
    scenarios = data.get("scenarios", {})
    if not isinstance(scenarios, dict) or not (
        scenarios.get("completed") is None or type(scenarios["completed"]) is int
    ):
        return "a completed-scenario count that is not a whole number"
    return None


def load_output_metadata(case_dir: Path) -> OutputMetadata:
    """Load output/{training,simulation,policy} metadata (empty if absent).

    ``training``/``simulation`` still read the sibling ``metadata.json``
    file; ``policy`` reads the self-describing ``manifest.bin`` checkpoint
    instead (see :func:`_load_policy_metadata`) — that subdirectory carries
    no ``metadata.json`` of its own.
    """

    simulation = _read_report_json(
        case_dir / "output" / "simulation" / "metadata.json",
        "output/simulation/metadata.json",
        "the dashboard leaves out what the run recorded about its simulation.",
        _simulation_metadata_problem,
    )

    return OutputMetadata(
        training=read_novomodelo_training_metadata(case_dir / "output"),
        simulation=simulation,
        policy=_load_policy_metadata(case_dir),
    )


@dataclasses.dataclass
class GenericConstraintData:
    """Generic constraints as novomodelo resolved them (optional).

    ``resolved`` is the comparison-ready view (see
    :func:`novomodelo_bridge.novomodelo.constraint_expr.resolve_generic_constraints`):
    coefficients and bounds read from the run's echo. Sense-free shape:
    ``constraints`` dicts carry no ``sense`` key, and ``bounds`` has nullable
    ``bound_lower``/``bound_upper`` endpoint columns instead of a single
    ``bound`` column (see
    :mod:`novomodelo_bridge.core.generic_constraint_format`). Readers derive the
    displayed direction from the endpoints via
    :func:`novomodelo_bridge.dashboard.tabs.constraints_utils.derive_constraint_shape`.
    """

    resolved: ResolvedGenericConstraints

    @property
    def constraints(self) -> list[dict]:
        return self.resolved.constraints

    @property
    def bounds(self) -> pd.DataFrame:
        return self.resolved.bounds.to_pandas()


def load_generic_constraints(case_dir: Path) -> GenericConstraintData:
    """Load ``constraints/generic_constraints.json`` resolved against the run's
    echo, by ``novomodelo_readers.read_novomodelo_resolved_generic_constraints``."""
    return GenericConstraintData(
        read_novomodelo_resolved_generic_constraints(case_dir, case_dir / "output")
    )


def _phase_note(training_available: bool, simulation_available: bool) -> str:
    """Log suffix naming the phase a case lacks; empty when it has both."""
    if not training_available and not simulation_available:
        return " (no training or simulation output)"
    if not simulation_available:
        return " (training-only; no simulation)"
    if not training_available:
        return " (simulation-only; no training)"
    return ""


# ---------------------------------------------------------------------------
# DashboardData dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DashboardData:
    """All data sources required to render the Novomodelo dashboard.

    Holds the eight ``load_*``-produced sub-structs verbatim (``temporal``,
    ``entities``, ``simulation``, ``scenario``, ``performance``,
    ``stochastic``, ``metadata``, ``gc``) plus the handful of values derived
    at load time that belong to none of them. Each flat ``data.<field>`` name a
    tab module reads is a read-only ``@property`` forwarding to its sub-struct.
    Populated via ``DashboardData.load(case_dir)``.
    """

    case_dir: Path
    case_name: str

    # Training — small single-file parquet loaded eagerly as pandas; empty
    # when the run skipped training.
    conv: pd.DataFrame

    # True when ``conv`` has rows. When ``False`` the training tab is hidden.
    training_available: bool
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
        """Load all dashboard data from a Novomodelo case directory.

        Thin orchestrator: each cohesive section is read by a dedicated
        ``load_*`` loader returning a sub-struct, which :class:`DashboardData`
        holds directly (see the flat-field ``@property`` pass-throughs below). Only
        ``stages.json`` is required: :func:`_read_stages` raises
        ``NovomodeloOutputError`` when it is absent or unusable. Every other
        file is optional: an absent one falls back to an empty DataFrame /
        dict inside its loader, and a JSON file that cannot be used to one
        with a ``report-file-unreadable`` warning, so a training-only and a
        simulation-only case both load.
        """
        logger.info("Loading dashboard data from %s", case_dir)

        conv = load_training_convergence(case_dir)

        temporal = load_temporal_context(case_dir)
        entities = load_entity_metadata(case_dir, temporal.stages_data)
        simulation = load_simulation_data(case_dir)
        scenario = load_scenario_inputs(case_dir)
        performance = load_solver_performance(case_dir, conv)
        stochastic = load_stochastic_data(case_dir)
        metadata = load_output_metadata(case_dir)
        gc = load_generic_constraints(case_dir)

        case_name = case_dir.resolve().name
        training_available = not conv.empty
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
            _phase_note(training_available, simulation_available),
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
            training_available=training_available,
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
    # Flat-field pass-throughs — read-only ``@property`` views onto the held
    # sub-structs above, for the ``data.<field>`` names the tab modules read.
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
    def hydro_bus_generation_lf(self) -> pl.LazyFrame:
        return self.simulation.hydro_bus_generation_lf

    @property
    def lhs_frames(self) -> dict[str, pl.LazyFrame]:
        """The simulation scans generic-constraint LHS terms read from."""
        sim = self.simulation
        return {
            "hydros": sim.hydros_lf,
            "thermals": sim.thermals_lf,
            "exchanges": sim.exchanges_lf,
            "pumping_stations": sim.pumping_stations_lf,
            "hydro_bus_generation": sim.hydro_bus_generation_lf,
        }

    @property
    def costs(self) -> pd.DataFrame:
        return self.simulation.costs

    @property
    def scenario_weights(self) -> ScenarioWeights:
        return self.simulation.scenario_weights

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
        return self.entities.non_fictitious_bus_ids

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
    def run_metadata(self) -> dict:
        """Metadata of the run's first phase: training, or simulation when the
        run skipped training. Both carry ``software_version``, ``started_at``
        and ``duration_seconds``."""
        return self.metadata.training or self.metadata.simulation

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
    def gc_resolved(self) -> ResolvedGenericConstraints:
        return self.gc.resolved

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

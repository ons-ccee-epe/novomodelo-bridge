#!/usr/bin/env python3
r"""Analyze and summarize novomodelo simulation results.

Usage:
    python scripts/analyze_results.py example/convertido/
    python scripts/analyze_results.py example/convertido/ --report costs
    python scripts/analyze_results.py example/convertido/ \
        --report hydro-detail --hydro 156
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_scenarios(case_dir: Path, entity: str) -> pd.DataFrame:
    """Load all scenario parquet files for an entity, concatenated."""
    sim_dir = case_dir / "output" / "simulation" / entity
    frames = []
    for scenario_dir in sorted(sim_dir.iterdir()):
        if scenario_dir.is_dir() and scenario_dir.name.startswith("scenario_id="):
            t = pq.read_table(scenario_dir / "data.parquet")
            frames.append(t.to_pandas())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_names(case_dir: Path) -> dict:
    """Load entity name mappings from system JSON files."""
    names = {}
    for entity, key in [
        ("hydros", "hydros"),
        ("buses", "buses"),
        ("thermals", "thermals"),
        ("lines", "lines"),
        ("non_controllable_sources", "non_controllable_sources"),
    ]:
        path = case_dir / "system" / f"{entity}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            for item in data.get(key, []):
                names[(entity, item["id"])] = item.get("name", str(item["id"]))
    return names


def name(names: dict, entity: str, eid: int) -> str:
    return names.get((entity, eid), str(eid))


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def report_overview(case_dir: Path) -> None:
    """High-level summary of the simulation."""
    names_map = load_names(case_dir)

    # Training convergence
    conv = pq.read_table(
        case_dir / "output" / "training" / "convergence.parquet"
    ).to_pandas()
    last = conv.iloc[-1]
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Iterations:   {int(last['iteration'])}")
    print(f"  Lower bound:  {last['lower_bound']:.4e}")
    print(
        f"  Upper bound:  {last['upper_bound_mean']:.4e} +/- "
        f"{last['upper_bound_std']:.4e}"
    )
    print(f"  Gap:          {last['gap_percent']:.2f}%")
    print()

    # Costs breakdown (average across scenarios)
    costs = load_all_scenarios(case_dir, "costs")
    n_scenarios = costs["scenario_id"].nunique()
    cost_cols = [
        "thermal_cost",
        "deficit_cost",
        "excess_cost",
        "spillage_cost",
        "curtailment_cost",
        "exchange_cost",
        "inflow_penalty_cost",
        "hydro_violation_cost",
        "generic_violation_cost",
        "storage_violation_cost",
        "turbined_cost",
    ]
    print("=" * 70)
    print(f"COST BREAKDOWN (avg over {n_scenarios} scenarios, sum over stages)")
    print("=" * 70)
    for col in cost_cols:
        if col in costs.columns:
            avg = costs.groupby("scenario_id")[col].sum().mean()
            if abs(avg) > 0.001:
                print(f"  {col:<35s} {avg:>15.2f}")
    total = costs.groupby("scenario_id")["total_cost"].sum().mean()
    print(f"  {'TOTAL':<35s} {total:>15.2f}")
    print()

    # Bus summary
    buses = load_all_scenarios(case_dir, "buses")
    print("=" * 70)
    print("BUS SUMMARY (avg over scenarios, sum over stages/blocks)")
    print("=" * 70)
    print(
        f"  {'Bus':<15s} {'Load(GWh)':>12s} {'Deficit(GWh)':>14s} "
        f"{'Excess(GWh)':>14s} {'AvgSpot':>10s}"
    )
    for bus_id in sorted(buses["bus_id"].unique()):
        b = buses[buses["bus_id"] == bus_id]
        avg_load = b.groupby("scenario_id")["load_mwh"].sum().mean() / 1e3
        avg_def = b.groupby("scenario_id")["deficit_mwh"].sum().mean() / 1e3
        avg_exc = b.groupby("scenario_id")["excess_mwh"].sum().mean() / 1e3
        avg_spot = b["spot_price"].mean()
        bname = name(names_map, "buses", bus_id)
        print(
            f"  {bname:<15s} {avg_load:>12.0f} {avg_def:>14.1f} "
            f"{avg_exc:>14.1f} {avg_spot:>10.2f}"
        )
    print()

    # Hydro summary
    hydros = load_all_scenarios(case_dir, "hydros")
    print("=" * 70)
    print("HYDRO AGGREGATE (avg over scenarios, sum over stages/blocks)")
    print("=" * 70)
    h_scen = hydros.groupby("scenario_id").agg(
        gen_gwh=("generation_mwh", lambda x: x.sum() / 1e3),
        spillage_total=("spillage_m3s", "sum"),
        inflow_slack=("inflow_nonnegativity_slack_m3s", "sum"),
        stor_viol=("storage_violation_below_hm3", "sum"),
    )
    print(f"  Total hydro generation:       {h_scen['gen_gwh'].mean():>12.0f} GWh")
    print(
        f"  Total spillage:               {h_scen['spillage_total'].mean():>12.0f} m3/s"
    )
    print(
        f"  Total inflow nonneg slack:    {h_scen['inflow_slack'].mean():>12.0f} m3/s"
    )
    print(f"  Total storage violation:      {h_scen['stor_viol'].mean():>12.2f} hm3")

    n_spilling = (
        hydros.groupby(["scenario_id", "hydro_id"])["spillage_m3s"]
        .sum()
        .reset_index()
        .groupby("scenario_id")
        .apply(lambda g: (g["spillage_m3s"] > 0).sum())
        .mean()
    )
    n_slack = (
        hydros.groupby(["scenario_id", "hydro_id"])["inflow_nonnegativity_slack_m3s"]
        .sum()
        .reset_index()
        .groupby("scenario_id")
        .apply(lambda g: (g["inflow_nonnegativity_slack_m3s"] > 0).sum())
        .mean()
    )
    print(
        f"  Hydros spilling (avg):        {n_spilling:>12.0f} / "
        f"{hydros['hydro_id'].nunique()}"
    )
    print(
        f"  Hydros with inflow slack:     {n_slack:>12.0f} / "
        f"{hydros['hydro_id'].nunique()}"
    )
    print()

    # Exchange summary
    exchanges = load_all_scenarios(case_dir, "exchanges")
    print("=" * 70)
    print("EXCHANGE SUMMARY (avg over scenarios, per line)")
    print("=" * 70)
    print(f"  {'Line':<12s} {'AvgDirect':>12s} {'AvgReverse':>12s} {'AvgNet':>12s}")
    for lid in sorted(exchanges["line_id"].unique()):
        ex = exchanges[exchanges["line_id"] == lid]
        avg_d = ex["direct_flow_mw"].mean()
        avg_r = ex["reverse_flow_mw"].mean()
        avg_n = ex["net_flow_mw"].mean()
        lname = name(names_map, "lines", lid)
        print(f"  {lname:<12s} {avg_d:>12.1f} {avg_r:>12.1f} {avg_n:>12.1f}")
    print()

    # NCS summary
    ncs = load_all_scenarios(case_dir, "non_controllables")
    print("=" * 70)
    print("NCS SUMMARY (avg over scenarios, sum over stages/blocks)")
    print("=" * 70)
    ncs_scen = ncs.groupby("scenario_id").agg(
        gen_gwh=("generation_mwh", lambda x: x.sum() / 1e3),
        curtail_gwh=("curtailment_mwh", lambda x: x.sum() / 1e3),
        avail_total=("available_mw", "sum"),
    )
    print(f"  Total NCS generation:       {ncs_scen['gen_gwh'].mean():>12.0f} GWh")
    print(f"  Total NCS curtailment:      {ncs_scen['curtail_gwh'].mean():>12.0f} GWh")
    curtail_pct = (
        ncs_scen["curtail_gwh"].mean()
        / max(ncs_scen["gen_gwh"].mean() + ncs_scen["curtail_gwh"].mean(), 1)
        * 100
    )
    print(f"  Curtailment rate:           {curtail_pct:>11.1f}%")
    print()

    # Thermal summary
    thermals = load_all_scenarios(case_dir, "thermals")
    t_scen = thermals.groupby("scenario_id").agg(
        gen_gwh=("generation_mwh", lambda x: x.sum() / 1e3),
        cost_total=("generation_cost", "sum"),
    )
    print("=" * 70)
    print("THERMAL SUMMARY (avg over scenarios)")
    print("=" * 70)
    print(f"  Total thermal generation:   {t_scen['gen_gwh'].mean():>12.0f} GWh")
    print(f"  Total thermal cost:         {t_scen['cost_total'].mean():>12.2f}")


def report_spillage(case_dir: Path) -> None:
    """Detailed spillage report per hydro plant."""
    names_map = load_names(case_dir)
    hydros = load_all_scenarios(case_dir, "hydros")

    spill_by_hydro = (
        hydros.groupby(["scenario_id", "hydro_id"])["spillage_m3s"]
        .sum()
        .reset_index()
        .groupby("hydro_id")["spillage_m3s"]
        .mean()
        .sort_values(ascending=False)
    )

    print("=" * 70)
    print("SPILLAGE BY HYDRO (avg m3/s across scenarios, top 20)")
    print("=" * 70)
    print(f"  {'ID':>4s}  {'Name':<20s} {'AvgSpillage':>14s}")
    for hid in spill_by_hydro.head(20).index:
        val = spill_by_hydro[hid]
        if val > 0:
            hname = name(names_map, "hydros", hid)
            print(f"  {hid:>4d}  {hname:<20s} {val:>14.1f}")


def report_curtailment(case_dir: Path) -> None:
    """Detailed NCS curtailment report."""
    names_map = load_names(case_dir)
    ncs = load_all_scenarios(case_dir, "non_controllables")

    curt_by_ncs = (
        ncs.groupby(["scenario_id", "non_controllable_id"])
        .agg(
            gen_mwh=("generation_mwh", "sum"),
            curtail_mwh=("curtailment_mwh", "sum"),
            avail=("available_mw", "sum"),
        )
        .reset_index()
        .groupby("non_controllable_id")
        .mean(numeric_only=True)
    )
    curt_by_ncs["curtail_pct"] = (
        curt_by_ncs["curtail_mwh"]
        / (curt_by_ncs["gen_mwh"] + curt_by_ncs["curtail_mwh"]).clip(lower=1)
        * 100
    )

    print("=" * 70)
    print("NCS CURTAILMENT (avg across scenarios)")
    print("=" * 70)
    print(f"  {'ID':>4s}  {'Name':<20s} {'GenGWh':>10s} {'CurtGWh':>10s} {'Curt%':>8s}")
    for nid in curt_by_ncs.sort_values("curtail_mwh", ascending=False).index:
        row = curt_by_ncs.loc[nid]
        if row["curtail_mwh"] > 0:
            nname = name(names_map, "non_controllable_sources", nid)
            print(
                f"  {nid:>4d}  {nname:<20s} "
                f"{row['gen_mwh'] / 1e3:>10.0f} {row['curtail_mwh'] / 1e3:>10.0f} "
                f"{row['curtail_pct']:>7.1f}%"
            )


def report_costs(case_dir: Path) -> None:
    """Per-stage cost breakdown."""
    costs = load_all_scenarios(case_dir, "costs")

    # Average across scenarios, per stage
    avg = costs.groupby("stage_id").mean(numeric_only=True)

    print("=" * 70)
    print("COST BY STAGE (avg across scenarios)")
    print("=" * 70)
    cols = [
        "thermal_cost",
        "spillage_cost",
        "curtailment_cost",
        "exchange_cost",
        "excess_cost",
        "inflow_penalty_cost",
        "deficit_cost",
        "total_cost",
    ]
    header = f"{'Stage':>6s}"
    for c in cols:
        short = c.replace("_cost", "").replace("_", "")[:10]
        header += f" {short:>12s}"
    print(header)
    for sid in avg.index[:20]:
        row = f"{sid:>6d}"
        for c in cols:
            val = avg.loc[sid, c] if c in avg.columns else 0
            row += f" {val:>12.1f}"
        print(row)
    if len(avg) > 20:
        print(f"  ... ({len(avg)} total stages)")


def report_exchanges(case_dir: Path) -> None:
    """Exchange flow analysis with capacity utilization."""
    names_map = load_names(case_dir)
    exchanges = load_all_scenarios(case_dir, "exchanges")

    # Load line bounds for capacity
    lb_path = case_dir / "constraints" / "line_bounds.parquet"
    lb = pq.read_table(lb_path).to_pandas() if lb_path.exists() else None

    print("=" * 70)
    print("EXCHANGE FLOW ANALYSIS (avg across scenarios)")
    print("=" * 70)

    for lid in sorted(exchanges["line_id"].unique()):
        lname = name(names_map, "lines", lid)
        ex = exchanges[exchanges["line_id"] == lid]

        avg_by_stage = ex.groupby("stage_id").agg(
            direct=("direct_flow_mw", "mean"),
            reverse=("reverse_flow_mw", "mean"),
        )

        # Capacity utilization if bounds available
        if lb is not None:
            line_lb = lb[lb["line_id"] == lid]
            if not line_lb.empty:
                merged = avg_by_stage.join(
                    line_lb.set_index("stage_id")[["direct_mw", "reverse_mw"]]
                )
                merged["direct_util"] = (
                    merged["direct"] / merged["direct_mw"].clip(lower=0.1) * 100
                )
                merged["reverse_util"] = (
                    merged["reverse"] / merged["reverse_mw"].clip(lower=0.1) * 100
                )
                avg_d_util = merged["direct_util"].mean()
                avg_r_util = merged["reverse_util"].mean()
                max_d_util = merged["direct_util"].max()
                max_r_util = merged["reverse_util"].max()

                print(f"  Line {lid} ({lname}):")
                print(
                    f"    Direct:  avg={avg_by_stage['direct'].mean():>8.1f} MW, "
                    f"util avg={avg_d_util:.1f}%, max={max_d_util:.1f}%"
                )
                print(
                    f"    Reverse: avg={avg_by_stage['reverse'].mean():>8.1f} MW, "
                    f"util avg={avg_r_util:.1f}%, max={max_r_util:.1f}%"
                )
            else:
                print(
                    f"  Line {lid} ({lname}): avg direct="
                    f"{avg_by_stage['direct'].mean():.1f}, "
                    f"reverse={avg_by_stage['reverse'].mean():.1f}"
                )
        print()


def report_hydro_detail(case_dir: Path, hydro_id: int) -> None:
    """Detailed view of a single hydro plant."""
    names_map = load_names(case_dir)
    hydros = load_all_scenarios(case_dir, "hydros")

    h = hydros[hydros["hydro_id"] == hydro_id]
    if h.empty:
        print(f"Hydro {hydro_id} not found.")
        return

    hname = name(names_map, "hydros", hydro_id)
    print("=" * 70)
    print(f"HYDRO DETAIL: {hname} (id={hydro_id})")
    print("=" * 70)

    # Block 0 only for stage-level view
    h0 = h[h["block_id"] == 0]
    avg = h0.groupby("stage_id").mean(numeric_only=True)

    print(
        f"  {'Stage':>6s} {'Storage':>10s} {'Inflow':>10s} {'Turbined':>10s} "
        f"{'Spillage':>10s} {'GenMW':>10s} {'InfSlack':>10s} {'WaterVal':>10s}"
    )
    for sid in avg.index[:30]:
        r = avg.loc[sid]
        print(
            f"  {sid:>6d} {r['storage_final_hm3']:>10.1f} {r['inflow_m3s']:>10.1f} "
            f"{r['turbined_m3s']:>10.1f} {r['spillage_m3s']:>10.1f} "
            f"{r['generation_mw']:>10.1f} {r['inflow_nonnegativity_slack_m3s']:>10.1f} "
            f"{r['water_value_per_hm3']:>10.0f}"
        )
    if len(avg) > 30:
        print(f"  ... ({len(avg)} total stages)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

REPORTS = {
    "overview": report_overview,
    "spillage": report_spillage,
    "curtailment": report_curtailment,
    "costs": report_costs,
    "exchanges": report_exchanges,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze novomodelo simulation results.",
    )
    parser.add_argument(
        "case_dir", type=Path, help="Path to the novomodelo case directory."
    )
    parser.add_argument(
        "--report",
        "-r",
        choices=list(REPORTS.keys()) + ["hydro-detail", "all"],
        default="overview",
        help="Report type (default: overview).",
    )
    parser.add_argument("--hydro", type=int, help="Hydro ID for hydro-detail report.")
    args = parser.parse_args()

    case_dir = args.case_dir
    if not (case_dir / "output" / "simulation").exists():
        print(f"Error: no simulation output in {case_dir}", file=sys.stderr)
        sys.exit(1)

    if args.report == "hydro-detail":
        if args.hydro is None:
            print("Error: --hydro required for hydro-detail report", file=sys.stderr)
            sys.exit(1)
        report_hydro_detail(case_dir, args.hydro)
    elif args.report == "all":
        for rname, rfunc in REPORTS.items():
            rfunc(case_dir)
            print()
    else:
        REPORTS[args.report](case_dir)


if __name__ == "__main__":
    main()

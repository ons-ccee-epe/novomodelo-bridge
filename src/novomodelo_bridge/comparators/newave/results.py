"""Results comparison engine: aligns the source model output with Novomodelo simulation
means.

Reads the source model MEDIAS / pmo.dat output and Novomodelo simulation parquets, aligns
entities via ``EntityAlignment``, and computes per-variable absolute and relative
differences.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from novomodelo_bridge.comparators.model import PercentileData, ResultComparison
from novomodelo_bridge.comparators.newave.alignment import EntityAlignment
from novomodelo_bridge.novomodelo.case_io import case_dir_for

if TYPE_CHECKING:
    from novomodelo_bridge.comparators.dataset import ComparisonDataset
    from novomodelo_bridge.newave.case import NewaveCase
    from novomodelo_bridge.newave.id_map import NewaveIdMap

_LOG = logging.getLogger(__name__)


def _compute_diff(
    nw_value: float, novomodelo_value: float
) -> tuple[float, float | None]:
    """Compute absolute and relative diff."""
    abs_diff = abs(nw_value - novomodelo_value)
    rel_diff: float | None = None
    if abs(nw_value) > 1e-10:
        rel_diff = abs_diff / abs(nw_value)
    return abs_diff, rel_diff


def _make_result(
    entity_type: str,
    entity_name: str,
    newave_code: int,
    novomodelo_id: int,
    stage: int,
    variable: str,
    nw_value: float,
    novomodelo_value: float,
) -> ResultComparison:
    """Build a ResultComparison with computed diffs."""
    abs_diff, rel_diff = _compute_diff(nw_value, novomodelo_value)
    return ResultComparison(
        entity_type=entity_type,
        entity_name=entity_name,
        newave_code=newave_code,
        novomodelo_id=novomodelo_id,
        stage=stage,
        variable=variable,
        newave_value=nw_value,
        novomodelo_value=novomodelo_value,
        abs_diff=abs_diff,
        rel_diff=rel_diff,
    )


# Minimum turbined flow (m³/s) for the derived gen/turbined productivity to be
# meaningful. At/near zero turbining, generation is also ~0, so the ratio is an
# undefined 0/0 — those stages are filtered out of the productivity comparison.
_PRODUCTIVITY_TURB_EPS: float = 1.0e-6

# MEDIAS variable name -> our standard variable name.
_HYDRO_VAR_MAP: dict[str, str] = {
    "VARMUH": "storage_final_hm3",
    "GHIDUH": "generation_mw",
    "QTURUH": "turbined_m3s",
    "QVERTUH": "spillage_m3s",
    "QINCRUH": "inflow_m3s",
    "PIVARM": "water_value_per_hm3",
    "VEVAPUH": "evaporation_m3s",
    # Novomodelo emits water withdrawal as an input constant (target) in
    # ``constraints/hydro_bounds.parquet`` rather than as a per-stage
    # simulation result; ``read_novomodelo_hydro_withdrawal`` joins it onto
    # ``novomodelo_hydro`` so this map entry still resolves.
    "VRETIRUH": "withdrawal_m3s",
    # QAFLUH = total inflow at the plant: incremental + Σ upstream
    # (turbined + spilled). Computed Novomodelo-side by
    # ``read_novomodelo_hydro_total_flows``; merged into ``novomodelo_hydro``.
    "QAFLUH": "total_inflow_m3s",
}

_SYSTEM_VAR_MAP: dict[str, str] = {
    "CMO": "spot_price",
    "DEFT": "deficit_mw",
}


def _nw_stage_offset(nw_df: pl.DataFrame) -> int:
    """Return the minimum stage number in a source-model MEDIAS DataFrame.

    The source model v29+ MEDIAS columns are numbered from the study start month (e.g. 3
    for March).  We subtract this offset to convert to 0-based stage indices matching
    Novomodelo convention.
    """
    stages = nw_df["stage"].drop_nulls()
    if stages.is_empty():
        return 1
    return int(stages.min())


def _build_gen_max_overlay(
    nw_hydro: pl.DataFrame,
    novomodelo_lp_gen_max: pl.DataFrame,
    nw_names: dict[int, str],
    novomodelo_meta: dict[int, dict],
    nw_offset: int,
) -> pl.DataFrame:
    """Build per-(entity_id, stage_id) generation upper-bound overlay.

    Returns columns ``entity_id``, ``stage_id``, ``nw_ghmax_fphc_mw`` (the source model
    constant-FPH max generation, MWmes from MEDIAS-USIH), and ``novomodelo_lp_gen_max_mw``
    (Novomodelo LP upper bound at block 0). Either column may be null when the source side is
    missing a value.
    """
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.Int64,
            "stage_id": pl.Int64,
            "nw_ghmax_fphc_mw": pl.Float64,
            "novomodelo_lp_gen_max_mw": pl.Float64,
        }
    )
    if novomodelo_lp_gen_max.is_empty() and nw_hydro.is_empty():
        return empty

    # The source model GHMAX_FPHC → (novomodelo_id, stage_0based) lookup via name match.
    novomodelo_by_name: dict[str, int] = {
        meta["name"].strip().upper(): eid for eid, meta in novomodelo_meta.items()
    }
    nw_code_to_novomodelo_id: dict[int, int] = {}
    for nw_code, nw_name in nw_names.items():
        cid = novomodelo_by_name.get(nw_name.strip().upper())
        if cid is not None:
            nw_code_to_novomodelo_id[nw_code] = cid

    nw_rows: list[dict[str, float | int]] = []
    if not nw_hydro.is_empty():
        for row in nw_hydro.iter_rows(named=True):
            if row["value"] is None:
                continue
            var = str(row["variable"]).strip().upper()
            if var != "GHMAX_FPHC":
                continue
            code = int(row["newave_code"])
            cid = nw_code_to_novomodelo_id.get(code)
            if cid is None:
                continue
            stage = int(row["stage"]) - nw_offset
            nw_rows.append(
                {
                    "entity_id": cid,
                    "stage_id": stage,
                    "nw_ghmax_fphc_mw": float(row["value"]),
                }
            )

    nw_df = (
        pl.DataFrame(nw_rows).cast(
            {
                "entity_id": pl.Int64,
                "stage_id": pl.Int64,
                "nw_ghmax_fphc_mw": pl.Float64,
            }
        )
        if nw_rows
        else pl.DataFrame(
            schema={
                "entity_id": pl.Int64,
                "stage_id": pl.Int64,
                "nw_ghmax_fphc_mw": pl.Float64,
            }
        )
    )

    if novomodelo_lp_gen_max.is_empty():
        return nw_df

    return nw_df.join(
        novomodelo_lp_gen_max, on=["entity_id", "stage_id"], how="full", coalesce=True
    )


# MEDIAS reports VIOL_POS_VRETIRUH / VIOL_NEG_VRETIRUH and VIOL_POS_EVAP / VIOL_NEG_EVAP
# as monthly volumes in hm³ (same convention as VRETIRUH / VEVAPUH).  Novomodelo emits the
# matching slacks in m³/s.  The source model's internal reports round the month-length
# factor to 2.63 (≈ 730 h × 3600 s / 1e6), so dividing by 2.63 yields apples-to-apples
# flows matching what ``_compare_hydros`` already does for the VEVAPUH / VRETIRUH
# realized series.
#
# WITHDRAWAL pos/neg are SWAPPED on purpose: Novomodelo's sign convention for the withdrawal
# slack is the inverse of the source model's, so the column the source model calls
# ``VIOL_POS_VRETIRUH`` lines up physically with Novomodelo's
# ``water_withdrawal_violation_neg_m3s`` (and vice versa).  Mapping them this way keeps
# the comparison HTML's "Withdrawal Slack Pos / Neg" panels self-consistent under the
# source model's labelling — the display labels in ``_HYDRO_NOVOMODELO_ONLY_VARIABLES`` /
# ``report_builder.slack_specs`` are correspondingly swapped so each panel shows the
# column that matches its header.  Evaporation pos/neg share the source model's
# convention and don't need the swap.
_NW_HYDRO_SLACK_VARS: dict[str, str] = {
    "VIOL_POS_VRETIRUH": "water_withdrawal_violation_neg_m3s",
    "VIOL_NEG_VRETIRUH": "water_withdrawal_violation_pos_m3s",
    "VIOL_POS_EVAP": "evaporation_violation_pos_m3s",
    "VIOL_NEG_EVAP": "evaporation_violation_neg_m3s",
}
_NW_HYDRO_SLACK_COLUMNS: tuple[str, ...] = tuple(_NW_HYDRO_SLACK_VARS.values())


def _compute_nw_hydro_slacks(
    nw_hydro: pl.DataFrame,
    nw_names: dict[int, str],
    novomodelo_meta: dict[int, dict],
) -> pl.DataFrame:
    """Build a per-(entity_id, stage_id) frame of the source model hydro slacks.

    Surfaces ``VIOL_POS_VRETIRUH`` / ``VIOL_NEG_VRETIRUH`` (water withdrawal) and
    ``VIOL_POS_EVAP`` / ``VIOL_NEG_EVAP`` (evaporation) from MEDIAS-USIH aligned to
    Novomodelo's ``entity_id`` and ``stage_id`` (0-based) conventions and converted from
    hm³/month to m³/s with the same /2.63 factor used by :func:`_compare_hydros` for the
    realized withdrawal / evaporation series. The chart layer joins this against the
    matching Novomodelo slack columns so the Hydro Operation tab can render the source model
    alongside Novomodelo on per-bus and SIN-total slack panels.
    """
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.Int64,
            "stage_id": pl.Int64,
            "water_withdrawal_violation_pos_m3s": pl.Float64,
            "water_withdrawal_violation_neg_m3s": pl.Float64,
            "evaporation_violation_pos_m3s": pl.Float64,
            "evaporation_violation_neg_m3s": pl.Float64,
        }
    )
    if nw_hydro.is_empty() or not nw_names or not novomodelo_meta:
        return empty

    novomodelo_by_name: dict[str, int] = {
        meta["name"].strip().upper(): eid for eid, meta in novomodelo_meta.items()
    }
    nw_code_to_novomodelo_id: dict[int, int] = {}
    for nw_code, nw_name in nw_names.items():
        cid = novomodelo_by_name.get(nw_name.strip().upper())
        if cid is not None:
            nw_code_to_novomodelo_id[nw_code] = cid
    if not nw_code_to_novomodelo_id:
        return empty

    offset = _nw_stage_offset(nw_hydro)
    rows: dict[tuple[int, int], dict[str, float]] = {}
    for row in nw_hydro.iter_rows(named=True):
        if row["value"] is None:
            continue
        var_raw = str(row["variable"]).strip().upper()
        col = _NW_HYDRO_SLACK_VARS.get(var_raw)
        if col is None:
            continue
        code = int(row["newave_code"])
        cid = nw_code_to_novomodelo_id.get(code)
        if cid is None:
            continue
        stage = int(row["stage"]) - offset
        rows.setdefault((cid, stage), {})[col] = float(row["value"]) / 2.63

    if not rows:
        return empty

    return (
        pl.DataFrame(
            [
                {
                    "entity_id": eid,
                    "stage_id": sid,
                    **{col: cols.get(col, 0.0) for col in _NW_HYDRO_SLACK_COLUMNS},
                }
                for (eid, sid), cols in rows.items()
            ]
        )
        .cast(
            {
                "entity_id": pl.Int64,
                "stage_id": pl.Int64,
                "water_withdrawal_violation_pos_m3s": pl.Float64,
                "water_withdrawal_violation_neg_m3s": pl.Float64,
                "evaporation_violation_pos_m3s": pl.Float64,
                "evaporation_violation_neg_m3s": pl.Float64,
            }
        )
        .sort("entity_id", "stage_id")
    )


def _compare_hydros(
    nw_hydro: pl.DataFrame,
    novomodelo_hydro: pl.DataFrame,
    nw_names: dict[int, str],
    novomodelo_meta: dict[int, dict],
) -> list[ResultComparison]:
    """Compare hydro results by matching plant names."""
    results: list[ResultComparison] = []

    offset = _nw_stage_offset(nw_hydro)

    # Build Novomodelo name→(id, min_storage) lookup.
    novomodelo_by_name: dict[str, tuple[int, float]] = {}
    for eid, meta in novomodelo_meta.items():
        name_upper = meta["name"].strip().upper()
        min_stor = meta.get("min_storage_hm3", 0.0)
        novomodelo_by_name[name_upper] = (eid, min_stor)

    # Match the source model codes to Novomodelo IDs by name.
    matched: dict[
        int, tuple[int, str, float]
    ] = {}  # nw_code→(novomodelo_id, name, min_stor)
    for nw_code, nw_name in nw_names.items():
        name_upper = nw_name.strip().upper()
        hit = novomodelo_by_name.get(name_upper)
        if hit is not None:
            matched[nw_code] = (hit[0], nw_name.strip(), hit[1])

    # Build the source model lookup: (nw_code, stage, variable) -> value
    nw_lookup: dict[tuple[int, int, str], float] = {}
    for row in nw_hydro.iter_rows(named=True):
        if row["value"] is None:
            continue
        code = int(row["newave_code"])
        if code not in matched:
            continue
        stage = int(row["stage"]) - offset
        var = str(row["variable"]).strip().upper()
        mapped = _HYDRO_VAR_MAP.get(var)
        if mapped is None:
            continue
        nw_lookup[(code, stage, mapped)] = float(row["value"])

    # Reconstruct realized evaporation and withdrawal from the source model LP slack
    # columns. VEVAPUH/VRETIRUH are reported as the *scheduled* values; VIOL_POS_* /
    # VIOL_NEG_* track over- and under-application respectively. Realized = scheduled +
    # POS − NEG (same convention as Novomodelo's matrix.rs water-balance row). Reading the
    # slacks directly from the long-format ``nw_hydro`` frame avoids adding them to
    # ``_HYDRO_VAR_MAP`` (which would spuriously emit ResultComparison rows for the
    # slacks themselves).
    nw_slacks: dict[tuple[int, int, str], float] = {}
    for row in nw_hydro.iter_rows(named=True):
        if row["value"] is None:
            continue
        var_raw = str(row["variable"]).strip().upper()
        if var_raw not in (
            "VIOL_POS_EVAP",
            "VIOL_NEG_EVAP",
            "VIOL_POS_VRETIRUH",
            "VIOL_NEG_VRETIRUH",
        ):
            continue
        code = int(row["newave_code"])
        if code not in matched:
            continue
        stage = int(row["stage"]) - offset
        nw_slacks[(code, stage, var_raw)] = float(row["value"])

    _SLACK_PAIRS = {
        "evaporation_m3s": ("VIOL_POS_EVAP", "VIOL_NEG_EVAP"),
        "withdrawal_m3s": ("VIOL_POS_VRETIRUH", "VIOL_NEG_VRETIRUH"),
    }
    for key in list(nw_lookup.keys()):
        code, stage, mapped = key
        slack_vars = _SLACK_PAIRS.get(mapped)
        if slack_vars is None:
            continue
        pos = nw_slacks.get((code, stage, slack_vars[0]), 0.0)
        neg = nw_slacks.get((code, stage, slack_vars[1]), 0.0)
        nw_lookup[key] = nw_lookup[key] + pos - neg

    # Derive the source model total outflow = turbined + spillage (no MEDIAS code).
    derived: dict[tuple[int, int, str], float] = {}
    for (code, stage, var), val in nw_lookup.items():
        if var != "turbined_m3s":
            continue
        spill = nw_lookup.get((code, stage, "spillage_m3s"))
        if spill is None:
            continue
        derived[(code, stage, "outflow_m3s")] = val + spill
    nw_lookup.update(derived)

    # Build Novomodelo lookup: (entity_id, stage, variable) -> value
    novomodelo_lookup: dict[tuple[int, int, str], float] = {}
    hydro_value_cols = list(_HYDRO_VAR_MAP.values()) + ["outflow_m3s"]
    for row in novomodelo_hydro.iter_rows(named=True):
        eid = int(row["entity_id"])
        sid = int(row["stage_id"])
        for col in hydro_value_cols:
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                novomodelo_lookup[(eid, sid, col)] = round(float(val), 2)

    # Compare matched pairs.
    for nw_code, (novomodelo_id, name, min_stor) in sorted(matched.items()):
        for (code, stage, var), nw_val in sorted(nw_lookup.items()):
            if code != nw_code:
                continue
            novomodelo_val = novomodelo_lookup.get((novomodelo_id, stage, var))
            if novomodelo_val is None:
                continue

            # The source model reports useful storage (storage - vol_min). Add
            # min_storage to align with Novomodelo absolute storage.
            if var == "storage_final_hm3":
                nw_val = nw_val + min_stor
            elif var in ("evaporation_m3s", "withdrawal_m3s"):
                # MEDIAS-USIH VEVAPUH and VRETIRUH are reported as monthly *volume* in
                # hm³, not flow. Novomodelo emits m³/s. The source model rounds the
                # month-length factor to 2.63 in its internal output reporting (≈ 730 h
                # × 3600 s / 10⁶); using the same rounded constant here makes the
                # comparison apples-to-apples. The converter side keeps the exact 2.628
                # (see network.py:C_M3S2HM3) because the input-data conversion has no
                # analogous rounding. Handled per-variable rather than via a V*-prefix
                # heuristic to avoid mis-scaling future MEDIAS columns that happen to
                # start with V.
                nw_val = nw_val / 2.63

            results.append(
                _make_result(
                    "hydro",
                    name,
                    nw_code,
                    novomodelo_id,
                    stage,
                    var,
                    nw_val,
                    novomodelo_val,
                )
            )

    # --- Derived: operational hydro productivity = generation / turbined ---
    # The effective ρ a plant achieves, in MW per m³/s. Stages where either
    # model turbined ~0 are filtered out (gen/turbined is 0/0 there — undefined
    # and noisy). Emitted as a "hydro" variable so it flows into both the
    # per-variable summary table and the Hydro Details per-plant tab.
    prod_stages: dict[int, set[int]] = {}
    for code, stage, var in nw_lookup:
        if var in ("generation_mw", "turbined_m3s"):
            prod_stages.setdefault(code, set()).add(stage)

    for nw_code, (novomodelo_id, name, _min_stor) in sorted(matched.items()):
        for stage in sorted(prod_stages.get(nw_code, ())):
            nw_gen = nw_lookup.get((nw_code, stage, "generation_mw"))
            nw_turb = nw_lookup.get((nw_code, stage, "turbined_m3s"))
            cb_gen = novomodelo_lookup.get((novomodelo_id, stage, "generation_mw"))
            cb_turb = novomodelo_lookup.get((novomodelo_id, stage, "turbined_m3s"))
            if nw_gen is None or nw_turb is None or cb_gen is None or cb_turb is None:
                continue
            # Filter turbined == 0 points out (both sides must turbine).
            if nw_turb <= _PRODUCTIVITY_TURB_EPS or cb_turb <= _PRODUCTIVITY_TURB_EPS:
                continue
            results.append(
                _make_result(
                    "hydro",
                    name,
                    nw_code,
                    novomodelo_id,
                    stage,
                    "productivity_mw_per_m3s",
                    round(nw_gen / nw_turb, 4),
                    round(cb_gen / cb_turb, 4),
                )
            )

    return results


def _compare_thermals(
    nw_thermal: pl.DataFrame,
    novomodelo_thermal: pl.DataFrame,
    nw_names: dict[int, str],
    novomodelo_meta: dict[int, dict],
) -> list[ResultComparison]:
    """Compare thermal results by matching plant names."""
    results: list[ResultComparison] = []

    offset = _nw_stage_offset(nw_thermal)

    # Name-based matching.
    novomodelo_by_name: dict[str, int] = {}
    for eid, meta in novomodelo_meta.items():
        novomodelo_by_name[meta["name"].strip().upper()] = eid

    matched: dict[int, tuple[int, str]] = {}  # nw_code→(novomodelo_id, name)
    for nw_code, nw_name in nw_names.items():
        hit = novomodelo_by_name.get(nw_name.strip().upper())
        if hit is not None:
            matched[nw_code] = (hit, nw_name.strip())

    nw_lookup: dict[tuple[int, int], float] = {}
    for row in nw_thermal.iter_rows(named=True):
        if row["value"] is None:
            continue
        code = int(row["newave_code"])
        if code not in matched:
            continue
        stage = int(row["stage"]) - offset
        nw_lookup[(code, stage)] = float(row["value"])

    novomodelo_lookup: dict[tuple[int, int], float] = {}
    for row in novomodelo_thermal.iter_rows(named=True):
        eid = int(row["entity_id"])
        sid = int(row["stage_id"])
        val = row.get("generation_mw")
        if val is not None:
            novomodelo_lookup[(eid, sid)] = round(float(val), 2)

    for nw_code, (novomodelo_id, name) in sorted(matched.items()):
        for (code, stage), nw_val in sorted(nw_lookup.items()):
            if code != nw_code:
                continue
            novomodelo_val = novomodelo_lookup.get((novomodelo_id, stage))
            if novomodelo_val is None:
                continue
            results.append(
                _make_result(
                    "thermal",
                    name,
                    nw_code,
                    novomodelo_id,
                    stage,
                    "generation_mw",
                    nw_val,
                    novomodelo_val,
                )
            )

    return results


def _compare_buses(
    nw_system: pl.DataFrame,
    novomodelo_bus: pl.DataFrame,
    nw_names: dict[int, str],
    novomodelo_meta: dict[int, dict],
) -> list[ResultComparison]:
    """Compare bus/subsystem results by matching names."""
    results: list[ResultComparison] = []

    offset = _nw_stage_offset(nw_system)

    # Name-based matching.
    novomodelo_by_name: dict[str, int] = {}
    for eid, meta in novomodelo_meta.items():
        novomodelo_by_name[meta["name"].strip().upper()] = eid

    matched: dict[int, tuple[int, str]] = {}  # nw_code→(novomodelo_id, name)
    for nw_code, nw_name in nw_names.items():
        hit = novomodelo_by_name.get(nw_name.strip().upper())
        if hit is not None:
            matched[nw_code] = (hit, nw_name.strip())

    nw_lookup: dict[tuple[int, int, str], float] = {}
    for row in nw_system.iter_rows(named=True):
        if row["value"] is None:
            continue
        code = int(row["newave_code"])
        if code not in matched:
            continue
        stage = int(row["stage"]) - offset
        var = str(row["variable"]).strip().upper()
        mapped = _SYSTEM_VAR_MAP.get(var, var.lower())
        nw_lookup[(code, stage, mapped)] = float(row["value"])

    novomodelo_lookup: dict[tuple[int, int, str], float] = {}
    for row in novomodelo_bus.iter_rows(named=True):
        eid = int(row["entity_id"])
        sid = int(row["stage_id"])
        for col in ("spot_price", "deficit_mw"):
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                novomodelo_lookup[(eid, sid, col)] = round(float(val), 2)

    for nw_code, (novomodelo_id, name) in sorted(matched.items()):
        for (code, stage, var), nw_val in sorted(nw_lookup.items()):
            if code != nw_code:
                continue
            novomodelo_val = novomodelo_lookup.get((novomodelo_id, stage, var))
            if novomodelo_val is None:
                continue
            results.append(
                _make_result(
                    "bus",
                    name,
                    nw_code,
                    novomodelo_id,
                    stage,
                    var,
                    nw_val,
                    novomodelo_val,
                )
            )

    return results


def _compare_convergence(
    nw_conv: pl.DataFrame,
    novomodelo_conv: pl.DataFrame,
) -> list[ResultComparison]:
    """Compare convergence data."""
    results: list[ResultComparison] = []

    def _load(df: pl.DataFrame) -> dict[int, dict[str, float]]:
        out: dict[int, dict[str, float]] = {}
        for row in df.iter_rows(named=True):
            lb = row["lower_bound"]
            ub = row["upper_bound_mean"]
            # Skip iterations where either bound is missing — robust against
            # pmo.dat layouts that inewave parses partially (NaNs in tail rows).
            if lb is None or ub is None:
                continue
            out[int(row["iteration"])] = {
                "lower_bound": float(lb),
                "upper_bound_mean": float(ub),
            }
        return out

    nw_lookup = _load(nw_conv)
    novomodelo_lookup = _load(novomodelo_conv)

    for it in sorted(set(nw_lookup) & set(novomodelo_lookup)):
        for var in ("lower_bound", "upper_bound_mean"):
            nw_val = nw_lookup[it][var]
            novomodelo_val = novomodelo_lookup[it][var]
            results.append(
                _make_result(
                    "convergence",
                    f"iteration_{it}",
                    it,
                    it,
                    it,
                    var,
                    nw_val,
                    novomodelo_val,
                )
            )

    return results


_LINE_VAR_MAP: dict[str, str] = {
    "INTERC": "net_flow_mw",
}


def _compare_lines(
    nw_intercambio: pl.DataFrame,
    novomodelo_line: pl.DataFrame,
    alignment: EntityAlignment,
    nw_offset: int,
) -> list[ResultComparison]:
    """Compare the source model intercâmbio to Novomodelo net line flow.

    The source model NWLISTOP ``int*.out`` files report directional flow (mean MW over
    the month) per (from, to) submarket pair. The ``alignment`` contains ``LineEntity``
    objects already matched by submarket-pair to Novomodelo line IDs (see
    ``build_entity_alignment``).

    Pre-study int*.out stages (the source model writes all absolute calendar months,
    including those before the study horizon) are filtered out by passing the
    MEDIAS-derived ``nw_offset`` — these rows would otherwise misalign with Novomodelo's
    ``stage_id`` numbering, which starts at the first study month.
    """
    results: list[ResultComparison] = []
    if nw_intercambio.is_empty() or novomodelo_line.is_empty():
        return results

    # (nw_de, nw_para) → (novomodelo_line_id, name)
    matched: dict[tuple[int, int], tuple[int, str]] = {}
    for ln in alignment.lines:
        matched[(ln.newave_de, ln.newave_para)] = (ln.novomodelo_line_id, ln.name)

    # The source model lookup: (novomodelo_line_id, stage_0based) -> mean MW.
    nw_lookup: dict[tuple[int, int], float] = {}
    for row in nw_intercambio.iter_rows(named=True):
        if row["value"] is None:
            continue
        key = (int(row["from_submarket_code"]), int(row["to_submarket_code"]))
        match = matched.get(key)
        if match is None:
            continue
        novomodelo_line_id, _name = match
        stage_0based = int(row["stage"]) - nw_offset
        if stage_0based < 0:
            continue
        nw_lookup[(novomodelo_line_id, stage_0based)] = float(row["value"])

    # Novomodelo lookup: (entity_id, stage_id) -> net_flow_mw.
    novomodelo_lookup: dict[tuple[int, int], float] = {}
    for row in novomodelo_line.iter_rows(named=True):
        eid = int(row["entity_id"])
        sid = int(row["stage_id"])
        val = row.get("net_flow_mw")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        novomodelo_lookup[(eid, sid)] = float(val)

    # Emit comparison rows in (line, stage) order.
    by_line: dict[int, str] = {ln.novomodelo_line_id: ln.name for ln in alignment.lines}
    code_by_line: dict[int, tuple[int, int]] = {
        ln.novomodelo_line_id: (ln.newave_de, ln.newave_para) for ln in alignment.lines
    }
    for novomodelo_line_id in sorted(by_line.keys()):
        for (eid, sid), nw_val in sorted(nw_lookup.items()):
            if eid != novomodelo_line_id:
                continue
            novomodelo_val = novomodelo_lookup.get((eid, sid))
            if novomodelo_val is None:
                continue
            de, _para = code_by_line.get(novomodelo_line_id, (0, 0))
            results.append(
                _make_result(
                    "line",
                    by_line[novomodelo_line_id],
                    de,
                    novomodelo_line_id,
                    sid,
                    "net_flow_mw",
                    nw_val,
                    round(novomodelo_val, 2),
                )
            )
    return results


_SYSTEM_SPILLAGE_VAR_MAP: dict[str, str] = {
    "VERTOT": "spill_energy_total_mw",
    "VERTCONT": "spill_energy_reservoir_mw",
    "VERTFIO": "spill_energy_rorov_mw",
}


def _compare_system_spillage(
    nw_sin: pl.DataFrame,
    novomodelo_spill_energy: pl.DataFrame,
) -> list[ResultComparison]:
    """Compare system spillage in MWmes between the source model SIN and Novomodelo.

    The source model MEDIAS-SIN reports ``VERTOT`` (total), ``VERTcont`` (reservoir
    cascades), and ``VERTfio`` (run-of-river) — all in MWmes (stage-mean MW). Novomodelo
    values come from :func:`read_novomodelo_spillage_energy` which already produces
    stage-mean MW per category.
    """
    results: list[ResultComparison] = []
    if nw_sin.is_empty() or novomodelo_spill_energy.is_empty():
        return results

    offset = _nw_stage_offset(nw_sin)

    # The source model side: (stage_0based, mapped_var) -> value.
    nw_lookup: dict[tuple[int, str], float] = {}
    for row in nw_sin.iter_rows(named=True):
        if row["value"] is None:
            continue
        var = str(row["variable"]).strip().upper()
        mapped = _SYSTEM_SPILLAGE_VAR_MAP.get(var)
        if mapped is None:
            continue
        stage = int(row["stage"]) - offset
        nw_lookup[(stage, mapped)] = float(row["value"])

    novomodelo_cols = {
        "spill_energy_total_mw": "total_mw",
        "spill_energy_reservoir_mw": "reservoir_mw",
        "spill_energy_rorov_mw": "rorov_mw",
    }
    novomodelo_lookup: dict[tuple[int, str], float] = {}
    for row in novomodelo_spill_energy.iter_rows(named=True):
        sid = int(row["stage_id"])
        for mapped, col in novomodelo_cols.items():
            val = row.get(col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            novomodelo_lookup[(sid, mapped)] = float(val)

    for (stage, mapped), nw_val in sorted(nw_lookup.items()):
        novomodelo_val = novomodelo_lookup.get((stage, mapped))
        if novomodelo_val is None:
            continue
        results.append(
            _make_result(
                "system_spillage",
                "SIN",
                0,
                0,
                stage,
                mapped,
                nw_val,
                novomodelo_val,
            )
        )
    return results


def compare_results(
    case: NewaveCase,
    id_map: NewaveIdMap,
    alignment: EntityAlignment,
    novomodelo_output_dir: Path,
    tolerance: float = 1e-2,
) -> ComparisonDataset:
    """Compare the source model output results against Novomodelo simulation means.

    Entities are matched by **name** (case-insensitive) rather than by
    converter-assigned IDs, so the comparison works even when the Novomodelo
    case was built by a different tool.

    Internally still builds the ``list[ResultComparison]`` and the
    ``PercentileData`` producer struct, then assembles and returns the canonical
    :class:`~novomodelo_bridge.comparators.dataset.ComparisonDataset` (the percentile
    frames and raw rows live in the dataset's render-only metadata).

    Returns
    -------
    ComparisonDataset
        The validated canonical comparison dataset.

    Parameters
    ----------
    case:
        Parsed the source model case (for names, cadastro, and locating pmo.dat).
    id_map:
        Entity ID mapping (used only for productivity fallback).
    alignment:
        Pre-built entity alignment (used only for productivity).
    novomodelo_output_dir:
        Path to Novomodelo output directory.
    tolerance:
        Relative tolerance for results comparison (informational).

    """
    from novomodelo_bridge.comparators.newave.alignment import read_reference_names
    from novomodelo_bridge.comparators.newave.readers import (
        read_fpha_grid,
        read_fpha_planes,
        read_medias_hydro,
        read_medias_market,
        read_medias_sin,
        read_medias_system,
        read_medias_thermal,
        read_newave_net_load,
        read_newave_tim_iterations,
        read_newave_tim_stages,
        read_nwlistop_intercambio,
        read_pmo_convergence,
        read_pmo_cost_breakdown,
        read_pmo_productivity_detail,
    )
    from novomodelo_bridge.newave.converters.hydro import read_cadastro
    from novomodelo_bridge.novomodelo.readers import (
        read_novomodelo_bus_aggregates,
        read_novomodelo_bus_means,
        read_novomodelo_bus_metadata,
        read_novomodelo_bus_percentiles,
        read_novomodelo_convergence,
        read_novomodelo_cost_breakdown,
        read_novomodelo_fpha_planes,
        read_novomodelo_hydro_bus_labels,
        read_novomodelo_hydro_means,
        read_novomodelo_hydro_metadata,
        read_novomodelo_hydro_per_stage_bounds,
        read_novomodelo_hydro_percentiles,
        read_novomodelo_hydro_total_flows,
        read_novomodelo_hydro_withdrawal,
        read_novomodelo_iteration_timing,
        read_novomodelo_line_bounds,
        read_novomodelo_line_means,
        read_novomodelo_line_percentiles,
        read_novomodelo_lp_max_generation,
        read_novomodelo_productivity_detail,
        read_novomodelo_spillage_energy,
        read_novomodelo_stage_costs,
        read_novomodelo_thermal_means,
        read_novomodelo_thermal_metadata,
        read_novomodelo_thermal_percentiles,
        read_novomodelo_training_duration,
    )

    results: list[ResultComparison] = []

    # Read entity names from both sides.
    nw_hydro_names, nw_thermal_names, nw_bus_names = read_reference_names(case)
    novomodelo_hydro_meta = read_novomodelo_hydro_metadata(novomodelo_output_dir)
    # ``read_novomodelo_hydro_metadata`` carries plant physics only;
    # merge in the plant->bus *label* from the 0.13 hydro_bus_generation
    # partition so ``dataset.metadata["novomodelo_hydro_meta"]`` -- the single
    # channel report_builder.py already threads into the per-bus hydro chart
    # helpers -- keeps carrying a bus label without those callers changing.
    # A plant absent from the partition (e.g. it has no simulation output at
    # all) simply gets no "bus_ids" key, matching the legacy "no bus" skip.
    for hid, bus_ids in read_novomodelo_hydro_bus_labels(novomodelo_output_dir).items():
        if hid in novomodelo_hydro_meta:
            # Sorted list, not the reader's frozenset: this lands in the
            # JSON-serialized metadata side-table, which rejects a frozenset.
            novomodelo_hydro_meta[hid]["bus_ids"] = sorted(bus_ids)
    novomodelo_thermal_meta = read_novomodelo_thermal_metadata(novomodelo_output_dir)
    novomodelo_bus_meta = read_novomodelo_bus_metadata(novomodelo_output_dir)

    # The source-model result files (MEDIAS-*.CSV, nwlistop-*.out) live directly
    # in the case directory (no saidas/ subfolder).
    source_dir = case.files.directory

    nw_offset = 0
    nw_max_stage_1based: int | None = None
    novomodelo_hydro = pl.DataFrame()
    nw_hydro = pl.DataFrame()
    nw_hydro_slacks = pl.DataFrame()

    # --- Hydro comparison ---
    if source_dir.is_dir():
        nw_hydro = read_medias_hydro(source_dir)
        novomodelo_hydro = read_novomodelo_hydro_means(novomodelo_output_dir)
        if not novomodelo_hydro.is_empty():
            # Merge derived per-(hydro, stage) quantities:
            # - total_inflow_m3s (≡ QAFLUH): incremental + Σ upstream outflow
            # - withdrawal_m3s (≡ VRETIRUH): input target from hydro_bounds
            total_flows = read_novomodelo_hydro_total_flows(
                novomodelo_output_dir, novomodelo_hydro
            )
            if not total_flows.is_empty():
                novomodelo_hydro = novomodelo_hydro.join(
                    total_flows, on=["entity_id", "stage_id"], how="left"
                )
            withdrawal = read_novomodelo_hydro_withdrawal(novomodelo_output_dir)
            if not withdrawal.is_empty():
                novomodelo_hydro = novomodelo_hydro.join(
                    withdrawal, on=["entity_id", "stage_id"], how="left"
                )
            # Reconstruct *realized* evaporation and withdrawal flows from
            # the LP slacks (Novomodelo's matrix.rs water-balance row uses
            #    +ζ × pos_slack − ζ × neg_slack = ζ × (base − scheduled)
            # so realized = scheduled + pos − neg).
            # ``evaporation_m3s`` from the simulation parquet is already
            # the LP-output (realized) flow — adding pos/neg slacks
            # represents the same correction applied symmetrically on
            # both sides so the comparison is apples-to-apples.
            recon_cols = {
                "evaporation_m3s": (
                    "evaporation_violation_pos_m3s",
                    "evaporation_violation_neg_m3s",
                ),
                "withdrawal_m3s": (
                    "water_withdrawal_violation_pos_m3s",
                    "water_withdrawal_violation_neg_m3s",
                ),
            }
            adjustments: list[pl.Expr] = []
            for base, (pos, neg) in recon_cols.items():
                if (
                    base in novomodelo_hydro.columns
                    and pos in novomodelo_hydro.columns
                    and neg in novomodelo_hydro.columns
                ):
                    adjustments.append(
                        (
                            pl.col(base).fill_null(0.0)
                            + pl.col(pos).fill_null(0.0)
                            - pl.col(neg).fill_null(0.0)
                        ).alias(base)
                    )
            if adjustments:
                novomodelo_hydro = novomodelo_hydro.with_columns(adjustments)
        if not nw_hydro.is_empty():
            nw_offset = _nw_stage_offset(nw_hydro)
            stages_col = nw_hydro["stage"].drop_nulls()
            if not stages_col.is_empty():
                max_val = stages_col.max()
                if max_val is not None:
                    nw_max_stage_1based = int(max_val)  # type: ignore[arg-type]
        if not novomodelo_hydro.is_empty():
            # Generation upper-bound overlay (the source model GHMAX_FPHC vs Novomodelo LP).
            gen_max_overlay = _build_gen_max_overlay(
                nw_hydro,
                read_novomodelo_lp_max_generation(novomodelo_output_dir),
                nw_hydro_names,
                novomodelo_hydro_meta,
                nw_offset,
            )
            if not gen_max_overlay.is_empty():
                novomodelo_hydro = novomodelo_hydro.join(
                    gen_max_overlay, on=["entity_id", "stage_id"], how="left"
                )
        if not nw_hydro.is_empty() and not novomodelo_hydro.is_empty():
            _LOG.info("Comparing hydro results...")
            results.extend(
                _compare_hydros(
                    nw_hydro, novomodelo_hydro, nw_hydro_names, novomodelo_hydro_meta
                )
            )
        nw_hydro_slacks = _compute_nw_hydro_slacks(
            nw_hydro, nw_hydro_names, novomodelo_hydro_meta
        )

        # --- Thermal comparison ---
        nw_thermal = read_medias_thermal(source_dir)
        novomodelo_thermal = read_novomodelo_thermal_means(novomodelo_output_dir)
        if not nw_thermal.is_empty() and not novomodelo_thermal.is_empty():
            _LOG.info("Comparing thermal results...")
            results.extend(
                _compare_thermals(
                    nw_thermal,
                    novomodelo_thermal,
                    nw_thermal_names,
                    novomodelo_thermal_meta,
                )
            )

        # --- Bus/system comparison ---
        nw_system = read_medias_system(source_dir)
        novomodelo_bus = read_novomodelo_bus_means(novomodelo_output_dir)
        if not nw_system.is_empty() and not novomodelo_bus.is_empty():
            _LOG.info("Comparing bus results...")
            results.extend(
                _compare_buses(
                    nw_system, novomodelo_bus, nw_bus_names, novomodelo_bus_meta
                )
            )

        # --- Line interchange comparison (NWLISTOP int*.out) --- nw_offset is the
        # MEDIAS-derived study-start offset (e.g., 9 for a September-start study).
        # int*.out files emit absolute the source model stages including pre-study
        # calendar months with all-zero values, so we reuse this offset to filter them
        # out and align the remainder with Novomodelo's 0-based stage_id.
        nw_intercambio = read_nwlistop_intercambio(source_dir)
        novomodelo_line = read_novomodelo_line_means(novomodelo_output_dir)
        if not nw_intercambio.is_empty() and not novomodelo_line.is_empty():
            _LOG.info("Comparing line interchange...")
            results.extend(
                _compare_lines(nw_intercambio, novomodelo_line, alignment, nw_offset)
            )
    else:
        _LOG.warning(
            "NEWAVE source result files not found in the case directory; "
            "skipping MEDIAS comparison."
        )
        nw_intercambio = pl.DataFrame()
        novomodelo_line = pl.DataFrame()

    # --- Convergence comparison ---
    nw_conv = read_pmo_convergence(case.files.directory)
    novomodelo_conv = read_novomodelo_convergence(novomodelo_output_dir)
    if not nw_conv.is_empty() and not novomodelo_conv.is_empty():
        _LOG.info("Comparing convergence data...")
        results.extend(_compare_convergence(nw_conv, novomodelo_conv))

    # --- Productivity detail (static conversion-fidelity check) --- Per-plant the
    # source model pmo productivities vs the *static* productivities novomodelo-bridge
    # computes from the same HIDR cadastro + cascade, plus the converted building blocks
    # — assembled for the Productivity tab.
    _LOG.info("Building productivity detail...")
    nw_prod_detail = read_pmo_productivity_detail(case.files.directory)
    novomodelo_prod_detail = read_novomodelo_productivity_detail(novomodelo_output_dir)
    cb_accumulated: dict[int, float] = {}
    try:
        nw_cadastro = read_cadastro(case)
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to read HIDR cadastro for productivity building blocks")
        nw_cadastro = pd.DataFrame()
    if not nw_cadastro.empty:
        try:
            from novomodelo_bridge.newave.converters.constraints import (
                compute_accumulated_integrated_productivities,
            )

            confhd_df = case.confhd.usinas
            cb_accumulated = compute_accumulated_integrated_productivities(
                nw_cadastro, confhd_df
            )
        except Exception:  # noqa: BLE001
            _LOG.warning("Failed to compute accumulated productivities for the tab")
    from novomodelo_bridge.comparators.analyze import build_productivity_detail

    productivity_detail = build_productivity_detail(
        alignment, nw_prod_detail, nw_cadastro, novomodelo_prod_detail, cb_accumulated
    )

    # --- Production-function (FPHA) comparison --- Both solvers' fitted
    # generation hyperplanes, evaluated on a shared (V, Q) grid. Populated only
    # when both sides fitted FPHA planes; otherwise the three frames stay None
    # and the Productivity tab shows only the constant-productivity content.
    fpha_metrics: pl.DataFrame | None = None
    fpha_surface: pl.DataFrame | None = None
    fpha_spill: pl.DataFrame | None = None
    cb_fpha = read_novomodelo_fpha_planes(novomodelo_output_dir)
    if cb_fpha is not None:
        nw_fpha = read_fpha_planes(case.files.directory)
        nw_fpha_grid = read_fpha_grid(case.files.directory)
        if nw_fpha is not None and nw_fpha_grid is not None:
            _LOG.info("Comparing fitted production functions (FPHA)...")
            from novomodelo_bridge.comparators.analyze import build_fpha_comparison

            # Surfaces are sampled at the source model's own (V, Q) fitting-grid
            # nodes (light + faithful to the model's resolution).
            fpha_metrics, fpha_surface, fpha_spill = build_fpha_comparison(
                nw_fpha, nw_fpha_grid, cb_fpha, alignment.hydros
            )

    # --- Cost breakdown --- The source model typically runs a shorter horizon than
    # Novomodelo.  Restrict Novomodelo's cost sum to the source model's stage range so the totals
    # compare like- for-like.  ``nw_max_stage_1based`` is the largest stage label
    # appearing in MEDIAS files; convert to Novomodelo's 0-based stage_id by subtracting the
    # source model start-month offset.
    nw_max_stage_0based: int | None = None
    if nw_max_stage_1based is not None:
        nw_max_stage_0based = nw_max_stage_1based - nw_offset

    _LOG.info(
        "Reading cost breakdowns (NEWAVE max stage_0based=%s)...", nw_max_stage_0based
    )
    nw_costs = read_pmo_cost_breakdown(case.files.directory)
    novomodelo_costs = read_novomodelo_cost_breakdown(
        novomodelo_output_dir, max_stage_id=nw_max_stage_0based
    )
    novomodelo_stage_costs = read_novomodelo_stage_costs(novomodelo_output_dir)
    if nw_max_stage_0based is not None and not novomodelo_stage_costs.is_empty():
        novomodelo_stage_costs = novomodelo_stage_costs.filter(
            pl.col("stage_id") <= nw_max_stage_0based
        )

    novomodelo_hydro_per_stage_bounds = read_novomodelo_hydro_per_stage_bounds(
        novomodelo_output_dir
    )
    if (
        nw_max_stage_0based is not None
        and not novomodelo_hydro_per_stage_bounds.is_empty()
    ):
        novomodelo_hydro_per_stage_bounds = novomodelo_hydro_per_stage_bounds.filter(
            pl.col("stage_id") <= nw_max_stage_0based
        )
    if nw_max_stage_0based is not None and not nw_hydro_slacks.is_empty():
        nw_hydro_slacks = nw_hydro_slacks.filter(
            pl.col("stage_id") <= nw_max_stage_0based
        )

    # --- Bus-level energy balance ---
    _LOG.info("Computing bus-level aggregates...")
    bus_aggregates = read_novomodelo_bus_aggregates(novomodelo_output_dir)
    nw_market = pl.DataFrame()
    nw_sin = pl.DataFrame()
    if source_dir.is_dir():
        nw_market = read_medias_market(source_dir)
        nw_sin = read_medias_sin(source_dir)

    # --- the source model deterministic net load (load - NCS from sistema.dat) ---
    nw_net_load = read_newave_net_load(case.files)

    # --- Percentile statistics ---
    _LOG.info("Computing Novomodelo percentile statistics...")
    hydro_pct = read_novomodelo_hydro_percentiles(novomodelo_output_dir)
    thermal_pct = read_novomodelo_thermal_percentiles(novomodelo_output_dir)
    bus_pct = read_novomodelo_bus_percentiles(novomodelo_output_dir)
    line_pct = read_novomodelo_line_percentiles(novomodelo_output_dir)

    # --- Line bounds (per stage) and line metadata for the Network tab ---
    line_bounds = read_novomodelo_line_bounds(novomodelo_output_dir)
    lines_json_path = case_dir_for(novomodelo_output_dir) / "system" / "lines.json"
    line_meta: list[dict] = []
    if lines_json_path.exists():
        try:
            import json as _json

            line_meta = _json.loads(lines_json_path.read_text()).get("lines", [])
        except Exception:  # noqa: BLE001
            _LOG.warning("Failed to read lines.json")

    # The source model typically reports a shorter horizon than Novomodelo.  Truncate all
    # Novomodelo-side per-stage DataFrames to the source model's max stage so the report's
    # charts compare like-for-like across tabs (energy balance, hydro operation, plant
    # details, etc.).  Truncation is a no-op when the source model stage data was
    # unavailable.
    if nw_max_stage_0based is not None:

        def _truncate(df: pl.DataFrame) -> pl.DataFrame:
            if df.is_empty() or "stage_id" not in df.columns:
                return df
            return df.filter(pl.col("stage_id") <= nw_max_stage_0based)

        novomodelo_hydro = _truncate(novomodelo_hydro)
        hydro_pct = _truncate(hydro_pct)
        thermal_pct = _truncate(thermal_pct)
        bus_pct = _truncate(bus_pct)
        bus_aggregates = _truncate(bus_aggregates)
        line_pct = _truncate(line_pct)
        novomodelo_line = _truncate(novomodelo_line)

    # --- Performance timings ---
    nw_tim_iters = read_newave_tim_iterations(case.files.directory)
    nw_tim_stages = read_newave_tim_stages(case.files.directory)
    novomodelo_training_seconds = read_novomodelo_training_duration(
        novomodelo_output_dir
    )
    novomodelo_iter_timing = read_novomodelo_iteration_timing(novomodelo_output_dir)

    # --- Generic constraints LHS comparison (RE / AGRINT / VminOP) ---
    # The Novomodelo case directory is one level above the simulation output dir
    # (novomodelo_output_dir ends in ``output/``).  Both the constraint
    # definitions and the per-stage bound parquet live in the case's
    # ``constraints/`` subdirectory.
    from novomodelo_bridge.comparators.constraints import (
        evaluate_lhs_novomodelo,
        load_generic_constraint_bounds,
        load_generic_constraints,
    )
    from novomodelo_bridge.comparators.newave.constraints import (
        apply_vminop_useful_energy,
        evaluate_lhs_newave,
    )

    novomodelo_case_dir = case_dir_for(novomodelo_output_dir)
    gc_constraints = load_generic_constraints(novomodelo_case_dir)
    gc_bounds_df = load_generic_constraint_bounds(novomodelo_case_dir)
    if gc_constraints and source_dir.is_dir():
        _LOG.info(
            "Comparing %d generic constraints (RE/AGRINT/VminOP)...",
            len(gc_constraints),
        )
        gc_lhs_nw = evaluate_lhs_newave(
            gc_constraints,
            nw_hydro,
            nw_intercambio,
            alignment,
            id_map,
            nw_offset,
        )
        gc_lhs_cb = evaluate_lhs_novomodelo(gc_constraints, novomodelo_output_dir)
        # VminOP constraints bound stored energy; re-express their LHS/bound as useful
        # stored energy (MWmonth) so they compare like-for-like against The source
        # model's per-REE EARMF from MEDIAS-REE.CSV.  RE/AGRINT are untouched.
        gc_bounds_df, gc_lhs_nw, gc_lhs_cb = apply_vminop_useful_energy(
            gc_constraints,
            gc_bounds_df,
            gc_lhs_nw,
            gc_lhs_cb,
            novomodelo_case_dir,
            novomodelo_output_dir,
            nw_hydro,
            id_map,
            nw_offset,
        )
    else:
        gc_lhs_nw = pl.DataFrame()
        gc_lhs_cb = pl.DataFrame()

    # --- System spillage in MWmes ---
    novomodelo_spill_energy = read_novomodelo_spillage_energy(novomodelo_output_dir)
    if nw_max_stage_0based is not None and not novomodelo_spill_energy.is_empty():
        novomodelo_spill_energy = novomodelo_spill_energy.filter(
            pl.col("stage_id") <= nw_max_stage_0based
        )
    if not nw_sin.is_empty() and not novomodelo_spill_energy.is_empty():
        _LOG.info("Comparing system spillage energy...")
        results.extend(_compare_system_spillage(nw_sin, novomodelo_spill_energy))

    pctiles = PercentileData(
        hydro=hydro_pct,
        thermal=thermal_pct,
        bus=bus_pct,
        bus_aggregates=bus_aggregates,
        nw_convergence=nw_conv,
        novomodelo_convergence=novomodelo_conv,
        nw_market=nw_market,
        nw_net_load=nw_net_load,
        nw_sin=nw_sin,
        novomodelo_hydro_means=novomodelo_hydro,
        novomodelo_bus_meta=novomodelo_bus_meta,
        novomodelo_hydro_meta=novomodelo_hydro_meta,
        nw_bus_names=nw_bus_names,
        nw_hydro_names=nw_hydro_names,
        nw_costs=nw_costs,
        novomodelo_costs=novomodelo_costs,
        novomodelo_stage_costs=novomodelo_stage_costs,
        novomodelo_hydro_per_stage_bounds=novomodelo_hydro_per_stage_bounds,
        nw_hydro_slacks=nw_hydro_slacks,
        nw_offset=nw_offset,
        nw_max_stage=nw_max_stage_0based,
        novomodelo_spillage_energy=novomodelo_spill_energy,
        line=line_pct,
        line_bounds=line_bounds,
        line_meta=line_meta,
        nw_line_means=nw_intercambio,
        nw_tim_iterations=nw_tim_iters,
        nw_tim_stages=nw_tim_stages,
        novomodelo_training_seconds=novomodelo_training_seconds,
        novomodelo_iteration_timing=novomodelo_iter_timing,
        gc_constraints=gc_constraints,
        gc_bounds=gc_bounds_df,
        gc_lhs_newave=gc_lhs_nw,
        gc_lhs_novomodelo=gc_lhs_cb,
        productivity_detail=productivity_detail,
        fpha_metrics=fpha_metrics,
        fpha_surface=fpha_surface,
        fpha_spill=fpha_spill,
    )

    _LOG.info("Results comparison: %d total comparisons", len(results))
    from novomodelo_bridge.comparators.analyze import build_results_dataset

    return build_results_dataset(results, pctiles, tolerance)

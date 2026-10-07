"""Network entity converter: maps the source model bus and line data to Cobre network
JSON."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence

import pandas as pd
import pyarrow as pa

from cobre_bridge.cobre import schemas as cobre_schemas
from cobre_bridge.core.pandas_utils import is_na
from cobre_bridge.core.penalties import PCORTEOL, PEXC, PINT, hydro_penalty_costs
from cobre_bridge.newave.case import NewaveCase
from cobre_bridge.newave.horizon import POST_STUDY_YEAR, historical_start_date
from cobre_bridge.newave.id_map import NewaveIdMap

_LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Penalty conversion constants
# --------------------------------------------------------------------------
#
# Source: The source model User Manual v30 (the source model/ONS, 2023), section 3.24
# "Penalidades
# (Ex.: Penalid.dat)" and the internal-default tables on pages 87–88.
#
# Time-aspect summary (re-derived from cobre's SDDP LP builder, matrix.rs):
#
# Every cobre penalty coefficient in penalties.json is multiplied by some
# hours quantity before entering the LP objective. The variable it sits on
# (MW, m³/s, or hm³) and the time multiplier (block_hours, total_stage_hours,
# or none) together determine the unit the user-facing value must be in.
# The pattern is "(penalty × hours) × variable_value = R$":
#
# Family A — Power columns (MW), per block. Variable carries MW for one block.
#   Cobre: `objective[col] = penalty × block_hours`
#   Cost  = (penalty × block_h) × MW = penalty × MWh → penalty unit R$/MWh.
#   Affected: bus.deficit_segments[].cost, bus.excess_cost, line.exchange_cost,
# ncs.curtailment_cost, hydro.generation_violation_below_cost. → Conversion from the
# source model R$/MWh: **direct** (no productivity factor).
#
# Family B — Flow columns (m³/s), per block. Variable carries m³/s for one block.
#   Cobre: `objective[col] = penalty × block_hours`
# Cost  = (penalty × block_h) × m³/s.  For this to equal R$ the penalty must be R$/(m³/s
# · h). The source model supplies R$/MWh; the per-flow-per-hour
#   form requires multiplying by ρ [MW/(m³/s)]:
#     R$/MWh × MW/(m³/s) = R$/(m³/s · h).
#   Affected: hydro.spillage_cost, hydro.turbined_cost, hydro.diversion_cost,
#   hydro.outflow_violation_(below|above)_cost, hydro.turbined_violation_below_cost.
#   → Conversion: **× ρ_avg** (`PROD_MEDIA_SIN`).
#
# Family C — Flow columns (m³/s), per stage. Same as B but cobre uses
# `total_stage_hours` instead of `block_hours`.
#   Affected: hydro.water_withdrawal_violation_(pos|neg)_cost,
#   hydro.evaporation_violation_(pos|neg)_cost,
# hydro.inflow_nonnegativity_cost. Cobre's docstring on evaporation_violation_cost says
# "$/mm" but the actual LP column (matrix.rs) reads f_evap_plus/minus as flow
# rates in m³/s — same unit as withdrawal. The "_m3s" suffix in the simulation output
# `evaporation_violation_pos_m3s` confirms this. → Conversion: **× ρ_max_acum**
# (`MAX_PRODTACUM_SIN`) for DESVIO and evaporation (per source-model manual p.87); **×
# ρ_avg** for the others.
#
# Family D — Volume columns (hm³), per stage. Variable carries hm³ once per stage.
#   Cobre: `objective[col] = penalty` (no time multiplier).
#   Cost  = penalty × hm³ = R$ → penalty unit R$/hm³.
#   Affected: hydro.storage_violation_below_cost, hydro.filling_target_violation_cost
#   (both currently NOT wired into the LP — slot is dormant).
#   Conversion: 1 hm³ × ρ → MWh of energy-equivalent is `(1e6 m³ / 3600 s/h) × ρ`
#   = 277.78 × ρ MWh (purely volumetric — 730h convention cancels). So
#     cobre_coef [R$/hm³] = source_R$/MWh × ρ × HM3_TO_MWH_PER_RHO
#   with HM3_TO_MWH_PER_RHO = 1e6/3600 ≈ 277.78.
#   Both slots are now DERIVED from the deficit cost via ρ_max_acum, not
#   hard-coded: storage_violation_below_cost = 10 × MAX_CUSTO_DEFICIT × ρ_max_acum
#   × HM3_TO_MWH_PER_RHO (the evaporation level, i.e. the greatest hydro penalty —
#   it equals evaporation_cost × HM3_TO_MWH_PER_RHO); filling_target_violation_cost
#   = 0.9 × MAX_CUSTO_DEFICIT × ρ_max_acum × HM3_TO_MWH_PER_RHO (a little below the
#   deficit cost). Neither is a hard-coded placeholder any more.
#
# The source model's 730 h-per-month convention vs cobre's actual calendar block_hours
# (672–744 h) introduces only a ±2% numerical drift in absolute LP cost for the Families
# above — *because each is converted consistently*: flow/power penalties carry no
# fixed-month factor (cobre integrates them with the real per-stage block_hours), and
# volume penalties carry no time multiplier (the 730 cancels in the pure-volumetric HM3
# → MWh conversion). When that holds, all costs scale together and merit order is
# preserved.
#
# ⚠️ CAVEAT — the assumption fails for any energy/STOCK quantity that cobre then
# prices *with* a `× block_hours` time multiplier. There the fixed 730 does NOT
# cancel: the converted energy uses 730 while cobre integrates the slack with
# the actual month hours, so the effective penalty drifts by block_hours/730 and
# CAN flip merit order. This bit the VminOP generic constraint (security curve):
# its LHS `Σ ρ_acum·storage` was left in ρ_acum·hm³ (≈ 2.628× the true MWmonth)
# while cobre priced the slack `× block_hours`, pushing the effective curve
# penalty above the deficit cost so cobre deficited instead of drawing down.
# Fixed by converting ρ_acum to MWmonth/hm³ *per stage* — see
# `converters/constraints.py:_vminop_energy_factor`. Any future LP constraint or
# penalty on a stock (storage/energy) priced `× block_hours` must do the same.
#
# Merit order from the source model micro-penalty values ("the source model
# individualizado"
# column, manual §3.24 p.88, v30 defaults):
#   p_INT (0.000273) < p_PFIO = p_EVERT (0.000300)
#   < p_TURB (0.000333) < p_CORTEOL (0.000344) < p_EXC (0.000355)

# The source model halves the intercâmbio penalty on lines that touch a fictitious
# submercado (e.g. NOFICT1). Rationale: a fictitious node is a routing-only hop with no
# demand or generation of its own, so a real → fict → real path would otherwise
# accumulate twice the penalty of an equivalent direct real → real link. The 0.5
# discount restores cost-parity between the two topologies.  Emitted as the per-line
# `exchange_cost` override defined in lines.schema.json; absence falls back to the
# global `PINT` value.
_PINT_FICTITIOUS_DISCOUNT = 0.5

# --- Soft fallback for ELETRI when PENALID is silent ----------------------- The source
# model's behaviour when ELETRI is absent: use the constraint only in final simulation,
# not in policy. Cobre can't represent that nuance, so we keep the slack enabled with a
# high penalty (10 × MAX_DEFICIT, matching The source model's evaporation/FPHA default
# magnitude).
_ELETRI_HIGH_MULT = 10.0


def _build_canonical_pair_to_line_id(
    case: NewaveCase,
) -> dict[tuple[int, int], int]:
    """Build the canonical (src, tgt) -> line_id mapping from sistema.dat.

    Scans ALL rows of ``sistema.limites_intercambio`` (all dates) to discover
    the full set of interchange pairs.  This is the single authoritative source
    used by ``convert_lines`` and ``convert_line_bounds`` to guarantee
    consistent line IDs.
    """
    sistema = case.sistema
    limites_df = sistema.limites_intercambio
    if limites_df is None or limites_df.empty:
        return {}

    all_pairs: set[tuple[int, int]] = set()
    for _, row in limites_df.iterrows():
        de = int(row["submercado_de"])
        para = int(row["submercado_para"])
        src, tgt = (de, para) if de < para else (para, de)
        all_pairs.add((src, tgt))

    return {pair: lid for lid, pair in enumerate(sorted(all_pairs))}


def convert_buses(case: NewaveCase, id_map: NewaveIdMap) -> dict:
    """Convert the source model subsystem data to a Cobre ``buses.json`` dict.

    Reads ``sistema.dat`` from *case*.  Each subsystem (including
    fictitious ones) becomes a bus.  Deficit segments are extracted from
    ``Sistema.custo_deficit``.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Pre-built ID mapping for bus IDs.
    """
    sistema = case.sistema
    deficit_df = sistema.custo_deficit

    if deficit_df is None:
        raise ValueError(
            "sistema.dat contains no deficit cost data (custo_deficit is None)"
        )

    # Columns: codigo_submercado, nome_submercado, ficticio,
    # patamar_deficit, custo, corte
    buses_by_code: dict[int, dict] = {}

    for _, row in deficit_df.iterrows():
        code = int(row["codigo_submercado"])
        name = str(row["nome_submercado"]).strip()

        if code not in buses_by_code:
            buses_by_code[code] = {
                "newave_code": code,
                "name": name,
                "segments": [],
            }

        corte = row.get("corte")
        depth_mw: float | None = (
            float(corte) if corte is not None and not is_na(corte) else None
        )
        cost_raw = row["custo"]
        cost = float(cost_raw) if not is_na(cost_raw) else None
        buses_by_code[code]["segments"].append(
            {
                "patamar": int(row["patamar_deficit"]),
                "depth_mw": depth_mw,
                "cost": cost,
            }
        )

    # Find the reference deficit cost (first non-NaN, non-zero cost across
    # all subsystems) to use as a fallback for fictitious subsystems.
    fallback_cost = 0.0
    for info in buses_by_code.values():
        for seg in info["segments"]:
            if seg["cost"] is not None and seg["cost"] > 0:
                fallback_cost = seg["cost"]
                break
        if fallback_cost > 0:
            break

    # Buses model network subsystems, which have no commissioning date; treat them
    # as in service since the historical record (Cobre uses the date only as a
    # canonical-ordering key, tiebroken by id).
    op_date = historical_start_date(case.dger)

    buses: list[dict] = []
    for code, info in buses_by_code.items():
        segs = sorted(info["segments"], key=lambda s: s["patamar"])
        active_segs = [s for s in segs if s["cost"] is not None and s["cost"] > 0]
        if not active_segs:
            active_segs = [{"cost": fallback_cost, "depth_mw": None}]

        deficit_segments: list[dict] = []
        for i, seg in enumerate(active_segs):
            is_last = i == len(active_segs) - 1
            deficit_segments.append(
                {
                    "depth_mw": None if is_last else seg["depth_mw"],
                    "cost": seg["cost"],
                }
            )

        bus_entry: dict = {
            "id": id_map.bus_id(code),
            "name": info["name"],
            "operational_start_date": op_date,
            "deficit_segments": deficit_segments,
        }
        buses.append(bus_entry)

    buses.sort(key=lambda b: b["id"])

    return {
        "$schema": cobre_schemas.schema_url_for("system/buses.json"),
        "buses": buses,
    }


def convert_bus_penalty_overrides(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> pa.Table | None:
    """Build ``constraints/penalty_overrides_bus.parquet`` for fictitious buses.

    The source model fictitious submarkets (``custo_deficit.ficticio``) are pure
    transshipment nodes: no real load/generation, and the source model forbids energy
    excess there.  Cobre has no hard per-bus "no excess" flag, so we override the
    per-bus ``excess_cost`` (sparse, per stage) to the deficit cost — symmetric
    with unserved energy — without which Cobre dumps surplus energy at the
    fictitious node for ~free.

    Returns ``None`` when the case has no fictitious submarkets (no file
    emitted, mirroring the sparse-override contract).

    Returns
    -------
    pyarrow.Table | None
        Columns: ``bus_id`` (INT32), ``stage_id`` (INT32),
        ``excess_cost`` (DOUBLE) — one row per (fictitious bus, stage).
    """
    sistema = case.sistema
    deficit_df = sistema.custo_deficit

    if deficit_df is None or deficit_df.empty or "ficticio" not in deficit_df.columns:
        return None

    fic_mask = deficit_df["ficticio"].fillna(False).astype(bool)
    fictitious_codes = {
        int(code) for code in deficit_df.loc[fic_mask, "codigo_submercado"].unique()
    }
    if not fictitious_codes:
        return None

    # Excess at a fictitious node is penalised at the deficit cost — symmetric
    # with unserved energy, and equal to the deficit_segments cost convert_buses
    # assigns to these buses. Uses the reference (first non-null, positive)
    # deficit cost from custo_deficit.
    deficit_costs = [
        float(c)
        for c in deficit_df["custo"].tolist()
        if c is not None and not is_na(c) and float(c) > 0.0
    ]
    if not deficit_costs:
        return None
    excess_cost = deficit_costs[0]

    total_stages = case.horizon.total_stages

    bus_ids: list[int] = []
    stage_ids: list[int] = []
    costs: list[float] = []
    for code in sorted(fictitious_codes):
        try:
            bus_id = id_map.bus_id(code)
        except KeyError:
            continue
        for stage_id in range(total_stages):
            bus_ids.append(bus_id)
            stage_ids.append(stage_id)
            costs.append(excess_cost)

    if not bus_ids:
        return None

    schema = pa.schema(
        [
            pa.field("bus_id", pa.int32()),
            pa.field("stage_id", pa.int32()),
            pa.field("excess_cost", pa.float64()),
        ]
    )
    return pa.table(
        {
            "bus_id": pa.array(bus_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "excess_cost": pa.array(costs, type=pa.float64()),
        },
        schema=schema,
    )


def _assign_flow_direction(
    caps: dict[str, float], de: int, para: int, sentido: int, valor: float
) -> None:
    """Assign *valor* into the direct/reverse slot of *caps* in place.

    ``sentido == 0`` is the first inewave block (de -> para) and
    ``sentido == 1`` is the second (para -> de); canonicalized against the
    (src, tgt) ordering with src < tgt used as the dict key.
    """
    if de < para:
        if sentido == 0:
            caps["direct_mw"] = valor
        else:
            caps["reverse_mw"] = valor
    else:
        if sentido == 0:
            caps["reverse_mw"] = valor
        else:
            caps["direct_mw"] = valor


def convert_lines(case: NewaveCase, id_map: NewaveIdMap) -> dict:
    """Convert the source model interchange limits to a Cobre ``lines.json`` dict.

    Reads ``sistema.dat`` from *case*.  Each directional interchange
    pair becomes a line using the first study month's limits as static
    capacities.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Pre-built ID mapping for bus IDs.
    """
    sistema = case.sistema
    limites_df = sistema.limites_intercambio

    if limites_df is None or limites_df.empty:
        return {
            "$schema": cobre_schemas.schema_url_for("system/lines.json"),
            "lines": [],
        }

    # Set of the source model codes flagged as fictitious in sistema.custo_deficit. Used
    # below to halve the exchange penalty on lines that touch a fictitious bus (see
    # `_PINT_FICTITIOUS_DISCOUNT`).
    fictitious_codes: set[int] = set()
    deficit_df = sistema.custo_deficit
    if (
        deficit_df is not None
        and not deficit_df.empty
        and "ficticio" in deficit_df.columns
    ):
        fic_mask = deficit_df["ficticio"].fillna(False).astype(bool)
        fictitious_codes = {
            int(code) for code in deficit_df.loc[fic_mask, "codigo_submercado"].unique()
        }

    # Use the study start month from dger.dat as the reference for static
    # capacities.  sistema.dat always contains full calendar years, so
    # pre-study months (before mes_inicio_estudo) may have NaN values.
    from datetime import datetime as _dt

    dger = case.dger
    study_start_dt = _dt(dger.ano_inicio_estudo, dger.mes_inicio_estudo, 1)
    first_month = limites_df[limites_df["data"] == study_start_dt]
    if first_month.empty:
        first_month = limites_df.dropna(subset=["valor"])
        if not first_month.empty:
            first_date = first_month["data"].min()
            first_month = limites_df[limites_df["data"] == first_date]

    # Build a (source, target) -> {direct_mw, reverse_mw} structure.
    # In inewave's parsed SISTEMA.DAT, sentido == 0 is the first block
    # (de -> para, i.e. A->B) and sentido == 1 is the second block
    # (para -> de, i.e. B->A).
    # We normalise all pairs so source_code < target_code to deduplicate.
    pair_map: dict[tuple[int, int], dict[str, float]] = {}

    for _, row in first_month.iterrows():
        de = int(row["submercado_de"])
        para = int(row["submercado_para"])
        valor = float(row["valor"])
        sentido = int(row["sentido"])

        # Canonical key: smaller ID first.
        src, tgt = (de, para) if de < para else (para, de)
        key = (src, tgt)

        if key not in pair_map:
            pair_map[key] = {"direct_mw": 0.0, "reverse_mw": 0.0}

        _assign_flow_direction(pair_map[key], de, para, sentido, valor)

    # Use the shared canonical mapping for consistent line IDs.
    canonical_map = _build_canonical_pair_to_line_id(case)

    # Interconnections have no commissioning date; treat every line as in service
    # since the historical record (Cobre uses the date only as a canonical-ordering
    # key, tiebroken by id).
    op_date = historical_start_date(dger)

    lines: list[dict] = []
    for (src, tgt), line_id in sorted(canonical_map.items(), key=lambda x: x[1]):
        caps = pair_map.get((src, tgt), {"direct_mw": 0.0, "reverse_mw": 0.0})
        src_bus = id_map.bus_id(src)
        tgt_bus = id_map.bus_id(tgt)
        src_name = _subsystem_name_from_id(src)
        tgt_name = _subsystem_name_from_id(tgt)
        line_entry: dict = {
            "id": line_id,
            "name": f"{src_name}_{tgt_name}",
            "operational_start_date": op_date,
            "source_bus_id": src_bus,
            "target_bus_id": tgt_bus,
            "capacity": {
                "direct_mw": caps["direct_mw"],
                "reverse_mw": caps["reverse_mw"],
            },
        }
        if src in fictitious_codes or tgt in fictitious_codes:
            line_entry["exchange_cost"] = PINT * _PINT_FICTITIOUS_DISCOUNT
        lines.append(line_entry)

    return {
        "$schema": cobre_schemas.schema_url_for("system/lines.json"),
        "lines": lines,
    }


def _read_penalid_costs(case: NewaveCase) -> dict[str, float]:
    """Pull ``{variable_name: first non-null R$/MWh value}`` from PENALID.DAT.

    Falls back to an empty dict if the file is absent or unparseable. Each PENALID
    variable can have per-REE / per-patamar values; we pick the first non-null R$/MWh
    entry as the global default the same way the source model does for REE-aggregated
    penalty handling.
    """
    if case.files.penalid is None:
        return {}
    try:
        penalid = case.penalid
    except (OSError, ValueError) as exc:
        _LOG.warning("penalid.dat could not be parsed (%s); using defaults.", exc)
        return {}

    pen_df = penalid.penalidades
    if pen_df is None or pen_df.empty:
        return {}

    out: dict[str, float] = {}
    for var in pen_df["variavel"].unique():
        rows = pen_df[(pen_df["variavel"] == var) & pen_df["valor_R$_MWh"].notna()]
        if not rows.empty:
            out[str(var).strip()] = float(rows.iloc[0]["valor_R$_MWh"])
    return out


def _own_productivities(
    hydros_dict: dict, productivities: dict[int, float]
) -> list[float]:
    """Return the list of per-hydro own productivities for averaging.

    Preferred source is the `productivities` map (new contract). Falls back
    to the legacy `hydros.json:generation.productivity_mw_per_m3s` field per
    entry so older test fixtures keep working.
    """
    out: list[float] = []
    for h in hydros_dict.get("hydros", []):
        prod = 0.0
        if "id" in h:
            prod = productivities.get(int(h["id"]), 0.0)
        if prod <= 0.0:
            legacy = h.get("generation", {}).get("productivity_mw_per_m3s")
            if legacy is not None:
                prod = float(legacy)
        if prod > 0:
            out.append(prod)
    return out


# Hydro penalty columns whose value scales with productivity (ρ_avg or
# ρ_max_acum) and is therefore stage-varying. ``generation_violation_below_cost``
# (energy-domain) never differs per stage and is excluded. After the
# deficit-derived rewrite ``storage_violation_below_cost`` always scales with
# ρ_max_acum (both the VOLMIN and default paths), so it is included.
# ``filling_target_violation_cost`` now also scales with ρ_max_acum, but is
# deliberately kept OUT of the per-stage override: its LP slot is dormant, so it
# is populated once from the base ρ_max_acum rather than re-emitted per stage.
_RHO_SCALED_HYDRO_COLUMNS: tuple[str, ...] = (
    "spillage_cost",
    "turbined_cost",
    "diversion_cost",
    "storage_violation_below_cost",
    "turbined_violation_below_cost",
    "outflow_violation_below_cost",
    "outflow_violation_above_cost",
    "evaporation_violation_cost",
    "water_withdrawal_violation_cost",
    "inflow_nonnegativity_cost",
)


def convert_penalties(
    case: NewaveCase,
    hydros_dict: dict,
    productivities: dict[int, float] | None = None,
    *,
    max_accumulated_productivity: float | None = None,
    prod_media_sin: float | None = None,
) -> dict:
    """Generate a Cobre ``penalties.json`` dict from the source model data.

    Faithful to the source model User Manual v30 section 3.24:

    - Bus deficit segments come from ``sistema.custo_deficit`` directly
      (R$/MWh on both sides, no conversion).
    - PENALID-sourced flow-domain penalties are converted to cobre's
      coefficient slot via ``× ρ`` (where ρ is ``PROD_MEDIA_SIN`` or
      ``MAX_PRODTACUM_SIN`` per source-model conversion table on page 87).
    - The micro-penalties (``pINT``, ``pEVERT``, ``pTURB``, ``pCORTEOL``,
      ``pEXC``, ``pCDESV``) are the source model's hard-coded internal defaults (page
      88, current v30 values). They are written directly to cobre and preserve the
      source model's merit order: exchange < spillage < FPHA < curtailment < excess.
    - Evaporation, storage-floor and filling-target slots without a PENALID
      source are DERIVED from the deficit cost via ``ρ_max_acum``: evaporation and
      storage at ``10 × MAX_CUSTO_DEFICIT × ρ_max_acum`` (the manual p.87 level;
      storage additionally ``× HM3_TO_MWH_PER_RHO`` for its R$/hm³ slot) and
      filling at ``0.9 × MAX_CUSTO_DEFICIT × ρ_max_acum × HM3_TO_MWH_PER_RHO``. The
      storage and filling slots are dormant in cobre's LP today but emitted so the
      file is ready when cobre wires them in.

    Parameters
    ----------
    case:
        Parsed the source model case.
    hydros_dict:
        The already-converted ``hydros.json`` dict (used for reservoir
        useful-volume weights and the productivity fallback).
    productivities:
        ``{hydro_id: own_productivity_mw_per_m3s}`` for each hydro. Required
        because productivity moved out of ``hydros.json:generation`` on
        cobre HEAD.
    max_accumulated_productivity:
        Optional ``MAX_PRODTACUM_SIN`` override. When omitted, defaults to
        ``max(productivities)`` — a coarse approximation; callers with
        access to the cascade DAG should pass the true accumulated max.
    """
    sistema = case.sistema
    deficit_df = sistema.custo_deficit

    # Primary deficit cost: first subsystem, first patamar.
    primary_deficit_cost = 0.0
    max_deficit_cost = 0.0
    if deficit_df is not None and not deficit_df.empty:
        first_sub = deficit_df.sort_values(["codigo_submercado", "patamar_deficit"])
        primary_deficit_cost = float(first_sub.iloc[0]["custo"])
        max_deficit_cost = float(deficit_df["custo"].max())

    penalid_costs = _read_penalid_costs(case)
    productivities = productivities or {}

    # ρ_avg = PROD_MEDIA_SIN: The source model converts the PENALID R$/MWh penalties
    # (VAZMIN, TURBMN, TURBMX, VOLMIN — manual table p.87) with the mean **PRODT**
    # (equivalent productivity vol_min→vol_max) over ALL existing plants including
    # zeros. Validated against pmo.dat's applied penalties (0.6299 ↔ penalty 821.78).
    # Pass it in via ``prod_media_sin`` (see ``hydro.compute_prodt_sin_mean``). The
    # legacy fallback — mean of the 65%-reference own ρ, ρ>0 only — is ~4% high; kept
    # only for callers/tests that don't supply ``prod_media_sin``.
    own_prods = _own_productivities(hydros_dict, productivities)
    rho_avg = (
        prod_media_sin
        if prod_media_sin is not None
        else (sum(own_prods) / len(own_prods) if own_prods else 1.0)
    )

    # ρ_max_acum = MAX_PRODTACUM_SIN: used by the source model for DESVIO and the
    # evaporation default. When the caller doesn't supply the true cascade accumulated
    # max we approximate by `max(own_prods)`; the caller in `pipeline.py` passes the
    # real value computed from the cascade DAG.
    rho_max_acum = (
        max_accumulated_productivity
        if max_accumulated_productivity is not None
        else (max(own_prods) if own_prods else rho_avg)
    )

    # The ρ-scaled hydro penalty block is computed by a pure helper so the
    # global (base) penalties.json and the per-stage override parquet built by
    # ``convert_hydro_penalty_overrides`` apply byte-identical formulas — see
    # ``hydro_penalty_costs``.
    hydro_costs = hydro_penalty_costs(
        rho_avg=rho_avg,
        rho_max_acum=rho_max_acum,
        penalid_costs=penalid_costs,
        max_deficit_cost=max_deficit_cost,
    )

    # -------------------------------------------------------------------- The source
    # model micro-penalty defaults (page 88, current v30) — energy-domain (R$/MWh), pass
    # through directly (no productivity multiplier, hence not stage-varying and not part
    # of the hydro override).
    # --------------------------------------------------------------------
    excess_cost = PEXC
    exchange_cost = PINT
    curtailment_cost = PCORTEOL

    return {
        "$schema": cobre_schemas.schema_url_for("penalties.json"),
        "bus": {
            "deficit_segments": [
                {
                    "cost": primary_deficit_cost,
                    "depth_mw": None,
                }
            ],
            "excess_cost": excess_cost,
        },
        "hydro": hydro_costs,
        "line": {
            "exchange_cost": exchange_cost,
        },
        "non_controllable_source": {
            "curtailment_cost": curtailment_cost,
        },
    }


def convert_hydro_penalty_overrides(
    case: NewaveCase,
    hydro_ids: Sequence[int],
    base_hydro_penalties: Mapping[str, float],
    per_stage_rho_avg: Sequence[float],
    per_stage_rho_max_acum: Sequence[float],
) -> pa.Table | None:
    """Build ``constraints/penalty_overrides_hydro.parquet`` (per-stage ρ).

    The source model converts its flow-domain hydro penalties (spillage, turbined,
    diversion, the outflow/turbined/storage/withdrawal/evaporation slacks) using the
    **SIN-aggregate** productivity constants ``PROD_MEDIA_SIN`` and
    ``MAX_PRODTACUM_SIN``. Those constants are *not* fixed across the horizon:
    each plant's equivalent productivity tracks its seasonal reference volume
    (VOLREF_SAZ) and any CFUGA/CMONT tailrace/forebay overrides, so the SIN
    mean / accumulated-max shift stage to stage. cobre-bridge already ships the
    per-stage per-plant ρ in ``system/hydro_energy_productivity.parquet``; this
    override makes the **penalty** conversion use the same per-stage ρ, instead
    of a single static fleet mean — keeping the two coherent.

    The override is SIN-uniform (one value per stage, applied to every hydro, matching
    the source model's use of a single SIN constant) and **sparse**: a row is emitted
    only for ``(hydro, stage)`` pairs and columns whose value actually differs from the
    global ``penalties.json`` default. ``generation_violation_below_cost``
    (energy-domain, ρ-independent) and ``filling_target_violation_cost``
    (ρ_max_acum-derived but deliberately excluded from the per-stage override —
    its LP slot is dormant) are never emitted.

    Parameters
    ----------
    case:
        Parsed the source model case (re-reads ``sistema`` for the max deficit cost and
        PENALID for the violation-slack base rates).
    hydro_ids:
        Every Cobre hydro id the SIN-uniform override must cover. Sorted
        ascending internally so output obeys the ``(hydro_id, stage_id)``
        ordering contract.
    base_hydro_penalties:
        The global ``penalties.json:hydro`` block — used as the diff baseline
        so only genuinely stage-varying values are emitted.
    per_stage_rho_avg:
        ``PROD_MEDIA_SIN[s]`` — mean own ρ over online plants at stage ``s``.
    per_stage_rho_max_acum:
        ``MAX_PRODTACUM_SIN[s]`` — max accumulated cascade ρ at stage ``s``.

    Returns
    -------
    pyarrow.Table | None
        Columns ``hydro_id`` (INT32), ``stage_id`` (INT32), and one DOUBLE
        column per ρ-scaled penalty that varies. ``None`` when nothing differs
        from the base (e.g. no seasonal/temporal productivity effects).
    """
    if not hydro_ids or not per_stage_rho_avg:
        return None
    n_stages = len(per_stage_rho_avg)
    if len(per_stage_rho_max_acum) != n_stages:
        raise ValueError(
            "per_stage_rho_avg and per_stage_rho_max_acum must be the same "
            f"length ({n_stages} vs {len(per_stage_rho_max_acum)})"
        )

    sistema = case.sistema
    deficit_df = sistema.custo_deficit
    max_deficit_cost = (
        float(deficit_df["custo"].max())
        if deficit_df is not None and not deficit_df.empty
        else 0.0
    )
    penalid_costs = _read_penalid_costs(case)

    # Recompute the full hydro penalty block per stage and keep only ρ-scaled
    # columns that differ from the global base (sparse-override contract).
    stage_overrides: list[tuple[int, dict[str, float]]] = []
    for s in range(n_stages):
        stage_costs = hydro_penalty_costs(
            rho_avg=per_stage_rho_avg[s],
            rho_max_acum=per_stage_rho_max_acum[s],
            penalid_costs=penalid_costs,
            max_deficit_cost=max_deficit_cost,
        )
        diff: dict[str, float] = {}
        for col in _RHO_SCALED_HYDRO_COLUMNS:
            base_v = base_hydro_penalties.get(col)
            new_v = stage_costs.get(col)
            if base_v is None or new_v is None:
                continue
            if not math.isclose(new_v, base_v, rel_tol=1e-12, abs_tol=0.0):
                diff[col] = new_v
        if diff:
            stage_overrides.append((s, diff))

    if not stage_overrides:
        return None

    present_cols = [
        c for c in _RHO_SCALED_HYDRO_COLUMNS if any(c in d for _, d in stage_overrides)
    ]

    hydro_id_col: list[int] = []
    stage_id_col: list[int] = []
    value_cols: dict[str, list[float | None]] = {c: [] for c in present_cols}
    for hid in sorted(int(h) for h in hydro_ids):
        for stage_id, diff in stage_overrides:
            hydro_id_col.append(hid)
            stage_id_col.append(stage_id)
            for c in present_cols:
                value_cols[c].append(diff.get(c))

    schema = pa.schema(
        [pa.field("hydro_id", pa.int32()), pa.field("stage_id", pa.int32())]
        + [pa.field(c, pa.float64()) for c in present_cols]
    )
    data: dict[str, pa.Array] = {
        "hydro_id": pa.array(hydro_id_col, type=pa.int32()),
        "stage_id": pa.array(stage_id_col, type=pa.int32()),
    }
    for c in present_cols:
        data[c] = pa.array(value_cols[c], type=pa.float64())
    return pa.table(data, schema=schema)


def convert_line_bounds(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> pa.Table:
    """Convert the source model interchange limits to a Cobre ``line_bounds.parquet``
    table, folding in the per-block exchange factors as absolute-MW override rows.

    Reads ``sistema.dat::limites_intercambio`` and ``dger.dat`` to produce one
    stage-level base row per (line, stage) with ``block_id = None`` and direct/
    reverse MW bounds. Cobre rule 36 treats ``block_id = None`` as a key
    distinct from ``Some(b)``, so this base row is load-bearing: without it, a
    stage whose blocks are all uniform would fall back to ``lines.json``'s
    declared (stage-0) capacity for every later stage. It also reads
    ``patamar.dat::intercambio_patamares`` — formerly emitted as a standalone
    per-block-factor JSON document, now deleted — and folds each per-block
    multiplicative factor into an absolute-MW override row, ``direct_mw =
    base_direct_mw × direct_factor`` (and the reverse equivalent), per cobre
    decision 10. A block row is emitted only where it differs
    from the base (i.e. the factor is not 1.0 for both directions); a
    line-stage whose blocks are all uniform gets no block rows, since the base
    row alone is equivalent.

    The canonical pair logic (``src < tgt``) and line ID assignment exactly
    mirror ``convert_lines`` so that line IDs are consistent.  Interchange
    limits have no seasonalize flag, so post-study base bounds freeze at the
    last study stage's value — but the per-block factors do **not** share
    that freeze: they seasonally repeat the last study year's per-calendar-
    month pattern instead (unchanged from the deleted factor emitter), so a
    post-study stage can still carry a non-trivial block override even though
    its base bounds are frozen at the (possibly zero) freeze-point value.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Entity ID map.  Used to resolve subsystem codes to Cobre bus IDs
        (indirectly, via the same canonical-pair ordering used in
        ``convert_lines``).

    Returns
    -------
    pyarrow.Table
        Columns: ``line_id`` (INT32), ``stage_id`` (INT32),
        ``direct_mw`` (DOUBLE), ``reverse_mw`` (DOUBLE), ``block_id``
        (INT32, nullable — ``None`` for the stage-level base row).
    """
    sistema = case.sistema
    limites_df: pd.DataFrame | None = sistema.limites_intercambio

    _LINE_BOUNDS_SCHEMA = pa.schema(
        [
            pa.field("line_id", pa.int32(), nullable=False),
            pa.field("stage_id", pa.int32(), nullable=False),
            pa.field("block_id", pa.int32(), nullable=True),
            pa.field("direct_mw", pa.float64(), nullable=False),
            pa.field("reverse_mw", pa.float64(), nullable=False),
        ]
    )

    if limites_df is None or limites_df.empty:
        return pa.table(
            {
                "line_id": pa.array([], type=pa.int32()),
                "stage_id": pa.array([], type=pa.int32()),
                "block_id": pa.array([], type=pa.int32()),
                "direct_mw": pa.array([], type=pa.float64()),
                "reverse_mw": pa.array([], type=pa.float64()),
            },
            schema=_LINE_BOUNDS_SCHEMA,
        )

    horizon = case.horizon
    start_month = horizon.start_month
    start_year = horizon.start_year
    study_months = horizon.study_months
    total_stages = horizon.total_stages

    # Study end boundary: first month *after* the study horizon.
    study_end_year = start_year + (start_month - 1 + study_months) // 12
    study_end_month = ((start_month - 1 + study_months) % 12) + 1

    pair_to_line_id = _build_canonical_pair_to_line_id(case)

    # Build per-date lookup:
    # {(src, tgt, year, cal_month) -> {direct_mw, reverse_mw}}
    date_lookup: dict[tuple[int, int, int, int], dict[str, float]] = {}

    for _, row in limites_df.iterrows():
        de = int(row["submercado_de"])
        para = int(row["submercado_para"])
        valor = float(row["valor"])
        sentido = int(row["sentido"])
        dt = row["data"]
        yr = int(dt.year)
        cal_month = int(dt.month)

        src, tgt = (de, para) if de < para else (para, de)
        key = (src, tgt, yr, cal_month)

        if key not in date_lookup:
            date_lookup[key] = {"direct_mw": 0.0, "reverse_mw": 0.0}

        _assign_flow_direction(date_lookup[key], de, para, sentido, valor)

    # Build last-year lookup for post-study:
    # {(src, tgt, cal_month) -> {direct_mw, reverse_mw}} — use the latest year.
    last_year_per_key: dict[tuple[int, int, int], tuple[int, dict[str, float]]] = {}
    for (src, tgt, yr, cal_month), caps in date_lookup.items():
        key3 = (src, tgt, cal_month)
        existing = last_year_per_key.get(key3)
        if existing is None or yr > existing[0]:
            last_year_per_key[key3] = (yr, caps)
    last_year_lookup: dict[tuple[int, int, int], dict[str, float]] = {
        k: v for k, (_, v) in last_year_per_key.items()
    }

    # Per-block factors (cobre decision 10): fold
    # ``patamar.dat::intercambio_patamares`` into per-block direct/reverse
    # multipliers, keyed like the base lookup above so block rows derive as
    # base × factor.
    patamar = case.patamar
    factors_df: pd.DataFrame | None = patamar.intercambio_patamares

    # {(src, tgt, year, cal_month, block_id) -> factor}; block_id is 0-based
    # (patamar is 1-based in the source).
    direct_factor_map: dict[tuple[int, int, int, int, int], float] = {}
    reverse_factor_map: dict[tuple[int, int, int, int, int], float] = {}
    num_blocks = 0

    if factors_df is not None and not factors_df.empty:
        all_blocks: set[int] = set()
        for _, row in factors_df.iterrows():
            de = int(row["submercado_de"])
            para = int(row["submercado_para"])
            val = float(row["valor"])
            block_id = int(row["patamar"]) - 1
            dt = row["data"]
            yr = int(dt.year)
            cal_month = int(dt.month)
            all_blocks.add(block_id)

            src, tgt = (de, para) if de < para else (para, de)
            key = (src, tgt, yr, cal_month, block_id)
            if de < para:
                direct_factor_map[key] = val
            else:
                reverse_factor_map[key] = val

        num_blocks = max(all_blocks) + 1

    # Last-year seasonal lookups for post-study factor stages:
    # {(src, tgt, cal_month, block_id) -> factor}
    last_yr_direct_factor: dict[tuple[int, int, int, int], tuple[int, float]] = {}
    last_yr_reverse_factor: dict[tuple[int, int, int, int], tuple[int, float]] = {}

    for (src, tgt, yr, cal_month, block_id), val in direct_factor_map.items():
        k4 = (src, tgt, cal_month, block_id)
        existing = last_yr_direct_factor.get(k4)
        if existing is None or yr > existing[0]:
            last_yr_direct_factor[k4] = (yr, val)

    for (src, tgt, yr, cal_month, block_id), val in reverse_factor_map.items():
        k4 = (src, tgt, cal_month, block_id)
        existing = last_yr_reverse_factor.get(k4)
        if existing is None or yr > existing[0]:
            last_yr_reverse_factor[k4] = (yr, val)

    last_direct_factor: dict[tuple[int, int, int, int], float] = {
        k: v for k, (_, v) in last_yr_direct_factor.items()
    }
    last_reverse_factor: dict[tuple[int, int, int, int], float] = {
        k: v for k, (_, v) in last_yr_reverse_factor.items()
    }

    def _block_factors(
        src: int, tgt: int, y: int, m: int, is_post_study: bool
    ) -> list[tuple[int, float, float]]:
        """Per-block (block_id, direct_factor, reverse_factor) for one
        (src, tgt, year, month), applying the same post-study freeze-to-last-
        seasonal-year fallback used for the base bounds above."""
        out: list[tuple[int, float, float]] = []
        for block_id in range(num_blocks):
            if is_post_study:
                d_factor = last_direct_factor.get((src, tgt, m, block_id), 1.0)
                r_factor = last_reverse_factor.get((src, tgt, m, block_id), 1.0)
            else:
                key_lookup = (src, tgt, y, m, block_id)
                d_factor = direct_factor_map.get(
                    key_lookup,
                    last_direct_factor.get((src, tgt, m, block_id), 1.0),
                )
                r_factor = reverse_factor_map.get(
                    key_lookup,
                    last_reverse_factor.get((src, tgt, m, block_id), 1.0),
                )
            out.append((block_id, d_factor, r_factor))
        return out

    rows_line_id: list[int] = []
    rows_stage_id: list[int] = []
    rows_direct: list[float] = []
    rows_reverse: list[float] = []
    rows_block_id: list[int | None] = []
    block_rows_emitted = 0

    # Last study stage's (year, calendar month). Interchange limits have no
    # seasonalize flag, so the post-study tail freezes at this stage's value
    # rather than repeating the last study year's seasonal pattern.
    ls_y = start_year + (start_month - 1 + study_months - 1) // 12
    ls_m = ((start_month - 1 + study_months - 1) % 12) + 1

    for pair, line_id in sorted(pair_to_line_id.items(), key=lambda x: x[1]):
        src, tgt = pair
        freeze_caps = date_lookup.get((src, tgt, ls_y, ls_m)) or last_year_lookup.get(
            (src, tgt, ls_m), {"direct_mw": 0.0, "reverse_mw": 0.0}
        )
        y, m = start_year, start_month
        for stage_id in range(total_stages):
            is_post_study = (y > study_end_year) or (
                y == study_end_year and m >= study_end_month
            )

            if is_post_study:
                caps = freeze_caps
            else:
                caps = date_lookup.get((src, tgt, y, m))
                if caps is None:
                    caps = last_year_lookup.get(
                        (src, tgt, m), {"direct_mw": 0.0, "reverse_mw": 0.0}
                    )

            base_direct = caps["direct_mw"]
            base_reverse = caps["reverse_mw"]

            # Stage-level base row (block_id = None) — kept unchanged and
            # unconditionally, per cobre rule 36.
            rows_line_id.append(line_id)
            rows_stage_id.append(stage_id)
            rows_direct.append(base_direct)
            rows_reverse.append(base_reverse)
            rows_block_id.append(None)

            # Per-block override rows, only where the factor differs from the
            # base (i.e. is not 1.0 in both directions).
            for block_id, d_factor, r_factor in _block_factors(
                src, tgt, y, m, is_post_study
            ):
                if d_factor == 1.0 and r_factor == 1.0:
                    continue
                rows_line_id.append(line_id)
                rows_stage_id.append(stage_id)
                rows_direct.append(base_direct * d_factor)
                rows_reverse.append(base_reverse * r_factor)
                rows_block_id.append(block_id)
                block_rows_emitted += 1

            m += 1
            if m > 12:
                m = 1
                y += 1

    _LOG.info(
        "line_bounds: emitted %d per-block override row(s) folded from "
        "patamar.dat exchange factors (cobre decision 10), alongside %d "
        "stage-level base row(s).",
        block_rows_emitted,
        len(rows_line_id) - block_rows_emitted,
    )

    return pa.table(
        {
            "line_id": pa.array(rows_line_id, type=pa.int32()),
            "stage_id": pa.array(rows_stage_id, type=pa.int32()),
            "block_id": pa.array(rows_block_id, type=pa.int32()),
            "direct_mw": pa.array(rows_direct, type=pa.float64()),
            "reverse_mw": pa.array(rows_reverse, type=pa.float64()),
        },
        schema=_LINE_BOUNDS_SCHEMA,
    )


def _in_study_horizon(
    dt: object, start_year: int, start_month: int, total_stages: int
) -> bool:
    """Return True if *dt* falls within the study + post-study horizon.

    Year == ``POST_STUDY_YEAR`` marks a post-study entry (inewave convention).
    """
    try:
        yr = int(dt.year)  # type: ignore[union-attr]
        mo = int(dt.month)  # type: ignore[union-attr]
    except (AttributeError, TypeError, ValueError):
        return False
    if yr == POST_STUDY_YEAR:
        return True
    stage_id = (yr - start_year) * 12 + (mo - start_month)
    return 0 <= stage_id < total_stages


def _build_ncs_group_to_id(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> dict[tuple[int, int], int]:
    """Build the canonical (codigo_submercado, indice_bloco) -> ncs_id mapping.

    Applies the same horizon filtering and bus_id validation used by
    ``convert_non_controllable_sources``.  This is the single authoritative
    NCS group mapping shared by ``convert_ncs_factors`` and
    ``convert_ncs_stats``.
    """
    sistema = case.sistema
    df_ncs: pd.DataFrame | None = sistema.geracao_usinas_nao_simuladas
    if df_ncs is None or df_ncs.empty:
        return {}

    horizon = case.horizon
    start_month = horizon.start_month
    start_year = horizon.start_year
    total_stages = horizon.total_stages

    df_filtered = df_ncs[
        df_ncs["data"].apply(
            lambda dt: _in_study_horizon(dt, start_year, start_month, total_stages)
        )
    ].copy()
    groups = df_filtered.groupby(["codigo_submercado", "indice_bloco"], sort=True)

    result: dict[tuple[int, int], int] = {}
    ncs_id = 0
    for (sub_code, bloco), _group in groups:
        sub_code_int = int(sub_code)
        try:
            id_map.bus_id(sub_code_int)
        except KeyError:
            continue
        result[(sub_code_int, int(bloco))] = ncs_id
        ncs_id += 1

    return result


def convert_non_controllable_sources(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> dict:
    """Convert the source model non-simulated generation to a Cobre NCS entity JSON
    dict.

    Reads ``sistema.dat::geracao_usinas_nao_simuladas``.  Each unique
    ``(codigo_submercado, indice_bloco)`` pair becomes one NCS entity.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Entity ID map.  Used to resolve subsystem codes to 0-based Cobre bus
        IDs.

    Returns
    -------
    dict
        JSON-serializable dict with key ``"non_controllable_sources"``
        containing a list of NCS entity dicts.
    """
    sistema = case.sistema
    df_ncs: pd.DataFrame | None = sistema.geracao_usinas_nao_simuladas

    if df_ncs is None or df_ncs.empty:
        return {
            "$schema": cobre_schemas.schema_url_for(
                "system/non_controllable_sources.json"
            ),
            "non_controllable_sources": [],
        }

    horizon = case.horizon
    start_month = horizon.start_month
    start_year = horizon.start_year
    total_stages = horizon.total_stages

    # Filter to study + post-study horizon only.
    # Rows with year == 9999 are post-study entries in inewave convention.
    df_filtered = df_ncs[
        df_ncs["data"].apply(
            lambda dt: _in_study_horizon(dt, start_year, start_month, total_stages)
        )
    ].copy()

    # The source model carries no per-source commissioning date for the aggregated
    # non-controllable generation; treat every NCS as in service since the
    # historical record (Cobre uses the date only as a canonical-ordering key,
    # tiebroken by id).
    op_date = historical_start_date(case.dger)

    # Columns: codigo_submercado, indice_bloco, fonte, data, valor
    # Group by (codigo_submercado, indice_bloco) — each unique pair is one NCS.
    ncs_list: list[dict] = []
    ncs_id = 0

    groups = df_filtered.groupby(["codigo_submercado", "indice_bloco"], sort=True)

    for (sub_code, bloco), group in groups:
        sub_code_int = int(sub_code)
        try:
            bus_id = id_map.bus_id(sub_code_int)
        except KeyError:
            _LOG.warning(
                "Subsystem code %d from geracao_usinas_nao_simuladas not in "
                "id_map; skipping NCS (indice_bloco=%s)",
                sub_code_int,
                bloco,
            )
            continue

        # fonte: use the first non-null value in the group.
        fonte_series = group["fonte"].dropna()
        fonte = str(fonte_series.iloc[0]).strip() if not fonte_series.empty else "NCS"

        # max_generation_mw: maximum non-NaN value across all rows in the group.
        valores = pd.to_numeric(group["valor"], errors="coerce")
        valid_vals = valores.dropna()
        max_gen = float(valid_vals.max()) if not valid_vals.empty else 0.0

        ncs_list.append(
            {
                "id": ncs_id,
                "name": f"{fonte}_{sub_code_int}",
                "operational_start_date": op_date,
                "bus_id": bus_id,
                "max_generation_mw": max_gen,
                # The source model pre-nets `geracao_usinas_nao_simuladas` from MERC
                # before the dispatch LP runs, so the aggregate is implicitly must-run.
                # Setting allow_curtailment=False instructs Cobre's LP to pin dispatch
                # to the realized availability for every scenario; otherwise the LP
                # discovers that curtailing NCS is one of the cheapest slacks and
                # produces a +15 % hydro / -23 % spillage divergence vs the source model
                # on this case family. See docs/findings/ncs-must-run-treatment.md.
                "allow_curtailment": False,
            }
        )
        ncs_id += 1

    return {
        "$schema": cobre_schemas.schema_url_for("system/non_controllable_sources.json"),
        "non_controllable_sources": ncs_list,
    }


def convert_ncs_factors(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> dict:
    """Convert patamar.dat NCS block factors to a Cobre non_controllable_factors dict.

    Reads ``patamar.dat::usinas_nao_simuladas``.  NCS entity IDs are assigned
    using the same ``(codigo_submercado, indice_bloco)`` sorted grouping as
    ``convert_non_controllable_sources`` in this module.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Entity ID map.  Used for subsystem code validation.

    Returns
    -------
    dict
        JSON-serializable dict with key ``"non_controllable_factors"``.
    """
    patamar_file = case.patamar
    df: pd.DataFrame | None = patamar_file.usinas_nao_simuladas

    if df is None or df.empty:
        return {
            "$schema": cobre_schemas.schema_url_for(
                "scenarios/non_controllable_factors.json"
            ),
            "non_controllable_factors": [],
        }

    horizon = case.horizon
    start_month = horizon.start_month
    start_year = horizon.start_year
    study_months = horizon.study_months
    total_stages = horizon.total_stages

    study_end_year = start_year + (start_month - 1 + study_months) // 12
    study_end_month = ((start_month - 1 + study_months) % 12) + 1

    # Filter to study + post-study horizon (year == 9999 is post-study).
    df = df[
        df["data"].apply(
            lambda dt: _in_study_horizon(dt, start_year, start_month, total_stages)
        )
    ].copy()

    if df.empty:
        return {
            "$schema": cobre_schemas.schema_url_for(
                "scenarios/non_controllable_factors.json"
            ),
            "non_controllable_factors": [],
        }

    # Columns: codigo_submercado, indice_bloco, data, patamar, valor
    # The ``patamar`` field is a GLOBAL running index across all NCS sources:
    # source 1 -> patamares 1..P, source 2 -> P+1..2P, etc. (P = number of load blocks).
    # The per-source block ordinal is therefore ``(patamar - 1) % P``, NOT ``patamar -
    # 1`` — using the raw index parks every source past the first on out-of-range
    # blocks, flattening its per-block NCS profile (so Cobre could not reshape per-block
    # load the way the source model does, the so_se divergence).
    num_blocks = patamar_file.numero_patamares or 1

    # Build per-(sub_code, bloco, yr, cal_month, block_id) -> factor lookup.
    NcsKey = tuple  # (sub_code, bloco, yr, cal_month, block_id)
    factor_map: dict[NcsKey, float] = {}

    for _, row in df.iterrows():
        sub_code = int(row["codigo_submercado"])
        bloco = int(row["indice_bloco"])
        block_id = (int(row["patamar"]) - 1) % num_blocks
        val = float(row["valor"])
        dt = row["data"]
        yr = int(dt.year)
        cal_month = int(dt.month)
        factor_map[(sub_code, bloco, yr, cal_month, block_id)] = val

    # Last-year seasonal fallback.
    last_yr_map: dict[tuple[int, int, int, int], tuple[int, float]] = {}
    for (sub_code, bloco, yr, cal_month, block_id), val in factor_map.items():
        k4 = (sub_code, bloco, cal_month, block_id)
        existing = last_yr_map.get(k4)
        if existing is None or yr > existing[0]:
            last_yr_map[k4] = (yr, val)

    last_factor: dict[tuple[int, int, int, int], float] = {
        k: v for k, (_, v) in last_yr_map.items()
    }

    # Use the shared canonical NCS group -> ID mapping to guarantee consistency
    # with convert_non_controllable_sources and convert_ncs_stats.
    ncs_group_map = _build_ncs_group_to_id(case, id_map)

    results: list[dict] = []

    for (sub_code, bloco), ncs_id in sorted(ncs_group_map.items(), key=lambda x: x[1]):
        y, m = start_year, start_month
        for stage_id in range(total_stages):
            is_post_study = (y > study_end_year) or (
                y == study_end_year and m >= study_end_month
            )

            block_factors: list[dict] = []
            for block_id in range(num_blocks):
                if is_post_study:
                    factor = last_factor.get((sub_code, bloco, m, block_id), 1.0)
                else:
                    factor = factor_map.get(
                        (sub_code, bloco, y, m, block_id),
                        last_factor.get((sub_code, bloco, m, block_id), 1.0),
                    )
                block_factors.append(
                    {
                        "block_id": block_id,
                        "factor": max(factor, 1e-6),
                    }
                )

            results.append(
                {
                    "ncs_id": ncs_id,
                    "stage_id": stage_id,
                    "block_factors": block_factors,
                }
            )

            m += 1
            if m > 12:
                m = 1
                y += 1

    return {
        "$schema": cobre_schemas.schema_url_for(
            "scenarios/non_controllable_factors.json"
        ),
        "non_controllable_factors": results,
    }


def convert_ncs_stats(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> pa.Table:
    """Convert sistema.dat NCS generation to ``non_controllable_stats.parquet``.

    Produces the stochastic availability model for each NCS entity.  Since the source
    model NCS generation is deterministic, ``std`` is always 0.0 and ``mean`` is the
    availability factor: ``available_mw / max_generation_mw``.

    NCS IDs are assigned using the same ``(codigo_submercado, indice_bloco)``
    sorted grouping as ``convert_non_controllable_sources``.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Entity ID map (unused directly; kept for API consistency).

    Returns
    -------
    pyarrow.Table
        Columns: ``ncs_id`` (INT32), ``stage_id`` (INT32),
        ``mean`` (DOUBLE), ``std`` (DOUBLE).
    """
    _NCS_STATS_SCHEMA = pa.schema(
        [
            pa.field("ncs_id", pa.int32()),
            pa.field("stage_id", pa.int32()),
            pa.field("mean", pa.float64()),
            pa.field("std", pa.float64()),
        ]
    )

    sistema = case.sistema
    df_raw: pd.DataFrame | None = sistema.geracao_usinas_nao_simuladas

    if df_raw is None or df_raw.empty:
        return pa.table(
            {
                "ncs_id": pa.array([], type=pa.int32()),
                "stage_id": pa.array([], type=pa.int32()),
                "mean": pa.array([], type=pa.float64()),
                "std": pa.array([], type=pa.float64()),
            },
            schema=_NCS_STATS_SCHEMA,
        )

    horizon = case.horizon
    start_month = horizon.start_month
    start_year = horizon.start_year
    study_months = horizon.study_months
    total_stages = horizon.total_stages

    study_end_year = start_year + (start_month - 1 + study_months) // 12
    study_end_month = ((start_month - 1 + study_months) % 12) + 1

    # Columns: codigo_submercado, indice_bloco, fonte, data, valor
    # Build per-(sub_code, bloco, yr, cal_month) -> valor lookup.
    # year == 9999 rows are stored with key yr=9999 for post-study seasonal
    # repeat logic handled below.
    BoundsKey = tuple  # (sub_code, bloco, yr, cal_month)
    bounds_map: dict[BoundsKey, float] = {}

    for _, row in df_raw.iterrows():
        val_raw = row["valor"]
        if is_na(val_raw):
            continue
        val = float(val_raw)
        sub_code = int(row["codigo_submercado"])
        bloco = int(row["indice_bloco"])
        dt = row["data"]
        yr = int(dt.year)
        cal_month = int(dt.month)
        bounds_map[(sub_code, bloco, yr, cal_month)] = val

    # Build last-year seasonal fallback (for post-study and missing study stages).
    # Use year == 9999 rows preferentially as post-study entries; otherwise use
    # the highest real year available.
    last_yr_bounds: dict[tuple[int, int, int], tuple[int, float]] = {}
    for (sub_code, bloco, yr, cal_month), val in bounds_map.items():
        k3 = (sub_code, bloco, cal_month)
        existing = last_yr_bounds.get(k3)
        if existing is None or yr > existing[0]:
            last_yr_bounds[k3] = (yr, val)

    last_bounds: dict[tuple[int, int, int], float] = {
        k: v for k, (_, v) in last_yr_bounds.items()
    }

    # Use the shared canonical NCS group -> ID mapping.
    ncs_group_map = _build_ncs_group_to_id(case, id_map)

    # Compute max_generation_mw per NCS entity.
    max_gen_per_ncs: dict[int, float] = {}
    for (sub_code, bloco), ncs_id in ncs_group_map.items():
        vals = [
            v
            for (sc, bl, _yr, _cm), v in bounds_map.items()
            if sc == sub_code and bl == bloco
        ]
        max_gen_per_ncs[ncs_id] = max(vals) if vals else 0.0

    rows_ncs_id: list[int] = []
    rows_stage_id: list[int] = []
    rows_mean: list[float] = []
    rows_std: list[float] = []

    for (sub_code, bloco), ncs_id in sorted(ncs_group_map.items(), key=lambda x: x[1]):
        max_gen = max_gen_per_ncs[ncs_id]

        y, m = start_year, start_month
        for stage_id in range(total_stages):
            is_post_study = (y > study_end_year) or (
                y == study_end_year and m >= study_end_month
            )

            if is_post_study:
                gen_mw = last_bounds.get((sub_code, bloco, m))
            else:
                gen_mw = bounds_map.get((sub_code, bloco, y, m))
                if gen_mw is None:
                    gen_mw = last_bounds.get((sub_code, bloco, m))

            if gen_mw is not None:
                mean = gen_mw / max_gen if max_gen > 0 else 0.0
                mean = max(0.0, min(1.0, mean))
                rows_ncs_id.append(ncs_id)
                rows_stage_id.append(stage_id)
                rows_mean.append(mean)
                rows_std.append(0.0)

            m += 1
            if m > 12:
                m = 1
                y += 1

    return pa.table(
        {
            "ncs_id": pa.array(rows_ncs_id, type=pa.int32()),
            "stage_id": pa.array(rows_stage_id, type=pa.int32()),
            "mean": pa.array(rows_mean, type=pa.float64()),
            "std": pa.array(rows_std, type=pa.float64()),
        },
        schema=_NCS_STATS_SCHEMA,
    )


def _subsystem_name_from_id(subsystem_code: int) -> str:
    """Return a short name string for a subsystem code (fallback to str)."""
    return str(subsystem_code)

"""The source model's micro-penalty defaults and the ρ-scaled hydro penalty block."""

from __future__ import annotations

from collections.abc import Mapping

from cobre_bridge.core.units import HM3_TO_MWH_PER_RHO

# --- the source model micro-penalties (page 88, current v30) -------------------------
# Energy-domain (R$/MWh) — passed through to cobre without conversion. Flow-domain
# (multiplied by ρ_avg before emission) — see `_PEVERT` group.
#
# These are the source model's v30 values verbatim: tiny (~1e-4 R$/MWh) regularization
# costs whose only role is to break LP ties in a fixed merit order (exchange < spillage
# < … < excess), well below any operational or deterrent cost.
PINT = 0.000273  # intercâmbio  → line.exchange_cost
PCORTEOL = 0.000344  # corte geração eólica → ncs.curtailment_cost
PEXC = 0.000355  # excesso de energia → bus.excess_cost

# Flow-domain (R$/MWh equivalent, multiplied by ρ_avg before emission). Cobre's
# `hydro.spillage_cost` covers ALL spillage (reservoir + run-of-river). In the
# *individualized* model (manual §3.24, p.88, "the source model individualizado" column)
# BOTH controllable (pEVERT) and run-of-river (pPFIO) spillage use the same base
# 0.000300 — only the REE-aggregated ("the source model equivalente") column raises
# pEVERT to 0.000327. Cobre cases are individualized, so anchor on 0.000300.
_PEVERT = 0.000300  # vertimento controlável → hydro.spillage_cost
_PTURB = 0.000333  # turbinamento → hydro.turbined_cost (applied to every hydro)
_PCDESV = 0.000300  # volume desviado → hydro.diversion_cost

# --- the source model hard-coded internal defaults (no user input via PENALID) -------
# Page 87: evaporation and FPHA folga both derive from MAX_CUSTO_DEFICIT, and The source
# model's manual prescribes a 10× multiplier — the evap/FPHA folga slack is ~10× more
# expensive per MWh-equivalent than the deficit cost, putting these physical-law
# constraints (water-cycle physics, water-supply requirements) at the top of the merit
# order so the LP violates them only as a last resort. We apply that 10× faithfully (it
# doubles as the PENALID fallback below, and `_ELETRI_HIGH_MULT` reuses the same
# magnitude).
_EVAPORATION_MULT = 10.0

# NOTE: when PENALID supplies TURBMN, VAZMIN, TURBMX with the same R$/MWh value
# (typical the source model convention), the resulting
# turbined/outflow-below/outflow-above slack costs share an LP coefficient. Reintroduce
# distinct spacing here if HiGHS degeneracy resurfaces.

# --- Cobre Family-D fields not yet wired into the LP -----------------------
# Storage-floor and filling-target violation costs are declared on cobre's schema
# but `lp_builder/matrix.rs` does NOT use them in the objective (all 0.0 at build
# time). We still emit faithful values so the case is ready for the day cobre
# wires them in — DERIVED from the deficit cost via ρ_max_acum (energy-equivalent,
# × HM3_TO_MWH_PER_RHO), exactly like `evaporation_violation_cost`, rather than
# hard-coded placeholders.
#
# storage_violation_below_cost is priced at the evaporation / FPHA-folga tier —
# the manual p.87 hard-physical-violation level, 10 × MAX_CUSTO_DEFICIT — so its
# converted R$/hm³ slot equals the evaporation energy-equivalent
# (evaporation_cost × HM3_TO_MWH_PER_RHO). That makes the storage floor the most
# expensive hydro penalty: the LP draws a reservoir below its floor only as a
# last resort.
_STORAGE_VIOLATION_DEFICIT_MULT = 10.0
# filling_target_violation_cost sits a little below the deficit cost
# (energy-equivalent): missing a filling target is undesirable but cheaper than
# load deficit, hence 0.9 × MAX_CUSTO_DEFICIT.
_FILLING_TARGET_DEFICIT_FRACTION = 0.9

# --- Inflow non-negativity penalty anchor ---------------------------------
# inflow_nonnegativity_cost is set to ``water_withdrawal_violation_cost + 1``
# (R$/m³/s) so that the LP can never use the inflow slack as a cheaper
# substitute for failing the deterministic withdrawal schedule.
#
# The previous "1 % above max(flow slacks)" rule scaled the slack to be a
# soft margin, which is the wrong design for this column: the inflow
# non-negativity slack physically *generates* water in the LP — every other
# flow slack just *absorbs* a constraint violation — so giving it a price
# even infinitesimally cheaper than some violation makes it exchangeable
# with that violation.  The +1 R$/m³/s constant offset is just enough to
# break the tie deterministically against withdrawal (the strictest
# slack with the largest spread on the case set we calibrate against).
#
# This anchor still doesn't eliminate the exploit (the slack remains a
# free water column in principle); ``modeling.inflow_non_negativity.method``
# = ``truncation`` is the real fix.  This setting is the conservative
# choice for the ``penalty`` method.
_INFLOW_NN_OFFSET_R_PER_M3S = 1.0


def hydro_penalty_costs(
    *,
    rho_avg: float,
    rho_max_acum: float,
    penalid_costs: Mapping[str, float],
    max_deficit_cost: float,
) -> dict[str, float]:
    """Compute the ρ-scaled hydro penalty block for one (ρ_avg, ρ_max_acum).

    Pure function shared by :func:`convert_penalties` (global/base defaults in
    ``penalties.json``) and :func:`convert_hydro_penalty_overrides` (per-stage
    overrides in ``constraints/penalty_overrides_hydro.parquet``). Keeping a
    single formula site guarantees the two paths never drift: the override is
    exactly the base recomputed with a different productivity pair.

    Convention from the source model manual p.87:

    - ``DESVIO`` / ``VOLMIN`` → ``× MAX_PRODTACUM_SIN`` (water withdrawal,
      storage floor — the agreed criterion, matching evaporation).
    - ``VAZMIN`` / ``TURBMN`` / ``TURBMX`` → ``× PROD_MEDIA_SIN``.
    - ``GHMIN`` → energy-domain slack, no productivity multiplier.
    - Micro-penalties ``pEVERT`` / ``pTURB`` / ``pCDESV`` → ``× PROD_MEDIA_SIN``.

    The returned dict preserves the exact key order of ``penalties.json:hydro``.
    """
    evaporation_tier_mwh = _EVAPORATION_MULT * max_deficit_cost
    desvio_mwh = penalid_costs.get("DESVIO", evaporation_tier_mwh)
    vazmin_mwh = penalid_costs.get("VAZMIN", evaporation_tier_mwh)
    ghmin_mwh = penalid_costs.get("GHMIN", evaporation_tier_mwh)
    turbmn_mwh = penalid_costs.get("TURBMN", evaporation_tier_mwh)
    turbmx_mwh = penalid_costs.get("TURBMX", evaporation_tier_mwh)

    water_withdrawal_cost = desvio_mwh * rho_max_acum
    outflow_below_cost = vazmin_mwh * rho_avg
    outflow_above_cost = turbmx_mwh * rho_avg
    turbined_below_cost = turbmn_mwh * rho_avg
    generation_below_cost = ghmin_mwh  # energy-domain, no productivity factor

    # Storage floor / filling target: cobre's Family-D slots are dormant in the
    # LP today (priced 0.0) but we still populate them with faithful, DERIVED
    # values so the case is ready when cobre wires them in.
    #
    # Conversion from the source model R$/MWh to cobre R$/hm³ is purely volumetric:
    # 1 hm³ of stored water released through the cascade yields
    #   1e6 m³ × ρ MW/(m³/s) × 1/3600 s/h = (1e6/3600) × ρ MWh
    # so cobre_coef = P_R$_MWh × ρ × HM3_TO_MWH_PER_RHO. The 730h/month assumption
    # cancels out — this is dimensional energy-equivalence, not a per-hour rate.
    # ρ here is ρ_max_acum (MAX_PRODTACUM_SIN), per the agreed criterion and
    # matching the evaporation / water-withdrawal slacks (manual p.87).
    #
    # ⚠️ This volumetric (730-cancelling) form is correct ONLY because cobre
    # prices these Family-D slots with NO time multiplier (`objective = penalty`).
    # If they are ever wired into the LP with a `× block_hours` term (like the
    # generic/VminOP slack), this conversion becomes wrong: the slack would then
    # need per-stage hours, exactly like
    # `constraints.py:_vminop_energy_factor`. Keep these two facts in lockstep.
    #
    # storage_violation_below_cost is priced at the evaporation tier
    # (10 × MAX_CUSTO_DEFICIT) — the greatest hydro penalty. With no PENALID
    # VOLMIN the default equals evaporation_cost × HM3_TO_MWH_PER_RHO; a PENALID
    # VOLMIN rate overrides the 10× multiplier but keeps the same ρ_max_acum
    # treatment.
    volmin_mwh = penalid_costs.get("VOLMIN")
    storage_below_cost = (
        volmin_mwh * rho_max_acum * HM3_TO_MWH_PER_RHO
        if volmin_mwh is not None
        else _STORAGE_VIOLATION_DEFICIT_MULT
        * max_deficit_cost
        * rho_max_acum
        * HM3_TO_MWH_PER_RHO
    )

    # Evaporation violation: no PENALID variable. The source model manual p.87
    # prescribes `(K × MAX_CUSTO_DEFICIT × MAX_PRODTACUM_SIN) / C_M3S2HM3` with K=10,
    # which we apply faithfully (K == `_EVAPORATION_MULT`) so the deterrent magnitude
    # sits above the PENALID-sourced operational slacks (typically
    # water_withdrawal_violation_cost). The /C_M3S2HM3 step is dropped (only needed for
    # the source model's per-hm³ slot).
    evaporation_cost = _EVAPORATION_MULT * max_deficit_cost * rho_max_acum

    # filling_target_violation_cost: no PENALID source — always derived. Same
    # volumetric ρ_max_acum × HM3_TO_MWH_PER_RHO form as the storage floor, but a
    # little below the deficit cost (0.9×) rather than at the evaporation tier.
    filling_target_cost = (
        _FILLING_TARGET_DEFICIT_FRACTION
        * max_deficit_cost
        * rho_max_acum
        * HM3_TO_MWH_PER_RHO
    )

    # Flow-domain micro-penalties (× ρ_avg per source-model individualized conversion).
    spillage_cost = _PEVERT * rho_avg
    turbined_cost = _PTURB * rho_avg
    diversion_cost = _PCDESV * rho_avg

    # Inflow non-negativity: The source model has no PENALID variable for this. Cobre's
    # default is 1000 R$/(m³/s · h), which is far below the operationally- significant
    # flow-domain slacks above (turbined / outflow / evaporation / water-withdrawal).
    # When the LP can choose between letting incremental natural inflow go negative (a
    # non-physical phenomenon — implies upstream is somehow "absorbing water" the
    # cascade can't account for) and violating another flow constraint, we want it to
    # *always* prefer fixing the other constraint first. Anchor inflow non-negativity
    # penalty to the withdrawal slack plus a 1 R$/m³/s tie-breaker.
    inflow_nonnegativity_cost = water_withdrawal_cost + _INFLOW_NN_OFFSET_R_PER_M3S

    return {
        "spillage_cost": spillage_cost,
        "turbined_cost": turbined_cost,
        "diversion_cost": diversion_cost,
        "storage_violation_below_cost": storage_below_cost,
        "filling_target_violation_cost": filling_target_cost,
        "turbined_violation_below_cost": turbined_below_cost,
        "outflow_violation_below_cost": outflow_below_cost,
        "outflow_violation_above_cost": outflow_above_cost,
        "generation_violation_below_cost": generation_below_cost,
        "evaporation_violation_cost": evaporation_cost,
        "water_withdrawal_violation_cost": water_withdrawal_cost,
        "inflow_nonnegativity_cost": inflow_nonnegativity_cost,
    }

"""Generic constraints converter: VminOP (minimum stored energy) from curva.dat
and electric constraints from restricao-eletrica.csv.

Converts the source model minimum stored energy constraints into Novomodelo generic
constraints. Each REE with entries in ``curva.dat`` becomes one generic constraint whose
expression is a weighted sum of ``hydro_storage_final`` variables (the end-of-stage
stored volume the source model's security curve bounds), with weights equal to the
accumulated cascade productivities.

Electric constraints from ``restricao-eletrica.csv`` (discovered via
``indices.csv``) are converted into Novomodelo generic constraints with
``hydro_generation`` and ``line_exchange`` variables.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

import pandas as pd
import pyarrow as pa

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from novomodelo_bridge.core.generic_constraint_builder import (
    ConstraintIdAllocator,
    GenericConstraintBuilder,
    GenericConstraintResult,
)
from novomodelo_bridge.core.productivity import (
    compute_productivity,
    stored_energy_productivity,
)
from novomodelo_bridge.core.units import C_M3S2HM3, MONTH_HOURS
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.converters.hydro import (
    _apply_permanent_overrides,
    compute_per_stage_own_integrated_productivities,
)
from novomodelo_bridge.newave.converters.temporal import _month_hours
from novomodelo_bridge.newave.id_map import NewaveIdMap
from novomodelo_bridge.newave.plants import active_hydros
from novomodelo_bridge.newave.switches import switch_off_diagnostic
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas
from novomodelo_bridge.novomodelo.scalar_parameters import rho_acum_name

_LOG = logging.getLogger(__name__)

_CATEGORY = "Special constraints"


@dataclass
class _VminopSkip:
    """One REE whose curva.dat VminOP entry could not be expressed."""

    ree_code: int
    ree_name: str
    reason: str


@dataclass
class _VminopPenaltyClamp:
    """One REE whose curva.dat VminOP penalty was <= 0 and clamped to 1000.0."""

    ree_code: int
    ree_name: str
    declared: float
    applied: float


@dataclass
class _ElectricTermSkip:
    """One electric-constraint expression term dropped while translating it."""

    constraint_code: int
    kind: Literal["hydro-unmapped", "no-line", "malformed"]
    source: Literal["formula", "RE.DAT"]
    hydro_code: int | None = None
    sys_from: int | None = None
    sys_to: int | None = None
    raw: str | None = None


@dataclass
class _ElectricConstraintSkip:
    """One electric constraint code dropped entirely (no terms, or no bounds)."""

    constraint_code: int
    reason: str


@dataclass
class _AgrintTermSkip:
    """One AGRINT group term dropped for lack of a matching interchange line."""

    group_id: int
    sys_a: int
    sys_b: int


@dataclass
class _AgrintGroupSkip:
    """One AGRINT group dropped entirely (no valid terms after line lookup)."""

    group_id: int


def _vminop_energy_factor(start_year: int, start_month: int, stage: int) -> float:
    """Per-stage ``ρ_acum·hm³ → MWmonth`` divisor for the VminOP energy units.

    ρ_acum is in MW/(m³/s), so ``ρ_acum · storage[hm³]`` overstates the true stored
    energy (MWmonth) by the hm³↔(m³/s)·month factor. That factor scales with the month's
    length, and novomodelo prices the VminOP slack with each stage's **real** ``block_hours``
    (672–744 h — actual calendar months), **not** the source model's fixed 730 h
    convention. Using the actual month length here makes the effective slack penalty
    resolve to the intended R$/MWh on every stage; the fixed ``C_M3S2HM3`` would leave a
    ±~9 % per-stage error (worst in February). Equals ``C_M3S2HM3`` exactly for a 730 h
    month.
    """
    total = start_month - 1 + stage
    hours = _month_hours(start_year + total // 12, total % 12 + 1)
    return C_M3S2HM3 * hours / MONTH_HOURS


class VminopResult(NamedTuple):
    """Result of :func:`convert_vminop_constraints`.

    Field order matches the legacy 4-tuple so existing index/destructure callers
    keep working; new code should use the names. ``constraints_dict`` is the full
    ``{"$schema", "constraints": [...]}`` object; ``bounds`` is the full 5-column
    F3 bounds table (``block_id`` always null — VminOP is stage-level, never
    per-block).
    """

    constraints_dict: dict
    bounds: pa.Table
    referenced_hydro_ids: list[int]
    rho_acum_overrides: dict[int, list[float]]


def _build_hydro_downstream_map(
    confhd_df: pd.DataFrame,
    cadastro: pd.DataFrame | None = None,
) -> dict[int, int | None]:
    """Return ``{plant_code: downstream_code}`` for every real (non-FICT) plant.

    The map collapses any chain of fictitious plants between a real plant and
    its next real downstream — see
    :func:`novomodelo_bridge.newave.converters.fict_cascade.resolve_cascade` for the
    resolution rules.  ``downstream_code`` is ``None`` when the cascade
    terminates at the sea.

    Parameters
    ----------
    confhd_df:
        ``Confhd.usinas``.
    cadastro:
        ``Hidr.cadastro`` (with MODIF.DAT overrides applied).  Required for a
        correct map: fictitious plants are now identified structurally from
        productivity (:func:`plants.fictitious_codes`), so without it the
        resolver cannot tell a fictitious accounting node from a real plant and
        the cascade stops short.  Defaults to empty only for callers that
        operate on FICT-free plant sets (e.g. unit tests).
    """
    from novomodelo_bridge.newave.converters.fict_cascade import resolve_cascade

    if cadastro is None:
        cadastro = pd.DataFrame()
    resolutions = resolve_cascade(confhd_df, cadastro)
    return {code: r.downstream_code for code, r in resolutions.items()}


def _build_hydro_to_ree(
    confhd_df: pd.DataFrame, cadastro: pd.DataFrame
) -> dict[int, int]:
    """Return {plant_code: ree_code} for existing non-fictitious plants."""
    non_fict = active_hydros(confhd_df, cadastro)
    return {int(r["codigo_usina"]): int(r["ree"]) for _, r in non_fict.iterrows()}


def compute_accumulated_integrated_productivities(
    cadastro: pd.DataFrame,
    confhd_df: pd.DataFrame,
) -> dict[int, float]:
    """Cascade-sum of per-plant stored-energy (EARM) productivity.

    Public, stable seam: the results comparator
    (:mod:`novomodelo_bridge.comparators.newave.results`) calls this to build its
    productivity-detail tab, so the ``(cadastro, confhd_df) -> {code: rho}``
    contract is shared across the converter↔comparator boundary. It is *not*
    interchangeable with :func:`compute_accumulated_productivities` (which uses
    the gen=ρ·Q point value); keep both distinct.

    Companion to :func:`compute_accumulated_productivities` but uses the
    EARM-flavored ρ that matches pmo.dat's
    ``produtibilidade_equivalente_volmin_volmax`` — the volmin→volmax integral
    for monthly reservoirs and the point at ``volume_referencia`` for
    run-of-river / special plants (see :func:`stored_energy_productivity`) —
    instead of the gen=ρ·Q point value at v_65.  Returns the cascade-summed
    values keyed by plant code.  Used as the base-case static check in
    :func:`convert_vminop_constraints` — the per-stage version is computed
    separately and accounts for CFUGA/CMONT overrides.
    """
    from novomodelo_bridge.newave.converters.fict_cascade import resolve_cascade

    resolutions = resolve_cascade(confhd_df, cadastro)
    downstream_map: dict[int, int | None] = {
        code: r.downstream_code for code, r in resolutions.items()
    }

    own_prod: dict[int, float] = {}
    for code, resolution in resolutions.items():
        if code in cadastro.index:
            own_prod[code] = stored_energy_productivity(cadastro.loc[code])
        else:
            own_prod[code] = 0.0
        own_prod[code] += resolution.fict_rho_sum

    return _cascade_sum(downstream_map, own_prod)


def compute_accumulated_productivities(
    cadastro: pd.DataFrame,
    confhd_df: pd.DataFrame,
) -> dict[int, float]:
    """Compute accumulated cascade productivity for each hydro plant.

    The accumulated productivity of a plant is its own productivity plus the
    accumulated productivity of its downstream plant.  This is computed by
    traversing the cascade DAG from downstream (sea-level sinks) to upstream.

    The cascade chain is resolved with
    :func:`novomodelo_bridge.newave.converters.fict_cascade.resolve_cascade`, so any
    fictitious plants between a real plant and its next real downstream are
    collapsed: the FICTs' ρ_eq is folded into the upstream real plant's own
    ρ_eq and the next real plant becomes the direct downstream.  This makes
    novomodelo-bridge's accumulated productivity reproduce the source model's
    ``produtibilidade_acumulada_calculo_earm`` from ``pmo.dat``.

    Parameters
    ----------
    cadastro:
        The ``Hidr.cadastro`` DataFrame (with MODIF.DAT overrides applied).
    confhd_df:
        The ``Confhd.usinas`` DataFrame.

    Returns
    -------
    dict[int, float]
        Mapping from plant code to accumulated productivity in MW/(m³/s).
    """
    from novomodelo_bridge.newave.converters.fict_cascade import resolve_cascade

    resolutions = resolve_cascade(confhd_df, cadastro)
    downstream_map: dict[int, int | None] = {
        code: r.downstream_code for code, r in resolutions.items()
    }

    own_prod: dict[int, float] = {}
    for code, resolution in resolutions.items():
        if code in cadastro.index:
            own_prod[code] = compute_productivity(cadastro.loc[code])
        else:
            own_prod[code] = 0.0
        own_prod[code] += resolution.fict_rho_sum

    return _cascade_sum(downstream_map, own_prod)


def _cascade_sum(
    downstream_map: dict[int, int | None],
    own_value: dict[int, float],
) -> dict[int, float]:
    """Topological cascade sum: result[h] = own_value[h] + result[downstream(h)].

    Plants with no downstream (sea sinks) keep their own value. Used by both
    the single-scalar :func:`compute_accumulated_productivities` and the
    per-stage :func:`compute_per_stage_acc_productivities`.
    """
    result: dict[int, float] = {}

    def _accumulate(code: int) -> float:
        if code in result:
            return result[code]
        ds = downstream_map.get(code)
        ds_acc = _accumulate(ds) if ds is not None else 0.0
        result[code] = own_value.get(code, 0.0) + ds_acc
        return result[code]

    for code in downstream_map:
        _accumulate(code)
    return result


def compute_max_prodtacum_sin(case: NewaveCase) -> float | None:
    """Return ``MAX_PRODTACUM_SIN`` = max accumulated productivity at altura máxima.

    The source model converts the DESVIO ("outros usos da água" / water-withdrawal) and
    evaporation penalties with the **maximum accumulated cascade productivity** (manual
    §3.24). pmo.dat shows these penalties are **fixed** over the whole study — the max
    is ITAIPU's cascade, which carries no CFUGA/CMONT temporal override, so it never
    moves — and the per-plant productivity is taken at the **maximum height** (vol_max),
    i.e. The source model's ``produtibilidade_acumulada_calculo_altura_maxima``. On the
    example case this returns ≈ 6.4458, matching pmo's "PENALIDADE POR VIOLACAO DOS
    OUTROS USOS" (19 149.55 → implied 6.4371) and Pmo's 6.4420; the legacy 65%-reference
    accumulated max was 6.3542 (~1.4% low).

    Falls back to ``None`` when the source model files cannot be read (mocked tests).
    """
    from novomodelo_bridge.newave.converters.fict_cascade import resolve_cascade

    try:
        cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)
        confhd_df = case.confhd.usinas
    except (OSError, ValueError, AttributeError, TypeError, KeyError):
        return None

    resolutions = resolve_cascade(confhd_df, cadastro)
    downstream_map: dict[int, int | None] = {
        code: r.downstream_code for code, r in resolutions.items()
    }
    own_max: dict[int, float] = {}
    for code, resolution in resolutions.items():
        if code in cadastro.index:
            hreg = cadastro.loc[code]
            useful = float(hreg["volume_maximo"]) - float(hreg["volume_minimo"])
            own_max[code] = compute_productivity(hreg, useful_volume_override=useful)
        else:
            own_max[code] = 0.0
        own_max[code] += resolution.fict_rho_sum

    acc = _cascade_sum(downstream_map, own_max)
    return max(acc.values()) if acc else None


def compute_per_stage_acc_productivities(
    confhd_df: pd.DataFrame,
    per_stage_own: dict[int, list[float]],
    cadastro: pd.DataFrame | None = None,
) -> dict[int, list[float]]:
    """Cascade-sum per-stage own productivities into per-stage ρ_acum lists.

    For each plant ``h`` and stage ``s``:
        ρ_acum(h, s) = ρ_eq_own(h, s) + ρ_acum(downstream(h), s)

    The cascade topology is static; only the leaf values change per stage.
    Stages are processed independently so a CFUGA override on plant ``A``
    correctly shifts ρ_acum for every plant strictly upstream of ``A`` from
    the override stage onwards, while leaving siblings unaffected.

    Parameters
    ----------
    confhd_df:
        ``Confhd.usinas`` — supplies the downstream-plant topology.
    per_stage_own:
        ``{plant_code: [own_ρ_eq per stage]}`` — output of
        :func:`hydro.compute_per_stage_own_productivities`.

    Returns
    -------
    dict[int, list[float]]
        ``{plant_code: [ρ_acum per stage]}`` for every plant present in both
        ``confhd_df`` (existing, non-fictitious) and ``per_stage_own``.
    """
    downstream_map = _build_hydro_downstream_map(confhd_df, cadastro)
    if not per_stage_own:
        return {}

    # All per-stage lists must share the same length — pick the first.
    sample = next(iter(per_stage_own.values()))
    n_stages = len(sample)

    result: dict[int, list[float]] = {code: [0.0] * n_stages for code in downstream_map}
    for s in range(n_stages):
        own_at_s: dict[int, float] = {
            code: (per_stage_own[code][s] if code in per_stage_own else 0.0)
            for code in downstream_map
        }
        acc_at_s = _cascade_sum(downstream_map, own_at_s)
        for code, v in acc_at_s.items():
            result[code][s] = v
    return result


def _warn_if_non_fixa_penalization(configuracoes_penalizacao: list[Any] | None) -> bool:
    """Warn when curva.dat selects a *non-FIXA* security-curve penalization.

    The penalization-config line of curva.dat carries, as its first field, the ``TIPO DE
    PENALIZACAO``: ``0`` = **FIXA** (a fixed per-violation penalty at the curve cost),
    anything else = the source model's iterative / variable penalization (the
    ``ETAPA-2`` adjustment).

    Novomodelo-bridge models the **FIXA** convention: a per-stage VminOP (minimum stored
    energy) slack penalized at the fixed curve cost.  This is the faithful equivalent of
    the source model FIXA, so a FIXA deck needs no warning.  It does **not** reproduce
    the iterative / variable penalization, so a non-FIXA deck will show an expected
    VminOP violation-penalty difference.

    Emits an INFO confirmation for FIXA and a WARNING for non-FIXA.  Returns
    ``True`` only when the (non-FIXA) warning fired (for testability).
    """
    if not configuracoes_penalizacao:
        return False
    try:
        tipo = int(configuracoes_penalizacao[0])
    except (TypeError, ValueError, IndexError):
        return False
    if tipo == 0:
        emit(
            Diagnostic(
                code="vminop-penalization-fixa",
                severity=Severity.INFO,
                category=_CATEGORY,
                title="Security curve uses FIXA penalization",
                summary=(
                    "curva.dat selects TIPO DE PENALIZACAO = 0 (FIXA): matches "
                    "novomodelo-bridge's VminOP modelling (a per-stage curve slack "
                    "at the fixed penalty), so no VminOP violation-penalty "
                    "difference is expected from the curve handling."
                ),
            ),
            logger=_LOG,
        )
        return False
    emit(
        Diagnostic(
            code="vminop-penalization-not-fixa",
            severity=Severity.WARNING,
            category=_CATEGORY,
            title="Security curve uses non-FIXA penalization",
            summary=(
                f"curva.dat selects TIPO DE PENALIZACAO = {tipo} (non-FIXA): "
                "novomodelo-bridge models the FIXA convention and does not "
                "reproduce NEWAVE's iterative / variable curve penalization, "
                "so a VminOP violation-penalty difference is expected."
            ),
            remediation="Check curva.dat's TIPO DE PENALIZACAO field.",
        ),
        logger=_LOG,
    )
    return True


def _curve_seasonalizes(configuracoes_penalizacao: list[Any] | None) -> bool:
    """Return whether the security curve is seasonalized in the post-study period.

    The penalization-config line of ``curva.dat`` carries, as its **third** field,
    the post-study (``Período Estático Final``) seasonalization flag for the
    security curve: ``1`` repeats the last study year's monthly curve across the
    static final period, anything else freezes December's value across the tail (the
    source model manual table, p.32-33). Returns ``False`` (freeze — the manual default)
    when the field is absent or unparseable.
    """
    try:
        return int(configuracoes_penalizacao[2]) == 1  # type: ignore[index]
    except (TypeError, ValueError, IndexError):
        return False


def _is_stored_energy_reservoir(cadastro: pd.DataFrame, code: int) -> bool:
    """True iff the source model counts plant ``code``'s storage in a REE's stored
    energy.

    The source model's stored-energy (EARM) and VminOP accounting include **only
    monthly-regulating reservoirs** (``tipo_regulacao == "M"``) that have usable storage
    (``volume_maximo > volume_minimo``).  Run-of-river (``"D"``) and special-regime
    (``"S"``) plants are excluded even when their *accumulated* cascade productivity is
    positive from downstream powerhouses — e.g. ITAIPU (``"S"``) and JIRAU (``"D"``)
    carry storage but are not part of the source model's stored energy.  This reproduces
    the plant set of pmo.dat's ``produtibilidade_acumulada_calculo_earm`` column.

    Narrower than a plain useful-volume test (``volume_maximo > volume_minimo``
    alone): that broader criterion does not gate on ``tipo_regulacao``.
    """
    if code not in cadastro.index:
        return False
    if str(cadastro.loc[code, "tipo_regulacao"]).strip() != "M":
        return False
    vol_min = float(cadastro.loc[code, "volume_minimo"])
    vol_max = float(cadastro.loc[code, "volume_maximo"])
    return vol_max - vol_min > 0.0


def convert_vminop_constraints(
    case: NewaveCase,
    id_map: NewaveIdMap,
    allocator: ConstraintIdAllocator | None = None,
) -> VminopResult | None:
    """Convert curva.dat VminOP constraints to Novomodelo generic constraints.

    Expressions are emitted using the novomodelo HEAD ``@name`` sigil, with one
    ``@rho_acum_h{hydro_id}`` per hydro term.  The accumulated productivity used both
    for the per-stage RHS bound and (via the fourth return value) for the
    ``@rho_acum_h{id}`` override is the cascade-summed *integrated* productivity — the
    source model's stored-energy / EARM convention — which differs from novomodelo's default
    point ρ_acum (gen = ρ·Q coefficient) by up to ~10% on plants with non-trivial head
    swing.

    Only monthly-regulating reservoirs with usable storage participate in a REE's
    expression (see :func:`_is_stored_energy_reservoir`); run-of-river and
    special-regime plants are excluded to match the source model's EARM plant set.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Entity ID mapping.
    allocator:
        Shared id source; defaults to a fresh :class:`ConstraintIdAllocator`
        (ids from 0) when omitted. The pipeline threads one instance across
        every emitter so ids stay contiguous across VminOP/electric/AGRINT.

    Returns
    -------
    tuple[dict, pa.Table, list[int], dict[int, list[float]]] | None
        ``(constraints_dict, bounds_table, referenced_hydro_ids,
        rho_acum_per_stage_overrides)`` or ``None`` if ``curva.dat`` is
        absent.  The fourth element maps novomodelo hydro_id → per-stage
        integrated ρ_acum and is fed into ``build_scalar_parameters`` so
        the LP coefficient at ``@rho_acum_h{id}`` matches the RHS bound.
    """
    if case.curva is None:
        _LOG.debug("curva.dat not found; skipping VminOP constraints.")
        return None

    # Honor dger.dat's `curva_aversao` switch: The source model itself disables the
    # risk-aversion curve when this flag is 0, even if curva.dat is on disk. Mirror that
    # here so the converted novomodelo case matches the source model's behavior. When the
    # field is absent (None), preserve historical behavior and emit the constraints —
    # only an explicit 0 disables them.
    dger = case.dger
    if dger.curva_aversao == 0:
        _LOG.info(
            "dger.dat curva_aversao=0; NEWAVE disables the risk-aversion"
            " curve, so skipping VminOP constraints."
        )
        return None

    curva = case.curva
    curva_df = curva.curva_seguranca
    if curva_df is None or curva_df.empty:
        return None

    # Novomodelo's per-stage curve slack reproduces the source model's FIXA penalization
    # (TIPO DE PENALIZACAO = 0); warn only when the deck selects a non-FIXA mode, which
    # the bridge does not reproduce.
    _warn_if_non_fixa_penalization(curva.configuracoes_penalizacao)

    # curva.dat's penalization line carries, as its third field, the post-study
    # seasonalization flag for the security curve (the "Período Estático Final"
    # rule, manual p.32-33). When set, the static final period repeats the last
    # study year's monthly curve; otherwise it freezes December's value across the
    # whole tail. Default (parse failure / absent) is freeze, matching the manual.
    seasonalize_curve_post_study = _curve_seasonalizes(curva.configuracoes_penalizacao)

    penalty_df = curva.custos_penalidades
    confhd = case.confhd
    hidr = case.hidr
    ree_file = case.ree

    cadastro = _apply_permanent_overrides(hidr.cadastro, case)
    confhd_df = confhd.usinas

    _horizon = case.horizon
    start_month = _horizon.start_month
    start_year = _horizon.start_year
    num_stages = _horizon.total_stages

    # The source model uses the *integrated* productivity (ρ_esp · (1/useful) ·
    # ∫_vmin^vmax h(V) dV) to evaluate stored energy / VminOP, which is the
    # `produtibilidade_acumulada_calculo_earm` column in pmo.dat — different from the
    # gen = ρ·Q point productivity at v_65 that the LP uses.  We compute that integrated
    # cascade here and use it both for the per-stage RHS bound *and* (via the return
    # value) to override the `@rho_acum_h{id}` scalar parameter in
    # generic_parameters.json so the LP's constraint coefficient matches the source
    # model.  Without the override the LHS would use novomodelo's default point ρ_acum and
    # silently drift from the RHS by ~10% on plants with non-trivial head swing.
    acc_prod = compute_accumulated_integrated_productivities(cadastro, confhd_df)
    per_stage_own_int = compute_per_stage_own_integrated_productivities(case)
    per_stage_acc = compute_per_stage_acc_productivities(
        confhd_df, per_stage_own_int, cadastro
    )

    # Convert ρ_acum from MW/(m³/s) to MWmonth/hm³ so the VminOP LHS (Σ ρ_acum ·
    # hydro_storage[hm³], and the matching RHS, both built from per_stage_acc) is the
    # *true* stored energy in MWmonth rather than ρ_acum·hm³, which overstates it by the
    # hm³↔(m³/s)·month factor (≈ 2.628). The VminOP slack penalty is a R$/MWh value
    # (penalid.dat); without this conversion the slack is that factor too large, so the
    # effective curve-violation cost rises above the deficit cost and the LP prefers
    # deficit to violating the security curve — i.e. Novomodelo hoards water under scarcity
    # instead of drawing it down like the source model.
    #
    # The factor is applied *per stage* via :func:`_vminop_energy_factor`, which
    # uses each stage's real month length (novomodelo prices the slack as
    # ``penalty × block_hours`` with the actual 672–744 h durations, not the
    # fixed 730 h). The factor cancels between LHS and RHS, so the binding
    # storage level is unchanged; only the slack's energy units (and thus the
    # effective R$/MWh penalty) are corrected.
    per_stage_acc = {
        code: [
            v / _vminop_energy_factor(start_year, start_month, s)
            for s, v in enumerate(values)
        ]
        for code, values in per_stage_acc.items()
    }

    hydro_to_ree = _build_hydro_to_ree(confhd_df, cadastro)

    ree_hydros: dict[int, list[int]] = defaultdict(list)
    for code, ree_code in hydro_to_ree.items():
        ree_hydros[ree_code].append(code)

    # Build penalty lookup: ree_code -> penalty value
    penalty_map: dict[int, float] = {}
    if penalty_df is not None and not penalty_df.empty:
        for _, row in penalty_df.iterrows():
            penalty_map[int(row["codigo_ree"])] = float(row["penalidade"])

    # REE names from ree.dat
    ree_names: dict[int, str] = {}
    ree_df = ree_file.rees
    if ree_df is not None:
        for _, row in ree_df.iterrows():
            ree_names[int(row["codigo"])] = str(row["nome"]).strip()

    # REEs that have constraints in curva.dat
    constraint_rees = sorted(curva_df["codigo_ree"].unique())

    allocator = allocator or ConstraintIdAllocator()
    builder = GenericConstraintBuilder(allocator)
    all_referenced_ids: set[int] = set()

    # Loop-accumulate-then-emit-once: one record per skipped/clamped REE,
    # emitted after the loop so the per-REE detail survives finalize_diagnostics'
    # (code, summary) de-dup (see converters/thermal.py's _GtminRecord idiom).
    vminop_skips: list[_VminopSkip] = []
    penalty_clamps: list[_VminopPenaltyClamp] = []

    for ree_code in constraint_rees:
        ree_code = int(ree_code)
        hydros_in_ree = ree_hydros.get(ree_code, [])

        if not hydros_in_ree:
            vminop_skips.append(
                _VminopSkip(
                    ree_code=ree_code,
                    ree_name=ree_names.get(ree_code, str(ree_code)),
                    reason="no hydro plants",
                )
            )
            continue

        # Build expression: sum of
        # @rho_acum_h{novomodelo_id} * hydro_storage_final(novomodelo_id).
        # The coefficient is resolved by novomodelo to its own ρ_acum at solve time.
        # We collect the participating plants here; the RHS bound for each
        # stage is computed below using the per-stage ρ_acum so LHS and RHS
        # stay aligned through CFUGA/CMONT temporal overrides.
        terms: list[str] = []
        active_plant_codes: list[int] = []
        referenced_ids: list[int] = []

        for plant_code in sorted(hydros_in_ree):
            # The source model's stored energy (EARM) — and thus the VminOP expression —
            # counts only monthly-regulating reservoirs with usable storage. Gate on
            # that, not on ``acc_prod > 0`` alone: run-of-river / special-regime plants
            # such as ITAIPU and JIRAU carry a positive *accumulated* productivity from
            # downstream powerhouses yet are not part of the source model's stored
            # energy.
            if not _is_stored_energy_reservoir(cadastro, plant_code):
                continue
            if acc_prod.get(plant_code, 0.0) <= 0.0:
                continue
            try:
                novomodelo_id = id_map.hydro_id(plant_code)
            except KeyError:
                continue
            terms.append(
                f"@{rho_acum_name(novomodelo_id)} * hydro_storage_final({novomodelo_id})"
            )
            referenced_ids.append(novomodelo_id)
            active_plant_codes.append(plant_code)

        # Pre-compute (vol_min, vol_max) per active plant once.
        plant_volumes: dict[int, tuple[float, float]] = {
            code: (
                float(cadastro.loc[code, "volume_minimo"]),
                float(cadastro.loc[code, "volume_maximo"]),
            )
            for code in active_plant_codes
        }

        # Existence check: at any stage there must be at least one plant
        # contributing positive ρ_acum × usable volume. Use stage 0 as a
        # representative; if everything is zero there, every later stage is
        # too (cascade sum is monotone in own values).
        max_energy_stage0 = sum(
            per_stage_acc.get(code, [0.0])[0] * vmax
            for code, (_vmin, vmax) in plant_volumes.items()
        )
        if not terms or max_energy_stage0 <= 0.0:
            vminop_skips.append(
                _VminopSkip(
                    ree_code=ree_code,
                    ree_name=ree_names.get(ree_code, "?"),
                    reason="no valid terms for VminOP expression",
                )
            )
            continue

        def _rhs_at(stage: int, pct: float) -> float:
            """Compute (pct/100)·useful(stage) + dead(stage)."""
            useful_s = 0.0
            dead_s = 0.0
            for code, (vmin, vmax) in plant_volumes.items():
                ps = per_stage_acc.get(code)
                ap_s = ps[stage] if ps is not None and 0 <= stage < len(ps) else 0.0
                useful_s += ap_s * (vmax - vmin)
                dead_s += ap_s * vmin
            return (pct / 100.0) * useful_s + dead_s

        expression = " + ".join(terms)
        ree_name = ree_names.get(ree_code, str(ree_code))
        penalty = penalty_map.get(ree_code, 1000.0)
        if penalty <= 0.0:
            penalty_clamps.append(
                _VminopPenaltyClamp(
                    ree_code=ree_code,
                    ree_name=ree_name,
                    declared=penalty,
                    applied=1000.0,
                )
            )
            penalty = 1000.0

        all_referenced_ids.update(referenced_ids)

        # Build per-stage bounds from curva_df.
        # curva.dat only covers the study period. Extend into the post-study
        # ("Período Estático Final") per source-model's rule (manual table p.32-33):
        # seasonalize (repeat the last study year's monthly curve) when
        # curva.dat's third penalization field is set, otherwise freeze December's
        # value across the whole tail. ``seasonalize_curve_post_study`` carries the
        # flag.
        study_months = _horizon.study_months
        ree_curva = curva_df[curva_df["codigo_ree"] == ree_code].sort_values("data")

        # First pass: emit in-study slots; build the per-calendar-month seasonal
        # map and a per-stage map (the latter supplies the freeze value).
        slots: list[tuple[int, int | None, float | None, float | None]] = []
        seasonal_pct: dict[int, float] = {}  # calendar_month -> last percentage
        pct_by_stage: dict[int, float] = {}
        for _, crow in ree_curva.iterrows():
            dt: datetime = crow["data"]
            stage_id = (dt.year - start_year) * 12 + (dt.month - start_month)
            if stage_id < 0 or stage_id >= num_stages:
                continue

            percentage = float(crow["valor"])
            rhs = _rhs_at(stage_id, percentage)
            slots.append((stage_id, None, rhs, None))

            seasonal_pct[dt.month] = percentage  # last value wins
            pct_by_stage[stage_id] = percentage

        # Second pass: fill uncovered stages. In-study gaps (and the post-study
        # tail when seasonalized) use the seasonal pattern; otherwise the tail
        # freezes at the last in-study stage's percentage (December of the last
        # study year, forward-filled if that exact stage had no explicit row).
        if seasonal_pct:
            covered = set(pct_by_stage)
            in_study = [s for s in pct_by_stage if s < study_months]
            freeze_pct = pct_by_stage[max(in_study)] if in_study else None
            for stage_id in range(num_stages):
                if stage_id in covered:
                    continue
                if stage_id >= study_months and not seasonalize_curve_post_study:
                    pct = freeze_pct
                else:
                    cal_month = ((start_month - 1 + stage_id) % 12) + 1
                    pct = seasonal_pct.get(cal_month)
                if pct is not None:
                    rhs = _rhs_at(stage_id, pct)
                    slots.append((stage_id, None, rhs, None))

        builder.add(
            name=f"VminOP_{ree_name}",
            description=f"Minimum stored energy for REE {ree_code} ({ree_name})",
            expression=expression,
            slack={"enabled": True, "penalty": penalty},
            slots=slots,
        )

    if vminop_skips:
        emit(
            Diagnostic(
                code="vminop-ree-not-expressible",
                severity=Severity.WARNING,
                category=_CATEGORY,
                title=f"VminOP constraint not expressible ({len(vminop_skips)} REE(s))",
                summary=(
                    f"{len(vminop_skips)} REE(s) with a curva.dat VminOP entry "
                    "could not be expressed as a generic constraint; skipping."
                ),
                table=DiagnosticTable(
                    columns=["REE", "Name", "Reason"],
                    rows=[[s.ree_code, s.ree_name, s.reason] for s in vminop_skips],
                    justify=["right", "left", "left"],
                ),
            ),
            logger=_LOG,
        )
    if penalty_clamps:
        emit(
            Diagnostic(
                code="vminop-penalty-nonpositive",
                severity=Severity.WARNING,
                category=_CATEGORY,
                title=f"VminOP penalty not positive ({len(penalty_clamps)} REE(s))",
                summary=(
                    f"{len(penalty_clamps)} REE(s) declare a curva.dat VminOP "
                    "penalty that is not positive; using 1000.0 instead."
                ),
                table=DiagnosticTable(
                    columns=["REE", "Name", "Declared", "Applied"],
                    rows=[
                        [c.ree_code, c.ree_name, c.declared, c.applied]
                        for c in penalty_clamps
                    ],
                    justify=["right", "left", "right", "right"],
                ),
                remediation="Check curva.dat's penalty column for these REEs.",
            ),
            logger=_LOG,
        )

    result = builder.result()
    if result is None:
        return None

    constraints_dict = {
        "$schema": novomodelo_schemas.schema_url_for(
            "constraints/generic_constraints.json"
        ),
        "constraints": result.constraints,
    }

    _LOG.info(
        "Generated %d VminOP constraints with %d stage bounds.",
        len(result.constraints),
        result.bounds.num_rows,
    )

    # Build the per_stage override for rho_acum_h{id}: map novomodelo hydro_id to
    # the integrated cascade ρ per stage. Only emit overrides for plants
    # actually referenced by a VminOP expression so unaffected hydros keep
    # novomodelo's default `computed` resolution.
    rho_acum_overrides: dict[int, list[float]] = {}
    for plant_code, per_stage_values in per_stage_acc.items():
        try:
            novomodelo_id = id_map.hydro_id(plant_code)
        except KeyError:
            continue
        if novomodelo_id not in all_referenced_ids:
            continue
        rho_acum_overrides[novomodelo_id] = list(per_stage_values)

    return VminopResult(
        constraints_dict,
        result.bounds,
        sorted(all_referenced_ids),
        rho_acum_overrides,
    )


# ---------------------------------------------------------------------------
# Electric constraints from restricao-eletrica.csv
# ---------------------------------------------------------------------------

_UNBOUNDED_THRESHOLD = -1.0e29  # values below this are treated as -inf


def _find_restricao_eletrica(directory: Path) -> Path | None:
    """Locate ``restricao-eletrica.csv`` via ``indices.csv`` in *directory*.

    Parses ``indices.csv`` case-insensitively, looking for the
    ``RESTRICAO-ELETRICA-ESPECIAL`` key.  Returns the resolved path or
    ``None`` if either file is absent.
    """
    indices_path: Path | None = None
    for entry in directory.iterdir():
        if entry.is_file() and entry.name.lower() == "indices.csv":
            indices_path = entry
            break

    if indices_path is None:
        return None

    try:
        with indices_path.open(encoding="latin-1") as fh:
            for line in fh:
                parts = [p.strip() for p in line.split(";")]
                if (
                    len(parts) >= 3
                    and parts[0].upper() == "RESTRICAO-ELETRICA-ESPECIAL"
                ):
                    filename = parts[2].strip()
                    if not filename:
                        return None
                    candidate = directory / filename
                    if candidate.exists():
                        return candidate
                    # Case-insensitive fallback
                    lower = filename.lower()
                    for entry in directory.iterdir():
                        if entry.is_file() and entry.name.lower() == lower:
                            return entry
                    return None
    except OSError:
        return None

    return None


def _parse_restricao_eletrica(
    path: Path,
) -> tuple[
    dict[int, str],
    dict[int, tuple[str, str]],
    list[tuple[int, str, str, int, float, float]],
]:
    """Parse ``restricao-eletrica.csv`` into its three sections.

    Returns
    -------
    expressions : dict[int, str]
        Mapping from constraint code to raw formula string.
    horizons : dict[int, tuple[str, str]]
        Mapping from constraint code to (PerIni, PerFin) as "YYYY/MM" strings.
    bounds : list[tuple[int, str, str, int, float, float]]
        Each element is (cod_rest, PerIni, PerFin, pat, lim_inf, lim_sup).
    """
    expressions: dict[int, str] = {}
    horizons: dict[int, tuple[str, str]] = {}
    bounds: list[tuple[int, str, str, int, float, float]] = []

    with path.open(encoding="latin-1") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("&"):
                continue

            parts = [p.strip() for p in line.split(";")]

            if parts[0].upper() == "RE" and len(parts) >= 3:
                cod = int(parts[1])
                formula = parts[2]
                expressions[cod] = formula

            elif parts[0].upper() == "RE-HORIZ-PER" and len(parts) >= 4:
                cod = int(parts[1])
                horizons[cod] = (parts[2], parts[3])

            elif parts[0].upper() == "RE-LIM-FORM-PER-PAT" and len(parts) >= 7:
                cod = int(parts[1])
                per_ini = parts[2]
                per_fin = parts[3]
                pat = int(parts[4])
                lim_inf = float(parts[5])
                lim_sup = float(parts[6])
                bounds.append((cod, per_ini, per_fin, pat, lim_inf, lim_sup))

    return expressions, horizons, bounds


def _build_line_id_map(case: NewaveCase) -> dict[tuple[int, int], int]:
    """Build the canonical (src, tgt) -> line_id mapping.

    Replicates the exact logic from ``convert_lines`` in network.py so that
    line IDs are consistent across both converters.

    Returns
    -------
    dict[tuple[int, int], int]
        Maps canonical (smaller_subsystem, larger_subsystem) to 0-based
        line ID.  Only pairs from ``sistema.dat`` are included.
    """
    sistema = case.sistema
    limites_df = sistema.limites_intercambio

    if limites_df is None or limites_df.empty:
        return {}

    dger = case.dger
    study_start_dt = datetime(dger.ano_inicio_estudo, dger.mes_inicio_estudo, 1)
    first_month = limites_df[limites_df["data"] == study_start_dt]
    if first_month.empty:
        non_nan_df = limites_df.dropna(subset=["valor"])
        if not non_nan_df.empty:
            first_month = limites_df[limites_df["data"] == non_nan_df["data"].min()]

    canonical_pairs: set[tuple[int, int]] = set()
    for _, row in first_month.iterrows():
        de = int(row["submercado_de"])
        para = int(row["submercado_para"])
        src, tgt = (de, para) if de < para else (para, de)
        canonical_pairs.add((src, tgt))

    return {pair: idx for idx, pair in enumerate(sorted(canonical_pairs))}


# Term regex: optional sign and float coefficient, then function name and args.
# The sign group captures a leading '+' or '-' (possibly preceded by whitespace).
_TERM_RE = re.compile(
    r"(?P<sign>[+-])?\s*(?P<coeff>\d+\.?\d*)?\s*\*?\s*"
    r"(?P<fn>ger_usih|ener_interc)\((?P<args>\d+(?:,\s*\d+)*)\)"
)


def _render_signed_expression(parsed_terms: list[tuple[float, str]]) -> str:
    """Render (coefficient, variable) pairs into a signed sum expression string.

    - Positive coefficient 1.0:  ``variable(id)``
    - Positive coefficient other: ``coeff * variable(id)``
    - Negative coefficient -1.0: ``- variable(id)``
    - Negative coefficient other: ``- abs(coeff) * variable(id)``
    - First term omits leading ``+``; subsequent positive terms use ``+ ``;
      negative terms use ``- `` as binary subtraction.
    """
    parts: list[str] = []
    for i, (coeff, var) in enumerate(parsed_terms):
        abs_coeff = abs(coeff)
        is_negative = coeff < 0.0
        term_body = var if abs_coeff == 1.0 else f"{abs_coeff} * {var}"
        if i == 0:
            parts.append(f"- {term_body}" if is_negative else term_body)
        else:
            parts.append(f"- {term_body}" if is_negative else f"+ {term_body}")
    return " ".join(parts)


def _parse_formula(
    formula: str,
    id_map: NewaveIdMap,
    line_id_map: dict[tuple[int, int], int],
    *,
    constraint_code: int,
    # Recording seam: `None` (the direct-unit-test default) means "do not
    # record"; convert_electric_constraints passes its own accumulator so the
    # dropped terms it reports carry which constraint they came from.
    skipped: list[_ElectricTermSkip] | None = None,
) -> str | None:
    """Translate a source-model RE formula into a Novomodelo expression string.

    Unknown plant codes or interchange pairs are skipped. Returns ``None`` if
    no valid terms remain after translation. See
    :func:`_render_signed_expression` for the output formatting rules.
    """
    # Each element: (effective_coeff: float, variable_str: str)
    parsed_terms: list[tuple[float, str]] = []

    for match in _TERM_RE.finditer(formula):
        sign = match.group("sign")
        coeff_str = match.group("coeff")
        coeff = float(coeff_str) if coeff_str else 1.0
        if sign == "-":
            coeff = -coeff
        fn = match.group("fn")
        args_str = match.group("args")
        args = [int(a.strip()) for a in args_str.split(",")]

        if fn == "ger_usih":
            plant_code = args[0]
            try:
                novomodelo_id = id_map.hydro_id(plant_code)
            except KeyError:
                if skipped is not None:
                    skipped.append(
                        _ElectricTermSkip(
                            constraint_code=constraint_code,
                            kind="hydro-unmapped",
                            source="formula",
                            hydro_code=plant_code,
                        )
                    )
                continue
            parsed_terms.append((coeff, f"hydro_generation({novomodelo_id})"))

        elif fn == "ener_interc":
            if len(args) < 2:
                if skipped is not None:
                    skipped.append(
                        _ElectricTermSkip(
                            constraint_code=constraint_code,
                            kind="malformed",
                            source="formula",
                            raw=args_str,
                        )
                    )
                continue
            from_sys, to_sys = args[0], args[1]
            src, tgt = (from_sys, to_sys) if from_sys < to_sys else (to_sys, from_sys)
            line_id = line_id_map.get((src, tgt))
            if line_id is None:
                if skipped is not None:
                    skipped.append(
                        _ElectricTermSkip(
                            constraint_code=constraint_code,
                            kind="no-line",
                            source="formula",
                            sys_from=from_sys,
                            sys_to=to_sys,
                        )
                    )
                continue

            # The source model's ``ener_interc(A, B)`` is the directional flow from A to
            # B as a non-negative variable.  Map to Novomodelo's
            # ``line_direct``/``line_reverse`` so the sign is encoded by the variable
            # choice rather than by negating the coefficient (the latter would dilute
            # the bound when LP solutions route in the non-canonical direction).  See
            # ``convert_agrint_constraints`` for the same rule applied to AGRINT terms.
            var = (
                f"line_direct({line_id})"
                if from_sys < to_sys
                else f"line_reverse({line_id})"
            )
            parsed_terms.append((coeff, var))

    if not parsed_terms:
        return None

    return _render_signed_expression(parsed_terms)


def _parse_yyyymm(period_str: str) -> tuple[int, int]:
    """Parse "YYYY/MM" into (year, month)."""
    parts = period_str.strip().split("/")
    return int(parts[0]), int(parts[1])


# Sentinel stage index returned when the individualizado cutoff date is
# unavailable: a value past any realistic horizon, so every stage is treated
# as individualised (no cutoff applied).
_NO_INDIVIDUALIZADO_CUTOFF = 9999


def _get_individualizado_cutoff(
    case: NewaveCase,
    start_year: int,
    start_month: int,
) -> int:
    """Return the first post-individualizado stage index.

    Reads REE.DAT for ``mes_fim_individualizado``/``ano_fim_individualizado``
    and converts to a 0-based stage index.  All REEs share the same cutoff
    date in production cases; we take the first REE's value.

    Returns the :data:`_NO_INDIVIDUALIZADO_CUTOFF` sentinel (a stage index
    beyond any real horizon) when the REE information is missing or unreadable.
    Because ``is_post_indiv = stage >= cutoff`` is then always False, the
    post-individualizado RE / seasonal-extrapolation bounds are dropped — so
    both fallback paths emit a structured diagnostic on the collect() sink
    rather than degrading silently.
    """
    try:
        ree_df = case.ree.rees
        if ree_df is None or ree_df.empty:
            emit(
                Diagnostic(
                    code="individualizado-cutoff-unknown",
                    severity=Severity.WARNING,
                    category=_CATEGORY,
                    title="Individualizado cutoff unknown",
                    summary=(
                        "REE.DAT has no entries; the individualizado cutoff "
                        "is unknown, so post-individualizado electric (RE) "
                        "bounds are not emitted."
                    ),
                ),
                logger=_LOG,
            )
            return _NO_INDIVIDUALIZADO_CUTOFF
        row = ree_df.iloc[0]
        end_month = int(row["mes_fim_individualizado"])
        end_year = int(row["ano_fim_individualizado"])
        return (end_year - start_year) * 12 + (end_month - start_month)
    except Exception:  # noqa: BLE001
        emit(
            Diagnostic(
                code="individualizado-cutoff-unknown",
                severity=Severity.WARNING,
                category=_CATEGORY,
                title="Individualizado cutoff unknown",
                summary=(
                    "Could not read the individualizado cutoff from REE.DAT; "
                    "post-individualizado electric (RE) bounds are not "
                    "emitted."
                ),
            ),
            logger=_LOG,
        )
        return _NO_INDIVIDUALIZADO_CUTOFF


def _parse_re_dat(
    case: NewaveCase,
    start_year: int,
    start_month: int,
    num_stages: int,
    num_patamares: int = 3,
) -> tuple[
    dict[int, list[int]],
    dict[int, dict[tuple[int, int], float]],
]:
    """Parse RE.DAT into plant sets and per-stage upper bound maps.

    Returns
    -------
    conjuntos : dict[int, list[int]]
        Mapping from constraint code to list of the source model plant codes.
    re_dat_bounds : dict[int, dict[tuple[int, int], float]]
        ``{constraint_code: {(stage_id, block_id): upper_bound}}``.
        ``patamar=0`` in RE.DAT is expanded to all blocks.  A bound covers
        only the stages its own row declares — the row's ``mes_fim``/
        ``ano_fim`` ends it, and stages no row covers carry no bound.
    """
    conjuntos: dict[int, list[int]] = {}
    re_dat_bounds: dict[int, dict[tuple[int, int], float]] = {}

    if case.files.re_dat is None:
        return conjuntos, re_dat_bounds

    try:
        re_file = case.re_dat
    except Exception:  # noqa: BLE001
        emit(
            Diagnostic(
                code="re-dat-unreadable",
                severity=Severity.WARNING,
                category=_CATEGORY,
                title="RE.DAT could not be parsed",
                summary="RE.DAT could not be parsed; RE constraints are skipped.",
            ),
            logger=_LOG,
        )
        return conjuntos, re_dat_bounds

    # Plant sets per constraint code.
    uc_df = re_file.usinas_conjuntos
    if uc_df is not None and not uc_df.empty:
        for _, row in uc_df.iterrows():
            code = int(row["conjunto"])
            plant = int(row["codigo_usina"])
            conjuntos.setdefault(code, []).append(plant)

    # Build per-constraint step-function bounds from restricoes.
    rest_df = re_file.restricoes
    if rest_df is None or rest_df.empty:
        return conjuntos, re_dat_bounds

    for code, grp in rest_df.groupby("conjunto"):
        code = int(code)
        # Collect changepoints: [(stage_id, patamar, value)]
        changepoints: list[tuple[int, int, float]] = []
        for _, row in grp.iterrows():
            mi = int(row["mes_inicio"])
            ai = int(row["ano_inicio"])
            mf = int(row["mes_fim"])
            af = int(row["ano_fim"])
            pat = int(row["patamar"])  # 0 = all blocks
            value = float(row["restricao"])

            # Expand date range into individual stages.
            y, m = ai, mi
            while (y, m) <= (af, mf):
                sid = (y - start_year) * 12 + (m - start_month)
                if 0 <= sid < num_stages:
                    changepoints.append((sid, pat, value))
                m += 1
                if m > 12:
                    m = 1
                    y += 1

        if not changepoints:
            continue

        # Patamar 0 sorts first, so a patamar-specific row wins its block.
        changepoints.sort()
        stage_bounds: dict[tuple[int, int], float] = {}

        for sid, pat, val in changepoints:
            if pat == 0:
                for b in range(num_patamares):
                    stage_bounds[(sid, b)] = val
            else:
                stage_bounds[(sid, pat - 1)] = val

        re_dat_bounds[code] = stage_bounds

    return conjuntos, re_dat_bounds


def convert_electric_constraints(
    case: NewaveCase,
    id_map: NewaveIdMap,
    allocator: ConstraintIdAllocator | None = None,
) -> GenericConstraintResult | None:
    """Convert electric constraints from restricao-eletrica.csv and RE.DAT.

    ``restricao-eletrica.csv`` provides constraints for the individualised
    period (first ~12 stages).  ``RE.DAT`` provides post-individualised
    bounds.  Both sources register every limit with a start and an end
    period, and a bound is emitted only for the stages its own registration
    covers: a constraint declared for part of the horizon is absent from the
    rest of it.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Entity ID mapping.
    allocator:
        Shared id source; defaults to a fresh :class:`ConstraintIdAllocator`
        (ids from 0) when omitted. The pipeline threads one instance across
        every emitter so IDs do not collide with VminOP constraints.

    Returns
    -------
    tuple[list[dict], pa.Table] | None
        A ``(constraints_list, bounds_table)`` pair, or ``None`` if no
        constraints are found.  ``bounds_table`` has schema
        ``(constraint_id: INT32, stage_id: INT32, block_id: INT32,
        bound_lower: DOUBLE, bound_upper: DOUBLE)`` — each row is
        single-sided (a ``<=`` ceiling or a ``>=`` floor, never both), so
        exactly one endpoint is populated per row.
    """
    switches = case.switches
    re_path = _find_restricao_eletrica(case.files.directory)
    if re_path is not None and not switches.restricao_eletrica.on:
        emit(switch_off_diagnostic(switches.restricao_eletrica), logger=_LOG)
        re_path = None
    has_re_dat = case.files.re_dat is not None
    if has_re_dat and not switches.re_dat.on:
        emit(switch_off_diagnostic(switches.re_dat), logger=_LOG)
        has_re_dat = False

    if re_path is None and not has_re_dat:
        _LOG.debug("No electric constraints found; skipping.")
        return None

    # Parse restricao-eletrica.csv (individualised period data).
    expressions: dict[int, str] = {}
    horizons: dict[int, tuple[str, str]] = {}
    bounds_rows: list[tuple[int, str, str, int, float, float]] = []

    if re_path is not None:
        expressions, horizons, bounds_rows = _parse_restricao_eletrica(re_path)

    _horizon = case.horizon
    start_month = _horizon.start_month
    start_year = _horizon.start_year
    num_anos = _horizon.num_anos
    num_stages = _horizon.total_stages

    patamar = case.patamar
    num_patamares: int = patamar.numero_patamares or 1

    # Individualised period cutoff.
    cutoff = _get_individualizado_cutoff(case, start_year, start_month)

    re_conjuntos: dict[int, list[int]] = {}
    re_dat_bounds: dict[int, dict[tuple[int, int], float]] = {}
    if has_re_dat:
        re_conjuntos, re_dat_bounds = _parse_re_dat(
            case, start_year, start_month, num_stages, num_patamares
        )

    if not expressions and not re_conjuntos:
        _LOG.debug("No electric constraints found; skipping.")
        return None

    line_id_map = _build_line_id_map(case)

    # Read ELETRI penalty from PENALID.DAT for slack costs. The source model manual v29
    # §3.24: ELETRI is energy-domain (R$/MWh) and applies only in individualized
    # periods. When absent in PENALID.DAT, the source model considers the constraint
    # only in the final simulation — novomodelo has no equivalent nuance, so we keep the
    # slack enabled with a high default (10 × MAX_DEFICIT, matching the source model's
    # evaporation/FPHA-folga magnitude) so RE_* constraints stay soft but very expensive
    # to violate.
    eletri_penalty: float | None = None
    if case.files.penalid is not None:
        try:
            penalid = case.penalid
            df_pen = penalid.penalidades
            if df_pen is not None and not df_pen.empty:
                eletri = df_pen[
                    (df_pen["variavel"] == "ELETRI")
                    & (df_pen["patamar_penalidade"] == 1)
                ]
                vals = eletri["valor_R$_MWh"].dropna()
                if not vals.empty:
                    eletri_penalty = float(vals.iloc[0])
        except (OSError, ValueError, KeyError) as exc:
            emit(
                Diagnostic(
                    code="electric-penalty-unreadable",
                    severity=Severity.WARNING,
                    category=_CATEGORY,
                    title="ELETRI penalty could not be read",
                    summary=(
                        "PENALID.DAT's ELETRI penalty could not be read; "
                        "falling back to 10x the maximum deficit cost, or "
                        "disabling the slack if that is also unavailable."
                    ),
                    notes=[str(exc)],
                ),
                logger=_LOG,
            )

    if eletri_penalty is None:
        # Fall back to 10 × MAX_DEFICIT (the source model evaporation-folga convention).
        try:
            _sis = case.sistema
            _def_df = _sis.custo_deficit
            if _def_df is not None and not _def_df.empty:
                eletri_penalty = 10.0 * float(_def_df["custo"].max())
        except (OSError, ValueError, KeyError):
            pass

    slack_config: dict = (
        {"enabled": True, "penalty": eletri_penalty}
        if eletri_penalty is not None and eletri_penalty > 0.0
        else {"enabled": False}
    )

    allocator = allocator or ConstraintIdAllocator()
    builder = GenericConstraintBuilder(allocator)

    # Index restricao-eletrica.csv bounds by constraint code.
    csv_bounds_by_code: dict[int, list[tuple[str, str, int, float, float]]] = (
        defaultdict(list)
    )
    for cod, per_ini, per_fin, pat, lim_inf, lim_sup in bounds_rows:
        csv_bounds_by_code[cod].append((per_ini, per_fin, pat, lim_inf, lim_sup))

    # All constraint codes from both sources.
    all_codes = sorted(set(expressions.keys()) | set(re_conjuntos.keys()))

    # Loop-accumulate-then-emit-once: one record per dropped term / skipped
    # constraint, grouped by kind and emitted once after the loop (see
    # converters/thermal.py's _GtminRecord idiom).
    elec_term_skips: list[_ElectricTermSkip] = []
    elec_constraint_skips: list[_ElectricConstraintSkip] = []

    for cod in all_codes:
        # --- Build expression ---
        has_csv = cod in expressions
        has_re_dat = cod in re_conjuntos

        if has_csv:
            novomodelo_expr = _parse_formula(
                expressions[cod],
                id_map,
                line_id_map,
                constraint_code=cod,
                skipped=elec_term_skips,
            )
        elif has_re_dat:
            # RE.DAT-only: expression is sum of hydro_generation for plants.
            terms: list[str] = []
            for plant_code in sorted(re_conjuntos[cod]):
                try:
                    novomodelo_id = id_map.hydro_id(plant_code)
                except KeyError:
                    elec_term_skips.append(
                        _ElectricTermSkip(
                            constraint_code=cod,
                            kind="hydro-unmapped",
                            source="RE.DAT",
                            hydro_code=plant_code,
                        )
                    )
                    continue
                terms.append(f"hydro_generation({novomodelo_id})")
            novomodelo_expr = " + ".join(terms) if terms else None
        else:
            continue

        if novomodelo_expr is None:
            elec_constraint_skips.append(
                _ElectricConstraintSkip(
                    constraint_code=cod, reason="no valid terms in expression"
                )
            )
            continue

        # --- Build bound maps for this constraint ---
        # csv_sup/csv_inf: restricao-eletrica.csv bounds by (stage, block)
        csv_sup: dict[tuple[int, int], float] = {}
        csv_inf: dict[tuple[int, int], float] = {}

        entries_for_code = csv_bounds_by_code.get(cod, [])
        horizon = horizons.get(cod)
        horiz_ini = _parse_yyyymm(horizon[0]) if horizon else (start_year, start_month)
        horiz_fin = (
            _parse_yyyymm(horizon[1]) if horizon else (start_year + num_anos - 1, 12)
        )

        for per_ini, per_fin, pat, lim_inf, lim_sup in entries_for_code:
            ini_year, ini_month = _parse_yyyymm(per_ini)
            fin_year, fin_month = _parse_yyyymm(per_fin)

            ini_year, ini_month = max(
                (ini_year, ini_month), horiz_ini, key=lambda t: t[0] * 12 + t[1]
            )
            fin_year, fin_month = min(
                (fin_year, fin_month), horiz_fin, key=lambda t: t[0] * 12 + t[1]
            )

            block_id = pat - 1  # 1-based -> 0-based

            y, m = ini_year, ini_month
            while (y, m) <= (fin_year, fin_month):
                sid = (y - start_year) * 12 + (m - start_month)
                if 0 <= sid < num_stages:
                    if lim_sup > _UNBOUNDED_THRESHOLD:
                        csv_sup[(sid, block_id)] = lim_sup
                    if lim_inf > _UNBOUNDED_THRESHOLD:
                        csv_inf[(sid, block_id)] = lim_inf
                m += 1
                if m > 12:
                    m = 1
                    y += 1

        # RE.DAT upper bounds for post-individualised stages.
        re_sup = re_dat_bounds.get(cod, {})

        # --- Determine senses ---
        has_sup = bool(csv_sup) or bool(re_sup)
        has_inf = bool(csv_inf)

        if not has_sup and not has_inf:
            elec_constraint_skips.append(
                _ElectricConstraintSkip(constraint_code=cod, reason="no bound data")
            )
            continue

        sup_id: int | None = None
        inf_id: int | None = None

        if has_sup:
            sup_id = builder.add_constraint(
                name=f"RE_{cod}",
                description=f"Electric constraint {cod}",
                expression=novomodelo_expr,
                slack=slack_config,
            )
        if has_inf:
            inf_id = builder.add_constraint(
                name=f"RE_{cod}",
                description=f"Electric constraint {cod}",
                expression=novomodelo_expr,
                slack=slack_config,
            )

        # --- Emit one row per declared (stage, block) bound ---
        for stage_id in range(num_stages):
            is_post_indiv = stage_id >= cutoff

            for block_id in range(num_patamares):
                # Upper bound.
                if sup_id is not None:
                    sup_val: float | None = None
                    if is_post_indiv and (stage_id, block_id) in re_sup:
                        sup_val = re_sup[(stage_id, block_id)]
                    elif (stage_id, block_id) in csv_sup:
                        sup_val = csv_sup[(stage_id, block_id)]
                    if sup_val is not None:
                        builder.add_bound_row(
                            sup_id,
                            stage_id=stage_id,
                            block_id=block_id,
                            lower=None,
                            upper=sup_val,
                        )

                # Lower bound.
                if inf_id is not None:
                    inf_val: float | None = None
                    if (stage_id, block_id) in csv_inf:
                        inf_val = csv_inf[(stage_id, block_id)]
                    if inf_val is not None:
                        builder.add_bound_row(
                            inf_id,
                            stage_id=stage_id,
                            block_id=block_id,
                            lower=inf_val,
                            upper=None,
                        )

    if elec_term_skips:
        by_kind: dict[str, list[_ElectricTermSkip]] = defaultdict(list)
        for skip in elec_term_skips:
            by_kind[skip.kind].append(skip)

        hydro_unmapped = by_kind.get("hydro-unmapped", [])
        if hydro_unmapped:
            emit(
                Diagnostic(
                    code="electric-constraint-hydro-unmapped",
                    severity=Severity.WARNING,
                    category=_CATEGORY,
                    title=(
                        "Electric constraint references an unmapped hydro "
                        f"plant ({len(hydro_unmapped)})"
                    ),
                    summary=(
                        f"{len(hydro_unmapped)} electric constraint term(s) "
                        "reference a hydro plant code with no matching novomodelo "
                        "entity; the term is dropped."
                    ),
                    table=DiagnosticTable(
                        columns=["Constraint", "Hydro code", "Source"],
                        rows=[
                            [s.constraint_code, s.hydro_code, s.source]
                            for s in hydro_unmapped
                        ],
                        justify=["right", "right", "left"],
                    ),
                ),
                logger=_LOG,
            )

        no_line = by_kind.get("no-line", [])
        if no_line:
            emit(
                Diagnostic(
                    code="electric-constraint-interchange-no-line",
                    severity=Severity.WARNING,
                    category=_CATEGORY,
                    title=(
                        "Electric constraint interchange has no matching line "
                        f"({len(no_line)})"
                    ),
                    summary=(
                        f"{len(no_line)} electric constraint interchange "
                        "term(s) reference a subsystem pair with no matching "
                        "novomodelo line; the term is dropped."
                    ),
                    table=DiagnosticTable(
                        columns=["Constraint", "From", "To"],
                        rows=[
                            [s.constraint_code, s.sys_from, s.sys_to] for s in no_line
                        ],
                        justify=["right", "right", "right"],
                    ),
                    remediation="Declare the pair as an interchange line.",
                ),
                logger=_LOG,
            )

        malformed = by_kind.get("malformed", [])
        if malformed:
            emit(
                Diagnostic(
                    code="electric-constraint-malformed-term",
                    severity=Severity.WARNING,
                    category=_CATEGORY,
                    title=(
                        f"Electric constraint has a malformed term ({len(malformed)})"
                    ),
                    summary=(
                        f"{len(malformed)} electric constraint interchange "
                        "term(s) declared fewer than 2 arguments; the term is "
                        "dropped."
                    ),
                    table=DiagnosticTable(
                        columns=["Constraint", "Term"],
                        rows=[[s.constraint_code, s.raw] for s in malformed],
                        justify=["right", "left"],
                    ),
                ),
                logger=_LOG,
            )

    if elec_constraint_skips:
        emit(
            Diagnostic(
                code="electric-constraint-skipped",
                severity=Severity.WARNING,
                category=_CATEGORY,
                title=f"Electric constraint skipped ({len(elec_constraint_skips)})",
                summary=(
                    f"{len(elec_constraint_skips)} electric constraint(s) "
                    "could not be converted; skipping."
                ),
                table=DiagnosticTable(
                    columns=["Constraint", "Reason"],
                    rows=[[s.constraint_code, s.reason] for s in elec_constraint_skips],
                    justify=["right", "left"],
                ),
            ),
            logger=_LOG,
        )

    result = builder.result()
    if result is None:
        return None

    _LOG.info(
        "Generated %d electric constraints with %d bounds "
        "(individualizado cutoff at stage %d).",
        len(result.constraints),
        result.bounds.num_rows,
        cutoff,
    )

    return result


# ---------------------------------------------------------------------------
# AGRINT.DAT — Exchange group constraints
# ---------------------------------------------------------------------------

_AGRINT_DEFAULT_PENALTY = 1000.0


def _parse_agrint(
    path: Path,
    num_patamares: int = 3,
) -> tuple[
    dict[int, list[tuple[int, int, float]]],  # group_id -> [(A, B, coeff)]
    list[tuple[int, int, int, int | None, int | None, list[float]]],
    # (group_id, mi, anoi, mf|None, anof|None, [lim per patamar])
]:
    """Parse AGRINT.DAT into group definitions and limit rows.

    Returns
    -------
    groups : dict[int, list[tuple[int, int, float]]]
        Mapping from group ID to a list of (A, B, coefficient) tuples.
    limits : list of 6-tuples
        Each element is
        (group_id, start_month, start_year, end_month|None, end_year|None,
         [lim_p1, lim_p2, lim_p3]).
    """
    groups: dict[int, list[tuple[int, int, float]]] = {}
    limits: list[tuple[int, int, int, int | None, int | None, list[float]]] = []

    in_groups_section = False
    in_limits_section = False

    with path.open(encoding="latin-1") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\r\n")

            stripped = line.strip()
            if not stripped or stripped.startswith("&"):
                continue

            upper = stripped.upper()

            # Section headers
            if "AGRUPAMENTOS" in upper:
                if "LIMITES" not in upper:
                    in_groups_section = True
                    in_limits_section = False
                    continue

            if "LIMITES POR GRUPO" in upper:
                in_groups_section = False
                in_limits_section = True
                continue

            # Skip format/header comment lines
            if stripped.startswith("#") or stripped.startswith("X"):
                continue

            if in_groups_section:
                # Terminator
                if stripped.startswith("999"):
                    in_groups_section = False
                    continue
                # Data line: right-aligned fixed-width columns
                # Format: XXX XXX XXX XX.XXXX
                # columns: group(1-3), A(5-7), B(9-11), coeff(13-20)
                try:
                    parts = stripped.split()
                    if len(parts) < 4:
                        continue
                    group_id = int(parts[0])
                    a = int(parts[1])
                    b = int(parts[2])
                    coeff = float(parts[3])
                    groups.setdefault(group_id, []).append((a, b, coeff))
                except (ValueError, IndexError):
                    continue

            elif in_limits_section:
                if stripped.startswith("999"):
                    break
                # Format: #AG MI ANOI MF ANOF LIM_P1 LIM_P2 LIM_P3 [description]
                # MF/ANOF are optional (open-ended if blank)
                try:
                    parts = stripped.split()
                    if len(parts) < 4:
                        continue
                    group_id = int(parts[0])
                    mi = int(parts[1])
                    anoi = int(parts[2])

                    # Detect whether MF/ANOF are present by trying to parse
                    # parts[3] and parts[4] as int before float limits.
                    # Limits always end with '.' in the file, so they parse as float.
                    # MF/ANOF are small integers (1-12 / 4-digit year).
                    mf: int | None = None
                    anof: int | None = None
                    lim_start = 3

                    # parts[3]: could be MF (int 1-12) or first limit (float w/ '.')
                    if "." not in parts[3] and len(parts) > 5:
                        # Could still be a year — check parts[4] too
                        try:
                            candidate_mf = int(parts[3])
                            candidate_anof = int(parts[4])
                            # Sanity: MF in [1,12], ANOF is a 4-digit year
                            if (
                                1 <= candidate_mf <= 12
                                and 1900 <= candidate_anof <= 9999
                            ):
                                mf = candidate_mf
                                anof = candidate_anof
                                lim_start = 5
                        except (ValueError, IndexError):
                            pass

                    lim_parts = parts[lim_start : lim_start + num_patamares]
                    if len(lim_parts) < 1:
                        continue
                    lims = [float(lp.rstrip(".")) for lp in lim_parts]
                    while len(lims) < num_patamares:
                        lims.append(lims[-1])
                    limits.append((group_id, mi, anoi, mf, anof, lims))
                except (ValueError, IndexError):
                    continue

    return groups, limits


def convert_agrint_constraints(
    case: NewaveCase,
    id_map: NewaveIdMap,
    allocator: ConstraintIdAllocator | None = None,
) -> GenericConstraintResult | None:
    """Convert AGRINT.DAT exchange group constraints to Novomodelo generic constraints.

    Each group in AGRINT.DAT defines a weighted sum of directional interchange
    flows.  Each group becomes one ``<=`` generic constraint with a high-default
    slack penalty.  Bounds are stored per (constraint_id, stage_id, block_id).

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Entity ID mapping (used indirectly via ``_build_line_id_map``).
    allocator:
        Shared id source; defaults to a fresh :class:`ConstraintIdAllocator`
        (ids from 0) when omitted. The pipeline threads one instance across
        every emitter so IDs do not collide with other constraints.

    Returns
    -------
    tuple[list[dict], pa.Table] | None
        A ``(constraints_list, bounds_table)`` pair, or ``None`` if
        ``agrint.dat`` is absent or contains no valid constraints.
        ``bounds_table`` has schema
        ``(constraint_id: INT32, stage_id: INT32, block_id: INT32,
        bound_lower: DOUBLE, bound_upper: DOUBLE)``. Every AGRINT constraint
        is ``<=``, so ``bound_lower`` is always null.
    """
    if case.files.agrint is None:
        _LOG.debug("agrint.dat not found; skipping AGRINT constraints.")
        return None
    if not case.switches.agrint.on:
        emit(switch_off_diagnostic(case.switches.agrint), logger=_LOG)
        return None

    patamar_ag = case.patamar
    num_patamares_ag: int = patamar_ag.numero_patamares or 1

    groups, limits = _parse_agrint(case.files.agrint, num_patamares_ag)
    if not groups or not limits:
        return None

    _horizon = case.horizon
    start_month = _horizon.start_month
    start_year = _horizon.start_year
    study_months = _horizon.study_months
    num_stages = _horizon.total_stages

    line_id_map = _build_line_id_map(case)

    allocator = allocator or ConstraintIdAllocator()
    builder = GenericConstraintBuilder(allocator)

    # Loop-accumulate-then-emit-once: one record per dropped term / empty
    # group, emitted once after the loop (see converters/thermal.py's
    # _GtminRecord idiom).
    agrint_term_skips: list[_AgrintTermSkip] = []
    agrint_group_skips: list[_AgrintGroupSkip] = []

    for group_id in sorted(groups.keys()):
        terms_raw = groups[group_id]

        # The source model's ``Interc(A→B)`` is the *non-negative directional* flow from
        # A to B (zero whenever physical flow goes B→A).  Novomodelo's ``line_direct(id)`` /
        # ``line_reverse(id)`` are the matching non-negative LP variables:
        # ``line_direct`` is the flow in the canonical (src<tgt) direction,
        # ``line_reverse`` the flow in the opposite direction.
        #
        # The naive substitution ``-line_exchange`` (signed net flow,
        # negated when reversed) would let the LHS go negative when one
        # term is flowing in its non-canonical direction — diluting the
        # bound at fictitious hubs and producing incorrect dispatch.
        parsed_terms: list[tuple[float, str]] = []
        for a, b, coeff in terms_raw:
            src, tgt = (a, b) if a < b else (b, a)
            line_id = line_id_map.get((src, tgt))
            if line_id is None:
                agrint_term_skips.append(
                    _AgrintTermSkip(group_id=group_id, sys_a=a, sys_b=b)
                )
                continue
            var = f"line_direct({line_id})" if a < b else f"line_reverse({line_id})"
            parsed_terms.append((coeff, var))

        if not parsed_terms:
            agrint_group_skips.append(_AgrintGroupSkip(group_id=group_id))
            continue

        expression = _render_signed_expression(parsed_terms)

        # Collect study-period limit rows for this group, keyed by (stage, block).
        # AGRINT has no seasonalize flag: The source model freezes its post-study limits
        # at the last study stage value and ignores any post-study-dated agrint.dat
        # entries (planned future expansion). The pmo.dat "LIMITES DOS AGRUPAMENTOS DE
        # INTERCAMBIO (MWmedio)" block confirms this — its POS row is flat at the last
        # study (December) value. So clamp to the study period here, then freeze the
        # tail.
        group_bounds: dict[tuple[int, int], float] = {}
        for grp, mi, anoi, mf, anof, lims in limits:
            if grp != group_id:
                continue

            # Determine effective end (open-ended => last study stage)
            if mf is None or anof is None:
                # Open-ended: run to the end of the study horizon
                end_y = start_year + (num_stages - 1 + start_month - 1) // 12
                end_m = (start_month - 1 + num_stages - 1) % 12 + 1
            else:
                end_y, end_m = anof, mf

            # Iterate month by month within [mi/anoi .. end_m/end_y], keeping
            # only the study period (post-study is handled by the freeze below).
            y, m = anoi, mi
            while (y, m) <= (end_y, end_m):
                stage_id = (y - start_year) * 12 + (m - start_month)
                if 0 <= stage_id < study_months:
                    for block_idx, lim_val in enumerate(lims):
                        group_bounds[(stage_id, block_idx)] = lim_val
                m += 1
                if m > 12:
                    m = 1
                    y += 1

        # Emit study-period slots.
        slots: list[tuple[int, int | None, float | None, float | None]] = [
            (stage_id, block_idx, None, lim_val)
            for (stage_id, block_idx), lim_val in group_bounds.items()
        ]

        # Freeze the last study stage value through the post-study tail.
        last_study_stage = study_months - 1
        if num_stages > study_months:
            for block_idx in sorted({b for (_, b) in group_bounds}):
                freeze_val = group_bounds.get((last_study_stage, block_idx))
                if freeze_val is None:
                    continue
                for stage_id in range(study_months, num_stages):
                    slots.append((stage_id, block_idx, None, freeze_val))

        builder.add(
            name=f"AGRINT_{group_id}",
            description=f"Exchange group constraint {group_id}",
            expression=expression,
            slack={"enabled": False},
            slots=slots,
        )

    if agrint_term_skips:
        emit(
            Diagnostic(
                code="agrint-interchange-no-line",
                severity=Severity.WARNING,
                category=_CATEGORY,
                title=(
                    "AGRINT interchange has no matching line "
                    f"({len(agrint_term_skips)})"
                ),
                summary=(
                    f"{len(agrint_term_skips)} AGRINT group term(s) reference "
                    "a subsystem pair with no matching novomodelo line; the term "
                    "is dropped."
                ),
                table=DiagnosticTable(
                    columns=["Group", "From", "To"],
                    rows=[[s.group_id, s.sys_a, s.sys_b] for s in agrint_term_skips],
                    justify=["right", "right", "right"],
                ),
                remediation="Declare the subsystem pair as an interchange line.",
            ),
            logger=_LOG,
        )
    if agrint_group_skips:
        emit(
            Diagnostic(
                code="agrint-group-empty",
                severity=Severity.WARNING,
                category=_CATEGORY,
                title=f"AGRINT group has no valid terms ({len(agrint_group_skips)})",
                summary=(
                    f"{len(agrint_group_skips)} AGRINT group(s) had no valid "
                    "terms after line lookup; skipping."
                ),
                table=DiagnosticTable(
                    columns=["Group"],
                    rows=[[s.group_id] for s in agrint_group_skips],
                    justify=["right"],
                ),
            ),
            logger=_LOG,
        )

    result = builder.result()
    if result is None:
        return None

    _LOG.info(
        "Generated %d AGRINT group constraints with %d bounds.",
        len(result.constraints),
        result.bounds.num_rows,
    )

    return result

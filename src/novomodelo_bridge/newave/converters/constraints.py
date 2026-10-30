"""Generic constraints converter: VminOP (minimum stored energy) from curva.dat
and electric constraints from restricao-eletrica.csv.

Converts the source model minimum stored energy constraints into Novomodelo generic
constraints. Each REE with entries in ``curva.dat`` becomes one generic constraint,
bounded on every curve stage, whose expression bounds the REE's useful stored energy
(MWh) — the ``hydro_useful_volume_final`` of its reservoirs weighted by novomodelo's
integrated accumulated productivity in MWh/hm³ — from below by the curve fraction of
its maximum useful stored energy (:mod:`novomodelo_bridge.core.stored_energy`).

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
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import pandas as pd
import pyarrow as pa

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from novomodelo_bridge.core.errors import BridgeError
from novomodelo_bridge.core.generic_constraint_builder import (
    ConstraintIdAllocator,
    GenericConstraintBuilder,
    GenericConstraintResult,
    Slot,
    hard_slack,
    soft_slack,
)
from novomodelo_bridge.core.pandas_utils import as_int
from novomodelo_bridge.core.productivity import (
    compute_productivity,
    stored_energy_productivity,
)
from novomodelo_bridge.core.source_text import read_source_text
from novomodelo_bridge.core.stored_energy import (
    StoredEnergyShare,
    cascade_share,
    stored_energy_floor,
)
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides
from novomodelo_bridge.newave.horizon import study_start
from novomodelo_bridge.newave.id_map import NewaveIdMap
from novomodelo_bridge.newave.plants import active_hydros
from novomodelo_bridge.newave.switches import switch_off_diagnostic
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Set as AbstractSet

    from novomodelo_bridge.newave.horizon import StudyHorizon

_LOG = logging.getLogger(__name__)

_CATEGORY = "Special constraints"

_VMINOP_DEFAULT_PENALTY = 1000.0


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


class VminopResult(NamedTuple):
    """Result of :func:`convert_vminop_constraints`.

    ``constraints_dict`` is the full ``{"$schema", "constraints": [...]}``
    object; ``bounds`` is the full 5-column interval bounds table (``block_id`` always
    null — VminOP is stage-level, never per-block). ``per_stage_parameters``
    maps each ``pct_vminop_ree{code}`` name the expressions reference to its
    per-stage curve fraction (one value per study stage, ``0.0`` where the REE
    has no curve), for ``build_scalar_parameters``.
    """

    constraints_dict: dict
    bounds: pa.Table
    referenced_hydro_ids: list[int]
    per_stage_parameters: dict[str, list[float]]


class _VminopRees(NamedTuple):
    """Per-REE inputs of the VminOP constraints, keyed by REE code."""

    plants: dict[int, list[int]]
    shares: dict[int, list[StoredEnergyShare]]
    penalties: dict[int, float]
    names: dict[int, str]


def vminop_pct_name(ree_code: int) -> str:
    """Canonical ``@name`` for the per-stage security-curve fraction of one REE."""
    return f"pct_vminop_ree{ree_code}"


def _vminop_ree_shares(
    cadastro: pd.DataFrame,
    id_map: NewaveIdMap,
    acc_prod: dict[int, float],
    hydro_to_ree: dict[int, int],
) -> dict[int, list[StoredEnergyShare]]:
    """Every REE's VminOP shares, by REE code in plant-code order.

    Each reservoir counts its whole cascade in its own REE, as the source
    model's stored energy does, including the plants below a cut into another
    REE. ``hydros.json`` links the cut plant to the real plant below its
    fictitious twin, so novomodelo's accumulated productivity already spans that
    cascade. Never split it at the REE boundary the way DECOMP's RHE does for
    a reservoir with an ``EZ`` record
    (:func:`~novomodelo_bridge.decomp.converters.rhe_shares.build_rhe_shares`): that moves
    stored energy between REEs, so water held upstream in one REE meets
    another REE's curve.
    """
    shares: dict[int, list[StoredEnergyShare]] = defaultdict(list)
    for code, ree_code in sorted(hydro_to_ree.items()):
        # Gate on the source model's stored-energy plant set, not on
        # ``acc_prod > 0`` alone: run-of-river and special-regime plants such as
        # ITAIPU and JIRAU carry a positive accumulated productivity from
        # downstream powerhouses yet store no energy in the source model.
        if not _is_stored_energy_reservoir(cadastro, code):
            continue
        if acc_prod.get(code, 0.0) <= 0.0:
            continue
        try:
            hid = id_map.hydro_id(code)
        except KeyError:
            continue
        shares[ree_code].append(cascade_share(hid))
    return shares


def _build_hydro_to_ree(
    confhd_df: pd.DataFrame, *, fictitious: AbstractSet[int]
) -> dict[int, int]:
    """Return {plant_code: ree_code} for existing non-fictitious plants."""
    non_fict = active_hydros(confhd_df, fictitious=fictitious)
    return {int(r["codigo_usina"]): int(r["ree"]) for _, r in non_fict.iterrows()}


def compute_accumulated_integrated_productivities(
    cadastro: pd.DataFrame,
    confhd_df: pd.DataFrame,
    *,
    fictitious: AbstractSet[int],
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

    Parameters
    ----------
    cadastro:
        The ``Hidr.cadastro`` DataFrame (with MODIF.DAT overrides applied).
    confhd_df:
        The ``Confhd.usinas`` DataFrame.
    fictitious:
        The case's fictitious plant codes (``NewaveCase.fictitious_codes``).
    """
    from novomodelo_bridge.newave.converters.fict_cascade import resolve_cascade

    resolutions = resolve_cascade(confhd_df, cadastro, fictitious=fictitious)
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
    *,
    fictitious: AbstractSet[int],
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
    fictitious:
        The case's fictitious plant codes (``NewaveCase.fictitious_codes``).

    Returns
    -------
    dict[int, float]
        Mapping from plant code to accumulated productivity in MW/(m³/s).
    """
    from novomodelo_bridge.newave.converters.fict_cascade import resolve_cascade

    resolutions = resolve_cascade(confhd_df, cadastro, fictitious=fictitious)
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

    Plants with no downstream (sea sinks) keep their own value. Shared by the
    accumulated-productivity helpers of this module.

    Raises
    ------
    BridgeError
        If a cycle is detected in the downstream map.
    """
    result: dict[int, float] = {}

    def _accumulate(code: int, path: tuple[int, ...]) -> float:
        if code in path:
            cycle = path[path.index(code) :] + (code,)
            cycle_path = " → ".join(str(c) for c in cycle)
            raise BridgeError(
                f"Cascade cycle detected: {cycle_path}. This usually indicates "
                "a fictitious plant was removed that breaks the cascade topology."
            )
        if code in result:
            return result[code]
        ds = downstream_map.get(code)
        ds_acc = _accumulate(ds, (*path, code)) if ds is not None else 0.0
        result[code] = own_value.get(code, 0.0) + ds_acc
        return result[code]

    for code in downstream_map:
        _accumulate(code, ())
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
        confhd_df = case.hydro_plants
        fictitious = case.fictitious_codes
    except (OSError, ValueError, AttributeError, TypeError, KeyError):
        return None

    resolutions = resolve_cascade(confhd_df, cadastro, fictitious=fictitious)
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
    if configuracoes_penalizacao is None:
        return False
    try:
        return int(configuracoes_penalizacao[2]) == 1
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

    Each REE yields one constraint, bounded on every stage with a curve value::

        Σ @rho_acum_mwh_h{id} * hydro_useful_volume_final(id)
            >= @pct_vminop_ree{code} * (Σ @earm_max_mwh_h{id})

    i.e. useful stored energy (MWh) at least the curve fraction of the maximum
    useful stored energy. The coefficients are novomodelo's *integrated*
    accumulated productivity in MWh/hm³ — the source model's stored-energy
    (EARM) convention, not the gen = ρ·Q point ρ_acum. novomodelo moves the
    dead-volume term of ``hydro_useful_volume_final`` onto the bound, so the
    row matches the source model's ``pct·useful + dead`` bound on the absolute
    storage. The slack is ``per_unit``: the curve penalty is a price per MWh
    of violation on every stage, whatever its hours.

    Only monthly-regulating reservoirs with usable storage participate in a REE's
    expression (see :func:`_is_stored_energy_reservoir`); run-of-river and
    special-regime plants are excluded to match the source model's EARM plant set.
    Each reservoir counts its whole cascade in its own REE (see
    :func:`_vminop_ree_shares`).

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
    VminopResult | None
        The constraints, their bounds, the referenced hydro ids and the
        per-stage ``pct_vminop_ree{code}`` parameters, or ``None`` when no
        VminOP constraint is emitted.
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

    rees = _read_vminop_rees(case, id_map, curva.custos_penalidades)

    allocator = allocator or ConstraintIdAllocator()
    builder = GenericConstraintBuilder(allocator)
    per_stage_parameters, all_referenced_ids = _add_vminop_constraints(
        builder,
        curva_df,
        rees,
        case.horizon,
        seasonalize=seasonalize_curve_post_study,
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

    return VminopResult(
        constraints_dict,
        result.bounds,
        sorted(all_referenced_ids),
        per_stage_parameters,
    )


def _read_vminop_rees(
    case: NewaveCase, id_map: NewaveIdMap, penalty_df: pd.DataFrame | None
) -> _VminopRees:
    """Each REE's plants, VminOP shares, curva.dat penalty and ree.dat name."""
    hidr = case.hidr
    ree_file = case.ree

    cadastro = _apply_permanent_overrides(hidr.cadastro, case)
    confhd_df = case.hydro_plants

    # Plants with no positive integrated cascade productivity store no energy.
    fictitious = case.fictitious_codes
    acc_prod = compute_accumulated_integrated_productivities(
        cadastro, confhd_df, fictitious=fictitious
    )

    hydro_to_ree = _build_hydro_to_ree(confhd_df, fictitious=fictitious)
    shares = _vminop_ree_shares(cadastro, id_map, acc_prod, hydro_to_ree)

    plants: dict[int, list[int]] = defaultdict(list)
    for code, ree_code in hydro_to_ree.items():
        plants[ree_code].append(code)

    penalties: dict[int, float] = {}
    if penalty_df is not None and not penalty_df.empty:
        for _, row in penalty_df.iterrows():
            penalties[int(row["codigo_ree"])] = float(row["penalidade"])

    names: dict[int, str] = {}
    ree_df = ree_file.rees
    if ree_df is not None:
        for _, row in ree_df.iterrows():
            names[int(row["codigo"])] = str(row["nome"]).strip()

    return _VminopRees(plants, shares, penalties, names)


def _add_vminop_constraints(
    builder: GenericConstraintBuilder,
    curva_df: pd.DataFrame,
    rees: _VminopRees,
    horizon: StudyHorizon,
    *,
    seasonalize: bool,
) -> tuple[dict[str, list[float]], set[int]]:
    """Add one VminOP constraint per curva.dat REE that can carry one.

    Returns the ``pct_vminop_ree{code}`` per-stage parameters and the hydro ids
    the added expressions reference.
    """
    per_stage_parameters: dict[str, list[float]] = {}
    all_referenced_ids: set[int] = set()

    # Loop-accumulate-then-emit-once: one record per skipped/clamped REE,
    # emitted after the loop so the per-REE detail survives finalize_diagnostics'
    # (code, summary) de-dup (see converters/thermal.py's _GtminRecord idiom).
    vminop_skips: list[_VminopSkip] = []
    penalty_clamps: list[_VminopPenaltyClamp] = []

    for ree_code in sorted(curva_df["codigo_ree"].unique()):
        ree_code = int(ree_code)
        hydros_in_ree = rees.plants.get(ree_code, [])

        if not hydros_in_ree:
            vminop_skips.append(
                _VminopSkip(
                    ree_code=ree_code,
                    ree_name=rees.names.get(ree_code, str(ree_code)),
                    reason="no hydro plants",
                )
            )
            continue

        shares = rees.shares.get(ree_code, [])
        if not shares:
            vminop_skips.append(
                _VminopSkip(
                    ree_code=ree_code,
                    ree_name=rees.names.get(ree_code, "?"),
                    reason="no valid terms for VminOP expression",
                )
            )
            continue

        pct_name = vminop_pct_name(ree_code)
        expression = stored_energy_floor([(1.0, s) for s in shares], pct_name)
        # Every hydro an ``@…_h{id}`` parameter names must be declared.
        referenced_ids = [int(h) for h in re.findall(r"@\w+?_h(\d+)\b", expression)]
        ree_name = rees.names.get(ree_code, str(ree_code))
        penalty = rees.penalties.get(ree_code, _VMINOP_DEFAULT_PENALTY)
        if penalty <= 0.0:
            penalty_clamps.append(
                _VminopPenaltyClamp(
                    ree_code=ree_code,
                    ree_name=ree_name,
                    declared=penalty,
                    applied=_VMINOP_DEFAULT_PENALTY,
                )
            )
            penalty = _VMINOP_DEFAULT_PENALTY

        ree_curva = curva_df[curva_df["codigo_ree"] == ree_code].sort_values("data")
        pct_by_stage = _vminop_pct_by_stage(ree_curva, horizon, seasonalize=seasonalize)
        if not pct_by_stage:
            continue

        builder.add(
            name=f"VminOP_{ree_name}",
            description=f"Minimum stored energy for REE {ree_code} ({ree_name})",
            expression=expression,
            slack=soft_slack(penalty, "per_unit"),
            slots=[(stage_id, None, 0.0, None) for stage_id in sorted(pct_by_stage)],
        )
        per_stage_parameters[pct_name] = [
            pct_by_stage.get(stage_id, 0.0) / 100.0
            for stage_id in range(horizon.total_stages)
        ]
        all_referenced_ids.update(referenced_ids)

    _emit_vminop_skips(vminop_skips)
    _emit_vminop_penalty_clamps(penalty_clamps)
    return per_stage_parameters, all_referenced_ids


def _vminop_pct_by_stage(
    ree_curva: pd.DataFrame, horizon: StudyHorizon, *, seasonalize: bool
) -> dict[int, float]:
    """One REE's security-curve percentage on each stage it binds, by stage id."""
    # Curve percentage per stage. curva.dat only covers the study period.
    # Extend into the post-study ("Período Estático Final") per source-model's
    # rule (manual table p.32-33): seasonalize (repeat the last study year's
    # monthly curve) when curva.dat's third penalization field is set,
    # otherwise freeze December's value across the whole tail.
    start_year = horizon.start_year
    start_month = horizon.start_month
    num_stages = horizon.total_stages
    study_months = horizon.study_months

    seasonal_pct: dict[int, float] = {}  # calendar_month -> last percentage
    pct_by_stage: dict[int, float] = {}
    for _, crow in ree_curva.iterrows():
        dt: datetime = crow["data"]
        stage_id = (dt.year - start_year) * 12 + (dt.month - start_month)
        if stage_id < 0 or stage_id >= num_stages:
            continue
        percentage = float(crow["valor"])
        seasonal_pct[dt.month] = percentage  # last value wins
        pct_by_stage[stage_id] = percentage

    # In-study gaps (and the post-study tail when seasonalized) use the
    # seasonal pattern; otherwise the tail freezes at the last in-study
    # stage's percentage (December of the last study year, forward-filled if
    # that exact stage had no explicit row).
    if seasonal_pct:
        in_study = [s for s in pct_by_stage if s < study_months]
        freeze_pct = pct_by_stage[max(in_study)] if in_study else None
        for stage_id in range(num_stages):
            if stage_id in pct_by_stage:
                continue
            if stage_id >= study_months and not seasonalize:
                pct = freeze_pct
            else:
                cal_month = ((start_month - 1 + stage_id) % 12) + 1
                pct = seasonal_pct.get(cal_month)
            if pct is not None:
                pct_by_stage[stage_id] = pct

    return pct_by_stage


def _emit_vminop_skips(skips: list[_VminopSkip]) -> None:
    """Emit the ``vminop-ree-not-expressible`` WARNING, one row per skipped REE."""
    if not skips:
        return
    emit(
        Diagnostic(
            code="vminop-ree-not-expressible",
            severity=Severity.WARNING,
            category=_CATEGORY,
            title=f"VminOP constraint not expressible ({len(skips)} REE(s))",
            summary=(
                f"{len(skips)} REE(s) with a curva.dat VminOP entry "
                "could not be expressed as a generic constraint; skipping."
            ),
            table=DiagnosticTable(
                columns=["REE", "Name", "Reason"],
                rows=[[s.ree_code, s.ree_name, s.reason] for s in skips],
                justify=["right", "left", "left"],
            ),
        ),
        logger=_LOG,
    )


def _emit_vminop_penalty_clamps(clamps: list[_VminopPenaltyClamp]) -> None:
    """Emit the ``vminop-penalty-nonpositive`` WARNING, one row per clamped REE."""
    if not clamps:
        return
    emit(
        Diagnostic(
            code="vminop-penalty-nonpositive",
            severity=Severity.WARNING,
            category=_CATEGORY,
            title=f"VminOP penalty not positive ({len(clamps)} REE(s))",
            summary=(
                f"{len(clamps)} REE(s) declare a curva.dat VminOP "
                f"penalty that is not positive; using {_VMINOP_DEFAULT_PENALTY} "
                "instead."
            ),
            table=DiagnosticTable(
                columns=["REE", "Name", "Declared", "Applied"],
                rows=[[c.ree_code, c.ree_name, c.declared, c.applied] for c in clamps],
                justify=["right", "left", "right", "right"],
            ),
            remediation="Check curva.dat's penalty column for these REEs.",
        ),
        logger=_LOG,
    )


# ---------------------------------------------------------------------------
# Electric constraints from restricao-eletrica.csv
# ---------------------------------------------------------------------------

_UNBOUNDED_THRESHOLD = -1.0e29  # values below this are treated as -inf


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

    for raw_line in read_source_text(path).splitlines():
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
    limites_df = case.interchange_limits

    if limites_df is None or limites_df.empty:
        return {}

    study_start_dt = datetime(*study_start(case.dger), 1)
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
            # ``_agrint_expression`` for the same rule applied to AGRINT terms.
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


def _stage_ids_between(
    first: tuple[int, int],
    last: tuple[int, int],
    *,
    start_year: int,
    start_month: int,
) -> Iterator[int]:
    """0-based stage id of every ``(year, month)`` from *first* through *last*.

    Months outside the horizon yield out-of-range ids; callers keep the stages
    they cover.
    """
    year, month = first
    while (year, month) <= last:
        yield (year - start_year) * 12 + (month - start_month)
        month += 1
        if month > 12:
            month = 1
            year += 1


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
    except Exception:
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

    try:
        re_file = case.re_dat
    except Exception:
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

    if re_file is None:
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

    for group_key, grp in rest_df.groupby("conjunto"):
        code = as_int(group_key)
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
            for sid in _stage_ids_between(
                (ai, mi), (af, mf), start_year=start_year, start_month=start_month
            ):
                if 0 <= sid < num_stages:
                    changepoints.append((sid, pat, value))

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
    re_path, has_re_dat = _enabled_electric_sources(case)
    if re_path is None and not has_re_dat:
        _LOG.debug("No electric constraints found; skipping.")
        return None

    sources, grid = _read_electric_sources(case, re_path, has_re_dat=has_re_dat)
    if not sources.expressions and not sources.re_conjuntos:
        _LOG.debug("No electric constraints found; skipping.")
        return None

    line_id_map = _build_line_id_map(case)

    eletri_penalty = _resolve_eletri_penalty(case)
    slack_config = (
        soft_slack(eletri_penalty, "per_hour")
        if eletri_penalty is not None and eletri_penalty > 0.0
        else hard_slack()
    )

    allocator = allocator or ConstraintIdAllocator()
    builder = GenericConstraintBuilder(allocator)

    # Loop-accumulate-then-emit-once: one record per dropped term / skipped
    # constraint, grouped by kind and emitted once after the loop (see
    # converters/thermal.py's _GtminRecord idiom).
    elec_term_skips: list[_ElectricTermSkip] = []
    elec_constraint_skips: list[_ElectricConstraintSkip] = []

    for cod in sorted(set(sources.expressions) | set(sources.re_conjuntos)):
        novomodelo_expr = _electric_expression(
            cod, sources, id_map, line_id_map, elec_term_skips
        )
        if novomodelo_expr is None:
            elec_constraint_skips.append(
                _ElectricConstraintSkip(
                    constraint_code=cod, reason="no valid terms in expression"
                )
            )
            continue

        bounds = _electric_bounds(cod, sources, grid.horizon)
        if not bounds.has_sup and not bounds.has_inf:
            elec_constraint_skips.append(
                _ElectricConstraintSkip(constraint_code=cod, reason="no bound data")
            )
            continue

        _add_electric_constraint(
            builder, cod, novomodelo_expr, slack_config, bounds, grid
        )

    _emit_electric_term_skips(elec_term_skips)
    _emit_electric_constraint_skips(elec_constraint_skips)

    result = builder.result()
    if result is None:
        return None

    _LOG.info(
        "Generated %d electric constraints with %d bounds "
        "(individualizado cutoff at stage %d).",
        len(result.constraints),
        result.bounds.num_rows,
        grid.cutoff,
    )

    return result


@dataclass(frozen=True)
class _ElectricSources:
    """restricao-eletrica.csv's and RE.DAT's electric constraints, by constraint code."""

    expressions: dict[int, str]
    horizons: dict[int, tuple[str, str]]
    csv_bounds: dict[int, list[tuple[str, str, int, float, float]]]
    re_conjuntos: dict[int, list[int]]
    re_dat_bounds: dict[int, dict[tuple[int, int], float]]


class _ElectricGrid(NamedTuple):
    """The stage/block grid electric bounds are laid on."""

    horizon: StudyHorizon
    num_patamares: int
    cutoff: int
    """First post-individualizado stage (:func:`_get_individualizado_cutoff`)."""


class _ElectricBounds(NamedTuple):
    """One electric constraint's limits by ``(stage_id, block_id)``, per source."""

    csv_sup: dict[tuple[int, int], float]
    csv_inf: dict[tuple[int, int], float]
    re_sup: dict[tuple[int, int], float]

    @property
    def has_sup(self) -> bool:
        return bool(self.csv_sup) or bool(self.re_sup)

    @property
    def has_inf(self) -> bool:
        return bool(self.csv_inf)


def _enabled_electric_sources(case: NewaveCase) -> tuple[Path | None, bool]:
    """restricao-eletrica.csv's path and whether RE.DAT is read, per dger.dat."""
    switches = case.switches
    re_path = case.files.restricao_eletrica
    if re_path is not None and not switches.restricao_eletrica.on:
        emit(switch_off_diagnostic(switches.restricao_eletrica), logger=_LOG)
        re_path = None
    has_re_dat = case.files.re_dat is not None
    if has_re_dat and not switches.re_dat.on:
        emit(switch_off_diagnostic(switches.re_dat), logger=_LOG)
        has_re_dat = False
    return re_path, has_re_dat


def _read_electric_sources(
    case: NewaveCase, re_path: Path | None, *, has_re_dat: bool
) -> tuple[_ElectricSources, _ElectricGrid]:
    """Parse restricao-eletrica.csv and RE.DAT, and the grid they are laid on."""
    expressions: dict[int, str] = {}
    horizons: dict[int, tuple[str, str]] = {}
    bounds_rows: list[tuple[int, str, str, int, float, float]] = []

    if re_path is not None:
        expressions, horizons, bounds_rows = _parse_restricao_eletrica(re_path)

    horizon = case.horizon
    patamar = case.patamar
    num_patamares: int = patamar.numero_patamares or 1
    cutoff = _get_individualizado_cutoff(case, horizon.start_year, horizon.start_month)

    re_conjuntos: dict[int, list[int]] = {}
    re_dat_bounds: dict[int, dict[tuple[int, int], float]] = {}
    if has_re_dat:
        re_conjuntos, re_dat_bounds = _parse_re_dat(
            case,
            horizon.start_year,
            horizon.start_month,
            horizon.total_stages,
            num_patamares,
        )

    csv_bounds: dict[int, list[tuple[str, str, int, float, float]]] = defaultdict(list)
    for cod, per_ini, per_fin, pat, lim_inf, lim_sup in bounds_rows:
        csv_bounds[cod].append((per_ini, per_fin, pat, lim_inf, lim_sup))

    sources = _ElectricSources(
        expressions, horizons, csv_bounds, re_conjuntos, re_dat_bounds
    )
    return sources, _ElectricGrid(horizon, num_patamares, cutoff)


def _resolve_eletri_penalty(case: NewaveCase) -> float | None:
    """The electric slack penalty in R$/MWh, or ``None`` when none is readable."""
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
            df_pen = penalid.penalidades if penalid is not None else None
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
            pass  # no default either: None makes the RE_* constraints hard

    return eletri_penalty


def _electric_expression(
    cod: int,
    sources: _ElectricSources,
    id_map: NewaveIdMap,
    line_id_map: dict[tuple[int, int], int],
    term_skips: list[_ElectricTermSkip],
) -> str | None:
    """Constraint ``cod``'s expression, from its restricao-eletrica.csv formula
    when it has one, else the sum of its RE.DAT plants' ``hydro_generation``.

    Every dropped term is recorded on *term_skips*; ``None`` when none is left.
    """
    if cod in sources.expressions:
        return _parse_formula(
            sources.expressions[cod],
            id_map,
            line_id_map,
            constraint_code=cod,
            skipped=term_skips,
        )

    terms: list[str] = []
    for plant_code in sorted(sources.re_conjuntos[cod]):
        try:
            novomodelo_id = id_map.hydro_id(plant_code)
        except KeyError:
            term_skips.append(
                _ElectricTermSkip(
                    constraint_code=cod,
                    kind="hydro-unmapped",
                    source="RE.DAT",
                    hydro_code=plant_code,
                )
            )
            continue
        terms.append(f"hydro_generation({novomodelo_id})")
    return " + ".join(terms) if terms else None


def _electric_bounds(
    cod: int, sources: _ElectricSources, horizon: StudyHorizon
) -> _ElectricBounds:
    """Constraint ``cod``'s restricao-eletrica.csv limits and RE.DAT ceilings.

    A csv limit is clipped to the constraint's ``RE-HORIZ-PER`` window, or to the
    study years when it declares none.
    """
    csv_sup: dict[tuple[int, int], float] = {}
    csv_inf: dict[tuple[int, int], float] = {}
    start_year = horizon.start_year
    start_month = horizon.start_month

    window = sources.horizons.get(cod)
    horiz_ini = _parse_yyyymm(window[0]) if window else (start_year, start_month)
    horiz_fin = (
        _parse_yyyymm(window[1]) if window else (start_year + horizon.num_anos - 1, 12)
    )

    for per_ini, per_fin, pat, lim_inf, lim_sup in sources.csv_bounds.get(cod, []):
        ini_year, ini_month = _parse_yyyymm(per_ini)
        fin_year, fin_month = _parse_yyyymm(per_fin)

        ini_year, ini_month = max(
            (ini_year, ini_month), horiz_ini, key=lambda t: t[0] * 12 + t[1]
        )
        fin_year, fin_month = min(
            (fin_year, fin_month), horiz_fin, key=lambda t: t[0] * 12 + t[1]
        )

        block_id = pat - 1  # 1-based -> 0-based

        for sid in _stage_ids_between(
            (ini_year, ini_month),
            (fin_year, fin_month),
            start_year=start_year,
            start_month=start_month,
        ):
            if 0 <= sid < horizon.total_stages:
                if lim_sup > _UNBOUNDED_THRESHOLD:
                    csv_sup[(sid, block_id)] = lim_sup
                if lim_inf > _UNBOUNDED_THRESHOLD:
                    csv_inf[(sid, block_id)] = lim_inf

    return _ElectricBounds(csv_sup, csv_inf, sources.re_dat_bounds.get(cod, {}))


def _add_electric_constraint(
    builder: GenericConstraintBuilder,
    cod: int,
    expression: str,
    slack: dict[str, object],
    bounds: _ElectricBounds,
    grid: _ElectricGrid,
) -> None:
    """Add ``RE_{cod}``'s ceiling and floor and one row per declared (stage, block).

    The ceiling is RE.DAT's from the individualizado cutoff on and the csv's
    elsewhere; the floor is the csv's.
    """
    sup_id: int | None = None
    inf_id: int | None = None

    if bounds.has_sup:
        sup_id = builder.add_constraint(
            name=f"RE_{cod}",
            description=f"Electric constraint {cod}",
            expression=expression,
            slack=slack,
        )
    if bounds.has_inf:
        inf_id = builder.add_constraint(
            name=f"RE_{cod}",
            description=f"Electric constraint {cod}",
            expression=expression,
            slack=slack,
        )

    for stage_id in range(grid.horizon.total_stages):
        is_post_indiv = stage_id >= grid.cutoff

        for block_id in range(grid.num_patamares):
            # Upper bound.
            if sup_id is not None:
                sup_val: float | None = None
                if is_post_indiv and (stage_id, block_id) in bounds.re_sup:
                    sup_val = bounds.re_sup[(stage_id, block_id)]
                elif (stage_id, block_id) in bounds.csv_sup:
                    sup_val = bounds.csv_sup[(stage_id, block_id)]
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
                if (stage_id, block_id) in bounds.csv_inf:
                    inf_val = bounds.csv_inf[(stage_id, block_id)]
                if inf_val is not None:
                    builder.add_bound_row(
                        inf_id,
                        stage_id=stage_id,
                        block_id=block_id,
                        lower=inf_val,
                        upper=None,
                    )


def _emit_electric_term_skips(skips: list[_ElectricTermSkip]) -> None:
    """Emit one WARNING per kind of dropped electric-constraint term."""
    if not skips:
        return
    by_kind: dict[str, list[_ElectricTermSkip]] = defaultdict(list)
    for skip in skips:
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
                    rows=[[s.constraint_code, s.sys_from, s.sys_to] for s in no_line],
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
                title=f"Electric constraint has a malformed term ({len(malformed)})",
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


def _emit_electric_constraint_skips(skips: list[_ElectricConstraintSkip]) -> None:
    """Emit the ``electric-constraint-skipped`` WARNING, one row per constraint."""
    if not skips:
        return
    emit(
        Diagnostic(
            code="electric-constraint-skipped",
            severity=Severity.WARNING,
            category=_CATEGORY,
            title=f"Electric constraint skipped ({len(skips)})",
            summary=(
                f"{len(skips)} electric constraint(s) could not be converted; skipping."
            ),
            table=DiagnosticTable(
                columns=["Constraint", "Reason"],
                rows=[[s.constraint_code, s.reason] for s in skips],
                justify=["right", "left"],
            ),
        ),
        logger=_LOG,
    )


# ---------------------------------------------------------------------------
# AGRINT.DAT — Exchange group constraints
# ---------------------------------------------------------------------------


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
    flows.  Each group becomes one hard ``<=`` generic constraint (no slack).
    Bounds are stored per (constraint_id, stage_id, block_id).

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

    horizon = case.horizon
    line_id_map = _build_line_id_map(case)

    allocator = allocator or ConstraintIdAllocator()
    builder = GenericConstraintBuilder(allocator)

    # Loop-accumulate-then-emit-once: one record per dropped term / empty
    # group, emitted once after the loop (see converters/thermal.py's
    # _GtminRecord idiom).
    agrint_term_skips: list[_AgrintTermSkip] = []
    agrint_group_skips: list[_AgrintGroupSkip] = []

    for group_id in sorted(groups.keys()):
        expression = _agrint_expression(
            group_id, groups[group_id], line_id_map, agrint_term_skips
        )
        if expression is None:
            agrint_group_skips.append(_AgrintGroupSkip(group_id=group_id))
            continue

        builder.add(
            name=f"AGRINT_{group_id}",
            description=f"Exchange group constraint {group_id}",
            expression=expression,
            slack=hard_slack(),
            slots=_agrint_slots(group_id, limits, horizon),
        )

    _emit_agrint_term_skips(agrint_term_skips)
    _emit_agrint_group_skips(agrint_group_skips)

    result = builder.result()
    if result is None:
        return None

    _LOG.info(
        "Generated %d AGRINT group constraints with %d bounds.",
        len(result.constraints),
        result.bounds.num_rows,
    )

    return result


def _agrint_expression(
    group_id: int,
    terms_raw: list[tuple[int, int, float]],
    line_id_map: dict[tuple[int, int], int],
    term_skips: list[_AgrintTermSkip],
) -> str | None:
    """Group ``group_id``'s weighted sum of directional interchange flows.

    Every term whose subsystem pair has no line is recorded on *term_skips*;
    ``None`` when none is left.
    """
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
            term_skips.append(_AgrintTermSkip(group_id=group_id, sys_a=a, sys_b=b))
            continue
        var = f"line_direct({line_id})" if a < b else f"line_reverse({line_id})"
        parsed_terms.append((coeff, var))

    if not parsed_terms:
        return None

    return _render_signed_expression(parsed_terms)


def _agrint_slots(
    group_id: int,
    limits: list[tuple[int, int, int, int | None, int | None, list[float]]],
    horizon: StudyHorizon,
) -> list[Slot]:
    """Group ``group_id``'s ``<=`` limit per (stage, block), study and post-study."""
    start_year = horizon.start_year
    start_month = horizon.start_month
    study_months = horizon.study_months
    num_stages = horizon.total_stages

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

        if mf is None or anof is None:
            # Open-ended: run to the end of the whole horizon; the study-period
            # filter below drops the post-study stages.
            end_y = start_year + (num_stages - 1 + start_month - 1) // 12
            end_m = (start_month - 1 + num_stages - 1) % 12 + 1
        else:
            end_y, end_m = anof, mf

        # Iterate month by month within [mi/anoi .. end_m/end_y], keeping
        # only the study period (post-study is handled by the freeze below).
        for stage_id in _stage_ids_between(
            (anoi, mi), (end_y, end_m), start_year=start_year, start_month=start_month
        ):
            if 0 <= stage_id < study_months:
                for block_idx, lim_val in enumerate(lims):
                    group_bounds[(stage_id, block_idx)] = lim_val

    # Emit study-period slots.
    slots: list[Slot] = [
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

    return slots


def _emit_agrint_term_skips(skips: list[_AgrintTermSkip]) -> None:
    """Emit the ``agrint-interchange-no-line`` WARNING, one row per dropped term."""
    if not skips:
        return
    emit(
        Diagnostic(
            code="agrint-interchange-no-line",
            severity=Severity.WARNING,
            category=_CATEGORY,
            title=f"AGRINT interchange has no matching line ({len(skips)})",
            summary=(
                f"{len(skips)} AGRINT group term(s) reference "
                "a subsystem pair with no matching novomodelo line; the term "
                "is dropped."
            ),
            table=DiagnosticTable(
                columns=["Group", "From", "To"],
                rows=[[s.group_id, s.sys_a, s.sys_b] for s in skips],
                justify=["right", "right", "right"],
            ),
            remediation="Declare the subsystem pair as an interchange line.",
        ),
        logger=_LOG,
    )


def _emit_agrint_group_skips(skips: list[_AgrintGroupSkip]) -> None:
    """Emit the ``agrint-group-empty`` WARNING, one row per dropped group."""
    if not skips:
        return
    emit(
        Diagnostic(
            code="agrint-group-empty",
            severity=Severity.WARNING,
            category=_CATEGORY,
            title=f"AGRINT group has no valid terms ({len(skips)})",
            summary=(
                f"{len(skips)} AGRINT group(s) had no valid "
                "terms after line lookup; skipping."
            ),
            table=DiagnosticTable(
                columns=["Group"],
                rows=[[s.group_id] for s in skips],
                justify=["right"],
            ),
        ),
        logger=_LOG,
    )

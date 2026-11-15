"""Generic constraints converter: VminOP (minimum stored energy) from curva.dat
and electric constraints from restricao-eletrica.csv.

Converts the source model minimum stored energy constraints into Novomodelo generic
constraints. Each REE with entries in ``curva.dat`` becomes one generic constraint,
bounded on every curve stage, whose expression bounds the REE's useful stored energy
(MWh) — the ``hydro_useful_volume_final`` of its reservoirs weighted by novomodelo's
integrated accumulated productivity in MWh/hm³ — from below by the curve fraction of
its maximum useful stored energy (:mod:`novomodelo_bridge.core.stored_energy`).

Electric constraints from ``restricao-eletrica.csv`` (discovered via
``indices.csv``) and ``re.dat`` are converted into Novomodelo generic constraints
over ``hydro_generation``, ``thermal_generation`` and the line flows, with each
formula's demand and constant terms folded into its limits.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import pandas as pd
import pyarrow as pa

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
    exception_reason,
    unreadable_file_diagnostic,
)
from novomodelo_bridge.core.electrical_expression import (
    CMO_DEMAND_TOKENS,
    CONST_TOKEN,
    PEQ_TOKEN,
    SE_TOKEN,
    Bucket,
    DataContext,
    MalformedElectricalExpression,
    ParsedTerm,
    UnrecognizedElectricalToken,
    UnsupportedElectricalExpression,
    classify_terms,
    demand_in_cmo_diagnostic,
    evaluate_terms,
    parse_linear_expression,
    term_text,
)
from novomodelo_bridge.core.errors import (
    BridgeError,
    FieldParseError,
    SourceFileError,
    UnsupportedSourceError,
)
from novomodelo_bridge.core.generic_constraint_builder import (
    ConstraintIdAllocator,
    GenericConstraintBuilder,
    GenericConstraintResult,
    Slot,
    hard_slack,
    soft_slack,
)
from novomodelo_bridge.core.generic_constraint_format import render_expression
from novomodelo_bridge.core.libs_index import unread_lines_diagnostic
from novomodelo_bridge.core.outside_terms import OutsideTerm
from novomodelo_bridge.core.pandas_utils import as_float, as_int, as_timestamp, is_na
from novomodelo_bridge.core.productivity import (
    compute_productivity,
    stored_energy_productivity,
)
from novomodelo_bridge.core.stored_energy import (
    StoredEnergyShare,
    cascade_share,
    stored_energy_floor,
)
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides
from novomodelo_bridge.newave.converters.network import line_ids
from novomodelo_bridge.newave.id_map import NewaveIdMap
from novomodelo_bridge.newave.plants import active_hydros
from novomodelo_bridge.newave.switches import switch_off_diagnostic
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from inewave.libs.restricoes import Restricoes
    from inewave.newave import Agrint

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


_TermSkipKind = Literal["hydro-idle", "hydro-unmapped", "thermal-unmapped", "no-line"]

_TERM_SKIP_CAUSE: dict[_TermSkipKind, OutsideTerm] = {
    "hydro-idle": OutsideTerm.NOT_GENERATING,
    "hydro-unmapped": OutsideTerm.ABSENT,
    "thermal-unmapped": OutsideTerm.ABSENT,
    "no-line": OutsideTerm.NO_LINE,
}


@dataclass
class _ElectricTermSkip:
    """One electric-constraint term on an entity outside the case, treated
    by its cause (:data:`_TERM_SKIP_CAUSE`): a *hydro-idle* plant of
    confhd.dat the case leaves out or a *no-line* subsystem pair leaves out
    only the term; an *-unmapped* plant leaves out the whole constraint."""

    constraint_code: int
    kind: _TermSkipKind
    source: Literal["restricao-eletrica.csv", "re.dat"]
    plant_code: int | None = None
    sys_from: int | None = None
    sys_to: int | None = None

    @property
    def drops_constraint(self) -> bool:
        return _TERM_SKIP_CAUSE[self.kind].drops_constraint


@dataclass
class _ElectricConstraintSkip:
    """One electric constraint left out: no term the case can carry
    (*termless*), no limit (*unbounded*), a *term* the case cannot carry, for
    *reason* (*unsupported*), or a term outside the case whose cause leaves
    the constraint out (*outside*, reported by its :class:`_ElectricTermSkip`)."""

    constraint_code: int
    kind: Literal["termless", "unbounded", "unsupported", "outside"]
    term: str | None = None
    reason: str | None = None


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


#: curva.dat ``TIPO DE PENALIZACAO`` values, as the file's own header names them.
_PENALIZATION_TYPES = {0: "FIXA", 1: "MAXPEN"}


def _warn_if_non_fixa_penalization(configuracoes_penalizacao: list[Any] | None) -> bool:
    """Warn when curva.dat selects a *non-FIXA* security-curve penalization.

    The penalization-config line of curva.dat carries, as its first field, the ``TIPO DE
    PENALIZACAO``: ``0`` = **FIXA** (a fixed per-violation penalty at the curve cost),
    ``1`` = **MAXPEN**.

    Novomodelo-bridge models the **FIXA** convention: a per-stage VminOP (minimum stored
    energy) slack penalized at the fixed curve cost.  This is the faithful equivalent of
    the source model FIXA, so a FIXA deck needs no warning.  It does **not** reproduce
    any other penalization, so such a deck will show an expected VminOP
    violation-penalty difference.

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
                title="Security curve penalized as in NEWAVE (FIXA)",
                summary=(
                    "curva.dat sets TIPO DE PENALIZACAO = 0 (FIXA), which matches "
                    "the converted case: each stage's shortfall below the "
                    "security curve is penalized at the curve cost."
                ),
            ),
            logger=_LOG,
        )
        return False
    label = _PENALIZATION_TYPES.get(tipo, f"type {tipo}")
    emit(
        Diagnostic(
            code="vminop-penalization-not-fixa",
            severity=Severity.WARNING,
            category=_CATEGORY,
            title=f"Security curve penalized as FIXA, not {label}",
            summary=(
                f"curva.dat sets TIPO DE PENALIZACAO = {tipo} ({label}), but the "
                "converted case penalizes each stage's shortfall below the "
                "security curve at the curve cost, NEWAVE's FIXA rule. The "
                "security-curve penalties, and the storage kept above the "
                "curve, can differ from NEWAVE's."
            ),
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

    rees = _read_vminop_rees(case, id_map, curva.custos_penalidades)
    _require_declared_rees(curva_df, rees.names)

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


def _require_declared_rees(curva_df: pd.DataFrame, ree_names: dict[int, str]) -> None:
    """Fail when the security curve names a REE that ree.dat does not declare.

    Every curve block belongs to a deck REE, so an undeclared code means
    curva.dat was misread; converting on would silently drop those REEs'
    VminOP constraints.
    """
    unknown = sorted(
        {int(code) for code in curva_df["codigo_ree"].unique()} - set(ree_names)
    )
    if not unknown:
        return
    label = "REE" if len(unknown) == 1 else "REEs"
    codes = ", ".join(str(code) for code in unknown)
    raise FieldParseError(
        f"curva.dat: the security curve names {label} {codes}, which ree.dat "
        "does not declare. The file was misread, or it names a REE the deck "
        "does not have; converting it would drop the security-curve (VminOP) "
        "constraints.",
        path="curva.dat",
        field="CURVA DE SEGURANCA",
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
                    ree_name=rees.names[ree_code],
                    reason="no hydro plant",
                )
            )
            continue

        shares = rees.shares.get(ree_code, [])
        if not shares:
            vminop_skips.append(
                _VminopSkip(
                    ree_code=ree_code,
                    ree_name=rees.names[ree_code],
                    reason="no plant stores energy",
                )
            )
            continue

        pct_name = vminop_pct_name(ree_code)
        expression = stored_energy_floor([(1.0, s) for s in shares], pct_name)
        # Every hydro an ``@…_h{id}`` parameter names must be declared.
        referenced_ids = [int(h) for h in re.findall(r"@\w+?_h(\d+)\b", expression)]
        ree_name = rees.names[ree_code]
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
    """Emit the ``vminop-ree-not-expressible`` note, one row per skipped REE."""
    if not skips:
        return
    emit(
        Diagnostic(
            code="vminop-ree-not-expressible",
            severity=Severity.INFO,
            category=_CATEGORY,
            title=f"Security curve of REEs without storage not converted ({len(skips)})",
            summary=(
                "The listed REEs with a curva.dat security curve have no "
                "plant that stores energy, so the curve's floor is zero and the "
                "conversion writes no constraint for them."
            ),
            table=DiagnosticTable(
                columns=["Code", "Name", "Reason"],
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
            title=f"Security-curve penalties not positive ({len(clamps)})",
            summary=(
                "The listed REEs give curva.dat a security-curve penalty "
                "that is not positive, so the conversion penalizes their "
                f"shortfall at {_VMINOP_DEFAULT_PENALTY:g} instead."
            ),
            table=DiagnosticTable(
                columns=["Code", "Name", "Declared penalty", "Applied penalty"],
                rows=[[c.ree_code, c.ree_name, c.declared, c.applied] for c in clamps],
                justify=["right", "left", "right", "right"],
            ),
            remediation="Give the listed REEs a positive penalty in curva.dat.",
        ),
        logger=_LOG,
    )


# ---------------------------------------------------------------------------
# Electric constraints from restricao-eletrica.csv
# ---------------------------------------------------------------------------

_UNBOUNDED_THRESHOLD = -1.0e29  # values below this are treated as -inf

_YearMonth = tuple[int, int]

_ELECTRIC_FIELD_LABELS = {
    "codigo_restricao": "constraint code",
    "formula": "formula",
    "data_inicio": "start period (YYYY/MM)",
    "data_fim": "end period (YYYY/MM)",
    "patamar": "load block",
}


class _ElectricLimit(NamedTuple):
    """One ``RE-LIM-FORM-PER-PAT`` row; a ``None`` limit is unbounded."""

    first: _YearMonth
    last: _YearMonth
    block: int  # 1-based patamar
    lower: float | None
    upper: float | None


def _read_restricao_eletrica(
    restricoes: Restricoes, path: Path
) -> tuple[
    dict[int, str],
    dict[int, tuple[_YearMonth, _YearMonth]],
    dict[int, list[_ElectricLimit]],
]:
    """restricao-eletrica.csv's formulas (``RE``), horizons (``RE-HORIZ-PER``)
    and limits (``RE-LIM-FORM-PER-PAT``), by constraint code.

    Raises:
        SourceFileError: a record lacks a field the conversion needs, or a
            limit is not a number, such as the manual's conditional ``se(...)``
            limits.
    """
    expressions = {
        as_int(row["codigo_restricao"]): str(row["formula"])
        for row in _electric_rows(
            restricoes.re(df=True), "RE", ("codigo_restricao", "formula"), path
        )
    }
    horizons = {
        as_int(row["codigo_restricao"]): (
            _year_month(row["data_inicio"]),
            _year_month(row["data_fim"]),
        )
        for row in _electric_rows(
            restricoes.re_horiz_per(df=True),
            "RE-HORIZ-PER",
            ("codigo_restricao", "data_inicio", "data_fim"),
            path,
        )
    }
    limits: dict[int, list[_ElectricLimit]] = defaultdict(list)
    for row in _electric_rows(
        restricoes.re_lim_form_per(df=True),
        "RE-LIM-FORM-PER-PAT",
        ("codigo_restricao", "data_inicio", "data_fim", "patamar"),
        path,
    ):
        cod = as_int(row["codigo_restricao"])
        limits[cod].append(
            _ElectricLimit(
                first=_year_month(row["data_inicio"]),
                last=_year_month(row["data_fim"]),
                block=as_int(row["patamar"]),
                lower=_electric_limit(row["limite_inferior"], cod, path),
                upper=_electric_limit(row["limite_superior"], cod, path),
            )
        )
    return expressions, horizons, dict(limits)


def _electric_rows(
    frame: pd.DataFrame,
    record: str,
    required: tuple[str, ...],
    path: Path,
) -> list[dict[str, object]]:
    """*frame*'s rows, as read by inewave's ``df=True`` accessor of *record*.

    Raises:
        SourceFileError: a row lacks one of the *required* columns; the reader
            reads a missing or unreadable field as ``None``.
    """
    if frame.empty:
        return []
    rows: list[dict[str, object]] = frame.to_dict("records")
    for row in rows:
        missing = [_ELECTRIC_FIELD_LABELS[col] for col in required if is_na(row[col])]
        if missing:
            raise SourceFileError(
                f"{path.name} has a {record} line without a readable "
                f"{', '.join(missing)}",
                path=str(path),
                field=record,
            )
    return rows


def _year_month(value: object) -> _YearMonth:
    stamp = as_timestamp(value)
    return stamp.year, stamp.month


def _electric_limit(value: object, cod: int, path: Path) -> float | None:
    """A ``RE-LIM-FORM-PER-PAT`` limit, ``None`` when unbounded: a blank field
    (manual §3.45.2) or a value at or below ``_UNBOUNDED_THRESHOLD``.

    Raises:
        SourceFileError: the limit is not a number.
    """
    if is_na(value) or not str(value).strip():
        return None
    try:
        limit = float(str(value))
    except ValueError as err:
        raise SourceFileError(
            f"{path.name}: constraint {cod} has the RE-LIM-FORM-PER-PAT limit "
            f"{str(value).strip()!r}, which is not a number; conditional limits "
            "are not converted",
            path=str(path),
            field="RE-LIM-FORM-PER-PAT",
        ) from err
    return None if limit <= _UNBOUNDED_THRESHOLD else limit


#: The demand terms of restricao-eletrica.csv formulas, folded into the
#: limits at the case's own block load.
# TRACKED NOVOMODELO-GAP (C36): the case has no load term in a generic
# constraint, so a bare demanda/demanda_sin multiplier is not added to the
# CMO as the source model does; electric-constraint-demand-in-cmo lists them.
_BUS_DEMAND_TOKENS = frozenset({"demanda", "val_demanda"})
_SYSTEM_DEMAND_TOKENS = frozenset({"demanda_sin", "val_demanda_sin"})

#: Why the case cannot carry a formula term, by token. A constraint with one
#: is left out whole: without the term its sum is not the source model's.
_UNSUPPORTED_TERM_REASONS = {
    "ener_comerc": "the case has no commercial energy variable",
    "ger_pee": "renewable parks are not converted",
    "ger_conjh": "the case has no unit-group generation variable",
    "disp_usih": "the case has no available-power variable",
    "carga_ande": "the ANDE load is not converted",
    PEQ_TOKEN: "small-plant series are not converted",
    SE_TOKEN: "conditional terms are not converted",
}

type _ElectricFormula = dict[Bucket, list[ParsedTerm]] | _ElectricConstraintSkip


@dataclass(frozen=True)
class _NewaveFormulaScope:
    """restricao-eletrica.csv as the scope of its own formulas: ``re(X)``
    names another of its constraints, the file declares no named expression
    or data alias, and every ``ger_usit`` is a decision."""

    formulas: Mapping[int, str]

    @property
    def gnl_thermal_codes(self) -> frozenset[int]:
        return frozenset()

    def expression_formula(self, name: str) -> str | None:
        return None

    def is_alias(self, name: str) -> bool:
        return False

    def restriction_formula(self, code: int) -> str | None:
        return self.formulas.get(code)


def _read_formulas(
    expressions: Mapping[int, str], path: Path
) -> dict[int, _ElectricFormula]:
    """Each restricao-eletrica.csv formula's terms by bucket, or the skip of
    its constraint when it has a term the case cannot carry.

    Raises:
        FieldParseError: a formula is malformed: unbalanced, with an unknown
            function or one given the wrong number of arguments, or with an
            ``re(X)`` that names no constraint of the file.
        UnsupportedSourceError: a formula holds algebra the conversion does
            not read yet, a constant factor written after its term.
    """
    scope = _NewaveFormulaScope(expressions)
    formulas: dict[int, _ElectricFormula] = {}
    for cod, text in expressions.items():
        try:
            terms = parse_linear_expression(text, scope)
            partition = classify_terms(terms)
        except UnrecognizedElectricalToken as err:
            formulas[cod] = _ElectricConstraintSkip(
                cod, "unsupported", err.name, "the file declares no such name"
            )
            continue
        except MalformedElectricalExpression as err:
            raise FieldParseError(
                f"{path.name}: the formula of constraint {cod} is malformed: {err}",
                path=str(path),
                field="RE",
            ) from err
        except UnsupportedElectricalExpression as err:
            raise UnsupportedSourceError(
                f"{path.name}: the formula of constraint {cod} is not converted "
                f"yet: {err}",
                path=str(path),
                field="RE",
            ) from err
        unsupported = next(
            (term for term in terms if term.token in _UNSUPPORTED_TERM_REASONS), None
        )
        formulas[cod] = (
            partition
            if unsupported is None
            else _ElectricConstraintSkip(
                cod,
                "unsupported",
                term_text(unsupported),
                _UNSUPPORTED_TERM_REASONS[unsupported.token],
            )
        )
    return formulas


class _ElectricExpression(NamedTuple):
    """One electric constraint's decision terms as a novomodelo expression,
    and the data terms its limits are folded on."""

    text: str
    data_terms: tuple[ParsedTerm, ...] = ()


def _formula_expression(
    cod: int,
    partition: dict[Bucket, list[ParsedTerm]],
    id_map: NewaveIdMap,
    line_id_map: dict[tuple[int, int], int],
    idle_hydros: AbstractSet[int],
    term_skips: list[_ElectricTermSkip],
) -> _ElectricExpression | _ElectricConstraintSkip:
    """Constraint ``cod``'s formula with every term on the case's entities,
    or its skip (:func:`_expression_or_skip`), also when a demand term names
    a subsystem the case lacks."""
    data_terms = partition[Bucket.B]
    declared = set(id_map.all_bus_ids)
    for term in data_terms:
        if term.token in _BUS_DEMAND_TOKENS and term.args[0] not in declared:
            return _ElectricConstraintSkip(
                cod,
                "unsupported",
                term_text(term),
                "names a subsystem sistema.dat does not declare",
            )
    skips: list[_ElectricTermSkip] = []
    variables = [
        (term.coefficient, variable)
        for term in partition[Bucket.A]
        if (
            variable := _decision_variable(
                cod, term, id_map, line_id_map, idle_hydros, skips
            )
        )
        is not None
    ]
    return _expression_or_skip(cod, variables, skips, term_skips, tuple(data_terms))


def _expression_or_skip(
    cod: int,
    variables: list[tuple[float, str]],
    skips: list[_ElectricTermSkip],
    term_skips: list[_ElectricTermSkip],
    data_terms: tuple[ParsedTerm, ...] = (),
) -> _ElectricExpression | _ElectricConstraintSkip:
    """Constraint ``cod``'s expression over *variables*, or its skip: a term
    in *skips* whose cause drops the constraint leaves it out (*outside*),
    and so does an empty sum (*termless*).

    The skips the outcome reports move to *term_skips*: those that left the
    constraint out, or every one of a constraint kept.
    """
    outside = [skip for skip in skips if skip.drops_constraint]
    if outside:
        term_skips.extend(outside)
        return _ElectricConstraintSkip(cod, "outside")
    text = render_expression(variables)
    if text is None:
        return _ElectricConstraintSkip(cod, "termless")
    term_skips.extend(skips)
    return _ElectricExpression(text, data_terms)


def _directional_flow(
    line_id_map: dict[tuple[int, int], int], a: int, b: int
) -> str | None:
    """The source model's non-negative flow from subsystem *a* to *b* as a
    novomodelo line variable, or ``None`` when no line joins the pair."""
    line_id = line_id_map.get((min(a, b), max(a, b)))
    if line_id is None:
        return None
    # The direction is the variable choice, never a negated coefficient or a
    # signed ``line_exchange``: either lets the sum go negative when a term
    # flows in its non-canonical direction, diluting the bound.
    return f"line_direct({line_id})" if a < b else f"line_reverse({line_id})"


def _decision_variable(
    cod: int,
    term: ParsedTerm,
    id_map: NewaveIdMap,
    line_id_map: dict[tuple[int, int], int],
    idle_hydros: AbstractSet[int],
    term_skips: list[_ElectricTermSkip],
) -> str | None:
    """*term*'s novomodelo variable, or ``None``, with the skip recorded on
    *term_skips*, when it names a plant or a subsystem pair outside the case."""
    if term.token == "ener_interc":
        sys_from, sys_to = term.args
        variable = _directional_flow(line_id_map, sys_from, sys_to)
        if variable is None:
            term_skips.append(
                _ElectricTermSkip(
                    cod,
                    "no-line",
                    "restricao-eletrica.csv",
                    sys_from=sys_from,
                    sys_to=sys_to,
                )
            )
        return variable

    (plant_code,) = term.args
    kind: _TermSkipKind
    if term.token == "ger_usih":
        lookup, kind, variable = (
            id_map.hydro_id,
            _unmapped_hydro_kind(plant_code, idle_hydros),
            "hydro_generation",
        )
    else:
        lookup, kind, variable = (
            id_map.thermal_id,
            "thermal-unmapped",
            "thermal_generation",
        )
    try:
        entity_id = lookup(plant_code)
    except KeyError:
        term_skips.append(
            _ElectricTermSkip(
                cod, kind, "restricao-eletrica.csv", plant_code=plant_code
            )
        )
        return None
    return f"{variable}({entity_id})"


def _unmapped_hydro_kind(
    plant_code: int, idle_hydros: AbstractSet[int]
) -> _TermSkipKind:
    return "hydro-idle" if plant_code in idle_hydros else "hydro-unmapped"


def _demand_context(
    loads: Mapping[tuple[int, int], Sequence[float]],
    id_map: NewaveIdMap,
    stage_id: int,
    block_id: int,
) -> DataContext:
    """The value of a formula's data term at one (stage, block): a
    subsystem's block load for ``demanda(X)`` and ``val_demanda(X)``, all
    subsystems' for their ``_sin`` forms, and 1 for a constant, whose value
    is its coefficient."""

    def bus_load(code: int) -> float:
        block_loads = loads.get((id_map.bus_id(code), stage_id))
        return 0.0 if block_loads is None else block_loads[block_id]

    def resolve(term: ParsedTerm) -> float:
        if term.token in _BUS_DEMAND_TOKENS:
            return bus_load(term.args[0])
        if term.token in _SYSTEM_DEMAND_TOKENS:
            return sum(bus_load(code) for code in id_map.all_bus_ids)
        if term.token == CONST_TOKEN:
            return 1.0
        raise ValueError(f"no value for the electric-constraint term {term_text(term)}")

    return resolve


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


_CUTOFF_EFFECT = (
    "The electric constraints take their restricao-eletrica.csv ceilings on "
    "every stage, without the re.dat ceilings NEWAVE applies after that period."
)


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
                    title="End of the individualized period unknown",
                    summary=(
                        "ree.dat lists no REE, so the conversion cannot tell "
                        f"where the individualized period ends. {_CUTOFF_EFFECT}"
                    ),
                ),
                logger=_LOG,
            )
            return _NO_INDIVIDUALIZADO_CUTOFF
        row = ree_df.iloc[0]
        end_month = int(row["mes_fim_individualizado"])
        end_year = int(row["ano_fim_individualizado"])
        return (end_year - start_year) * 12 + (end_month - start_month)
    except (OSError, ValueError, KeyError) as exc:
        emit(
            Diagnostic(
                code="individualizado-cutoff-unknown",
                severity=Severity.WARNING,
                category=_CATEGORY,
                title="End of the individualized period unknown",
                summary=(
                    "The end of the individualized period could not be read "
                    f"from ree.dat. {_CUTOFF_EFFECT}"
                ),
                notes=[f"Reason: {exception_reason(exc)}"],
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
    except Exception as exc:
        emit(
            unreadable_file_diagnostic(
                code="re-dat-unreadable",
                file_name="re.dat",
                category=_CATEGORY,
                effect="the case has none of its generation limits on plant sets.",
                exc=exc,
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
    *,
    block_loads: Mapping[tuple[int, int], Sequence[float]],
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
    block_loads:
        Each ``(bus_id, stage_id)``'s load in MW on each block
        (:func:`~novomodelo_bridge.newave.converters.stochastic.block_loads`),
        which the formulas' demand terms read.

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
    if not sources.formulas and not sources.re_conjuntos:
        _LOG.debug("No electric constraints found; skipping.")
        return None

    line_id_map = line_ids(case)
    idle_hydros = case.idle_hydro_codes

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
    demand_in_cmo: list[tuple[int, str]] = []

    data_context = partial(_demand_context, block_loads, id_map)
    for cod in sorted(set(sources.formulas) | set(sources.re_conjuntos)):
        expression = _electric_expression(
            cod, sources, id_map, line_id_map, idle_hydros, elec_term_skips
        )
        if isinstance(expression, _ElectricConstraintSkip):
            elec_constraint_skips.append(expression)
            continue

        bounds = _electric_bounds(cod, sources, grid.horizon)
        if not bounds.has_sup and not bounds.has_inf:
            elec_constraint_skips.append(_ElectricConstraintSkip(cod, "unbounded"))
            continue

        _add_electric_constraint(
            builder, cod, expression, slack_config, bounds, grid, data_context
        )
        demand_in_cmo.extend(
            (cod, term_text(term))
            for term in expression.data_terms
            if term.token in CMO_DEMAND_TOKENS
        )

    _emit_electric_term_skips(elec_term_skips)
    _emit_electric_constraint_skips(elec_constraint_skips)
    if demand_in_cmo:
        emit(
            demand_in_cmo_diagnostic(
                "electric-constraint-demand-in-cmo",
                demand_in_cmo,
                source="restricao-eletrica.csv",
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
        grid.cutoff,
    )

    return result


@dataclass(frozen=True)
class _ElectricSources:
    """restricao-eletrica.csv's and RE.DAT's electric constraints, by constraint code."""

    formulas: dict[int, _ElectricFormula]
    horizons: dict[int, tuple[_YearMonth, _YearMonth]]
    csv_bounds: dict[int, list[_ElectricLimit]]
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
    """Read restricao-eletrica.csv and RE.DAT, and the grid they are laid on."""
    formulas: dict[int, _ElectricFormula] = {}
    horizons: dict[int, tuple[_YearMonth, _YearMonth]] = {}
    csv_bounds: dict[int, list[_ElectricLimit]] = {}
    if re_path is not None and (restricoes := case.restricao_eletrica) is not None:
        unread = unread_lines_diagnostic(re_path, restricoes.data)
        if unread is not None:
            emit(unread, logger=_LOG)
        expressions, horizons, csv_bounds = _read_restricao_eletrica(
            restricoes, re_path
        )
        formulas = _read_formulas(expressions, re_path)

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

    sources = _ElectricSources(
        formulas, horizons, csv_bounds, re_conjuntos, re_dat_bounds
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
                        "The ELETRI penalty of penalid.dat could not be read, "
                        "so the conversion prices violations of the electric "
                        "constraints at ten times the highest deficit cost, or "
                        "makes them hard limits when sistema.dat gives none."
                    ),
                    notes=[f"Reason: {exception_reason(exc)}"],
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
    idle_hydros: AbstractSet[int],
    term_skips: list[_ElectricTermSkip],
) -> _ElectricExpression | _ElectricConstraintSkip:
    """Constraint ``cod``'s expression, from its restricao-eletrica.csv formula
    when it has one, else the sum of its RE.DAT plants' ``hydro_generation``,
    or the constraint's skip (:func:`_expression_or_skip`)."""
    formula = sources.formulas.get(cod)
    if isinstance(formula, _ElectricConstraintSkip):
        return formula
    if formula is not None:
        return _formula_expression(
            cod, formula, id_map, line_id_map, idle_hydros, term_skips
        )

    skips: list[_ElectricTermSkip] = []
    terms: list[tuple[float, str]] = []
    for plant_code in sorted(sources.re_conjuntos[cod]):
        try:
            novomodelo_id = id_map.hydro_id(plant_code)
        except KeyError:
            skips.append(
                _ElectricTermSkip(
                    constraint_code=cod,
                    kind=_unmapped_hydro_kind(plant_code, idle_hydros),
                    source="re.dat",
                    plant_code=plant_code,
                )
            )
            continue
        terms.append((1.0, f"hydro_generation({novomodelo_id})"))
    return _expression_or_skip(cod, terms, skips, term_skips)


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

    horiz_ini, horiz_fin = sources.horizons.get(
        cod, ((start_year, start_month), (start_year + horizon.num_anos - 1, 12))
    )

    for limit in sources.csv_bounds.get(cod, []):
        block_id = limit.block - 1  # 1-based -> 0-based
        for sid in _stage_ids_between(
            max(limit.first, horiz_ini),
            min(limit.last, horiz_fin),
            start_year=start_year,
            start_month=start_month,
        ):
            if 0 <= sid < horizon.total_stages:
                if limit.upper is not None:
                    csv_sup[(sid, block_id)] = limit.upper
                if limit.lower is not None:
                    csv_inf[(sid, block_id)] = limit.lower

    return _ElectricBounds(csv_sup, csv_inf, sources.re_dat_bounds.get(cod, {}))


def _add_electric_constraint(
    builder: GenericConstraintBuilder,
    cod: int,
    expression: _ElectricExpression,
    slack: dict[str, object],
    bounds: _ElectricBounds,
    grid: _ElectricGrid,
    data_context: Callable[[int, int], DataContext],
) -> None:
    """Add ``RE_{cod}``'s ceiling and floor and one row per declared (stage, block).

    The ceiling is RE.DAT's from the individualizado cutoff on and the csv's
    elsewhere; the floor is the csv's. Each limit is net of the expression's
    data terms at its (stage, block), read by *data_context*.
    """
    sup_id: int | None = None
    inf_id: int | None = None

    if bounds.has_sup:
        sup_id = builder.add_constraint(
            name=f"RE_{cod}",
            description=f"Electric constraint {cod}",
            expression=expression.text,
            slack=slack,
        )
    if bounds.has_inf:
        inf_id = builder.add_constraint(
            name=f"RE_{cod}",
            description=f"Electric constraint {cod}",
            expression=expression.text,
            slack=slack,
        )

    for stage_id in range(grid.horizon.total_stages):
        is_post_indiv = stage_id >= grid.cutoff

        for block_id in range(grid.num_patamares):
            cell = (stage_id, block_id)
            sup_val: float | None = None
            if sup_id is not None:
                sup_val = (
                    bounds.re_sup[cell]
                    if is_post_indiv and cell in bounds.re_sup
                    else bounds.csv_sup.get(cell)
                )
            inf_val = bounds.csv_inf.get(cell) if inf_id is not None else None
            if sup_val is None and inf_val is None:
                continue
            offset = evaluate_terms(
                expression.data_terms, data_context(stage_id, block_id)
            )
            if sup_id is not None and sup_val is not None:
                builder.add_bound_row(
                    sup_id,
                    stage_id=stage_id,
                    block_id=block_id,
                    lower=None,
                    upper=sup_val - offset,
                )
            if inf_id is not None and inf_val is not None:
                builder.add_bound_row(
                    inf_id,
                    stage_id=stage_id,
                    block_id=block_id,
                    lower=inf_val - offset,
                    upper=None,
                )


#: ``(code, title, summary)`` of each kind of plant term outside the case,
#: worded for what its cause's treatment does with the constraint.
_PLANT_TERM_SKIPS: dict[_TermSkipKind, tuple[str, str, str]] = {
    "hydro-idle": (
        "electric-constraint-term-idle-plant",
        "Electric-constraint terms on hydro plants that generate nothing left out",
        "The listed electric-constraint terms name a confhd.dat plant the case "
        "leaves out because it generates nothing: a fictitious plant, an NC "
        "plant or an NE plant with no row in exph.dat. The conversion leaves "
        "the term out and keeps the rest of the constraint.",
    ),
    "hydro-unmapped": (
        "electric-constraint-hydro-unmapped",
        "Electric constraints on hydro plants outside the case not converted",
        "The listed electric constraints have a term on a hydro plant the "
        "case does not hold: one confhd.dat does not declare, or an NE plant "
        "with unit rows but no filling row in exph.dat. The conversion leaves "
        "each constraint out rather than convert it without the term, so the "
        "case runs without these limits.",
    ),
    "thermal-unmapped": (
        "electric-constraint-thermal-unmapped",
        "Electric constraints on thermals outside the case not converted",
        "The listed electric constraints have a term on a thermal plant "
        "conft.dat does not declare. The conversion leaves each constraint "
        "out rather than convert it without the term, so the case runs "
        "without these limits.",
    ),
}


def _emit_electric_term_skips(skips: list[_ElectricTermSkip]) -> None:
    """Emit one diagnostic per kind of electric-constraint term outside the
    case, at its cause's severity."""
    if not skips:
        return
    by_kind: dict[_TermSkipKind, list[_ElectricTermSkip]] = defaultdict(list)
    for skip in skips:
        by_kind[skip.kind].append(skip)

    for kind, (code, title, summary) in _PLANT_TERM_SKIPS.items():
        plant_skips = by_kind.get(kind, [])
        if plant_skips:
            emit(
                Diagnostic(
                    code=code,
                    severity=_TERM_SKIP_CAUSE[kind].severity,
                    category=_CATEGORY,
                    title=f"{title} ({len(plant_skips)})",
                    summary=summary,
                    table=DiagnosticTable(
                        columns=["Constraint", "Plant code", "Source"],
                        rows=[
                            [s.constraint_code, s.plant_code, s.source]
                            for s in plant_skips
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
                severity=OutsideTerm.NO_LINE.severity,
                category=_CATEGORY,
                title=(
                    "Electric-constraint terms on subsystems without interchange "
                    f"({len(no_line)})"
                ),
                summary=(
                    "The listed electric-constraint interchange terms "
                    "name a subsystem pair with no interchange limit in "
                    "sistema.dat, so no flow runs between them and the "
                    "conversion leaves the term out."
                ),
                table=DiagnosticTable(
                    columns=["Constraint", "From", "To"],
                    rows=[[s.constraint_code, s.sys_from, s.sys_to] for s in no_line],
                    justify=["right", "right", "right"],
                ),
            ),
            logger=_LOG,
        )


def _emit_electric_constraint_skips(skips: list[_ElectricConstraintSkip]) -> None:
    """Emit the constraints left out: those with a term the case cannot carry
    (``electric-constraint-term-unsupported``), with no term it can
    (``electric-constraint-skipped``) and with no limit
    (``electric-constraint-unbounded``)."""
    unsupported = sorted(
        (s for s in skips if s.kind == "unsupported"), key=lambda s: s.constraint_code
    )
    termless = sorted(s.constraint_code for s in skips if s.kind == "termless")
    unbounded = sorted(s.constraint_code for s in skips if s.kind == "unbounded")
    if unsupported:
        emit(
            Diagnostic(
                code="electric-constraint-term-unsupported",
                severity=Severity.WARNING,
                category=_CATEGORY,
                title=(
                    "Electric constraints with an unsupported term not converted "
                    f"({len(unsupported)})"
                ),
                summary=(
                    "The listed electric constraints of restricao-eletrica.csv "
                    "have a term the case cannot carry, so the conversion leaves "
                    "them out and the case runs without these limits."
                ),
                table=DiagnosticTable(
                    columns=["Constraint", "Term", "Reason"],
                    rows=[[s.constraint_code, s.term, s.reason] for s in unsupported],
                    justify=["right", "left", "left"],
                ),
            ),
            logger=_LOG,
        )
    if termless:
        emit(
            Diagnostic(
                code="electric-constraint-skipped",
                severity=Severity.WARNING,
                category=_CATEGORY,
                title=f"Electric constraints without a term not converted ({len(termless)})",
                summary=(
                    "The listed electric constraints have no ger_usih, "
                    "ger_usit or ener_interc term the case can carry, so the "
                    "conversion leaves them out."
                ),
                table=DiagnosticTable(
                    columns=["Constraint"],
                    rows=[[code] for code in termless],
                    justify=["right"],
                ),
            ),
            logger=_LOG,
        )
    if unbounded:
        emit(
            Diagnostic(
                code="electric-constraint-unbounded",
                severity=Severity.INFO,
                category=_CATEGORY,
                title=f"Electric constraints without a limit not converted ({len(unbounded)})",
                summary=(
                    "The listed electric constraints have no limit on "
                    "any stage, so they cannot bind and the conversion leaves "
                    "them out."
                ),
                table=DiagnosticTable(
                    columns=["Constraint"],
                    rows=[[code] for code in unbounded],
                    justify=["right"],
                ),
            ),
            logger=_LOG,
        )


# ---------------------------------------------------------------------------
# AGRINT.DAT — Exchange group constraints
# ---------------------------------------------------------------------------


def _agrint_tables(
    agrint: Agrint, num_patamares: int
) -> tuple[
    dict[int, list[tuple[int, int, float]]],  # group_id -> [(A, B, coeff)]
    list[tuple[int, int, int, int | None, int | None, list[float]]],
    # (group_id, mi, anoi, mf|None, anof|None, [lim per patamar]), in file order
]:
    """``agrint.dat``'s groups and limit records, one limit per deck block.

    A blank block column repeats the previous block's limit.

    Raises:
        FieldParseError: a limit record whose first block column is blank.
    """
    groups_df = agrint.agrupamentos
    limits_df = agrint.limites_agrupamentos
    if groups_df is None or limits_df is None:
        return {}, []

    groups: dict[int, list[tuple[int, int, float]]] = {}
    for group_id, de, para, coefficient in zip(
        groups_df["agrupamento"],
        groups_df["submercado_de"],
        groups_df["submercado_para"],
        groups_df["coeficiente"],
        strict=True,
    ):
        groups.setdefault(as_int(group_id), []).append(
            (as_int(de), as_int(para), as_float(coefficient))
        )

    limits: list[tuple[int, int, int, int | None, int | None, list[float]]] = []
    record_number = (limits_df["patamar"] == 1).cumsum()
    for _, record in limits_df.groupby(record_number, sort=False):
        first = record.iloc[0]
        group_id = as_int(first["agrupamento"])
        values = (
            record.set_index("patamar")["valor"]
            .reindex(range(1, num_patamares + 1))
            .ffill()
        )
        if values.isna().any():
            raise FieldParseError(
                f"agrint.dat: a limit record of group {group_id} has no first block limit",
                path="agrint.dat",
                field="LIMITES POR GRUPO",
            )
        start = as_timestamp(first["data_inicio"])
        end = None if is_na(first["data_fim"]) else as_timestamp(first["data_fim"])
        limits.append(
            (
                group_id,
                start.month,
                start.year,
                None if end is None else end.month,
                None if end is None else end.year,
                [float(v) for v in values],
            )
        )
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
        Entity ID mapping (used indirectly via ``line_ids``).
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

    agrint = case.agrint
    if agrint is None:
        return None
    groups, limits = _agrint_tables(agrint, num_patamares_ag)
    if not groups or not limits:
        return None

    horizon = case.horizon
    line_id_map = line_ids(case)

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
    parsed_terms: list[tuple[float, str]] = []
    for a, b, coeff in terms_raw:
        var = _directional_flow(line_id_map, a, b)
        if var is None:
            term_skips.append(_AgrintTermSkip(group_id=group_id, sys_a=a, sys_b=b))
            continue
        parsed_terms.append((coeff, var))
    return render_expression(parsed_terms)


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
    """Emit the ``agrint-interchange-no-line`` note, one row per dropped term."""
    if not skips:
        return
    emit(
        Diagnostic(
            code="agrint-interchange-no-line",
            severity=Severity.INFO,
            category=_CATEGORY,
            title=f"agrint.dat terms on subsystems without interchange ({len(skips)})",
            summary=(
                "The listed agrint.dat group terms name a subsystem pair "
                "with no interchange limit in sistema.dat, so no flow runs "
                "between them and the conversion leaves the term out."
            ),
            table=DiagnosticTable(
                columns=["Group", "From", "To"],
                rows=[[s.group_id, s.sys_a, s.sys_b] for s in skips],
                justify=["right", "right", "right"],
            ),
        ),
        logger=_LOG,
    )


def _emit_agrint_group_skips(skips: list[_AgrintGroupSkip]) -> None:
    """Emit the ``agrint-group-empty`` note, one row per dropped group."""
    if not skips:
        return
    emit(
        Diagnostic(
            code="agrint-group-empty",
            severity=Severity.INFO,
            category=_CATEGORY,
            title=f"agrint.dat groups without interchange not converted ({len(skips)})",
            summary=(
                "The listed agrint.dat groups name only subsystem pairs "
                "with no interchange limit in sistema.dat, so the group's flow "
                "is zero and the conversion writes no constraint for it."
            ),
            table=DiagnosticTable(
                columns=["Group"],
                rows=[[s.group_id] for s in skips],
                justify=["right"],
            ),
        ),
        logger=_LOG,
    )

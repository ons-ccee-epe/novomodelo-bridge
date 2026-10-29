"""DECOMP's a-priori turbined-flow limit, per plant and stage.

Before it optimises, the source model caps each plant's turbined flow at
``min(qmaxT · MP · FD, Pdisp / ρ_eq)`` (its reference manual, §4.5.5):

- ``qmaxT`` is the head-limited flow
  (:func:`~novomodelo_bridge.core.turbine.tailrace_limited_flow`) of the stage's
  machine sets, ``q_ef = Σ n·q_nom`` at ``h_ef``, the unit-weighted mean of the
  ``hidr.dat`` rated heads. The forebay sits at the study's initial volume on
  every stage; the tailrace rises with the flow along the plant's ``polinjus``
  families, interpolated at the level of the plant downstream at its minimum
  volume, or along its ``hidr.dat`` downstream polynomials when ``polinjus``
  has none. When the plant's unregulated inflow, less what its reservoir can
  still store over the stage, exceeds that flow, ``qmaxT`` is the head-limited
  flow at that inflow instead.
- ``Pdisp / ρ_eq`` is the generator limit: the available power
  (installed × ``MP`` × ``FD``) over the equivalent productivity, the
  full-range mean for a reservoir and the reference volume's for a
  run-of-river plant.

Itaipu's limit is plant-wide, ``T = min(qmaxT, Σ Pdisp_g / ρ_eq)`` with no
``MP`` × ``FD`` factor on ``qmaxT``, where the source model sums its two
frequency halves (:attr:`TurbineLimits.plant_caps`); each unit group is capped
at ``min(T, Pdisp_g / ρ_eq)``, its own generator limit within the plant's.

The unregulated inflow is the plant's incremental inflow plus that of every
run-of-river plant upstream of it short of a reservoir; on the terminal stage
it is the fan's probability-weighted mean.

The source model fits each plant's FPHA over the turbined flows up to the same
limit with ``MP`` × ``FD`` left out, ``min(qmaxT, P_inst / ρ_eq)``, and
Itaipu's up to its ``T`` (:attr:`TurbineLimits.fit_maxima`).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
    format_stage_ranges,
)
from novomodelo_bridge.core.pandas_utils import as_frame_or_empty
from novomodelo_bridge.core.productivity import (
    KTURB_BY_TIPO_TURBINA,
    apply_hydraulic_loss,
    evaluate_cota,
)
from novomodelo_bridge.core.tailrace import (
    TailraceFamily,
    tailrace_families_from_hidr,
    tailrace_families_from_polinjus,
    tailrace_level_m,
)
from novomodelo_bridge.core.tolerances import relative_tolerance
from novomodelo_bridge.core.turbine import (
    head_limited_flow,
    tailrace_limited_flow,
    turbine_limit_gap_diagnostic,
)
from novomodelo_bridge.decomp.bounds_accumulator import BoundContribution
from novomodelo_bridge.decomp.converters.cadastro import effective_storage_range
from novomodelo_bridge.decomp.converters.cadastro.effective import is_reservoir
from novomodelo_bridge.decomp.converters.hydro.bounds import (
    _ITAIPU_CODE,
    _conjunto_count,
    _conjunto_machines,
)
from novomodelo_bridge.decomp.converters.hydro.entity import _downstream_operated
from novomodelo_bridge.decomp.converters.hydro.productivity import (
    _equivalent_productivity_mw_per_m3s,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
    from novomodelo_bridge.decomp.id_map import DecompIdMap
    from novomodelo_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)

#: The source model's convergence step for ``qmaxT`` (manual §4.5.5).
_FIXED_POINT_TOLERANCE_M3S = 1.0
#: A safety bound only: the iteration converges in a few steps on real decks.
_FIXED_POINT_MAX_ITERATIONS = 100


@dataclass(frozen=True)
class TurbineLimits:
    """Turbined-flow caps (m³/s), one per stage, keyed ``(code, unit_group_id)``.

    Every plant has unit group 0; Itaipu has one group per frequency half.
    *plant_caps* holds, by plant code, the per-stage plant-wide cap of a plant
    whose groups share one limit, tighter than their caps' sum. *fit_maxima*
    holds, by plant code, the per-stage upper end of the turbined flows the
    source model fits the plant's FPHA over.
    """

    caps: Mapping[tuple[int, int], tuple[float, ...]]
    plant_caps: Mapping[int, tuple[float, ...]] = field(default_factory=dict)
    fit_maxima: Mapping[int, tuple[float, ...]] = field(default_factory=dict)

    def cap(self, code: int, group_id: int, stage_index: int) -> float:
        return self.caps[(code, group_id)][stage_index]

    def group_count(self, code: int) -> int:
        return sum(1 for c, _ in self.caps if c == code)

    def plant_limit(self, code: int, stage_index: int) -> float:
        """The plant's cap at *stage_index*: its groups' summed caps, or its
        plant-wide cap when tighter."""
        total = sum(
            self.cap(code, group_id, stage_index)
            for group_id in range(self.group_count(code))
        )
        plant = self.plant_caps.get(code)
        return min(total, plant[stage_index]) if plant is not None else total

    def envelope(self, code: int, group_id: int) -> float:
        """The group's widest cap over the horizon."""
        return max(self.caps[(code, group_id)])


def tailrace_families(
    case: DecompCase, id_map: DecompIdMap
) -> dict[int, tuple[TailraceFamily, ...]]:
    """``{plant code: tailrace families}`` for every operated plant: its
    ``polinjus`` families, else its ``hidr.dat`` downstream polynomials."""
    polinjus = case.polinjus
    from_polinjus = (
        {}
        if polinjus is None
        else tailrace_families_from_polinjus(
            as_frame_or_empty(polinjus.hidreletrica_curvajusante(df=True)),
            as_frame_or_empty(
                polinjus.hidreletrica_curvajusante_polinomio_segmento(df=True)
            ),
        )
    )
    return {
        code: from_polinjus.get(code)
        or tailrace_families_from_hidr(case.hidr.loc[code], code)
        for code in id_map.hydro_codes
    }


def compute_turbine_limits(
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    *,
    initial_volumes: Mapping[int, float],
    incremental_inflows: Mapping[tuple[int, int], float],
    tailrace: Mapping[int, Sequence[TailraceFamily]],
    availability_factors: Mapping[tuple[int, int, int], float],
) -> TurbineLimits:
    """Every operated plant's per-stage turbined-flow caps.

    Args:
        initial_volumes: hm³ per plant code, the study's initial storage.
        incremental_inflows: m³/s per ``(code, stage_index)``
            (:func:`~novomodelo_bridge.decomp.scenarios.expected_incremental_inflows`).
        tailrace: families per plant code (:func:`tailrace_families`).
        availability_factors: ``MP × FD`` per ``(code, unit_group_id,
            stage_index)``
            (:func:`~novomodelo_bridge.decomp.converters.hydro.bounds.
            stage_availability_factors`).
    """
    operated = set(id_map.hydro_codes)
    downstream = {
        code: _downstream_operated(effective, code, operated)
        for code in id_map.hydro_codes
    }
    upstream: dict[int, list[int]] = {code: [] for code in id_map.hydro_codes}
    for code, below in downstream.items():
        if below is not None:
            upstream[below].append(code)

    caps: dict[tuple[int, int], list[float]] = {}
    plant_caps: dict[int, list[float]] = {}
    fit_maxima: dict[int, list[float]] = {}
    unconverged: list[list[object]] = []
    gap_stages: dict[tuple[int, str, str], list[int]] = {}
    for code in id_map.hydro_codes:
        for stage in calendar:
            stage_caps = _stage_caps(
                effective,
                code,
                stage,
                initial_volume_hm3=initial_volumes[code],
                unregulated_inflow_m3s=_unregulated_inflow(
                    effective, code, stage.index, upstream, incremental_inflows
                ),
                families=tailrace[code],
                downstream_code=downstream[code],
                availability_factors=availability_factors,
            )
            for group_id, cap in enumerate(stage_caps.groups):
                caps.setdefault((code, group_id), []).append(cap)
            if stage_caps.plant is not None:
                plant_caps.setdefault(code, []).append(stage_caps.plant)
            fit_maxima.setdefault(code, []).append(stage_caps.fit_max)
            if not stage_caps.converged:
                unconverged.append(
                    [code, _plant_name(effective, code), stage.index + 1]
                )
            for missing, used in stage_caps.gaps:
                gap_stages.setdefault((code, missing, used), []).append(stage.index)
    if unconverged:
        emit(_unconverged_diagnostic(unconverged), logger=_LOG)
    if gap_stages:
        emit(
            turbine_limit_gap_diagnostic(
                [
                    [
                        code,
                        _plant_name(effective, code),
                        missing,
                        used,
                        format_stage_ranges(stages),
                    ]
                    for (code, missing, used), stages in gap_stages.items()
                ]
            ),
            logger=_LOG,
        )
    return TurbineLimits(
        {key: tuple(values) for key, values in caps.items()},
        {code: tuple(values) for code, values in plant_caps.items()},
        {code: tuple(values) for code, values in fit_maxima.items()},
    )


def plant_turbine_limit_contributions(
    turbine_limits: TurbineLimits, id_map: DecompIdMap
) -> list[BoundContribution]:
    """The ``hydro`` ``turbined`` upper bounds carrying
    :attr:`TurbineLimits.plant_caps`, one per stage where the plant cap is
    below its groups' summed caps."""
    contributions: list[BoundContribution] = []
    for code, stage_caps in turbine_limits.plant_caps.items():
        for stage_index, cap in enumerate(stage_caps):
            groups_total = sum(
                turbine_limits.cap(code, group_id, stage_index)
                for group_id in range(turbine_limits.group_count(code))
            )
            if cap < groups_total - relative_tolerance(groups_total):
                contributions.append(
                    BoundContribution(
                        family="hydro",
                        entity_id=id_map.hydro_id(code),
                        stage_id=stage_index,
                        block_id=None,
                        axis="turbined",
                        lower=None,
                        upper=cap,
                        contributor="turbine_limit",
                    )
                )
    return contributions


def _unregulated_inflow(
    effective: EffectiveCadastro,
    code: int,
    stage_index: int,
    upstream: Mapping[int, Sequence[int]],
    incremental_inflows: Mapping[tuple[int, int], float],
) -> float:
    return incremental_inflows[(code, stage_index)] + sum(
        _unregulated_inflow(
            effective, above, stage_index, upstream, incremental_inflows
        )
        for above in upstream[code]
        if not is_reservoir(effective, above)
    )


class _StageCaps(NamedTuple):
    groups: list[float]
    plant: float | None
    fit_max: float
    converged: bool
    gaps: tuple[tuple[str, str], ...] = ()


def _stage_caps(
    effective: EffectiveCadastro,
    code: int,
    stage: OperativeStage,
    *,
    initial_volume_hm3: float,
    unregulated_inflow_m3s: float,
    families: Sequence[TailraceFamily],
    downstream_code: int | None,
    availability_factors: Mapping[tuple[int, int, int], float],
) -> _StageCaps:
    """One stage's caps: one per unit group, the plant-wide one of a split
    plant, and whether ``qmaxT`` converged."""
    s = stage.index
    hreg = effective.base.loc[code]
    machines = [
        _conjunto_machines(hreg, code, k, effective, s)
        for k in range(1, _conjunto_count(hreg, code, effective, s) + 1)
    ]
    n_units = sum(n for n, _, _ in machines)
    rated_flow = sum(n * q_nom for n, q_nom, _ in machines)
    if n_units <= 0 or rated_flow <= 0.0:
        if code == _ITAIPU_CODE:
            return _StageCaps([0.0] * len(machines), 0.0, 0.0, True)
        return _StageCaps([0.0], None, 0.0, True)
    nominal_head = _rated_head(hreg, [n for n, _, _ in machines])
    gaps: list[tuple[str, str]] = []
    unrated = _unrated_sets(hreg, len(machines))
    if unrated:
        gaps.append(
            (
                ", ".join(f"queda_nominal_conjunto_{k}" for k in unrated),
                "rated flow, not head-corrected",
            )
        )
    exponent = KTURB_BY_TIPO_TURBINA.get(int(hreg.get("tipo_turbina", 0) or 0), 0.5)

    v_lo, v_hi = effective_storage_range(effective, code, s)
    forebay = evaluate_cota(
        effective.cota_polynomial(code, s), min(max(initial_volume_hm3, v_lo), v_hi)
    )
    downstream_level = (
        None
        if downstream_code is None
        else evaluate_cota(
            effective.cota_polynomial(downstream_code, s),
            effective_storage_range(effective, downstream_code, s)[0],
        )
    )
    tipo_perda = int(hreg["tipo_perda"])
    perdas = effective.value(code, "perdas", s)

    def net_head_at(flow_m3s: float) -> float:
        return apply_hydraulic_loss(
            forebay - tailrace_level_m(families, flow_m3s, downstream_level),
            tipo_perda,
            perdas,
        )

    head_limited, converged = tailrace_limited_flow(
        net_head_at,
        nominal_head_m=nominal_head,
        nominal_flow_m3s=rated_flow,
        exponent=exponent,
        tolerance_m3s=_FIXED_POINT_TOLERANCE_M3S,
        max_iterations=_FIXED_POINT_MAX_ITERATIONS,
    )
    storage_room_hm3 = (
        max(v_hi - initial_volume_hm3, 0.0) if is_reservoir(effective, code) else 0.0
    )
    # hm³ → m³/s over the stage: × 1e6 m³ per hm³, ÷ the stage's seconds.
    minimum_outflow = unregulated_inflow_m3s - storage_room_hm3 * 1e6 / (
        stage.total_hours * 3600.0
    )
    if minimum_outflow > head_limited:
        head_limited = head_limited_flow(
            net_head_at(minimum_outflow),
            nominal_head_m=nominal_head,
            nominal_flow_m3s=rated_flow,
            exponent=exponent,
        )

    rho_eq = _equivalent_productivity_mw_per_m3s(
        effective,
        code,
        s,
        reference_volume_hm3=None if is_reservoir(effective, code) else v_lo,
    )
    if rho_eq <= 0.0:
        gaps.append(("positive equivalent productivity", "no generator limit"))
    if code == _ITAIPU_CODE:
        generator_limits = [
            _generator_limit(
                n * p_nom * availability_factors[(code, group_id, s)], rho_eq
            )
            for group_id, (n, _, p_nom) in enumerate(machines)
        ]
        plant_cap = min(head_limited, sum(generator_limits))
        return _StageCaps(
            [min(plant_cap, limit) for limit in generator_limits],
            plant_cap,
            plant_cap,
            converged,
            tuple(gaps),
        )
    factor = availability_factors[(code, 0, s)]
    installed = sum(n * p_nom for n, _, p_nom in machines)
    return _StageCaps(
        [min(head_limited * factor, _generator_limit(installed * factor, rho_eq))],
        None,
        min(head_limited, _generator_limit(installed, rho_eq)),
        converged,
        tuple(gaps),
    )


def _rated_head(hreg: pd.Series, units: Sequence[int]) -> float:
    """``h_ef`` (m), the unit-weighted mean of the machine sets' rated heads, or
    ``0.0`` (no head limit) when any set declares none.

    The heads stay the ``hidr.dat`` ones: ``AC ALTEFE`` is not consumed (see
    the TRACKED NOVOMODELO-GAP in :func:`convert_hydros`).
    """
    if _unrated_sets(hreg, len(units)):
        return 0.0
    heads = [
        float(hreg[f"queda_nominal_conjunto_{k}"]) for k in range(1, len(units) + 1)
    ]
    return sum(n * h for n, h in zip(units, heads, strict=True)) / sum(units)


def _unrated_sets(hreg: pd.Series, count: int) -> list[int]:
    """The machine sets among the first *count* that declare no positive
    rated head."""
    unrated = []
    for k in range(1, count + 1):
        head = hreg.get(f"queda_nominal_conjunto_{k}")
        if head is None or pd.isna(head) or float(head) <= 0.0:
            unrated.append(k)
    return unrated


def _generator_limit(available_mw: float, rho_eq: float) -> float:
    return available_mw / rho_eq if rho_eq > 0.0 else math.inf


def _plant_name(effective: EffectiveCadastro, code: int) -> str:
    return str(effective.base.loc[code, "nome_usina"]).strip()


def _unconverged_diagnostic(rows: list[list[object]]) -> Diagnostic:
    return Diagnostic(
        code="decomp-turbine-limit-unconverged",
        severity=Severity.WARNING,
        category="Hydro capacity",
        title=f"Turbined-flow limit did not converge ({len(rows)})",
        summary=(
            "The head-limited turbined flow of the listed plants did not settle "
            f"within {_FIXED_POINT_MAX_ITERATIONS} iterations; the last value "
            "is used as the stage's limit."
        ),
        table=DiagnosticTable(columns=["Code", "Name", "Stage"], rows=rows),
        remediation=(
            "Check the plants' tailrace polynomials and rated heads in hidr.dat "
            "and polinjus: a tailrace that rises steeply with the flow keeps the "
            "head and the flow from settling."
        ),
    )

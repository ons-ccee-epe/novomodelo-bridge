"""Long-term-mean (MLT) inflow incrementalization for the boundary FCF fold.

The boundary FCF's inflow-lag term (``pi_qafl``) prices the inflow *deviation
from the seasonal mean* — the source model's PAR(p) state is the increment
``Q - mu`` about the long-term mean ``mu`` (the MLT), not the absolute inflow
(reference manual §5.1.9.2: the energies are computed on *incremental*
inflows). novomodelo evaluates the loaded cut at its **raw** inflow-lag state, so the
mean must be folded into the cut RHS (``fcf/mapper.py::map_boundary_cuts``); this
module supplies that ``mu`` per plant per calendar month.

``mlt.dat`` (:class:`idecomp.decomp.Mlt`) carries the seasonal means per gauging
station (``posto``) in the **natural** basis — ``mu_nat[posto]`` accumulates down
each cascade (verified on the deck: every downstream station's MLT ≥ the sum of
its upstream stations'). The lag state, like every other inflow quantity in a
DECOMP deck, is **incremental** (each ``posto`` column of the inflow file already
holds the plant's own local increment — see
:func:`novomodelo_bridge.decomp.scenarios._incremental_values`). So the natural MLT
must be incrementalized to the same basis before it can be subtracted from the
raw incremental lag state:

    mu_incr[plant] = mu_nat[posto] - Σ_{upstream postos} mu_nat[upstream]

The attribution is :func:`novomodelo_bridge.core.incremental_inflow.
incremental_by_plant`, the one the source-model track uses, over the
*effective* operated cascade. DECOMP operates every plant (no
fictitious/non-operated intermediates), so the operated cascade *is* the full
topology.

**Zero-inflow stations.** A plant on the artificial gauge, which a
diversion-fed plant uses, has zero incremental inflow. So does a plant on a
station whose MLT is zero across all twelve months: subtracting a real
upstream station from that zero base would fabricate a spurious negative
increment (validated
against the deck's ``observacoes``: those plants read exactly 0).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from novomodelo_bridge.core.diagnostics import emit
from novomodelo_bridge.core.incremental_inflow import (
    artificial_pass_through,
    artificial_pass_through_diagnostic,
    incremental_by_plant,
)
from novomodelo_bridge.decomp.scenarios import _incremental_context

if TYPE_CHECKING:
    import pandas as pd

    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
    from novomodelo_bridge.decomp.id_map import DecompIdMap

_LOG = logging.getLogger(__name__)

#: The twelve calendar months an MLT table carries, in ``mes`` order.
_MONTHS: tuple[int, ...] = tuple(range(1, 13))


def _natural_by_posto(mlt_valores: pd.DataFrame) -> dict[int, dict[int, float]]:
    """Index ``mlt.dat`` as ``{posto: {calendar_month: mu_natural_m3s}}``.

    ``mlt_valores`` (:attr:`idecomp.decomp.Mlt.valores`) is a 12-row frame: a
    ``mes`` column plus one column per gauging station, the column *name* being
    the station code as a string. A station absent from the table simply never
    appears in the returned map (its ``mu`` defaults to zero downstream).
    """
    natural: dict[int, dict[int, float]] = {}
    month_of_row = {int(month): idx for idx, month in enumerate(mlt_valores["mes"])}
    for column in mlt_valores.columns:
        if column == "mes":
            continue
        try:
            posto = int(column)
        except (ValueError, TypeError):
            continue
        series = mlt_valores[column]
        natural[posto] = {
            month: float(series.iloc[row]) for month, row in month_of_row.items()
        }
    return natural


def build_incremental_mlt(
    mlt_valores: pd.DataFrame,
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
) -> dict[int, dict[int, float]]:
    """Per-plant incremental long-term-mean inflows, ``{hydro_id: {month: mu}}``.

    Returns the seasonal mean *incremental* inflow (m³/s) for every operated
    plant (0-based novomodelo ``hydro_id``) and calendar month (1..12), the ``mu`` the
    boundary FCF fold subtracts from the raw inflow-lag state
    (:func:`novomodelo_bridge.core.incremental_inflow.incremental_by_plant`). A plant
    on the artificial gauge, or on a station whose natural MLT is zero across
    all twelve months, is pinned to ``0``.

    A plant whose station is absent from ``mlt_valores`` is omitted from the
    result: the fold then leaves its lag terms unshifted (``mu = 0``), the same
    conservative default as an unmatched station downstream.
    """
    natural = {
        posto: np.array([months.get(month, 0.0) for month in _MONTHS])
        for posto, months in _natural_by_posto(mlt_valores).items()
    }
    zero_gauges = {posto for posto, series in natural.items() if not series.any()}
    station_by_code, parents = _incremental_context(effective, id_map)
    posto_of = {code: int(station) for code, station in station_by_code.items()}

    pairs = artificial_pass_through(posto_of, parents, id_map.hydro_codes)
    if pairs:
        name_of = {
            code: str(effective.base.loc[code, "nome_usina"]).strip()
            for pair in pairs
            for code in pair
        }
        emit(
            artificial_pass_through_diagnostic(
                [(code, name_of[code], up, name_of[up]) for code, up in pairs]
            ),
            logger=_LOG,
        )

    by_code = incremental_by_plant(
        natural, posto_of, parents, id_map.hydro_codes, zero_gauges=zero_gauges
    )
    return {
        id_map.hydro_id(code): {
            month: float(value) for month, value in zip(_MONTHS, series, strict=True)
        }
        for code, series in by_code.items()
    }


def coupling_lag_means(
    incremental_mlt: dict[int, dict[int, float]],
    coupling_month: int,
) -> dict[int, tuple[float, ...]]:
    """Align the incremental MLT to the boundary cut's inflow-lag axis.

    The boundary cut's inflow-lag coefficient ``pi_qafl`` at depth ``d`` (1..12)
    prices the inflow ``d`` calendar months before the coupling stage's month, so
    ``depth d -> calendar month (coupling_month - d)`` wrapped into 1..12
    (verified empirically: at an April coupling, depth 1 aligns to March, …,
    depth 12 to the previous April). Returns ``{hydro_id: (mu_depth1, …,
    mu_depth12)}`` — one 12-vector per plant, indexed by lag depth minus one,
    ready for :func:`novomodelo_bridge.decomp.fcf.mapper.map_boundary_cuts`'s
    ``inflow_lag_means`` argument.
    """
    aligned: dict[int, tuple[float, ...]] = {}
    for hydro_id, months in incremental_mlt.items():
        aligned[hydro_id] = tuple(
            months.get(((coupling_month - depth - 2) % 12) + 1, 0.0)
            for depth in range(12)
        )
    return aligned

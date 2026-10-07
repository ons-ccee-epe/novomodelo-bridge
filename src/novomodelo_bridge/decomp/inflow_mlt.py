"""Long-term-mean (MLT) inflow incrementalization for the boundary FCF fold.

The boundary FCF's inflow-lag term (``pi_qafl``) prices the inflow *deviation
from the seasonal mean* — the source model's PAR(p) state is the increment
``Q - mu`` about the long-term mean ``mu`` (the MLT), not the absolute inflow
(reference manual §5.1.9.2: the energies are computed on *incremental*
inflows). cobre evaluates the loaded cut at its **raw** inflow-lag state, so the
mean must be folded into the cut RHS (``fcf/mapper.py::map_boundary_cuts``); this
module supplies that ``mu`` per plant per calendar month.

``mlt.dat`` (:class:`idecomp.decomp.Mlt`) carries the seasonal means per gauging
station (``posto``) in the **natural** basis — ``mu_nat[posto]`` accumulates down
each cascade (verified on the deck: every downstream station's MLT ≥ the sum of
its upstream stations'). The lag state, like every other inflow quantity in a
DECOMP deck, is **incremental** (each ``posto`` column of the inflow file already
holds the plant's own local increment — see
:func:`cobre_bridge.decomp.scenarios._incremental_values`). So the natural MLT
must be incrementalized to the same basis before it can be subtracted from the
raw incremental lag state:

    mu_incr[posto] = mu_nat[posto] - Σ_{upstream postos} mu_nat[upstream]

The upstream set is built in **posto space** off the *effective* operated
cascade (mirroring the source-model side's
:func:`cobre_bridge.newave.converters.stochastic._build_upstream_postos`): every
operated plant's ``posto -> downstream-operated posto`` edge, deduplicated (plants
sharing a station collapse to one node) and self-loops skipped. DECOMP operates
every plant (no fictitious/non-operated intermediates), so the operated cascade
*is* the full topology — no walk-through step is needed, unlike the source-model
map.

**Zero-inflow (artificial) stations.** A diversion-fed plant is gauged on an
artificial station whose natural MLT is identically zero (e.g. the Billings /
Paraíba-do-Sul light system and the Belo Monte / Volta-Grande diversion all
share one such station): its water arrives via routing/``desvio`` channels, not
natural drainage, so its incremental inflow is zero. Subtracting a real upstream
station from that zero base would fabricate a spurious negative increment, so a
station whose MLT is zero across all twelve months is pinned to ``mu_incr = 0``
(validated against the deck's ``observacoes``: those plants read exactly 0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cobre_bridge.decomp.scenarios import _incremental_context

if TYPE_CHECKING:
    import pandas as pd

    from cobre_bridge.decomp.converters.cadastro import EffectiveCadastro
    from cobre_bridge.decomp.id_map import DecompIdMap

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


def _posto_upstream(
    effective: EffectiveCadastro, id_map: DecompIdMap
) -> dict[int, set[int]]:
    """``{downstream_posto: {upstream_posto, ...}}`` for the operated cascade.

    Built in posto space off :func:`_incremental_context`'s per-plant station
    column and direct operated-upstream links: for every operated plant with a
    direct operated-upstream sibling, add the edge ``upstream_posto ->
    downstream_posto`` (self-loops — a plant sharing its upstream's station —
    skipped, and duplicate edges collapsed by the ``set``). Mirrors the
    source-model side's :func:`cobre_bridge.newave.converters.stochastic.
    _build_upstream_postos`, minus its NE/NC walk-through step (DECOMP operates
    every plant, so no non-inflow intermediates sit between operated nodes).
    """
    station_by_code, parents = _incremental_context(effective, id_map)
    posto_of = {code: int(station_by_code[code]) for code in id_map.hydro_codes}
    upstream: dict[int, set[int]] = {}
    for downstream_code, upstream_codes in parents.items():
        downstream_posto = posto_of[downstream_code]
        for upstream_code in upstream_codes:
            upstream_posto = posto_of[upstream_code]
            if upstream_posto != downstream_posto:
                upstream.setdefault(downstream_posto, set()).add(upstream_posto)
    return upstream


def build_incremental_mlt(
    mlt_valores: pd.DataFrame,
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
) -> dict[int, dict[int, float]]:
    """Per-plant incremental long-term-mean inflows, ``{hydro_id: {month: mu}}``.

    Returns the seasonal mean *incremental* inflow (m³/s) for every operated
    plant (0-based cobre ``hydro_id``) and calendar month (1..12), the ``mu`` the
    boundary FCF fold subtracts from the raw inflow-lag state. Computed as
    ``mu_nat[posto] - Σ_{upstream postos} mu_nat`` (see the module docstring),
    with a station whose natural MLT is zero across all twelve months pinned to
    ``0`` (an artificial diversion-fed station carries no natural increment).

    A plant whose station is absent from ``mlt_valores`` is omitted from the
    result: the fold then leaves its lag terms unshifted (``mu = 0``), the same
    conservative default as an unmatched station downstream.
    """
    natural = _natural_by_posto(mlt_valores)
    upstream = _posto_upstream(effective, id_map)
    station_by_code, _ = _incremental_context(effective, id_map)

    zero_posto = {
        posto
        for posto, months in natural.items()
        if all(value == 0.0 for value in months.values())
    }

    means: dict[int, dict[int, float]] = {}
    for code in id_map.hydro_codes:
        posto = int(station_by_code[code])
        posto_natural = natural.get(posto)
        if posto_natural is None:
            continue
        hydro_id = id_map.hydro_id(code)
        if posto in zero_posto:
            means[hydro_id] = {month: 0.0 for month in _MONTHS}
            continue
        incremental: dict[int, float] = {}
        for month in _MONTHS:
            value = posto_natural.get(month, 0.0)
            for upstream_posto in upstream.get(posto, ()):
                value -= natural.get(upstream_posto, {}).get(month, 0.0)
            incremental[month] = value
        means[hydro_id] = incremental
    return means


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
    ready for :func:`cobre_bridge.decomp.fcf.mapper.map_boundary_cuts`'s
    ``inflow_lag_means`` argument.
    """
    aligned: dict[int, tuple[float, ...]] = {}
    for hydro_id, months in incremental_mlt.items():
        aligned[hydro_id] = tuple(
            months.get(((coupling_month - depth - 2) % 12) + 1, 0.0)
            for depth in range(12)
        )
    return aligned

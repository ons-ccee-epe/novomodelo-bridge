"""GNL ring round-trip identity test for the boundary FCF mapper.

``map_boundary_cuts(..., gnl_plan=...)`` prices each dated
``AnticipatedThermalState`` slot as ``Σ_p pi_gnl[col(s, p, l)] · h · f_p``:
``l`` the cut lag of the slot's delivery month, ``h`` the slot's own hours and
``f_p`` the source's patamar share of that month. A slot whose month the cut
does not price stays ``0.0``, and an already-commanded delivery folds into the
intercept. This module is the GNL counterpart of ``test_fcf_roundtrip.py``:
deck-independent, tier 2 (the ``novomodelo-python`` writer binding, no solver
binary).

The oracle re-derives the flat ``pi_gnl`` layout locally (:func:`_col`) and
finds every slot by scanning the *reloaded* ``entity_manifest``
(:func:`_slot_index`), so it never routes through the mapper's own
bookkeeping.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from novomodelo_bridge.core.units import MEAN_MONTH_HOURS
from novomodelo_bridge.decomp.fcf.cortes import BoundaryCuts
from novomodelo_bridge.decomp.fcf.mapper import (
    GnlCommandedDelivery,
    GnlDeliveryMonth,
    GnlRingPlan,
    GnlThermalTarget,
    month_key,
)
from tests._fcf_fixtures import (
    make_cortes_header,
    make_cut_record,
    make_id_map,
    make_manifest,
    make_slot,
    synthetic_roundtrip,
)
from tests.conftest import requires_novomodelo_python, requires_writer_binding

#: novomodelo `policy.fbs` entity_type codes, re-declared so the oracle's slot
#: lookup never depends on the mapper under test.
_HYDRO_STORAGE = 0
_ANTICIPATED_THERMAL_STATE = 2

#: 4 submercados x 3 patamares x 2 lags (width-24 `pi_gnl`), the shape of
#: the `mar-26-rv2` deck.
_N_PATAMARES = 3
_LAG_MAXIMO_GNL = 2

#: Distinct per column triple, so a wrong-column placement cannot match.
_SUBMERCADO_1_LAG_2_VALUES = (10.0, 20.0, 30.0)
_SUBMERCADO_3_LAG_1_VALUES = (100.0, 200.0, 300.0)

_MAY_FRACTIONS = (0.1, 0.3, 0.6)
_JUNE_FRACTIONS = (0.2, 0.5, 0.3)


def _col(submercado: int, patamar: int, lag: int) -> int:
    """Flat `pi_gnl` column for 1-based (submercado, patamar, lag)."""
    return ((submercado - 1) * _N_PATAMARES + (patamar - 1)) * _LAG_MAXIMO_GNL + (
        lag - 1
    )


def _slot_index(
    manifest: Sequence[Mapping[str, Any]],
    entity_type: int,
    entity_id: int,
    subindex: int,
) -> int | None:
    """Position of the reloaded manifest slot keyed by its triple."""
    for position, slot in enumerate(manifest):
        if (
            slot["entity_type"] == entity_type
            and slot["entity_id"] == entity_id
            and slot["subindex"] == subindex
        ):
            return position
    return None


def _make_pi_gnl() -> tuple[float, ...]:
    pi_gnl = [0.0] * (4 * _N_PATAMARES * _LAG_MAXIMO_GNL)
    for patamar, value in enumerate(_SUBMERCADO_1_LAG_2_VALUES, start=1):
        pi_gnl[_col(1, patamar, 2)] = value
    for patamar, value in enumerate(_SUBMERCADO_3_LAG_1_VALUES, start=1):
        pi_gnl[_col(3, patamar, 1)] = value
    return tuple(pi_gnl)


def _priced(
    pi_gnl: Sequence[float],
    submercado: int,
    lag: int,
    weights: Sequence[float],
) -> float:
    return math.fsum(
        pi_gnl[_col(submercado, patamar, lag)] * weight
        for patamar, weight in enumerate(weights, start=1)
    )


@requires_novomodelo_python
@requires_writer_binding
def test_synthetic_gnl_roundtrip_coefficient_identity(tmp_path: Path) -> None:
    """Priced, unpriced, and undated ring slots plus the commanded fold."""
    header = make_cortes_header(
        (),
        lag_maximo_gnl=_LAG_MAXIMO_GNL,
        n_patamares=_N_PATAMARES,
        submercado_codes=(1, 2, 3, 4),
    )
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # satisfies the storage guard
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0),  # undated
            make_slot(
                _ANTICIPATED_THERMAL_STATE,
                94,
                1,
                interval_start=20260606,
                interval_end=20260630,
            ),  # June: lag 2
            make_slot(
                _ANTICIPATED_THERMAL_STATE,
                95,
                0,
                interval_start=20260509,
                interval_end=20260516,
            ),  # May: lag 1
            make_slot(
                _ANTICIPATED_THERMAL_STATE,
                95,
                1,
                interval_start=20260425,
                interval_end=20260502,
            ),  # April: the cut's own month, unpriced
        ]
    )
    pi_gnl = _make_pi_gnl()
    rhs = 4.0
    record = make_cut_record(pi_varm=(), pi_qafl=(), pi_gnl=pi_gnl, rhs=rhs)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    commanded_mw = (500.0, 250.0, 0.0)
    gnl_plan = GnlRingPlan(
        targets=(GnlThermalTarget(94, 1), GnlThermalTarget(95, 3)),
        months={
            month_key(2026, 5): GnlDeliveryMonth(1, _MAY_FRACTIONS),
            month_key(2026, 6): GnlDeliveryMonth(2, _JUNE_FRACTIONS),
        },
        commanded=(GnlCommandedDelivery(3, month_key(2026, 5), commanded_mw, 168.0),),
    )

    reloaded = synthetic_roundtrip(
        tmp_path / "boundary",
        cuts,
        manifest,
        make_id_map(()),
        couplings=(),
        gnl_plan=gnl_plan,
    )

    entry = reloaded["stage_cuts"][0]
    reloaded_manifest = entry["entity_manifest"]
    (cut,) = (cut for cut in entry["cuts"] if cut["is_active"])
    coefficients = cut["coefficients"]

    def at(entity_id: int, subindex: int) -> float:
        position = _slot_index(
            reloaded_manifest, _ANTICIPATED_THERMAL_STATE, entity_id, subindex
        )
        assert position is not None
        return coefficients[position]

    june_hours = 24 * 24.0
    assert math.isclose(
        at(94, 1),
        _priced(pi_gnl, 1, 2, [june_hours * f for f in _JUNE_FRACTIONS]),
        rel_tol=1e-12,
    )
    assert math.isclose(
        at(95, 0),
        _priced(pi_gnl, 3, 1, [168.0 * f for f in _MAY_FRACTIONS]),
        rel_tol=1e-12,
    )
    assert at(95, 1) == 0.0
    assert at(94, 0) == 0.0

    commanded_fold = _priced(
        pi_gnl,
        3,
        1,
        [168.0 * f * mw for f, mw in zip(_MAY_FRACTIONS, commanded_mw, strict=True)],
    )
    assert math.isclose(
        cut["intercept"], rhs * MEAN_MONTH_HOURS + commanded_fold, rel_tol=1e-12
    )

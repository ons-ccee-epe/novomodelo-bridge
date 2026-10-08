"""GNL ring round-trip identity test for the boundary FCF mapper.

``map_boundary_cuts(..., gnl_plan=GnlRingPlan | None)`` places the per-block
hours-weighted patamar sum ``Σ_p pi_gnl[col(s, p, l)] · h_p`` (``math.fsum``)
onto the terminal ring's *covered* dated ``AnticipatedThermalState`` slots,
and the covered-lane filter drops that sum to ``0.0`` on any dated slot whose
``interval_start`` falls before the post-study horizon. Two existing
tests in ``tests/decomp/test_fcf_roundtrip.py`` prove the mapper's
storage/lag legs end to end via a synthetic ``map -> write -> load_policy``
round trip cross-checked against a *non-circular* oracle; this module is the
GNL ring's equivalent — deck-independent, tier-2 (needs the optional
``novomodelo-python`` wheel and its writer binding, but no real deck and no real
solver binary).

The oracle here never calls
:func:`novomodelo_bridge.decomp.fcf.mapper.map_boundary_cuts`'s private
``col``/``_resolve_gnl_targets``/``_index_gnl_ring`` helpers — routing the
oracle through the code under test would make this a circular,
Python-vs-Python check. Instead it re-derives the flat ``pi_gnl`` column
layout locally (:func:`_col`) and locates every slot by a linear scan over
the *reloaded* ``entity_manifest`` (:func:`_slot_index`) — the ground truth
for where the checkpoint writer actually placed each coefficient, never the
mapper's own bookkeeping. Mirrors ``test_fcf_roundtrip.py``'s
``_slot_index``/oracle discipline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from novomodelo_bridge.decomp.fcf.cortes import BoundaryCuts
from novomodelo_bridge.decomp.fcf.mapper import GnlRingPlan, GnlThermalTarget
from tests._fcf_fixtures import (
    make_cortes_header,
    make_cut_record,
    make_id_map,
    make_manifest,
    make_slot,
    synthetic_roundtrip,
)
from tests.conftest import requires_novomodelo_python, requires_writer_binding

#: novomodelo `policy.fbs` entity_type codes (mirrors `fcf/mapper.py`'s private
#: constants of the same name/value — re-declared here, never imported, so
#: this oracle's slot lookup never depends on the mapper under test).
_HYDRO_STORAGE = 0
_ANTICIPATED_THERMAL_STATE = 2

#: The synthetic case's GNL shape — 4 submercados x 3 patamares x 2 lags
#: (width-24 `pi_gnl`), mirroring `mar-26-rv2`'s real shape.
_N_PATAMARES = 3
_LAG_MAXIMO_GNL = 2

#: Distinct, nonzero source values at the two targeted (submercado, lag)
#: column triples — distinct so a wrong-column placement (e.g. reading the
#: submercado-3/lag-1 columns onto the submercado-1/lag-2 slot) is caught
#: rather than accidentally matching.
_SUBMERCADO_1_LAG_2_VALUES = (10.0, 20.0, 30.0)
_SUBMERCADO_3_LAG_1_VALUES = (100.0, 200.0, 300.0)

#: The post-study horizon: thermal 94's dated slot (20260501) is covered,
#: thermal 95's (20260401) is not.
_POST_HORIZON_START = 20260501

#: Non-uniform coupling per-block hours (patamar order) — deliberately
#: distinct across patamares so the round trip actually exercises the
#: per-block weighting, not merely the uniform-split special
#: case that collapses to the pre-fix plain-sum-times-total-hours value.
_COUPLING_BLOCK_HOURS = (100.0, 200.0, 300.0)


def _col(
    submercado: int,
    patamar: int,
    lag: int,
    *,
    n_patamares: int,
    lag_maximo_gnl: int,
) -> int:
    """Flat `pi_gnl` column for (submercado, patamar, lag), 1-based axes.

    Re-derives `fcf/mapper.py::_resolve_gnl_targets`'s private `col()`
    independently — never imported — so this oracle isn't a circular
    Python-vs-Python check.
    """
    return ((submercado - 1) * n_patamares + (patamar - 1)) * lag_maximo_gnl + (lag - 1)


def _slot_index(
    manifest: Sequence[Mapping[str, Any]],
    entity_type: int,
    entity_id: int,
    subindex: int,
) -> int | None:
    """Position of the manifest slot keyed `(entity_type, entity_id, subindex)`.

    A linear scan over the *reloaded* `entity_manifest` — the ground truth
    for where the checkpoint writer actually placed each mapped
    coefficient — never `fcf.mapper._index_manifest`/`_index_gnl_ring`,
    which are the mapper's own bookkeeping and would make this oracle
    circular. Mirrors `tests/decomp/test_fcf_roundtrip.py::_slot_index`.
    """
    for position, slot in enumerate(manifest):
        if (
            slot["entity_type"] == entity_type
            and slot["entity_id"] == entity_id
            and slot["subindex"] == subindex
        ):
            return position
    return None


def _make_pi_gnl() -> tuple[float, ...]:
    """The synthetic case's width-24 `pi_gnl`, nonzero at exactly the two
    targeted (submercado, lag) column triples.
    """
    width = 4 * _N_PATAMARES * _LAG_MAXIMO_GNL
    pi_gnl = [0.0] * width
    for patamar, value in zip((1, 2, 3), _SUBMERCADO_1_LAG_2_VALUES, strict=True):
        pi_gnl[
            _col(
                1,
                patamar,
                2,
                n_patamares=_N_PATAMARES,
                lag_maximo_gnl=_LAG_MAXIMO_GNL,
            )
        ] = value
    for patamar, value in zip((1, 2, 3), _SUBMERCADO_3_LAG_1_VALUES, strict=True):
        pi_gnl[
            _col(
                3,
                patamar,
                1,
                n_patamares=_N_PATAMARES,
                lag_maximo_gnl=_LAG_MAXIMO_GNL,
            )
        ] = value
    return tuple(pi_gnl)


@requires_novomodelo_python
@requires_writer_binding
def test_synthetic_gnl_roundtrip_coefficient_identity(tmp_path: Path) -> None:
    """AC 1/2/3 — the GNL ring's covered/non-covered/sentinel identity, no
    deck and no novomodelo binary.

    Maps a synthetic 3-slot `AnticipatedThermalState` ring (a thermal-94
    sentinel, a thermal-94 covered dated slot, and a thermal-95 non-covered
    dated slot) with a two-target `GnlRingPlan`, round-trips it via
    `synthetic_roundtrip`, and checks the reloaded coefficients — located by
    scanning the reloaded `entity_manifest`, never the mapper's own
    bookkeeping — against this module's independent `_col`/`math.fsum`
    oracle: the covered slot equals the per-block hours-weighted sum of its
    three source `pi_gnl` columns against `_COUPLING_BLOCK_HOURS`;
    the non-covered and sentinel slots are exactly `0.0`.
    """
    header = make_cortes_header(
        (),
        lag_maximo_gnl=_LAG_MAXIMO_GNL,
        n_patamares=_N_PATAMARES,
        submercado_codes=(1, 2, 3, 4),
    )
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # dummy; satisfies the storage guard
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0),  # sentinel (undated)
            make_slot(
                _ANTICIPATED_THERMAL_STATE, 94, 1, interval_start=20260501
            ),  # covered
            make_slot(
                _ANTICIPATED_THERMAL_STATE, 95, 0, interval_start=20260401
            ),  # non-covered
        ]
    )
    pi_gnl = _make_pi_gnl()
    record = make_cut_record(pi_varm=(), pi_qafl=(), pi_gnl=pi_gnl, rhs=0.0, cut_id=1)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    id_map = make_id_map(())
    gnl_plan = GnlRingPlan(
        (
            GnlThermalTarget(94, submercado=1, nl_lag=2),
            GnlThermalTarget(95, submercado=3, nl_lag=1),
        ),
        post_horizon_start=_POST_HORIZON_START,
    )

    reloaded = synthetic_roundtrip(
        tmp_path / "boundary",
        cuts,
        manifest,
        id_map,
        gnl_plan=gnl_plan,
        coupling_block_hours=_COUPLING_BLOCK_HOURS,
    )

    entry = reloaded["stage_cuts"][0]
    reloaded_manifest = entry["entity_manifest"]
    active_cuts = [cut for cut in entry["cuts"] if cut["is_active"]]
    assert len(active_cuts) == 1
    coefficients = active_cuts[0]["coefficients"]

    covered_position = _slot_index(reloaded_manifest, _ANTICIPATED_THERMAL_STATE, 94, 1)
    non_covered_position = _slot_index(
        reloaded_manifest, _ANTICIPATED_THERMAL_STATE, 95, 0
    )
    sentinel_position = _slot_index(
        reloaded_manifest, _ANTICIPATED_THERMAL_STATE, 94, 0
    )
    assert covered_position is not None
    assert non_covered_position is not None
    assert sentinel_position is not None

    # The authored coefficient is the per-block hours-weighted pi_gnl sum
    # `Σ_p pi_gnl[col(1, p, 2)] · coupling_block_hours[p]`, matching the
    # mapper's `math.fsum(pi_gnl[c] * h for c, h in zip(...))` order exactly.
    expected_covered_sum = math.fsum(
        pi_gnl[
            _col(
                1,
                patamar,
                2,
                n_patamares=_N_PATAMARES,
                lag_maximo_gnl=_LAG_MAXIMO_GNL,
            )
        ]
        * hours
        for patamar, hours in zip(
            range(1, _N_PATAMARES + 1), _COUPLING_BLOCK_HOURS, strict=True
        )
    )

    assert coefficients[covered_position] == expected_covered_sum
    assert coefficients[non_covered_position] == 0.0
    assert coefficients[sentinel_position] == 0.0

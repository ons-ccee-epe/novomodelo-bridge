"""Tests for the boundary-cut manifest-to-manifest mapper (``fcf/mapper.py``)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import pytest

from novomodelo_bridge.core.units import C_M3S2HM3, MONTH_HOURS
from novomodelo_bridge.decomp.fcf.bootstrap import TerminalManifest
from novomodelo_bridge.decomp.fcf.cortes import BoundaryCuts
from novomodelo_bridge.decomp.fcf.mapper import (
    DroppedTerm,
    GnlRingPlan,
    GnlThermalTarget,
    _resolve_gnl_targets,
    map_boundary_cuts,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from tests._fcf_fixtures import (
    make_boundary_cuts,
    make_cortes_header,
    make_cut_record,
    make_id_map,
    make_manifest,
    make_slot,
    synthetic_roundtrip,
)
from tests.conftest import requires_novomodelo_python

# Inflow-lag (pi_qafl) coefficients take an extra × C_M3S2HM3 beyond MONTH_HOURS
# (novomodelo's inflow-lag state is m³/s, not Hm³); storage/rhs take × MONTH_HOURS.
_LAG_FACTOR = MONTH_HOURS * C_M3S2HM3

# `_make_gnl_ring_fixture` always builds an `n_patamares=3` header; a uniform
# split of `MONTH_HOURS` across those 3 coupling blocks reproduces the
# plain-sum-times-total-hours GNL value exactly ÷ 3 (see
# `test_map_gnl_uniform_blocks_is_sum_over_n_patamares`).
_UNIFORM_GNL_BLOCK_HOURS = (MONTH_HOURS / 3, MONTH_HOURS / 3, MONTH_HOURS / 3)

# novomodelo `policy.fbs` entity_type codes — a stable external contract, restated
# locally rather than importing the mapper module's private constants.
_HYDRO_STORAGE = 0
_HYDRO_INFLOW_LAG = 1
_ANTICIPATED_THERMAL_STATE = 2
_HYDRO_TRANSIT_BUCKET = 3


def _gnl_row(width: int, nonzero: Mapping[int, float]) -> tuple[float, ...]:
    """A `pi_gnl` flat vector of `width` zeros with `nonzero` columns set."""
    row = [0.0] * width
    for column, value in nonzero.items():
        row[column] = value
    return tuple(row)


def test_map_storage_places_pi_varm_at_hydro_slots() -> None:
    id_map = make_id_map((10, 20))
    # Plant 10 (hydro_id 0) carries a full-width HydroInflowLag family
    # (subindices 0..11) so the default identity `lag_slot_of` (d -> d-1)
    # stays in range for every calendar depth 1..12 (all zero here, per
    # "lags zero" in the AC); plant 20 has no lag exposure in this fixture.
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # position 0: plant 10's storage
            make_slot(_HYDRO_STORAGE, 1, 0),  # position 1: plant 20's storage
            *[
                make_slot(_HYDRO_INFLOW_LAG, 0, lag)  # positions 2..13
                for lag in range(12)
            ],
            make_slot(_HYDRO_TRANSIT_BUCKET, 1, 2),  # position 14
        ]
    )
    cuts = make_boundary_cuts(
        (10, 20), (make_cut_record(pi_varm=(3.0, 5.0), rhs=100.0),)
    )

    result = map_boundary_cuts(cuts, manifest, id_map, cost_unit_hours=MONTH_HOURS)

    assert result.dropped == ()
    assert len(result.cuts) == 1
    mapped = result.cuts[0]
    assert len(mapped.coefficients) == manifest.state_dimension
    # Storage coefficients and the intercept are scaled to novomodelo cost units
    # by MONTH_HOURS (the source's ($·mês)/h -> $ conversion); zero slots stay 0.
    assert mapped.coefficients[0] == pytest.approx(3.0 * MONTH_HOURS)
    assert mapped.coefficients[1] == pytest.approx(5.0 * MONTH_HOURS)
    assert mapped.coefficients[2:] == (0.0,) * 13
    assert mapped.intercept == pytest.approx(100.0 * MONTH_HOURS)


def test_map_drops_source_only_plant_with_diagnostic_record() -> None:
    # Plant 20 is present in the source but unknown to the target id_map.
    id_map = make_id_map((10,))
    manifest = make_manifest([make_slot(_HYDRO_STORAGE, 0, 0)])
    cuts = make_boundary_cuts((10, 20), (make_cut_record(pi_varm=(3.0, 7.0)),))

    result = map_boundary_cuts(cuts, manifest, id_map, cost_unit_hours=MONTH_HOURS)

    # `dropped` reports the source coefficient in source units (diagnostic,
    # never scaled); the kept storage coefficient is scaled by MONTH_HOURS.
    assert result.dropped == (DroppedTerm(plant_code=20, beta=7.0),)
    assert len(result.cuts) == 1
    assert result.cuts[0].coefficients == (pytest.approx(3.0 * MONTH_HOURS),)


def test_map_lags_one_to_one() -> None:
    id_map = make_id_map((10,))
    # Full-width HydroInflowLag family (subindices 0..11) so identity
    # `lag_slot_of` stays in range for every calendar depth 1..12; only
    # depths 1 and 2 carry a nonzero source coefficient.
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # position 0
            *[
                make_slot(_HYDRO_INFLOW_LAG, 0, lag)  # positions 1..12
                for lag in range(12)
            ],
        ]
    )
    lags = (1.0, 2.0) + (0.0,) * 10
    cuts = make_boundary_cuts(
        (10,), (make_cut_record(pi_varm=(0.0,), pi_qafl=(lags,)),)
    )

    result = map_boundary_cuts(
        cuts, manifest, id_map, cost_unit_hours=MONTH_HOURS, lag_slot_of=lambda d: d - 1
    )

    assert result.dropped == ()
    mapped = result.cuts[0]
    assert mapped.coefficients[1] == pytest.approx(1.0 * _LAG_FACTOR)  # depth 1
    assert mapped.coefficients[2] == pytest.approx(2.0 * _LAG_FACTOR)  # depth 2


def test_map_inflow_lag_means_folds_rhs() -> None:
    # The seasonal-mean fold reduces the intercept by Σ placed_lag_coef·μ, so
    # the loaded cut prices the raw lag state as the deviation Q - μ. The
    # coefficients themselves are untouched.
    id_map = make_id_map((10,))
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),
            *[make_slot(_HYDRO_INFLOW_LAG, 0, lag) for lag in range(12)],
        ]
    )
    lags = (1.0, 2.0) + (0.0,) * 10
    cuts = make_boundary_cuts(
        (10,), (make_cut_record(pi_varm=(0.0,), pi_qafl=(lags,), rhs=5.0),)
    )
    means = {0: (100.0, 50.0) + (0.0,) * 10}

    result = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        lag_slot_of=lambda d: d - 1,
        inflow_lag_means=means,
    )

    mapped = result.cuts[0]
    # Coefficients unchanged by the fold.
    assert mapped.coefficients[1] == pytest.approx(1.0 * _LAG_FACTOR)
    assert mapped.coefficients[2] == pytest.approx(2.0 * _LAG_FACTOR)
    # Intercept reduced by Σ (placed_coef · μ) over the two nonzero depths.
    expected_fold = 1.0 * _LAG_FACTOR * 100.0 + 2.0 * _LAG_FACTOR * 50.0
    assert mapped.intercept == pytest.approx(5.0 * MONTH_HOURS - expected_fold)


def test_map_inflow_lag_means_none_is_noop() -> None:
    # Default (no means) leaves the intercept at the plain scaled rhs — the
    # pre-fold behaviour — byte-for-byte.
    id_map = make_id_map((10,))
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),
            *[make_slot(_HYDRO_INFLOW_LAG, 0, lag) for lag in range(12)],
        ]
    )
    lags = (1.0, 2.0) + (0.0,) * 10
    cuts = make_boundary_cuts(
        (10,), (make_cut_record(pi_varm=(0.0,), pi_qafl=(lags,), rhs=5.0),)
    )

    result = map_boundary_cuts(
        cuts, manifest, id_map, cost_unit_hours=MONTH_HOURS, lag_slot_of=lambda d: d - 1
    )

    assert result.cuts[0].intercept == pytest.approx(5.0 * MONTH_HOURS)


def test_map_inflow_lag_means_only_folds_placed_lags() -> None:
    # A plant with a storage slot but NO inflow-lag slots in the manifest has no
    # lag coefficient placed, so its mean must not fold — the fold can never
    # reference a term novomodelo will not apply. Plant 10 (hydro_id 0) carries the
    # full 12-slot lag family; plant 20 (hydro_id 1) has storage only.
    id_map = make_id_map((10, 20))
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),
            make_slot(_HYDRO_STORAGE, 1, 0),
            *[make_slot(_HYDRO_INFLOW_LAG, 0, lag) for lag in range(12)],
        ]
    )
    lags10 = (1.0,) + (0.0,) * 11
    lags20 = (4.0,) + (0.0,) * 11
    cuts = make_boundary_cuts(
        (10, 20),
        (make_cut_record(pi_varm=(0.0, 0.0), pi_qafl=(lags10, lags20), rhs=5.0),),
    )
    # Plant 20 (hydro_id 1) has a μ, but no lag slot → must not fold.
    means = {0: (100.0,) + (0.0,) * 11, 1: (999.0,) + (0.0,) * 11}

    result = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        lag_slot_of=lambda d: d - 1,
        inflow_lag_means=means,
    )

    # Only plant 10's depth-1 lag is placed and folded; plant 20's μ is inert.
    expected_fold = 1.0 * _LAG_FACTOR * 100.0
    assert result.cuts[0].intercept == pytest.approx(5.0 * MONTH_HOURS - expected_fold)


def test_map_inflow_lag_means_skips_dropped_plant() -> None:
    # A source-only plant (dropped, no target storage slot) must not fold its
    # μ: its lag terms are never placed, so its mean can carry no RHS shift.
    id_map = make_id_map((10,))  # plant 20 is unknown → dropped
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),
            *[make_slot(_HYDRO_INFLOW_LAG, 0, lag) for lag in range(12)],
        ]
    )
    lags10 = (1.0,) + (0.0,) * 11
    lags20 = (7.0,) + (0.0,) * 11
    cuts = make_boundary_cuts(
        (10, 20),
        (make_cut_record(pi_varm=(0.0, 0.0), pi_qafl=(lags10, lags20), rhs=5.0),),
    )
    # hydro_id 1 (plant 20) has a μ, but plant 20 is dropped, so it must not
    # affect the fold; only plant 10's (hydro_id 0) depth-1 μ folds.
    means = {0: (10.0,) + (0.0,) * 11, 1: (999.0,) + (0.0,) * 11}

    result = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        lag_slot_of=lambda d: d - 1,
        inflow_lag_means=means,
    )

    assert [term.plant_code for term in result.dropped] == [20]
    expected_fold = 1.0 * _LAG_FACTOR * 10.0
    assert result.cuts[0].intercept == pytest.approx(5.0 * MONTH_HOURS - expected_fold)


def test_map_complexo_replicates_coefficients_onto_components() -> None:
    # A NEWAVE complexo (code 99, not operated) resolves via CX onto two DECOMP
    # components (codes 10, 11 -> hydro_ids 0, 1). Its storage + inflow-lag
    # coefficients replicate onto EACH component (full value), and each folds
    # its own seasonal mean.
    id_map = make_id_map((10, 11))
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # position 0
            make_slot(_HYDRO_STORAGE, 1, 0),  # position 1
            *[make_slot(_HYDRO_INFLOW_LAG, 0, lag) for lag in range(12)],  # 2..13
            *[make_slot(_HYDRO_INFLOW_LAG, 1, lag) for lag in range(12)],  # 14..25
        ]
    )
    lags = (3.0,) + (0.0,) * 11
    cuts = make_boundary_cuts(
        (99,), (make_cut_record(pi_varm=(2.0,), pi_qafl=(lags,), rhs=5.0),)
    )
    means = {0: (100.0,) + (0.0,) * 11, 1: (50.0,) + (0.0,) * 11}

    result = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        lag_slot_of=lambda d: d - 1,
        inflow_lag_means=means,
        complexo_components={99: [10, 11]},
    )

    assert result.dropped == ()  # complexo resolved via CX, not dropped
    mapped = result.cuts[0]
    # Storage coefficient replicated onto BOTH components (full value each).
    assert mapped.coefficients[0] == pytest.approx(2.0 * MONTH_HOURS)
    assert mapped.coefficients[1] == pytest.approx(2.0 * MONTH_HOURS)
    # Depth-1 lag replicated onto both components' lag slots.
    assert mapped.coefficients[2] == pytest.approx(3.0 * _LAG_FACTOR)
    assert mapped.coefficients[14] == pytest.approx(3.0 * _LAG_FACTOR)
    # Each component folds its OWN seasonal mean: Σ placed_lag_coef · μ.
    expected_fold = 3.0 * _LAG_FACTOR * 100.0 + 3.0 * _LAG_FACTOR * 50.0
    assert mapped.intercept == pytest.approx(5.0 * MONTH_HOURS - expected_fold)


def test_map_complexo_dropped_when_no_operated_component() -> None:
    # A complexo whose CX components are all unknown to id_map resolves onto no
    # live target -> dropped (D3), like any source-only plant.
    id_map = make_id_map((10,))
    manifest = make_manifest([make_slot(_HYDRO_STORAGE, 0, 0)])
    cuts = make_boundary_cuts((99,), (make_cut_record(pi_varm=(2.0,)),))

    result = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        complexo_components={99: [20, 21]},
    )

    assert [term.plant_code for term in result.dropped] == [99]


def test_map_complexo_none_leaves_complexo_dropped() -> None:
    # Without a complexo map, a complexo code (not operated) is dropped exactly
    # as before — byte-for-byte the pre-CX behaviour.
    id_map = make_id_map((10,))
    manifest = make_manifest([make_slot(_HYDRO_STORAGE, 0, 0)])
    cuts = make_boundary_cuts((99,), (make_cut_record(pi_varm=(2.0,)),))

    result = map_boundary_cuts(cuts, manifest, id_map, cost_unit_hours=MONTH_HOURS)

    assert [term.plant_code for term in result.dropped] == [99]


def test_map_zeroes_buckets_and_gnl_ring() -> None:
    id_map = make_id_map((10,))
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # position 0
            make_slot(_HYDRO_TRANSIT_BUCKET, 0, 5),  # position 1
            make_slot(_ANTICIPATED_THERMAL_STATE, 0, 0),  # position 2
        ]
    )
    cuts = make_boundary_cuts((10,), (make_cut_record(pi_varm=(9.0,)),))

    result = map_boundary_cuts(cuts, manifest, id_map, cost_unit_hours=MONTH_HOURS)

    for mapped in result.cuts:
        assert mapped.coefficients[1] == 0.0
        assert mapped.coefficients[2] == 0.0
    assert result.cuts[0].coefficients[0] == pytest.approx(9.0 * MONTH_HOURS)


def test_map_rejects_out_of_range_lag_slot() -> None:
    id_map = make_id_map((10,))
    # Only one HydroInflowLag slot exists (subindex 0) -> valid range is {0}.
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),
            make_slot(_HYDRO_INFLOW_LAG, 0, 0),
        ]
    )
    cuts = make_boundary_cuts((10,), (make_cut_record(pi_varm=(1.0,)),))

    with pytest.raises(ValueError, match="out of range"):
        map_boundary_cuts(
            cuts, manifest, id_map, cost_unit_hours=MONTH_HOURS, lag_slot_of=lambda d: d
        )


@requires_novomodelo_python
def test_synthetic_roundtrip_preserves_coeffs(tmp_path: Path) -> None:
    """The mapper's storage + lag-depth-1 placement survives a real
    map -> write -> load_policy round trip, with no deck and no novomodelo
    binary — the reloaded cut's intercept and leading coefficients match
    the synthetic source record verbatim.
    """
    id_map = make_id_map((10,))
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # position 0
            *[
                make_slot(_HYDRO_INFLOW_LAG, 0, lag)  # positions 1..12
                for lag in range(12)
            ],
        ]
    )
    record = make_cut_record(
        pi_varm=(2.5,),
        pi_qafl=((0.75,) + (0.0,) * 11,),
        rhs=42.0,
    )
    cuts = make_boundary_cuts((10,), (record,))

    reloaded = synthetic_roundtrip(tmp_path / "boundary", cuts, manifest, id_map)

    reloaded_cut = reloaded["stage_cuts"][0]["cuts"][0]
    # pos 0 = storage (× MONTH_HOURS); pos 1 = inflow-lag depth 1 (× _LAG_FACTOR).
    assert reloaded_cut["intercept"] == pytest.approx(record.rhs * MONTH_HOURS)
    assert reloaded_cut["coefficients"][0:3] == pytest.approx(
        [2.5 * MONTH_HOURS, 0.75 * _LAG_FACTOR, 0.0]
    )


def _make_gnl_ring_fixture(
    pi_gnl: tuple[float, ...],
    *,
    post_horizon_start: int | None = None,
) -> tuple[BoundaryCuts, TerminalManifest, DecompIdMap, GnlRingPlan]:
    """The READBACK-shaped ring (thermal 94: sentinel + dated; thermal 95:
    dated; thermal 96: dated but untargeted) over a `P=3, L=2, S=4` GNL
    block, plus the `GnlRingPlan` targeting 94 (submercado 1, lag 2) and 95
    (submercado 3, lag 1). No source hydro plants — the GNL placement is
    independent of the storage/lag families, only their manifest guard
    (`>= 1 HydroStorage slot`) must be satisfied.
    """
    id_map = make_id_map(())
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # position 0: unrelated dummy
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0),  # position 1: sentinel
            make_slot(
                _ANTICIPATED_THERMAL_STATE, 94, 1, interval_start=20260501
            ),  # position 2: dated
            make_slot(
                _ANTICIPATED_THERMAL_STATE, 95, 0, interval_start=20260401
            ),  # position 3: dated
            make_slot(
                _ANTICIPATED_THERMAL_STATE, 96, 0, interval_start=20260601
            ),  # position 4: dated but untargeted by the plan
        ]
    )
    header = make_cortes_header(
        (), lag_maximo_gnl=2, n_patamares=3, submercado_codes=(1, 2, 3, 4)
    )
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=5.0)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    plan = GnlRingPlan(
        (GnlThermalTarget(94, 1, 2), GnlThermalTarget(95, 3, 1)),
        post_horizon_start=post_horizon_start,
    )
    return cuts, manifest, id_map, plan


def test_map_gnl_places_chain_rule_sum_on_dated_slots() -> None:
    # col(1,p,2) for p=1..3 -> flat indices 1, 3, 5; col(3,p,1) -> 12, 14, 16
    # (P=3, L=2): col(s,p,l) = ((s-1)*3 + (p-1))*2 + (l-1).
    pi_gnl = _gnl_row(24, {1: 0.1, 3: 0.2, 5: 0.3, 12: 1.0, 14: 2.0, 16: 4.0})
    cuts, manifest, id_map, plan = _make_gnl_ring_fixture(pi_gnl)

    result = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        gnl_plan=plan,
        coupling_block_hours=_UNIFORM_GNL_BLOCK_HOURS,
    )

    mapped = result.cuts[0]
    # Uniform per-block hours reproduce the plain sum ÷ n_patamares (3) —
    # see test_map_gnl_uniform_blocks_is_sum_over_n_patamares.
    assert mapped.coefficients[2] == pytest.approx(0.6 * MONTH_HOURS / 3)  # 94 dated
    assert mapped.coefficients[3] == pytest.approx(7.0 * MONTH_HOURS / 3)  # 95 dated


def test_map_gnl_covered_lane_populated_uncovered_lane_dropped() -> None:
    """The excised-ring shape: thermal 94's `20260501` dated slot is a
    class-3 signaled lane (`>= post_horizon_start`), covered, and keeps the
    chain-rule sum, nonzero and finite; thermal 95's `20260401` dated slot
    is an in-study delivery (before `post_horizon_start`), non-covered, and
    is dropped, staying at `0.0`, with a `GnlDroppedTerm` naming the
    in-study committed window that actually prices it. No class-4
    já-comandada slot exists in the fixture at all — novomodelo excises it from
    the ring entirely, so there is nothing to drop for it.
    """
    pi_gnl = _gnl_row(24, {1: 0.1, 3: 0.2, 5: 0.3, 12: 1.0, 14: 2.0, 16: 4.0})
    cuts, manifest, id_map, plan = _make_gnl_ring_fixture(
        pi_gnl, post_horizon_start=20260501
    )

    result = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        gnl_plan=plan,
        coupling_block_hours=_UNIFORM_GNL_BLOCK_HOURS,
    )

    mapped = result.cuts[0]
    assert mapped.coefficients[2] == pytest.approx(0.6 * MONTH_HOURS / 3)  # 94 covered
    assert math.isfinite(mapped.coefficients[2])
    assert mapped.coefficients[3] == 0.0  # thermal 95, uncovered -> dropped

    matches = [
        term
        for term in result.gnl_dropped
        if term.thermal_id == 95 and term.submercado == 3 and term.nl_lag == 1
    ]
    assert len(matches) == 1
    dropped = matches[0]
    assert dropped.coefficient == pytest.approx(7.0)
    assert "in-study committed window" in dropped.reason
    assert "K=0" not in dropped.reason
    assert "post-horizon lane" not in dropped.reason
    # The "all dated slots uncovered" case is the uncovered-drop above, not
    # the pre-existing "no dated ring slot for thermal" reason (that one is
    # reserved for a target with zero dated slots at all).
    assert not any(
        term.thermal_id == 95 and "no dated ring slot" in term.reason
        for term in result.gnl_dropped
    )


def test_resolve_gnl_targets_docstring_keys_on_interval_start() -> None:
    """AC 4 — the docstring keys the covered/non-covered split on the slot's
    ``interval_start`` (novomodelo's day-accurate delivery-stage start), not the
    retired single ``delivery_date`` or the K=0-lead-lane framing
    (source-text check)."""
    docstring = _resolve_gnl_targets.__doc__
    assert docstring is not None
    assert "interval_start" in docstring
    assert "delivery_date" not in docstring
    assert "K=0" not in docstring


def test_map_gnl_all_dated_slots_covered_drops_nothing() -> None:
    """The post-GAP-2 shape: `post_horizon_start` is enabled
    (non-`None`) but both targets' dated slots are `>= post_horizon_start`,
    so the covered-lane filter drops nothing. Locks the invariant that
    enabling the filter on an all-covered ring is a no-op versus
    `post_horizon_start=None` (`test_map_gnl_post_horizon_start_none_is_old_behavior`).
    """
    pi_gnl = _gnl_row(24, {1: 0.1, 3: 0.2, 5: 0.3, 12: 1.0, 14: 2.0, 16: 4.0})
    cuts, manifest, id_map, plan = _make_gnl_ring_fixture(
        pi_gnl, post_horizon_start=20260401
    )

    result = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        gnl_plan=plan,
        coupling_block_hours=_UNIFORM_GNL_BLOCK_HOURS,
    )

    mapped = result.cuts[0]
    assert mapped.coefficients[2] == pytest.approx(0.6 * MONTH_HOURS / 3)  # 94, dated
    assert mapped.coefficients[3] == pytest.approx(7.0 * MONTH_HOURS / 3)  # 95, dated
    assert not any("post-study horizon" in term.reason for term in result.gnl_dropped)


def test_map_gnl_post_horizon_start_none_is_old_behavior() -> None:
    """`post_horizon_start=None` (the default) disables the covered-lane
    filter entirely: both dated slots populate exactly as the pre-filter
    behavior, and no covered-lane drop fires.
    """
    pi_gnl = _gnl_row(24, {1: 0.1, 3: 0.2, 5: 0.3, 12: 1.0, 14: 2.0, 16: 4.0})
    cuts, manifest, id_map, plan = _make_gnl_ring_fixture(
        pi_gnl, post_horizon_start=None
    )

    result = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        gnl_plan=plan,
        coupling_block_hours=_UNIFORM_GNL_BLOCK_HOURS,
    )

    mapped = result.cuts[0]
    assert mapped.coefficients[2] == pytest.approx(0.6 * MONTH_HOURS / 3)  # 94, dated
    assert mapped.coefficients[3] == pytest.approx(7.0 * MONTH_HOURS / 3)  # 95, dated
    assert not any("post-study horizon" in term.reason for term in result.gnl_dropped)


def test_map_gnl_sentinel_and_nontarget_slots_stay_zero() -> None:
    pi_gnl = _gnl_row(24, {1: 0.1, 3: 0.2, 5: 0.3, 12: 1.0, 14: 2.0, 16: 4.0})
    cuts, manifest, id_map, plan = _make_gnl_ring_fixture(pi_gnl)

    result = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        gnl_plan=plan,
        coupling_block_hours=_UNIFORM_GNL_BLOCK_HOURS,
    )

    mapped = result.cuts[0]
    assert mapped.coefficients[1] == 0.0  # thermal 94, sentinel slot
    assert mapped.coefficients[4] == 0.0  # thermal 96, absent from the plan


def test_map_gnl_drops_submercado_without_thermal() -> None:
    # col(2,p,1) for p=1..3 -> flat indices 6, 8, 10 (submercado 2, lag 1);
    # neither target claims submercado 2, and submercados 1/3's own targeted
    # columns are all zero here.
    pi_gnl = _gnl_row(24, {6: 10.0, 8: 20.0, 10: 40.0})
    cuts, manifest, id_map, plan = _make_gnl_ring_fixture(pi_gnl)

    result = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        gnl_plan=plan,
        coupling_block_hours=_UNIFORM_GNL_BLOCK_HOURS,
    )

    matches = [
        term
        for term in result.gnl_dropped
        if term.thermal_id is None
        and term.submercado == 2
        and term.nl_lag == 1
        and term.reason == "no GNL thermal in submercado"
    ]
    assert len(matches) == 1
    assert matches[0].coefficient == pytest.approx(70.0)
    # Neither targeted thermal's dated slot was altered by the drop.
    assert result.cuts[0].coefficients[2] == 0.0
    assert result.cuts[0].coefficients[3] == 0.0


def test_map_gnl_drops_target_with_no_dated_slot() -> None:
    id_map = make_id_map(())
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0),  # sentinel only
        ]
    )
    header = make_cortes_header(
        (), lag_maximo_gnl=1, n_patamares=1, submercado_codes=(1,)
    )
    record = make_cut_record(pi_varm=(), pi_gnl=(5.0,))
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    plan = GnlRingPlan((GnlThermalTarget(94, 1, 1),))

    result = map_boundary_cuts(
        cuts, manifest, id_map, cost_unit_hours=MONTH_HOURS, gnl_plan=plan
    )

    assert any(
        term.thermal_id == 94 and "no dated ring slot" in term.reason
        for term in result.gnl_dropped
    )
    assert result.cuts[0].coefficients[1] == 0.0  # sentinel slot untouched


def test_map_gnl_plan_none_is_noop() -> None:
    id_map = make_id_map((10,))
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0, interval_start=20260501),
        ]
    )
    cuts = make_boundary_cuts((10,), (make_cut_record(pi_varm=(9.0,)),))

    result = map_boundary_cuts(cuts, manifest, id_map, cost_unit_hours=MONTH_HOURS)

    assert result.gnl_dropped == ()
    assert result.cuts[0].coefficients[0] == pytest.approx(9.0 * MONTH_HOURS)
    assert result.cuts[0].coefficients[1] == 0.0


def test_map_gnl_rejects_bad_pi_gnl_width() -> None:
    id_map = make_id_map(())
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0, interval_start=20260501),
        ]
    )
    header = make_cortes_header(
        (), lag_maximo_gnl=2, n_patamares=3, submercado_codes=(1,)
    )
    # n_patamares * lag_maximo_gnl == 6; width 7 is not a multiple of it.
    record = make_cut_record(pi_varm=(), pi_gnl=(0.0,) * 7)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    plan = GnlRingPlan((GnlThermalTarget(94, 1, 1),))

    with pytest.raises(ValueError, match="pi_gnl width"):
        map_boundary_cuts(
            cuts, manifest, id_map, cost_unit_hours=MONTH_HOURS, gnl_plan=plan
        )


def test_map_gnl_weights_pi_gnl_by_coupling_block_hours() -> None:
    """AC 1 — the GNL coefficient is the f64-exact per-block weighted sum
    `Σ_p pi_gnl[p] · h_p`, not a uniform-hours collapse."""
    pi_gnl = _gnl_row(24, {1: 0.1, 3: 0.2, 5: 0.3, 12: 1.0, 14: 2.0, 16: 4.0})
    cuts, manifest, id_map, plan = _make_gnl_ring_fixture(pi_gnl)
    h1, h2, h3 = 100.0, 200.0, 428.0

    result = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        gnl_plan=plan,
        coupling_block_hours=(h1, h2, h3),
    )

    mapped = result.cuts[0]
    assert mapped.coefficients[2] == pytest.approx(0.1 * h1 + 0.2 * h2 + 0.3 * h3)
    assert mapped.coefficients[3] == pytest.approx(1.0 * h1 + 2.0 * h2 + 4.0 * h3)


def test_map_gnl_uniform_blocks_is_sum_over_n_patamares() -> None:
    """AC 2 — uniform per-block hours give exactly `1/n_patamares` of the
    pre-fix plain-sum-times-total-hours value."""
    pi_gnl = _gnl_row(24, {1: 0.1, 3: 0.2, 5: 0.3, 12: 1.0, 14: 2.0, 16: 4.0})
    cuts, manifest, id_map, plan = _make_gnl_ring_fixture(pi_gnl)

    result = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        gnl_plan=plan,
        coupling_block_hours=_UNIFORM_GNL_BLOCK_HOURS,
    )

    mapped = result.cuts[0]
    assert mapped.coefficients[2] == pytest.approx(0.6 * MONTH_HOURS / 3)
    assert mapped.coefficients[3] == pytest.approx(7.0 * MONTH_HOURS / 3)


def test_map_gnl_rejects_block_hours_length_mismatch() -> None:
    """AC 3 — `len(coupling_block_hours) != n_patamares` (here 4 vs 3) raises,
    naming the observed vs expected length."""
    pi_gnl = _gnl_row(24, {1: 0.1, 3: 0.2, 5: 0.3})
    cuts, manifest, id_map, plan = _make_gnl_ring_fixture(pi_gnl)

    with pytest.raises(ValueError, match="coupling_block_hours"):
        map_boundary_cuts(
            cuts,
            manifest,
            id_map,
            cost_unit_hours=MONTH_HOURS,
            gnl_plan=plan,
            coupling_block_hours=(MONTH_HOURS / 4,) * 4,
        )


def test_map_gnl_requires_block_hours_when_placing() -> None:
    """AC 3 — `coupling_block_hours=None` with a placing `gnl_plan` raises,
    rather than silently falling back to the pre-fix plain-sum behaviour."""
    pi_gnl = _gnl_row(24, {1: 0.1, 3: 0.2, 5: 0.3})
    cuts, manifest, id_map, plan = _make_gnl_ring_fixture(pi_gnl)

    with pytest.raises(ValueError, match="coupling_block_hours"):
        map_boundary_cuts(
            cuts, manifest, id_map, cost_unit_hours=MONTH_HOURS, gnl_plan=plan
        )


def test_map_no_lag_slots_emits_keyed_inflow_lag_coefficients() -> None:
    # A DECOMP manifest carries no HydroInflowLag slots; with inflow_lag_depth=N
    # the mapper emits the lag terms keyed by hydro (for novomodelo's writer to
    # reserve + place) rather than into the storage-only coefficient vector, and
    # still folds the seasonal mean into the intercept.
    id_map = make_id_map((10,))
    manifest = make_manifest([make_slot(_HYDRO_STORAGE, 0, 0)])  # storage only
    lags = (1.0, 2.0) + (0.0,) * 10
    cuts = make_boundary_cuts(
        (10,), (make_cut_record(pi_varm=(0.0,), pi_qafl=(lags,), rhs=5.0),)
    )
    means = {0: (100.0, 50.0) + (0.0,) * 10}

    result = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        inflow_lag_means=means,
        inflow_lag_depth=3,
    )

    mapped = result.cuts[0]
    # Storage-aligned vector stays storage-only; the lag terms are keyed instead.
    assert len(mapped.coefficients) == manifest.state_dimension
    assert mapped.inflow_lag_coefficients[0] == pytest.approx(
        (1.0 * _LAG_FACTOR, 2.0 * _LAG_FACTOR, 0.0)
    )
    # Intercept folded by Σ placed_coef·μ over the emitted depths.
    expected_fold = 1.0 * _LAG_FACTOR * 100.0 + 2.0 * _LAG_FACTOR * 50.0
    assert mapped.intercept == pytest.approx(5.0 * MONTH_HOURS - expected_fold)


def test_map_no_lag_slots_zero_depth_emits_nothing() -> None:
    # No lag slots + the default inflow_lag_depth=0: no keyed coeffs and no fold
    # — byte-for-byte the pre-fix behaviour.
    id_map = make_id_map((10,))
    manifest = make_manifest([make_slot(_HYDRO_STORAGE, 0, 0)])
    lags = (1.0, 2.0) + (0.0,) * 10
    cuts = make_boundary_cuts(
        (10,), (make_cut_record(pi_varm=(0.0,), pi_qafl=(lags,), rhs=5.0),)
    )

    result = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        inflow_lag_means={0: (100.0, 50.0) + (0.0,) * 10},
    )

    mapped = result.cuts[0]
    assert mapped.inflow_lag_coefficients == {}
    assert mapped.intercept == pytest.approx(5.0 * MONTH_HOURS)

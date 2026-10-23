"""Tests for the boundary-cut manifest-to-manifest mapper (``fcf/mapper.py``)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from novomodelo_bridge.core.units import C_M3S2HM3, MEAN_MONTH_HOURS
from novomodelo_bridge.decomp.fcf.bootstrap import TerminalManifest
from novomodelo_bridge.decomp.fcf.cortes import BoundaryCuts
from novomodelo_bridge.decomp.fcf.mapper import (
    DroppedTerm,
    GnlCommandedDelivery,
    GnlDeliveryMonth,
    GnlDroppedTerm,
    GnlRingPlan,
    GnlThermalTarget,
    HydroCoupling,
    map_boundary_cuts,
    month_key,
)
from tests._fcf_fixtures import (
    make_boundary_cuts,
    make_cortes_header,
    make_couplings,
    make_cut_record,
    make_id_map,
    make_manifest,
    make_slot,
    synthetic_roundtrip,
)
from tests.conftest import requires_novomodelo_python

# Inflow-lag coefficients take × C_M3S2HM3 beyond the mean month: novomodelo's lag
# state is m³/s, the source's hm³ over a mean month.
_LAG_FACTOR = MEAN_MONTH_HOURS * C_M3S2HM3

# novomodelo `policy.fbs` entity_type codes — a stable external contract, restated
# locally rather than importing the mapper module's private constants.
_HYDRO_STORAGE = 0
_HYDRO_INFLOW_LAG = 1
_ANTICIPATED_THERMAL_STATE = 2
_HYDRO_TRANSIT_BUCKET = 3

_MAY = month_key(2026, 5)
_JUNE = month_key(2026, 6)
_MAY_FRACTIONS = (0.2, 0.3, 0.5)
_JUNE_FRACTIONS = (0.25, 0.25, 0.5)
_WEEK_HOURS = 168.0


def _lags(*leading: float) -> tuple[float, ...]:
    return tuple(leading) + (0.0,) * (12 - len(leading))


def _storage_and_lag_manifest(n_hydros: int) -> TerminalManifest:
    """One storage slot per hydro, then 12 lag slots per hydro."""
    return make_manifest(
        [
            *(make_slot(_HYDRO_STORAGE, hydro, 0) for hydro in range(n_hydros)),
            *(
                make_slot(_HYDRO_INFLOW_LAG, hydro, lag)
                for hydro in range(n_hydros)
                for lag in range(12)
            ),
        ]
    )


def _lag_position(n_hydros: int, hydro: int, depth: int) -> int:
    return n_hydros + hydro * 12 + depth - 1


def test_map_scales_intercept_and_storage_by_the_mean_month() -> None:
    id_map = make_id_map((10, 20))
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),
            make_slot(_HYDRO_STORAGE, 1, 0),
            make_slot(_HYDRO_TRANSIT_BUCKET, 1, 2),
        ]
    )
    cuts = make_boundary_cuts(
        (10, 20), (make_cut_record(pi_varm=(3.0, 5.0), rhs=100.0),)
    )

    result = map_boundary_cuts(cuts, manifest, make_couplings(cuts, id_map))

    mapped = result.cuts[0]
    assert mapped.coefficients == pytest.approx(
        (3.0 * MEAN_MONTH_HOURS, 5.0 * MEAN_MONTH_HOURS, 0.0)
    )
    assert mapped.intercept == pytest.approx(100.0 * MEAN_MONTH_HOURS)
    assert result.dropped == ()


def test_map_storage_floor_comes_off_the_intercept() -> None:
    id_map = make_id_map((10,))
    manifest = make_manifest([make_slot(_HYDRO_STORAGE, 0, 0)])
    cuts = make_boundary_cuts((10,), (make_cut_record(pi_varm=(2.0,), rhs=7.0),))

    result = map_boundary_cuts(
        cuts, manifest, make_couplings(cuts, id_map, storage_floor_hm3=150.0)
    )

    mapped = result.cuts[0]
    assert mapped.coefficients[0] == pytest.approx(2.0 * MEAN_MONTH_HOURS)
    assert mapped.intercept == pytest.approx((7.0 - 2.0 * 150.0) * MEAN_MONTH_HOURS)


def test_map_coupling_without_floor_takes_no_storage_term() -> None:
    id_map = make_id_map((10,))
    manifest = _storage_and_lag_manifest(1)
    cuts = make_boundary_cuts(
        (10,),
        (make_cut_record(pi_varm=(2.0,), pi_qafl=(_lags(0.5),), rhs=7.0),),
    )

    result = map_boundary_cuts(
        cuts, manifest, make_couplings(cuts, id_map, storage_floor_hm3=None)
    )

    mapped = result.cuts[0]
    assert mapped.coefficients[0] == 0.0
    assert mapped.coefficients[_lag_position(1, 0, 1)] == pytest.approx(
        0.5 * _LAG_FACTOR
    )
    assert mapped.intercept == pytest.approx(7.0 * MEAN_MONTH_HOURS)


def test_map_shared_cut_row_lands_on_every_taker() -> None:
    """Two hydros taking one cut row (a complex's components) both carry it."""
    manifest = _storage_and_lag_manifest(2)
    cuts = make_boundary_cuts(
        (99,),
        (make_cut_record(pi_varm=(4.0,), pi_qafl=(_lags(0.0, 1.5),), rhs=1.0),),
    )
    couplings = (HydroCoupling(0, 0, 10.0), HydroCoupling(1, 0, 20.0))

    result = map_boundary_cuts(cuts, manifest, couplings)

    mapped = result.cuts[0]
    assert mapped.coefficients[0] == pytest.approx(4.0 * MEAN_MONTH_HOURS)
    assert mapped.coefficients[1] == pytest.approx(4.0 * MEAN_MONTH_HOURS)
    for hydro in (0, 1):
        assert mapped.coefficients[_lag_position(2, hydro, 2)] == pytest.approx(
            1.5 * _LAG_FACTOR
        )
    assert mapped.intercept == pytest.approx(
        (1.0 - 4.0 * 10.0 - 4.0 * 20.0) * MEAN_MONTH_HOURS
    )
    assert result.dropped == ()


def test_map_places_lags_through_lag_slot_of() -> None:
    id_map = make_id_map((10,))
    manifest = _storage_and_lag_manifest(1)
    lags = tuple(float(depth) for depth in range(1, 13))
    cuts = make_boundary_cuts(
        (10,), (make_cut_record(pi_varm=(0.0,), pi_qafl=(lags,)),)
    )

    result = map_boundary_cuts(
        cuts,
        manifest,
        make_couplings(cuts, id_map),
        lag_slot_of=lambda depth: 12 - depth,
    )

    coefficients = result.cuts[0].coefficients
    for depth in range(1, 13):
        assert coefficients[_lag_position(1, 0, 13 - depth)] == pytest.approx(
            depth * _LAG_FACTOR
        )


def test_map_rejects_out_of_range_lag_slot() -> None:
    id_map = make_id_map((10,))
    manifest = _storage_and_lag_manifest(1)
    cuts = make_boundary_cuts((10,), (make_cut_record(pi_varm=(0.0,)),))

    with pytest.raises(ValueError, match=r"lag_slot_of\(12\) = 12"):
        map_boundary_cuts(
            cuts, manifest, make_couplings(cuts, id_map), lag_slot_of=lambda d: d
        )


def test_map_no_lag_slots_emits_keyed_inflow_lag_coefficients() -> None:
    id_map = make_id_map((10,))
    manifest = make_manifest([make_slot(_HYDRO_STORAGE, 0, 0)])
    cuts = make_boundary_cuts(
        (10,),
        (make_cut_record(pi_varm=(1.0,), pi_qafl=(_lags(0.5, 0.25, 9.0),)),),
    )

    result = map_boundary_cuts(
        cuts, manifest, make_couplings(cuts, id_map), inflow_lag_depth=2
    )

    assert result.cuts[0].inflow_lag_coefficients == {
        0: pytest.approx((0.5 * _LAG_FACTOR, 0.25 * _LAG_FACTOR))
    }


def test_map_no_lag_slots_zero_depth_emits_nothing() -> None:
    id_map = make_id_map((10,))
    manifest = make_manifest([make_slot(_HYDRO_STORAGE, 0, 0)])
    cuts = make_boundary_cuts(
        (10,), (make_cut_record(pi_varm=(1.0,), pi_qafl=(_lags(0.5),)),)
    )

    result = map_boundary_cuts(cuts, manifest, make_couplings(cuts, id_map))

    assert result.cuts[0].inflow_lag_coefficients == {}


def test_map_lag_offsets_fold_over_the_placed_slots() -> None:
    id_map = make_id_map((10,))
    manifest = _storage_and_lag_manifest(1)
    cuts = make_boundary_cuts(
        (10,),
        (make_cut_record(pi_varm=(0.0,), pi_qafl=(_lags(2.0, 3.0),), rhs=5.0),),
    )
    offsets = {0: _lags(10.0, -4.0)}

    result = map_boundary_cuts(
        cuts, manifest, make_couplings(cuts, id_map), inflow_lag_offsets=offsets
    )

    expected = 5.0 * MEAN_MONTH_HOURS + (2.0 * 10.0 - 3.0 * 4.0) * _LAG_FACTOR
    assert result.cuts[0].intercept == pytest.approx(expected)


def test_map_lag_offsets_fold_only_the_declared_depths() -> None:
    """Without lag slots, only depths ``1..inflow_lag_depth`` fold."""
    id_map = make_id_map((10,))
    manifest = make_manifest([make_slot(_HYDRO_STORAGE, 0, 0)])
    cuts = make_boundary_cuts(
        (10,),
        (make_cut_record(pi_varm=(0.0,), pi_qafl=(_lags(2.0, 3.0),), rhs=5.0),),
    )

    result = map_boundary_cuts(
        cuts,
        manifest,
        make_couplings(cuts, id_map),
        inflow_lag_depth=1,
        inflow_lag_offsets={0: _lags(10.0, -4.0)},
    )

    expected = 5.0 * MEAN_MONTH_HOURS + 2.0 * 10.0 * _LAG_FACTOR
    assert result.cuts[0].intercept == pytest.approx(expected)


def test_map_drops_untaken_cut_plant_and_folds_its_inflow() -> None:
    id_map = make_id_map((10,))
    manifest = make_manifest([make_slot(_HYDRO_STORAGE, 0, 0)])
    cuts = make_boundary_cuts(
        (10, 260),
        (
            make_cut_record(
                pi_varm=(1.0, 8.0),
                pi_qafl=(_lags(), _lags(0.5, 0.25)),
                rhs=3.0,
            ),
        ),
    )

    result = map_boundary_cuts(
        cuts,
        manifest,
        make_couplings(cuts, id_map),
        source_only_inflows={1: _lags(40.0, 20.0)},
    )

    assert result.dropped == (DroppedTerm(260, 8.0, inflow_folded=True),)
    mapped = result.cuts[0]
    assert mapped.coefficients == pytest.approx((1.0 * MEAN_MONTH_HOURS,))
    assert mapped.intercept == pytest.approx(
        3.0 * MEAN_MONTH_HOURS + (0.5 * 40.0 + 0.25 * 20.0) * _LAG_FACTOR
    )


def test_map_untaken_cut_plant_without_inflow_is_not_folded() -> None:
    id_map = make_id_map((10,))
    manifest = make_manifest([make_slot(_HYDRO_STORAGE, 0, 0)])
    cuts = make_boundary_cuts(
        (10, 260),
        (make_cut_record(pi_varm=(1.0, 8.0), pi_qafl=(_lags(), _lags(0.5))),),
    )

    result = map_boundary_cuts(cuts, manifest, make_couplings(cuts, id_map))

    assert result.dropped == (DroppedTerm(260, 8.0, inflow_folded=False),)
    assert result.cuts[0].intercept == 0.0


def test_map_source_only_inflows_need_all_twelve_depths() -> None:
    id_map = make_id_map((10,))
    manifest = make_manifest([make_slot(_HYDRO_STORAGE, 0, 0)])
    cuts = make_boundary_cuts((10, 260), (make_cut_record(pi_varm=(1.0, 8.0)),))

    with pytest.raises(ValueError, match="zip"):
        map_boundary_cuts(
            cuts,
            manifest,
            make_couplings(cuts, id_map),
            source_only_inflows={1: (1.0, 2.0)},
        )


def test_map_rejects_manifest_without_storage_slots() -> None:
    id_map = make_id_map((10,))
    manifest = make_manifest([make_slot(_HYDRO_INFLOW_LAG, 0, 0)])
    cuts = make_boundary_cuts((10,), (make_cut_record(pi_varm=(1.0,)),))

    with pytest.raises(ValueError, match="no HydroStorage slots"):
        map_boundary_cuts(cuts, manifest, make_couplings(cuts, id_map))


def test_map_rejects_storage_floor_on_hydro_without_slot() -> None:
    manifest = make_manifest([make_slot(_HYDRO_STORAGE, 0, 0)])
    cuts = make_boundary_cuts((10,), (make_cut_record(pi_varm=(1.0,)),))

    with pytest.raises(ValueError, match="hydro 3 takes a storage term"):
        map_boundary_cuts(cuts, manifest, (HydroCoupling(3, 0, 0.0),))


def test_map_keeps_record_provenance_for_inactive_cuts() -> None:
    id_map = make_id_map((10,))
    manifest = make_manifest([make_slot(_HYDRO_STORAGE, 0, 0)])
    cuts = make_boundary_cuts(
        (10,),
        (
            make_cut_record(pi_varm=(1.0,), cut_id=7, iteration=3),
            make_cut_record(
                pi_varm=(2.0,),
                cut_id=8,
                iteration=4,
                forward_pass_index=2,
                is_active=False,
            ),
        ),
    )

    result = map_boundary_cuts(cuts, manifest, make_couplings(cuts, id_map))

    assert [
        (c.cut_id, c.iteration, c.forward_pass_index, c.is_active) for c in result.cuts
    ] == [(7, 3, 1, True), (8, 4, 2, False)]


@requires_novomodelo_python
def test_synthetic_roundtrip_preserves_coeffs(tmp_path: Path) -> None:
    """Storage and lag-depth-1 placement survive map -> write -> load_policy."""
    id_map = make_id_map((10,))
    manifest = _storage_and_lag_manifest(1)
    record = make_cut_record(pi_varm=(2.5,), pi_qafl=(_lags(0.75),), rhs=42.0)
    cuts = make_boundary_cuts((10,), (record,))

    reloaded = synthetic_roundtrip(tmp_path / "boundary", cuts, manifest, id_map)

    reloaded_cut = reloaded["stage_cuts"][0]["cuts"][0]
    assert reloaded_cut["intercept"] == pytest.approx(42.0 * MEAN_MONTH_HOURS)
    assert reloaded_cut["coefficients"][0:3] == pytest.approx(
        [2.5 * MEAN_MONTH_HOURS, 0.75 * _LAG_FACTOR, 0.0]
    )


# ---------------------------------------------------------------------------
# GNL ring: P = 3 patamares, L = 2 lags, S = 4 submercados (24 columns).
# ---------------------------------------------------------------------------


def _col(submercado: int, patamar: int, lag: int) -> int:
    """The source's flat ``pi_gnl`` layout, submercado-major × patamar × lag."""
    return ((submercado - 1) * 3 + (patamar - 1)) * 2 + (lag - 1)


def _gnl_row(nonzero: Mapping[int, float]) -> tuple[float, ...]:
    row = [0.0] * 24
    for column, value in nonzero.items():
        row[column] = value
    return tuple(row)


def _gnl_cuts(pi_gnl: tuple[float, ...], *, rhs: float = 5.0) -> BoundaryCuts:
    header = make_cortes_header(
        (), lag_maximo_gnl=2, n_patamares=3, submercado_codes=(1, 2, 3, 4)
    )
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=rhs)
    return BoundaryCuts(header=header, boundary_stage=10, records=(record,))


def _gnl_manifest() -> TerminalManifest:
    return make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # 0: the storage-family guard
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0),  # 1: undated
            make_slot(
                _ANTICIPATED_THERMAL_STATE,
                94,
                1,
                interval_start=20260502,
                interval_end=20260509,
            ),  # 2: May
            make_slot(
                _ANTICIPATED_THERMAL_STATE,
                94,
                2,
                interval_start=20260601,
                interval_end=20260608,
            ),  # 3: a June-dated week the plan books to May
            make_slot(
                _ANTICIPATED_THERMAL_STATE,
                95,
                0,
                interval_start=20260606,
                interval_end=20260630,
            ),  # 4: June
            make_slot(
                _ANTICIPATED_THERMAL_STATE, 96, 0, interval_start=20260601
            ),  # 5: untargeted
            make_slot(
                _ANTICIPATED_THERMAL_STATE,
                95,
                1,
                interval_start=20260701,
                interval_end=20260708,
            ),  # 6: July, past the cut's lags
        ]
    )


def _gnl_plan(
    commanded: tuple[GnlCommandedDelivery, ...] = (),
    targets: tuple[GnlThermalTarget, ...] = (
        GnlThermalTarget(94, 1),
        GnlThermalTarget(95, 3),
    ),
) -> GnlRingPlan:
    return GnlRingPlan(
        targets=targets,
        months={
            _MAY: GnlDeliveryMonth(1, _MAY_FRACTIONS),
            _JUNE: GnlDeliveryMonth(2, _JUNE_FRACTIONS),
        },
        week_months={20260601: _MAY},
        commanded=commanded,
    )


def _slot_price(
    pi_gnl: tuple[float, ...],
    submercado: int,
    lag: int,
    fractions: tuple[float, ...],
    hours: float,
) -> float:
    return sum(
        pi_gnl[_col(submercado, patamar, lag)] * hours * fraction
        for patamar, fraction in enumerate(fractions, start=1)
    )


_PI_GNL = _gnl_row(
    {
        _col(1, 1, 1): 0.1,
        _col(1, 2, 1): 0.2,
        _col(1, 3, 1): 0.3,
        _col(1, 1, 2): 0.7,
        _col(1, 2, 2): 0.8,
        _col(1, 3, 2): 0.9,
        _col(3, 1, 2): 1.0,
        _col(3, 2, 2): 2.0,
        _col(3, 3, 2): 4.0,
    }
)


def test_map_gnl_prices_each_slot_by_its_delivery_month() -> None:
    result = map_boundary_cuts(
        _gnl_cuts(_PI_GNL), _gnl_manifest(), (), gnl_plan=_gnl_plan()
    )

    coefficients = result.cuts[0].coefficients
    may = _slot_price(_PI_GNL, 1, 1, _MAY_FRACTIONS, _WEEK_HOURS)
    assert coefficients[2] == pytest.approx(may)
    assert coefficients[3] == pytest.approx(may)
    assert coefficients[4] == pytest.approx(
        _slot_price(_PI_GNL, 3, 2, _JUNE_FRACTIONS, 24 * 24.0)
    )


def test_map_gnl_leaves_undated_untargeted_and_unpriced_slots_at_zero() -> None:
    result = map_boundary_cuts(
        _gnl_cuts(_PI_GNL), _gnl_manifest(), (), gnl_plan=_gnl_plan()
    )

    coefficients = result.cuts[0].coefficients
    assert (coefficients[0], coefficients[1], coefficients[5], coefficients[6]) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    assert result.gnl_dropped == (
        GnlDroppedTerm(
            95,
            3,
            None,
            0.0,
            "1 ring slot(s) deliver in a month the cut's GNL lags do not price",
        ),
    )


def test_map_gnl_week_without_booking_falls_back_to_its_calendar_month() -> None:
    plan = GnlRingPlan(
        targets=(GnlThermalTarget(94, 1),),
        months={
            _MAY: GnlDeliveryMonth(1, _MAY_FRACTIONS),
            _JUNE: GnlDeliveryMonth(2, _JUNE_FRACTIONS),
        },
    )

    result = map_boundary_cuts(_gnl_cuts(_PI_GNL), _gnl_manifest(), (), gnl_plan=plan)

    assert result.cuts[0].coefficients[3] == pytest.approx(
        _slot_price(_PI_GNL, 1, 2, _JUNE_FRACTIONS, _WEEK_HOURS)
    )


def test_map_gnl_folds_commanded_deliveries_into_the_intercept() -> None:
    commanded = (GnlCommandedDelivery(1, _MAY, (500.0, 500.0, 0.0), _WEEK_HOURS),)

    result = map_boundary_cuts(
        _gnl_cuts(_PI_GNL, rhs=5.0),
        _gnl_manifest(),
        (),
        gnl_plan=_gnl_plan(commanded=commanded),
    )

    expected_fold = sum(
        _PI_GNL[_col(1, patamar, 1)] * _WEEK_HOURS * fraction * mw
        for patamar, (fraction, mw) in enumerate(
            zip(_MAY_FRACTIONS, (500.0, 500.0, 0.0), strict=True), start=1
        )
    )
    assert result.cuts[0].intercept == pytest.approx(
        5.0 * MEAN_MONTH_HOURS + expected_fold
    )


def test_map_gnl_commanded_delivery_in_an_unpriced_month_adds_nothing() -> None:
    commanded = (
        GnlCommandedDelivery(1, month_key(2026, 4), (500.0,) * 3, _WEEK_HOURS),
    )

    result = map_boundary_cuts(
        _gnl_cuts(_PI_GNL, rhs=5.0),
        _gnl_manifest(),
        (),
        gnl_plan=_gnl_plan(commanded=commanded),
    )

    assert result.cuts[0].intercept == pytest.approx(5.0 * MEAN_MONTH_HOURS)


def test_map_gnl_drops_submercado_without_thermal() -> None:
    pi_gnl = _gnl_row({_col(2, 1, 1): 0.5, _col(2, 3, 1): 0.25})

    result = map_boundary_cuts(
        _gnl_cuts(pi_gnl), _gnl_manifest(), (), gnl_plan=_gnl_plan()
    )

    assert (
        GnlDroppedTerm(None, 2, 1, 0.75, "no GNL thermal in submercado")
        in result.gnl_dropped
    )


def test_map_gnl_drops_target_without_dated_slot() -> None:
    plan = _gnl_plan(targets=(GnlThermalTarget(97, 1),))

    result = map_boundary_cuts(_gnl_cuts(_PI_GNL), _gnl_manifest(), (), gnl_plan=plan)

    assert result.gnl_dropped[0] == GnlDroppedTerm(
        97, 1, None, 0.0, "no dated ring slot for thermal"
    )


def test_map_gnl_drops_target_outside_the_submercado_range() -> None:
    plan = _gnl_plan(targets=(GnlThermalTarget(94, 5),))

    result = map_boundary_cuts(_gnl_cuts(_PI_GNL), _gnl_manifest(), (), gnl_plan=plan)

    assert result.gnl_dropped[0] == GnlDroppedTerm(
        94, 5, None, 0.0, "submercado out of the pi_gnl range"
    )


def test_map_gnl_plan_none_leaves_the_ring_at_zero() -> None:
    result = map_boundary_cuts(_gnl_cuts(_PI_GNL), _gnl_manifest(), ())

    assert result.cuts[0].coefficients == (0.0,) * 7
    assert result.gnl_dropped == ()


def test_map_gnl_rejects_bad_pi_gnl_width() -> None:
    cuts = _gnl_cuts((0.0,) * 23)

    with pytest.raises(ValueError, match="pi_gnl width 23"):
        map_boundary_cuts(cuts, _gnl_manifest(), (), gnl_plan=_gnl_plan())

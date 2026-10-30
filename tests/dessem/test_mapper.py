"""Tests for dessem/exporters/fcf/mapper.py."""

import logging
from dataclasses import replace

import numpy as np
import pytest

from novomodelo_bridge.core.policy_dates import UNDATED
from novomodelo_bridge.dessem.exporters.fcf.core.inputs_reader import (
    BlockInfo,
    CaseInputs,
    HydroInfo,
    StageInfo,
    ThermalInfo,
)
from novomodelo_bridge.dessem.exporters.fcf.core.layout import CutBlockOffsets
from novomodelo_bridge.dessem.exporters.fcf.core.mapper import (
    DAYS_PER_YEAR,
    DECOMP_COST_DIVISOR,
    ENTITY_TYPE_ANTICIPATED_THERMAL,
    EXCLUDED_BUS_ID,
    SUBMARKET_COUNT,
    GnlDeliveries,
    MappingError,
    SourceGnlThermal,
    SourceHydro,
    discount_factors,
    downstream_positions,
    gnl_block,
    gnl_drop_audit,
    gnl_lag_months,
    match_source_hydros,
    negate_gnl_array,
    plant_codes,
    select_gnl_deliveries,
    select_submarket_buses,
    storage_plant_ids,
    submarket_for_bus,
    to_decomp_cost,
    to_decomp_costs,
)
from novomodelo_bridge.dessem.exporters.fcf.core.reader import (
    AffinePieceRecord,
    EntitySlotRecord,
    StageCutPool,
)


class TestCostConversion:
    """Test P2: cost division by 1000."""

    def test_to_decomp_cost(self) -> None:
        assert to_decomp_cost(1000.0) == 1.0
        assert to_decomp_cost(0.0) == 0.0
        assert to_decomp_cost(-1000.0) == -1.0

    def test_to_decomp_costs_array(self) -> None:
        values = np.array([1000.0, 2000.0, 3000.0])
        result = to_decomp_costs(values)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_decomp_cost_divisor_constant(self) -> None:
        assert DECOMP_COST_DIVISOR == 1000.0


class TestGnlSignConvention:
    """Test P4: GNL sign negation."""

    def test_negate_gnl_array(self) -> None:
        values = np.array([1.0, -2.0, 0.0])
        result = negate_gnl_array(values)
        assert list(result) == [-1.0, 2.0, 0.0]


class TestSubmarketMapping:
    """Test P6: submarket = bus_id + 1."""

    def test_valid_bus_ids(self) -> None:
        assert submarket_for_bus(0) == 1
        assert submarket_for_bus(1) == 2
        assert submarket_for_bus(2) == 3
        assert submarket_for_bus(3) == 4
        assert submarket_for_bus(4) == 5

    def test_excluded_bus_raises(self) -> None:
        with pytest.raises(MappingError, match="interconnection"):
            submarket_for_bus(EXCLUDED_BUS_ID)

    def test_out_of_range_bus_raises(self) -> None:
        with pytest.raises(MappingError):
            submarket_for_bus(6)
        with pytest.raises(MappingError):
            submarket_for_bus(-1)

    def test_select_submarket_buses(self, synthetic_inputs: CaseInputs) -> None:
        selected = select_submarket_buses(synthetic_inputs.buses)
        assert len(selected) == SUBMARKET_COUNT
        assert all(b.id != EXCLUDED_BUS_ID for b in selected)


class TestDiscountFactors:
    """Test P9: discount factors with 365.25 day basis."""

    def test_first_stage_is_one(self, synthetic_inputs: CaseInputs) -> None:
        factors = discount_factors(synthetic_inputs.stages, 0.12)
        assert factors[0] == pytest.approx(1.0)

    def test_factors_are_decreasing(self, synthetic_inputs: CaseInputs) -> None:
        factors = discount_factors(synthetic_inputs.stages, 0.12)
        for i in range(1, len(factors)):
            assert factors[i] < factors[i - 1]

    def test_empty_stages(self) -> None:
        assert discount_factors([], 0.12) == ()

    def test_invalid_rate_raises(self, synthetic_inputs: CaseInputs) -> None:
        with pytest.raises(MappingError, match="exceed -1"):
            discount_factors(synthetic_inputs.stages, -1.0)

    def test_days_per_year_constant(self) -> None:
        assert DAYS_PER_YEAR == 365.25


def _slot(entity_type: int, entity_id: int) -> EntitySlotRecord:
    return EntitySlotRecord(
        entity_type=entity_type,
        entity_id=entity_id,
        subindex=0,
        was_active=True,
        reference_date=UNDATED,
        interval_start=UNDATED,
        interval_end=UNDATED,
    )


def _hydro(hydro_id: int, downstream_id: int | None) -> HydroInfo:
    return HydroInfo(
        id=hydro_id,
        name=f"UHE_{hydro_id}",
        downstream_id=downstream_id,
        min_storage_hm3=0.0,
        max_storage_hm3=1.0,
    )


def _source(*hydro_ids: int) -> dict[int, SourceHydro]:
    return {i: SourceHydro(code=100 + i, name=f"UHE_{i}") for i in hydro_ids}


class TestMatchSourceHydros:
    """The deck must be the one the case was converted from."""

    def test_same_plants_and_names_pass(self) -> None:
        match_source_hydros([_hydro(0, None), _hydro(1, 0)], _source(0, 1))

    def test_different_plant_set_raises(self) -> None:
        with pytest.raises(
            MappingError, match="operates 3 hydro plants but the case holds 2"
        ):
            match_source_hydros([_hydro(0, None), _hydro(1, 0)], _source(0, 1, 2))

    def test_renamed_plant_raises(self) -> None:
        source = _source(0, 1) | {1: SourceHydro(code=101, name="OTHER")}
        with pytest.raises(MappingError, match="'UHE_1' in the case, 'OTHER'"):
            match_source_hydros([_hydro(0, None), _hydro(1, 0)], source)


def _gnl(thermal_id: int, name: str) -> ThermalInfo:
    return ThermalInfo(id=thermal_id, name=name, bus_id=0, lead_time_hours=1512.0)


class TestGnlLagMonths:
    """Premise P15: each GNL plant's lag months come from the deck's NL record."""

    def test_lags_follow_the_deck_nl_records_whatever_the_lead_time(self) -> None:
        thermals = [_gnl(94, "SANTA CRUZ"), _gnl(95, "PSERGIPE I")]
        source = [
            SourceGnlThermal(code=86, name="SANTA CRUZ", lag_months=2),
            SourceGnlThermal(code=224, name="PSERGIPE I", lag_months=3),
        ]

        assert gnl_lag_months(thermals, source) == (2, 3)

    def test_no_gnl_plants_gives_no_lags(self) -> None:
        assert gnl_lag_months([], []) == ()

    def test_plant_count_mismatch_raises(self) -> None:
        with pytest.raises(
            MappingError, match="declares 0 GNL plants but the case holds 1"
        ):
            gnl_lag_months([_gnl(94, "SANTA CRUZ")], [])

    def test_renamed_plant_raises(self) -> None:
        source = [SourceGnlThermal(code=86, name="OTHER", lag_months=2)]
        with pytest.raises(MappingError, match="'SANTA CRUZ' in the case"):
            gnl_lag_months([_gnl(94, "SANTA CRUZ")], source)

    def test_plant_without_nl_record_raises(self) -> None:
        source = [SourceGnlThermal(code=86, name="SANTA CRUZ", lag_months=None)]
        with pytest.raises(MappingError, match="GNL plant 86 .* has no NL record"):
            gnl_lag_months([_gnl(94, "SANTA CRUZ")], source)


def _delivery_slot(
    thermal_id: int, start: int, end: int, subindex: int = 0
) -> EntitySlotRecord:
    return EntitySlotRecord(
        entity_type=ENTITY_TYPE_ANTICIPATED_THERMAL,
        entity_id=thermal_id,
        subindex=subindex,
        was_active=True,
        reference_date=UNDATED,
        interval_start=start,
        interval_end=end,
    )


def _pool(slots: list[EntitySlotRecord], graph_stage_id: int = 0) -> StageCutPool:
    return StageCutPool(
        stage_id=graph_stage_id,
        node_id=-1,
        graph_stage_id=graph_stage_id,
        state_dimension=len(slots),
        capacity=1,
        warm_start_count=0,
        populated_count=1,
        cost_scale_factor=1.0,
        slots=tuple(slots),
        pieces=(),
    )


def _stage(stage_id: int, start: str, end: str, hours: tuple[float, ...]) -> StageInfo:
    return StageInfo(
        id=stage_id,
        start_date=start,
        end_date=end,
        blocks=tuple(
            BlockInfo(id=i, name=f"B{i}", hours=h) for i, h in enumerate(hours)
        ),
        storage_state=True,
        inflow_lags_state=False,
    )


def _weekly_stages() -> list[StageInfo]:
    return [
        _stage(0, "2026-08-15", "2026-08-22", (24.0, 72.0, 72.0)),
        _stage(1, "2026-08-22", "2026-08-29", (27.0, 68.0, 73.0)),
        _stage(2, "2026-08-29", "2026-09-05", (24.0, 72.0, 72.0)),
    ]


def _storage_slots() -> list[EntitySlotRecord]:
    return [_slot(0, hydro_id) for hydro_id in range(3)]


def _ring_pool() -> StageCutPool:
    """Three storage slots, then thermal 100's ring: the 22/08 week, then 29/08."""
    return _pool(
        [
            *_storage_slots(),
            _delivery_slot(100, 20260822, 20260829, subindex=0),
            _delivery_slot(100, 20260829, 20260905, subindex=1),
        ]
    )


class TestSelectGnlDeliveries:
    """The exported GNL coefficient is the one priced on the stage after the first."""

    def test_weekly_delivery_is_found_by_its_start_not_by_ring_subindex(self) -> None:
        deliveries = select_gnl_deliveries(
            _ring_pool(), [_gnl(100, "PLANT")], _weekly_stages()
        )

        assert deliveries.positions == (3,)
        assert deliveries.block_hours == (27.0, 68.0, 73.0)
        assert deliveries.hours == 168.0

    def test_selection_ignores_ring_order(self) -> None:
        pool = _pool(
            [
                *_storage_slots(),
                _delivery_slot(100, 20260829, 20260905, subindex=0),
                _delivery_slot(100, 20260822, 20260829, subindex=1),
            ]
        )

        deliveries = select_gnl_deliveries(pool, [_gnl(100, "PLANT")], _weekly_stages())

        assert deliveries.positions == (4,)

    def test_monthly_stage_divides_by_its_own_hours(self) -> None:
        stages = [
            _stage(0, "2026-08-29", "2026-09-05", (24.0, 72.0, 72.0)),
            _stage(1, "2026-09-05", "2026-10-01", (100.0, 250.0, 274.0)),
        ]
        pool = _pool([*_storage_slots(), _delivery_slot(100, 20260905, 20261001)])

        deliveries = select_gnl_deliveries(pool, [_gnl(100, "PLANT")], stages)

        assert deliveries.positions == (3,)
        assert deliveries.hours == 624.0

    def test_positions_follow_the_thermal_order_and_mark_the_missing_one(self) -> None:
        pool = _pool(
            [
                *_storage_slots(),
                _delivery_slot(101, 20260822, 20260829),
                _delivery_slot(100, 20260829, 20260905),
            ]
        )

        deliveries = select_gnl_deliveries(
            pool, [_gnl(100, "A"), _gnl(101, "B")], _weekly_stages()
        )

        assert deliveries.positions == (None, 3)

    def test_slot_hours_that_differ_from_the_stage_hours_raise(self) -> None:
        pool = _pool([*_storage_slots(), _delivery_slot(100, 20260822, 20260830)])

        with pytest.raises(
            MappingError, match=r"thermal 100 \(PLANT\).*\(192 h\).*stage 1.*168 h"
        ):
            select_gnl_deliveries(pool, [_gnl(100, "PLANT")], _weekly_stages())

    def test_reversed_interval_raises(self) -> None:
        pool = _pool([*_storage_slots(), _delivery_slot(100, 20260822, 20260815)])

        with pytest.raises(MappingError, match=r"\(-168 h\)"):
            select_gnl_deliveries(pool, [_gnl(100, "PLANT")], _weekly_stages())

    def test_two_slots_on_the_delivery_start_raise(self) -> None:
        pool = _pool(
            [
                *_storage_slots(),
                _delivery_slot(100, 20260822, 20260829, subindex=0),
                _delivery_slot(100, 20260822, 20260829, subindex=1),
            ]
        )

        with pytest.raises(
            MappingError, match=r"thermal 100 \(PLANT\) has 2 slots delivering from"
        ):
            select_gnl_deliveries(pool, [_gnl(100, "PLANT")], _weekly_stages())

    def test_single_stage_case_with_gnl_raises(self) -> None:
        with pytest.raises(MappingError, match="has 1 stage"):
            select_gnl_deliveries(
                _ring_pool(), [_gnl(100, "PLANT")], _weekly_stages()[:1]
            )

    def test_first_pool_off_stage_zero_raises(self) -> None:
        pool = _pool([*_storage_slots()], graph_stage_id=1)

        with pytest.raises(MappingError, match="pool is at stage 1"):
            select_gnl_deliveries(pool, [_gnl(100, "PLANT")], _weekly_stages())

    def test_malformed_delivery_start_raises(self) -> None:
        pool = _pool([*_storage_slots(), _delivery_slot(100, 20260230, 20260829)])

        with pytest.raises(
            MappingError, match=r"thermal 100 \(PLANT\).*20260230.*not a YYYYMMDD"
        ):
            select_gnl_deliveries(pool, [_gnl(100, "PLANT")], _weekly_stages())

    def test_undated_delivery_end_raises(self) -> None:
        pool = _pool([*_storage_slots(), _delivery_slot(100, 20260822, UNDATED)])

        with pytest.raises(MappingError, match=r"thermal 100 \(PLANT\).*end .*not a"):
            select_gnl_deliveries(pool, [_gnl(100, "PLANT")], _weekly_stages())

    def test_undated_slots_are_skipped(self) -> None:
        pool = _pool([*_storage_slots(), _delivery_slot(100, UNDATED, UNDATED)])

        deliveries = select_gnl_deliveries(pool, [_gnl(100, "PLANT")], _weekly_stages())

        assert deliveries.positions == (None,)

    def test_missing_delivery_keeps_the_stage(self) -> None:
        pool = _pool(_storage_slots())

        deliveries = select_gnl_deliveries(pool, [_gnl(100, "PLANT")], _weekly_stages())

        assert deliveries.positions == (None,)
        assert deliveries.hours == 168.0

    def test_no_gnl_thermal_gives_no_stage(self) -> None:
        deliveries = select_gnl_deliveries(_ring_pool(), [], _weekly_stages())

        assert deliveries == GnlDeliveries(stage=None, positions=())
        assert deliveries.block_hours == ()
        assert deliveries.hours == 0.0

    def test_no_gnl_thermal_is_never_refused(self) -> None:
        pool = _pool(_storage_slots(), graph_stage_id=1)

        deliveries = select_gnl_deliveries(pool, [], _weekly_stages()[:1])

        assert deliveries.stage is None


def _gnl_offsets(n_sbm_gnl: int, n_estagios: int, n_patamares: int) -> CutBlockOffsets:
    return CutBlockOffsets(
        rhs=0,
        pi_varm=1,
        pi_qdefp=2,
        pi_gnl=4,
        ncoef=4 + n_sbm_gnl * n_estagios * n_patamares,
        n_sbm_gnl=n_sbm_gnl,
        n_estagios=n_estagios,
        n_patamares=n_patamares,
    )


def _weekly_deliveries(*positions: int | None) -> GnlDeliveries:
    return GnlDeliveries(stage=_weekly_stages()[1], positions=positions)


class TestGnlBlock:
    """A GNL coefficient is divided by the delivery stage's hours, once per entry."""

    def test_weekly_block_holds_each_entry_value_on_every_patamar(self) -> None:
        coefficients = np.array([0.0, 0.0, 0.0, -33600.0, 0.0, -25200.0])

        block = gnl_block(_weekly_deliveries(3, 5), coefficients, _gnl_offsets(2, 3, 3))

        assert block.tolist() == [-200.0] * 3 + [0.0] * 6 + [-150.0] * 3 + [0.0] * 6

    def test_monthly_stage_divides_by_the_block_hours_sum(self) -> None:
        stage = _stage(1, "2026-09-05", "2026-10-01", (100.0, 250.0, 274.0))
        deliveries = GnlDeliveries(stage=stage, positions=(0,))

        block = gnl_block(deliveries, np.array([-62400.0]), _gnl_offsets(1, 1, 3))

        assert block.tolist() == [-100.0, -100.0, -100.0]

    def test_missing_position_leaves_its_entry_zero(self) -> None:
        coefficients = np.array([0.0, 0.0, 0.0, -33600.0, 0.0, -25200.0])

        block = gnl_block(
            _weekly_deliveries(None, 5), coefficients, _gnl_offsets(2, 3, 3)
        )

        assert block.tolist() == [0.0] * 9 + [-150.0] * 3 + [0.0] * 6

    def test_value_is_the_direct_quotient_not_a_reciprocal_multiply(self) -> None:
        block = gnl_block(
            _weekly_deliveries(0), np.array([-1000.0]), _gnl_offsets(1, 1, 3)
        )

        assert block.tolist() == [-5.9523809523809526] * 3

    def test_no_gnl_entries_give_an_empty_block(self) -> None:
        deliveries = GnlDeliveries(stage=None, positions=())

        block = gnl_block(deliveries, np.array([1.0]), _gnl_offsets(0, 3, 3))

        assert block.tolist() == []

    def test_position_count_that_differs_from_the_layout_raises(self) -> None:
        with pytest.raises(
            MappingError, match="1 GNL delivery positions but .* holds 2 GNL entries"
        ):
            gnl_block(_weekly_deliveries(3), np.zeros(6), _gnl_offsets(2, 3, 3))

    def test_position_past_the_coefficient_vector_raises(self) -> None:
        with pytest.raises(
            MappingError, match="coefficient 6 but the coefficient vector has 6"
        ):
            gnl_block(_weekly_deliveries(6), np.zeros(6), _gnl_offsets(1, 3, 3))

    def test_zero_hours_raise(self) -> None:
        stage = _stage(1, "2026-08-22", "2026-08-29", (0.0, 0.0, 0.0))
        deliveries = GnlDeliveries(stage=stage, positions=(0,))

        with pytest.raises(MappingError, match="0 hours, which cannot convert"):
            gnl_block(deliveries, np.array([-1.0]), _gnl_offsets(1, 1, 3))


_MAPPER_LOGGER = "novomodelo_bridge.dessem.exporters.fcf.core.mapper"


def _audit_slots() -> list[EntitySlotRecord]:
    """Storage at positions 0-2, then GNL slots 3-6: two thermals, two slots each."""
    return [
        *_storage_slots(),
        _delivery_slot(100, 20260108, 20260115, subindex=0),
        _delivery_slot(100, 20260115, 20260122, subindex=1),
        _delivery_slot(102, 20260115, 20260122, subindex=0),
        _delivery_slot(102, 20260108, 20260115, subindex=2),
    ]


def _audit_pools(
    slots: list[EntitySlotRecord], first_warm_start: int = 0
) -> dict[int, StageCutPool]:
    """Pools 0-2 of five pieces each; *first_warm_start* pieces of pool 0 are warm."""
    pieces = tuple(
        AffinePieceRecord(
            slot_index=k,
            iteration=k + 1,
            forward_pass_index=0,
            intercept=0.0,
            coefficients=np.zeros(len(slots)),
            is_active=True,
        )
        for k in range(5)
    )
    return {
        pool_id: replace(
            _pool(slots, pool_id),
            pieces=pieces,
            populated_count=5,
            warm_start_count=first_warm_start if pool_id == 0 else 0,
        )
        for pool_id in range(3)
    }


def _mapper_messages(
    caplog: pytest.LogCaptureFixture,
) -> list[tuple[str, str]]:
    return [
        (record.levelname, record.getMessage())
        for record in caplog.records
        if record.name == _MAPPER_LOGGER
    ]


class TestGnlDropAudit:
    """Premise P14 counts the GNL coefficients the written cuts leave out."""

    def test_only_the_first_pool_writes_its_delivered_positions(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pools = _audit_pools(_audit_slots())

        with caplog.at_level(logging.INFO, logger=_MAPPER_LOGGER):
            audit = gnl_drop_audit(pools, (0, 1, 2), _weekly_deliveries(3, 6))

        assert audit.per_pool == ((0, 10), (1, 20), (2, 20))
        assert audit.total == 50
        assert _mapper_messages(caplog) == [
            (
                "WARNING",
                "premise P14 drops 50 anticipated-thermal coefficients: "
                "pool 0=10, pool 1=20, pool 2=20",
            )
        ]

    def test_warm_start_pieces_are_not_written_cuts(self) -> None:
        pools = _audit_pools(_audit_slots(), first_warm_start=2)

        audit = gnl_drop_audit(pools, (0, 1, 2), _weekly_deliveries(3, 6))

        assert audit.per_pool == ((0, 6), (1, 20), (2, 20))

    def test_a_missing_delivery_counts_every_gnl_slot_of_the_first_pool(self) -> None:
        pools = _audit_pools(_audit_slots())

        audit = gnl_drop_audit(pools, (0, 1, 2), _weekly_deliveries(None, None))

        assert audit.per_pool == ((0, 20), (1, 20), (2, 20))

    def test_pools_outside_the_trunk_are_not_counted(self) -> None:
        pools = _audit_pools(_audit_slots())

        audit = gnl_drop_audit(pools, (0, 1), _weekly_deliveries(3, 6))

        assert audit.per_pool == ((0, 10), (1, 20))
        assert audit.total == 30

    def test_no_trunk_pool_drops_nothing(self) -> None:
        audit = gnl_drop_audit({}, (), GnlDeliveries(stage=None, positions=()))

        assert audit.per_pool == ()
        assert audit.total == 0

    def test_pools_without_gnl_slots_log_one_info_and_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pools = _audit_pools(_storage_slots())

        with caplog.at_level(logging.INFO, logger=_MAPPER_LOGGER):
            audit = gnl_drop_audit(
                pools, (0, 1, 2), GnlDeliveries(stage=None, positions=())
            )

        assert audit.total == 0
        assert _mapper_messages(caplog) == [
            (
                "INFO",
                "premise P14 drops no anticipated-thermal coefficients on this case",
            )
        ]


class TestStoragePlantIds:
    """Storage slots give the plant order of every per-plant record."""

    def test_ids_follow_slot_order(self) -> None:
        slots = [_slot(0, 3), _slot(0, 1), _slot(0, 2), _slot(2, 100)]

        assert storage_plant_ids(slots) == (3, 1, 2)

    def test_storage_after_another_axis_raises(self) -> None:
        with pytest.raises(MappingError, match="instead of leading the slots"):
            storage_plant_ids([_slot(2, 100), _slot(0, 1)])


class TestPlantCodes:
    """The plant-code record carries the deck code of each storage axis."""

    def test_codes_follow_plant_order(self) -> None:
        assert plant_codes((2, 0, 1), _source(0, 1, 2)) == (102, 100, 101)

    def test_storage_hydro_missing_from_deck_raises(self) -> None:
        with pytest.raises(
            MappingError, match="storage hydro ids not in the deck: \\[5\\]"
        ):
            plant_codes((0, 5), _source(0, 1))


class TestDownstreamPositions:
    """The mapcut downstream record holds 1-based positions, not plant codes."""

    def test_entries_are_positions_into_the_plant_list(self) -> None:
        # A chain 0 -> 1 -> 2 -> 3, plus a second plant feeding 2.
        hydros = [
            _hydro(0, 1),
            _hydro(1, 2),
            _hydro(2, 3),
            _hydro(3, None),
            _hydro(4, 2),
        ]

        assert downstream_positions(hydros, (0, 1, 2, 3, 4)) == (2, 3, 4, 0, 3)

    def test_entries_follow_plant_order_not_id_order(self) -> None:
        hydros = [_hydro(0, 2), _hydro(1, 0), _hydro(2, None)]

        assert downstream_positions(hydros, (2, 0, 1)) == (0, 1, 2)

    def test_unknown_downstream_hydro_raises(self) -> None:
        with pytest.raises(MappingError, match="names downstream hydro 9"):
            downstream_positions([_hydro(0, 9)], (0,))

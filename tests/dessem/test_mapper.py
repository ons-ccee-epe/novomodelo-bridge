"""Tests for dessem/exporters/fcf/mapper.py."""

import pytest

from novomodelo_bridge.dessem.exporters.fcf.core.inputs_reader import (
    CaseInputs,
    HydroInfo,
    ThermalInfo,
)
from novomodelo_bridge.dessem.exporters.fcf.core.mapper import (
    DAYS_PER_YEAR,
    DECOMP_COST_DIVISOR,
    EXCLUDED_BUS_ID,
    SUBMARKET_COUNT,
    MappingError,
    SourceGnlThermal,
    SourceHydro,
    discount_factors,
    downstream_positions,
    gnl_block_weights,
    gnl_lag_months,
    match_source_hydros,
    negate_gnl,
    negate_gnl_array,
    plant_codes,
    select_submarket_buses,
    storage_plant_ids,
    submarket_for_bus,
    to_decomp_cost,
    to_decomp_costs,
)
from novomodelo_bridge.dessem.exporters.fcf.core.reader import EntitySlotRecord


class TestCostConversion:
    """Test P2: cost division by 1000."""

    def test_to_decomp_cost(self) -> None:
        assert to_decomp_cost(1000.0) == 1.0
        assert to_decomp_cost(0.0) == 0.0
        assert to_decomp_cost(-1000.0) == -1.0

    def test_to_decomp_costs_array(self) -> None:
        import numpy as np

        values = np.array([1000.0, 2000.0, 3000.0])
        result = to_decomp_costs(values)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_decomp_cost_divisor_constant(self) -> None:
        assert DECOMP_COST_DIVISOR == 1000.0


class TestGnlSignConvention:
    """Test P4: GNL sign negation."""

    def test_negate_gnl_single(self) -> None:
        assert negate_gnl(1.0) == -1.0
        assert negate_gnl(-1.0) == 1.0
        assert negate_gnl(0.0) == 0.0

    def test_negate_gnl_array(self) -> None:
        import numpy as np

        values = np.array([1.0, -2.0, 0.0])
        result = negate_gnl_array(values)
        assert list(result) == [-1.0, 2.0, 0.0]


class TestGnlBlockWeights:
    """Test P5: GNL load block weighting."""

    def test_weights_sum_to_one(self, synthetic_inputs: CaseInputs) -> None:
        blocks = synthetic_inputs.stages[0].blocks
        weights = gnl_block_weights(blocks)
        assert sum(weights) == pytest.approx(1.0)

    def test_weights_proportional_to_hours(self, synthetic_inputs: CaseInputs) -> None:
        blocks = synthetic_inputs.stages[0].blocks
        weights = gnl_block_weights(blocks)
        total_hours = sum(b.hours for b in blocks)
        for i, block in enumerate(blocks):
            assert weights[i] == pytest.approx(block.hours / total_hours)

    def test_empty_blocks_raises(self) -> None:
        with pytest.raises(MappingError, match="empty block list"):
            gnl_block_weights([])


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


_SENTINEL = -2147483648


def _slot(entity_type: int, entity_id: int) -> EntitySlotRecord:
    return EntitySlotRecord(
        entity_type=entity_type,
        entity_id=entity_id,
        subindex=0,
        was_active=True,
        reference_date=_SENTINEL,
        interval_start=_SENTINEL,
        interval_end=_SENTINEL,
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

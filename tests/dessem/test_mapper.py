"""Tests for dessem/exporters/fcf/mapper.py."""

from pathlib import Path

import pytest

from novomodelo_bridge.dessem.exporters.fcf.core.inputs_reader import CaseInputs
from novomodelo_bridge.dessem.exporters.fcf.core.mapper import (
    DAYS_PER_YEAR,
    DECOMP_COST_DIVISOR,
    EXCLUDED_BUS_ID,
    SUBMARKET_COUNT,
    MappingError,
    discount_factors,
    gnl_block_weights,
    hydro_code_for_position,
    load_hydro_codes,
    negate_gnl,
    negate_gnl_array,
    select_submarket_buses,
    submarket_for_bus,
    to_decomp_cost,
    to_decomp_costs,
)


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


class TestHydroCodes:
    """Test hydro code loading and lookup."""

    def test_load_hydro_codes(self, tmp_path: Path) -> None:
        codes_file = tmp_path / "codes.json"
        codes_file.write_text('{"0": 101, "1": 102, "2": 103}')
        codes = load_hydro_codes(codes_file)
        assert codes == (101, 102, 103)

    def test_load_hydro_codes_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(MappingError, match="not found"):
            load_hydro_codes(tmp_path / "nonexistent.json")

    def test_load_hydro_codes_invalid_json(self, tmp_path: Path) -> None:
        codes_file = tmp_path / "codes.json"
        codes_file.write_text("not json")
        with pytest.raises(MappingError, match="not valid JSON"):
            load_hydro_codes(codes_file)

    def test_load_hydro_codes_not_object(self, tmp_path: Path) -> None:
        codes_file = tmp_path / "codes.json"
        codes_file.write_text("[1, 2, 3]")
        with pytest.raises(MappingError, match="JSON object"):
            load_hydro_codes(codes_file)

    def test_load_hydro_codes_non_contiguous(self, tmp_path: Path) -> None:
        codes_file = tmp_path / "codes.json"
        codes_file.write_text('{"0": 1, "2": 3}')  # Missing key "1"
        with pytest.raises(MappingError, match="not contiguous"):
            load_hydro_codes(codes_file)

    def test_hydro_code_for_position_valid(self) -> None:
        codes = (101, 102, 103)
        assert hydro_code_for_position(codes, 0) == 101
        assert hydro_code_for_position(codes, 2) == 103

    def test_hydro_code_for_position_out_of_range(self) -> None:
        codes = (101, 102, 103)
        with pytest.raises(MappingError, match="outside"):
            hydro_code_for_position(codes, 3)

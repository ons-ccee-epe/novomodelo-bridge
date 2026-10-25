"""Tests for dessem/exporters/fcf/layout.py."""

import pytest

from novomodelo_bridge.dessem.exporters.fcf.core.layout import (
    PHYSICAL_RECORD_COUNT,
    PHYSICAL_RECORD_FIRST,
    PHYSICAL_RECORD_LAST,
    RECORD_SIZE,
    TAMANHO_CORTE,
    LayoutError,
    assert_uniform_blocks,
    chain_pointer,
    cortdeco_block_offsets,
    cortdeco_ncoef,
    cortdeco_record_count,
    cut_head_indices,
    mapcut_record_count,
    travel_time_lag_starts,
)


class TestConstants:
    """Test layout constants."""

    def test_record_size(self) -> None:
        assert RECORD_SIZE == 48020

    def test_tamanho_corte(self) -> None:
        assert TAMANHO_CORTE == 26976

    def test_physical_record_span(self) -> None:
        assert PHYSICAL_RECORD_FIRST == 4
        assert PHYSICAL_RECORD_LAST == 17
        assert PHYSICAL_RECORD_COUNT == 14


class TestMapcutRecordCount:
    """Test mapcut record count calculation."""

    def test_minimal_case(self) -> None:
        count = mapcut_record_count((), 1, 1)
        # header(4) + physical(14) + tree(1) + stage(1) + reg 9 (1) + reg 10 (1)
        assert count == 4 + 14 + 1 + 1 + 1 + 1

    def test_travel_time_adds_one_record_per_plant_stage_and_lag(self) -> None:
        assert mapcut_record_count((3, 1, 3, 1), 2, 2) == mapcut_record_count(
            (), 2, 2
        ) + (4 + 2 + 4 + 2)

    def test_negative_values_raise(self) -> None:
        with pytest.raises(LayoutError):
            mapcut_record_count((-1,), 1, 1)
        with pytest.raises(LayoutError):
            mapcut_record_count((), -1, 1)


class TestTravelTimeLagStarts:
    """Reg 6's lag starts, as the reference DECOMP writes them."""

    def test_two_plants_with_different_lags(self) -> None:
        lags = (3, 1) * 4

        assert travel_time_lag_starts(lags, 2, 4) == (1, 13, 4, 14, 7, 15, 10, 16)

    def test_swapped_plant_order_numbers_the_shorter_lags_first(self) -> None:
        lags = (1, 3) * 4

        assert travel_time_lag_starts(lags, 2, 4) == (1, 5, 2, 8, 3, 11, 4, 14)

    def test_lag_count_must_match_plants_and_stages(self) -> None:
        with pytest.raises(LayoutError, match="2 plants and 3 stages"):
            travel_time_lag_starts((3, 1, 3, 1), 2, 3)


class TestCortdecoNcoef:
    """Test NCOEF calculation."""

    def test_basic_ncoef(self) -> None:
        # n_uhes=3, n_utv=0, max_lag=0, n_sbm_gnl=2, n_estagios=3, n_patamares=3
        ncoef = cortdeco_ncoef(3, 0, 0, 2, 3, 3)
        # 1 + 3 + 0 + 2*3*3 = 1 + 3 + 18 = 22
        assert ncoef == 22

    def test_ncoef_no_gnl(self) -> None:
        # n_uhes=5, n_utv=0, max_lag=0, n_sbm_gnl=0
        ncoef = cortdeco_ncoef(5, 0, 0, 0, 3, 3)
        assert ncoef == 6  # 1 + 5

    def test_negative_raises(self) -> None:
        with pytest.raises(LayoutError):
            cortdeco_ncoef(-1, 0, 0, 0, 0, 0)


class TestCortdecoBlockOffsets:
    """Test block offset calculation."""

    def test_offsets_basic(self) -> None:
        offsets = cortdeco_block_offsets(3, 0, 0, 2, 3, 3)
        assert offsets.rhs == 0
        assert offsets.pi_varm == 1
        assert offsets.pi_gnl == 4  # 1 + 3
        assert offsets.ncoef == 22

    def test_offsets_no_gnl(self) -> None:
        offsets = cortdeco_block_offsets(5, 0, 0, 0, 3, 3)
        assert offsets.pi_gnl == 6  # 1 + 5
        assert offsets.ncoef == 6


class TestCutHeadIndices:
    """Test cut head index calculation."""

    def test_basic_heads(self) -> None:
        # 12 cuts over 3 nodes, 5 total nodes
        heads = cut_head_indices(12, 3, 5)
        assert heads == (12, 11, 10, 0, 0)

    def test_single_node(self) -> None:
        heads = cut_head_indices(5, 1, 1)
        assert heads == (5,)

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(LayoutError):
            cut_head_indices(-1, 1, 1)
        with pytest.raises(LayoutError):
            cut_head_indices(10, 3, 2)  # n_nodes > n_nodes_total


class TestCortdecoRecordCount:
    """Test cortdeco record count."""

    def test_record_count(self) -> None:
        # P13: numero_cortes + 1
        assert cortdeco_record_count(100) == 101
        assert cortdeco_record_count(0) == 1

    def test_negative_raises(self) -> None:
        with pytest.raises(LayoutError):
            cortdeco_record_count(-1)


class TestChainPointer:
    """Test chain pointer calculation."""

    def test_first_records_zero(self) -> None:
        # First n_nodes records have pointer 0 (chain terminators)
        assert chain_pointer(0, 3) == 0
        assert chain_pointer(1, 3) == 0
        assert chain_pointer(2, 3) == 0

    def test_subsequent_records(self) -> None:
        # Record 3 points to 1, record 4 points to 2, etc.
        assert chain_pointer(3, 3) == 1
        assert chain_pointer(4, 3) == 2
        assert chain_pointer(5, 3) == 3

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(LayoutError):
            chain_pointer(-1, 3)
        with pytest.raises(LayoutError):
            chain_pointer(0, 0)


class TestAssertUniformBlocks:
    """Test uniform block assertion."""

    def test_uniform(self) -> None:
        result = assert_uniform_blocks((3, 3, 3))
        assert result == 3

    def test_single_stage(self) -> None:
        result = assert_uniform_blocks((5,))
        assert result == 5

    def test_non_uniform_raises(self) -> None:
        with pytest.raises(LayoutError, match="not uniform"):
            assert_uniform_blocks((3, 3, 4))

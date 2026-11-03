"""Tests for dessem/exporters/fcf/layout.py."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from novomodelo_bridge.dessem.exporters.fcf.core.layout import (
    PHYSICAL_RECORD_COUNT,
    PHYSICAL_RECORD_FIRST,
    PHYSICAL_RECORD_LAST,
    RECORD_SIZE,
    TAMANHO_CORTE,
    CutBlockOffsets,
    LayoutError,
    assert_gnl_support,
    assert_uniform_blocks,
    chain_pointer,
    cortdeco_block_offsets,
    cortdeco_gnl_offset,
    cortdeco_ncoef,
    cortdeco_record_count,
    cut_head_indices,
    gnl_positions,
    mapcut_record_count,
    sync_directory,
    travel_time_lag_starts,
)


def test_sync_directory_opens_no_directory_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: object) -> int:
        raise PermissionError("a directory cannot be opened as a file")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "open", refuse)

    sync_directory(tmp_path)


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
        assert (offsets.n_sbm_gnl, offsets.n_estagios, offsets.n_patamares) == (2, 3, 3)

    def test_offsets_no_gnl(self) -> None:
        offsets = cortdeco_block_offsets(5, 0, 0, 0, 3, 3)
        assert offsets.pi_gnl == 6  # 1 + 5
        assert offsets.ncoef == 6


class TestCutBlockOffsets:
    """Test the pi_gnl width invariant of CutBlockOffsets."""

    def test_post_init_width_mismatch_raises(self) -> None:
        with pytest.raises(LayoutError, match="17 wide"):
            CutBlockOffsets(
                rhs=0,
                pi_varm=1,
                pi_qdefp=4,
                pi_gnl=4,
                ncoef=21,
                n_sbm_gnl=2,
                n_estagios=3,
                n_patamares=3,
            )

    def test_post_init_negative_length_raises(self) -> None:
        # (-1) * (-1) * 1 equals the width of 1, so only the sign check can reject it
        with pytest.raises(LayoutError, match="non-negative"):
            CutBlockOffsets(
                rhs=0,
                pi_varm=1,
                pi_qdefp=4,
                pi_gnl=4,
                ncoef=5,
                n_sbm_gnl=-1,
                n_estagios=-1,
                n_patamares=1,
            )

    def test_post_init_accepts_no_gnl(self) -> None:
        offsets = CutBlockOffsets(
            rhs=0,
            pi_varm=1,
            pi_qdefp=4,
            pi_gnl=4,
            ncoef=4,
            n_sbm_gnl=0,
            n_estagios=3,
            n_patamares=3,
        )
        assert offsets.ncoef - offsets.pi_gnl == 0


class TestCortdecoGnlOffset:
    """Test pi_gnl addressing through the layout."""

    def test_gnl_offset_entry_one_patamar_two(self) -> None:
        offsets = cortdeco_block_offsets(3, 0, 0, 2, 3, 3)
        assert cortdeco_gnl_offset(offsets, 1, 0, 2) == 15  # 4 + (1*3 + 0)*3 + 2

    def test_gnl_offset_first_position_is_pi_gnl(self) -> None:
        offsets = cortdeco_block_offsets(3, 0, 0, 2, 3, 3)
        assert cortdeco_gnl_offset(offsets, 0, 0, 0) == offsets.pi_gnl

    def test_gnl_offset_last_position_is_last_slot(self) -> None:
        offsets = cortdeco_block_offsets(3, 0, 0, 2, 3, 3)
        assert cortdeco_gnl_offset(offsets, 1, 2, 2) == offsets.ncoef - 1

    @pytest.mark.parametrize(
        ("axis", "position"),
        [("entry", (2, 0, 0)), ("middle", (0, 3, 0)), ("patamar", (0, 0, 3))],
    )
    def test_gnl_offset_out_of_range_raises(
        self, axis: str, position: tuple[int, int, int]
    ) -> None:
        offsets = cortdeco_block_offsets(3, 0, 0, 2, 3, 3)
        with pytest.raises(LayoutError, match=axis):
            cortdeco_gnl_offset(offsets, *position)


class TestGnlPositions:
    """Test on-disk iteration of pi_gnl positions."""

    def test_gnl_positions_cover_pi_gnl_in_disk_order(self) -> None:
        offsets = cortdeco_block_offsets(3, 0, 0, 2, 3, 3)
        addresses = [
            cortdeco_gnl_offset(offsets, *position)
            for position in gnl_positions(2, 3, 3)
        ]
        assert addresses == list(range(4, 22))

    def test_gnl_positions_without_entries_yield_nothing(self) -> None:
        assert list(gnl_positions(0, 3, 3)) == []


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


GNL_OFFSETS = cortdeco_block_offsets(3, 0, 0, 2, 3, 3)


def _write_cortdeco(
    path: Path,
    records: int,
    n_nodes: int,
    gnl: dict[tuple[int, int, int, int], float],
    pointers: dict[int, int] | None = None,
) -> None:
    """Write raw cuts; `gnl` maps (record, entry, middle, patamar) to its value."""
    payload = bytearray()
    for index in range(records):
        pointer = (pointers or {}).get(index, chain_pointer(index, n_nodes))
        cut = np.zeros(GNL_OFFSETS.ncoef, dtype="<f8")
        for (record, *position), value in gnl.items():
            if record == index:
                cut[cortdeco_gnl_offset(GNL_OFFSETS, *position)] = value
        body = np.int32(pointer).astype("<i4").tobytes() + cut.tobytes()
        payload += body.ljust(TAMANHO_CORTE, b"\x00")
    path.write_bytes(payload)


def _node_zero_values(
    records: tuple[int, ...],
) -> dict[tuple[int, int, int, int], float]:
    return {
        (record, entry, 0, patamar): record + 0.5 * (entry + 1)
        for record in records
        for entry in range(2)
        for patamar in range(3)
    }


class TestAssertGnlSupport:
    """Test the byte-level guard on where pi_gnl may be nonzero."""

    def test_two_node_file_with_node_zero_values_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "cortdeco"
        _write_cortdeco(path, 5, 2, _node_zero_values((1, 3)))

        assert assert_gnl_support(path, 5, GNL_OFFSETS, 2) is None

    def test_file_without_gnl_values_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "cortdeco"
        _write_cortdeco(path, 5, 2, {})

        assert assert_gnl_support(path, 5, GNL_OFFSETS, 2) is None

    @pytest.mark.parametrize("value", [1.5, -0.0], ids=["nonzero", "negative-zero"])
    def test_value_outside_node_zero_raises(self, tmp_path: Path, value: float) -> None:
        path = tmp_path / "cortdeco"
        gnl = _node_zero_values((1, 3)) | {(2, 0, 0, 0): value}
        _write_cortdeco(path, 5, 2, gnl)

        with pytest.raises(LayoutError, match="first cut-building node") as caught:
            assert_gnl_support(path, 5, GNL_OFFSETS, 2)
        assert f"record 2's pi_gnl entry 0, middle 0, patamar 0 is {value!r}" in str(
            caught.value
        )

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (
                {(3, 1, 1, 2): 4.0},
                "record 3's pi_gnl entry 1, middle 1, patamar 2 is 4.0: only middle "
                "offset 0 carries a GNL coefficient",
            ),
            (
                {(3, 1, 0, 2): float("nan")},
                "record 3's pi_gnl entry 1, middle 0, patamar 2 is nan: "
                "a GNL coefficient must be finite",
            ),
            (
                {(1, 0, 0, 2): 9.0},
                "record 1's pi_gnl entry 0, middle 0: patamar 0 is 1.5 but "
                "patamar 2 is 9.0",
            ),
        ],
        ids=["middle-offset-1", "not-finite", "unequal-patamares"],
    )
    def test_node_zero_violation_raises(
        self,
        tmp_path: Path,
        change: dict[tuple[int, int, int, int], float],
        message: str,
    ) -> None:
        path = tmp_path / "cortdeco"
        _write_cortdeco(path, 5, 2, _node_zero_values((1, 3)) | change)

        with pytest.raises(LayoutError) as caught:
            assert_gnl_support(path, 5, GNL_OFFSETS, 2)
        assert message in str(caught.value)

    def test_equal_patamares_are_compared_by_bits(self, tmp_path: Path) -> None:
        path = tmp_path / "cortdeco"
        gnl = _node_zero_values((1, 3)) | {
            (1, 0, 0, 0): 0.0,
            (1, 0, 0, 1): -0.0,
            (1, 0, 0, 2): 0.0,
        }
        _write_cortdeco(path, 5, 2, gnl)

        with pytest.raises(LayoutError, match="patamar 1 is -0.0"):
            assert_gnl_support(path, 5, GNL_OFFSETS, 2)

    def test_one_node_file_feeds_the_extra_record(self, tmp_path: Path) -> None:
        path = tmp_path / "cortdeco"
        values = _node_zero_values((1,))
        gnl = values | {(2, *position[1:]): value for position, value in values.items()}
        _write_cortdeco(path, 3, 1, gnl)

        assert assert_gnl_support(path, 3, GNL_OFFSETS, 1) is None

    def test_two_node_file_refuses_the_extra_record(self, tmp_path: Path) -> None:
        path = tmp_path / "cortdeco"
        _write_cortdeco(path, 5, 2, _node_zero_values((1, 3)) | {(4, 0, 0, 0): 1.5})

        with pytest.raises(LayoutError, match="record 4's pi_gnl"):
            assert_gnl_support(path, 5, GNL_OFFSETS, 2)

    def test_without_gnl_entries_returns_before_opening_the_file(
        self, tmp_path: Path
    ) -> None:
        offsets = cortdeco_block_offsets(3, 0, 0, 0, 3, 3)

        assert assert_gnl_support(tmp_path / "missing", 5, offsets, 2) is None

    @pytest.mark.parametrize(
        ("pointers", "message"),
        [({3: 99}, "reached record 98"), ({1: 4}, "record 3 reached twice")],
        ids=["out-of-range", "repeated"],
    )
    def test_broken_chain_pointer_raises(
        self, tmp_path: Path, pointers: dict[int, int], message: str
    ) -> None:
        path = tmp_path / "cortdeco"
        _write_cortdeco(path, 5, 2, {}, pointers)

        with pytest.raises(LayoutError, match=message):
            assert_gnl_support(path, 5, GNL_OFFSETS, 2)

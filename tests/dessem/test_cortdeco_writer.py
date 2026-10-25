"""Tests for dessem/exporters/fcf/cortdeco_writer.py."""

from pathlib import Path

import numpy as np
import pytest

from novomodelo_bridge.dessem.exporters.fcf.core.cortdeco_writer import (
    CutInput,
    serialize_cut,
    write_cortdeco,
    zeroed_gnl_block,
)
from novomodelo_bridge.dessem.exporters.fcf.core.layout import (
    TAMANHO_CORTE,
    CutBlockOffsets,
    LayoutError,
)


@pytest.fixture
def simple_offsets() -> CutBlockOffsets:
    """Simple offsets for 3 hydros, no GNL."""
    return CutBlockOffsets(rhs=0, pi_varm=1, pi_qdefp=4, pi_gnl=4, ncoef=4)


@pytest.fixture
def offsets_with_gnl() -> CutBlockOffsets:
    """Offsets with GNL block: 3 hydros, 2*3*3=18 GNL slots."""
    return CutBlockOffsets(rhs=0, pi_varm=1, pi_qdefp=4, pi_gnl=4, ncoef=22)


class TestZeroedGnlBlock:
    """Test P14: zeroed GNL block creation."""

    def test_correct_size(self, offsets_with_gnl: CutBlockOffsets) -> None:
        block = zeroed_gnl_block(offsets_with_gnl)
        expected_size = offsets_with_gnl.ncoef - offsets_with_gnl.pi_gnl
        assert len(block) == expected_size

    def test_all_zeros(self, offsets_with_gnl: CutBlockOffsets) -> None:
        block = zeroed_gnl_block(offsets_with_gnl)
        assert np.all(block == 0.0)

    def test_dtype_float64(self, offsets_with_gnl: CutBlockOffsets) -> None:
        block = zeroed_gnl_block(offsets_with_gnl)
        assert block.dtype == np.float64


class TestCutInput:
    """Test CutInput dataclass."""

    def test_cut_input_frozen(self, simple_offsets: CutBlockOffsets) -> None:
        cut = CutInput(
            intercept=1000.0,
            pi_varm=np.array([1.0, 2.0, 3.0]),
            pi_qdefp=np.array([]),
            pi_gnl=np.array([]),
        )
        with pytest.raises(AttributeError):
            cut.intercept = 2000.0  # type: ignore[misc]  # intentional mutation to test frozen


class TestSerializeCut:
    """Test cut record serialization."""

    def test_produces_correct_size(self, simple_offsets: CutBlockOffsets) -> None:
        cut = CutInput(
            intercept=1000.0,
            pi_varm=np.array([1.0, 2.0, 3.0]),
            pi_qdefp=np.array([]),
            pi_gnl=np.array([]),
        )
        record = serialize_cut(cut, pointer=0, offsets=simple_offsets)
        assert len(record) == TAMANHO_CORTE

    def test_pointer_at_offset_zero(self, simple_offsets: CutBlockOffsets) -> None:
        cut = CutInput(
            intercept=1000.0,
            pi_varm=np.array([1.0, 2.0, 3.0]),
            pi_qdefp=np.array([]),
            pi_gnl=np.array([]),
        )
        record = serialize_cut(cut, pointer=42, offsets=simple_offsets)
        pointer = np.frombuffer(record[:4], dtype="<i4")[0]
        assert pointer == 42

    def test_p2_cost_division(self, simple_offsets: CutBlockOffsets) -> None:
        cut = CutInput(
            intercept=1000.0,  # Should become 1.0 on disk
            pi_varm=np.array([1000.0, 2000.0, 3000.0]),
            pi_qdefp=np.array([]),
            pi_gnl=np.array([]),
        )
        record = serialize_cut(cut, pointer=0, offsets=simple_offsets)

        # Read back the RHS (at offset 4, after 4-byte pointer)
        rhs = np.frombuffer(record[4:12], dtype="<f8")[0]
        assert rhs == pytest.approx(1.0)

    def test_mismatched_pi_varm_raises(self, simple_offsets: CutBlockOffsets) -> None:
        cut = CutInput(
            intercept=1000.0,
            pi_varm=np.array([1.0, 2.0]),  # Wrong size
            pi_qdefp=np.array([]),
            pi_gnl=np.array([]),
        )
        with pytest.raises(LayoutError, match="pi_varm"):
            serialize_cut(cut, pointer=0, offsets=simple_offsets)

    def test_pi_qdefp_sits_between_storage_and_gnl_unnegated(self) -> None:
        offsets = CutBlockOffsets(rhs=0, pi_varm=1, pi_qdefp=3, pi_gnl=5, ncoef=7)
        cut = CutInput(
            intercept=1000.0,
            pi_varm=np.array([1000.0, 2000.0]),
            pi_qdefp=np.array([-3000.0, -4000.0]),
            pi_gnl=np.array([0.0, 0.0]),
        )

        record = serialize_cut(cut, pointer=0, offsets=offsets)

        values = np.frombuffer(record, dtype="<f8", count=7, offset=4)
        assert values.tolist() == [1.0, 1.0, 2.0, -3.0, -4.0, 0.0, 0.0]

    def test_mismatched_pi_qdefp_raises(self) -> None:
        offsets = CutBlockOffsets(rhs=0, pi_varm=1, pi_qdefp=3, pi_gnl=5, ncoef=5)
        cut = CutInput(
            intercept=1000.0,
            pi_varm=np.array([1.0, 2.0]),
            pi_qdefp=np.array([1.0]),
            pi_gnl=np.array([]),
        )
        with pytest.raises(LayoutError, match="pi_qdefp has 1 entries"):
            serialize_cut(cut, pointer=0, offsets=offsets)

    def test_non_finite_pi_qdefp_names_its_block(self) -> None:
        offsets = CutBlockOffsets(rhs=0, pi_varm=1, pi_qdefp=3, pi_gnl=5, ncoef=5)
        cut = CutInput(
            intercept=1000.0,
            pi_varm=np.array([1.0, 2.0]),
            pi_qdefp=np.array([1.0, np.nan]),
            pi_gnl=np.array([]),
        )
        with pytest.raises(LayoutError, match="pi_qdefp position 1"):
            serialize_cut(cut, pointer=0, offsets=offsets)


class TestWriteCortdeco:
    """Test write_cortdeco function."""

    def test_write_produces_file(
        self, tmp_path: Path, simple_offsets: CutBlockOffsets
    ) -> None:
        cuts = [
            [
                CutInput(
                    intercept=1000.0 * (i + 1),
                    pi_varm=np.array([1.0, 2.0, 3.0]),
                    pi_qdefp=np.array([]),
                    pi_gnl=np.array([]),
                )
                for i in range(3)
            ]
        ]
        path = tmp_path / "cortdeco.rv0"

        write_cortdeco(cuts, path, numero_cortes=3, offsets=simple_offsets)

        assert path.exists()

    def test_p13_extra_record(
        self, tmp_path: Path, simple_offsets: CutBlockOffsets
    ) -> None:
        """P13: file has numero_cortes + 1 records."""
        cuts = [
            [
                CutInput(
                    intercept=1000.0,
                    pi_varm=np.array([1.0, 2.0, 3.0]),
                    pi_qdefp=np.array([]),
                    pi_gnl=np.array([]),
                )
                for _ in range(3)
            ]
        ]
        path = tmp_path / "cortdeco.rv0"

        write_cortdeco(cuts, path, numero_cortes=3, offsets=simple_offsets)

        size = path.stat().st_size
        expected_records = 3 + 1  # P13
        assert size == expected_records * TAMANHO_CORTE

    def test_empty_cuts_raises(
        self, tmp_path: Path, simple_offsets: CutBlockOffsets
    ) -> None:
        with pytest.raises(LayoutError, match="no cut-building nodes"):
            write_cortdeco(
                [], tmp_path / "cortdeco.rv0", numero_cortes=0, offsets=simple_offsets
            )

    def test_mismatched_numero_cortes_raises(
        self, tmp_path: Path, simple_offsets: CutBlockOffsets
    ) -> None:
        cuts = [
            [
                CutInput(
                    intercept=1000.0,
                    pi_varm=np.array([1.0, 2.0, 3.0]),
                    pi_qdefp=np.array([]),
                    pi_gnl=np.array([]),
                )
            ]
        ]
        with pytest.raises(LayoutError, match="numero_cortes"):
            write_cortdeco(
                cuts, tmp_path / "cortdeco.rv0", numero_cortes=5, offsets=simple_offsets
            )

    def test_p14_nonzero_gnl_raises(
        self, tmp_path: Path, offsets_with_gnl: CutBlockOffsets
    ) -> None:
        """P14: non-zero GNL coefficients should raise."""
        gnl_size = offsets_with_gnl.ncoef - offsets_with_gnl.pi_gnl
        cuts = [
            [
                CutInput(
                    intercept=1000.0,
                    pi_varm=np.array([1.0, 2.0, 3.0]),
                    pi_qdefp=np.array([]),
                    pi_gnl=np.ones(gnl_size),  # Non-zero!
                )
            ]
        ]
        with pytest.raises(LayoutError, match="P14"):
            write_cortdeco(
                cuts,
                tmp_path / "cortdeco.rv0",
                numero_cortes=1,
                offsets=offsets_with_gnl,
            )

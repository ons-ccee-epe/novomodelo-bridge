"""Tests for dessem/exporters/fcf/cortdeco_writer.py."""

from pathlib import Path

import numpy as np
import pytest

from novomodelo_bridge.dessem.exporters.fcf.core.cortdeco_writer import (
    CutInput,
    serialize_cut,
    write_cortdeco,
)
from novomodelo_bridge.dessem.exporters.fcf.core.layout import (
    TAMANHO_CORTE,
    CutBlockOffsets,
    LayoutError,
)


@pytest.fixture
def simple_offsets() -> CutBlockOffsets:
    """Simple offsets for 3 hydros, no GNL."""
    return CutBlockOffsets(
        rhs=0,
        pi_varm=1,
        pi_qdefp=4,
        pi_gnl=4,
        ncoef=4,
        n_sbm_gnl=0,
        n_estagios=3,
        n_patamares=3,
    )


@pytest.fixture
def offsets_with_gnl() -> CutBlockOffsets:
    """Offsets with GNL block: 3 hydros, 2*3*3=18 GNL slots."""
    return CutBlockOffsets(
        rhs=0,
        pi_varm=1,
        pi_qdefp=4,
        pi_gnl=4,
        ncoef=22,
        n_sbm_gnl=2,
        n_estagios=3,
        n_patamares=3,
    )


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
        offsets = CutBlockOffsets(
            rhs=0,
            pi_varm=1,
            pi_qdefp=3,
            pi_gnl=5,
            ncoef=7,
            n_sbm_gnl=1,
            n_estagios=2,
            n_patamares=1,
        )
        cut = CutInput(
            intercept=1000.0,
            pi_varm=np.array([1000.0, 2000.0]),
            pi_qdefp=np.array([-3000.0, -4000.0]),
            pi_gnl=np.array([-150.0, -150.0]),
        )

        record = serialize_cut(cut, pointer=0, offsets=offsets)

        values = np.frombuffer(record, dtype="<f8", count=7, offset=4)
        assert values.tolist() == [1.0, 1.0, 2.0, -3.0, -4.0, 150.0, 150.0]

    def test_p2_exempts_pi_gnl(self) -> None:
        offsets = CutBlockOffsets(
            rhs=0,
            pi_varm=1,
            pi_qdefp=3,
            pi_gnl=3,
            ncoef=5,
            n_sbm_gnl=1,
            n_estagios=2,
            n_patamares=1,
        )
        cut = CutInput(
            intercept=1000.0,
            pi_varm=np.array([1000.0, 2000.0]),
            pi_qdefp=np.array([]),
            pi_gnl=np.array([-200.0, 0.0]),
        )

        record = serialize_cut(cut, pointer=0, offsets=offsets)

        values = np.frombuffer(record, dtype="<f8", count=5, offset=4)
        assert values.tolist() == [1.0, 1.0, 2.0, 200.0, 0.0]

    def test_mismatched_pi_qdefp_raises(self) -> None:
        offsets = CutBlockOffsets(
            rhs=0,
            pi_varm=1,
            pi_qdefp=3,
            pi_gnl=5,
            ncoef=5,
            n_sbm_gnl=0,
            n_estagios=1,
            n_patamares=1,
        )
        cut = CutInput(
            intercept=1000.0,
            pi_varm=np.array([1.0, 2.0]),
            pi_qdefp=np.array([1.0]),
            pi_gnl=np.array([]),
        )
        with pytest.raises(LayoutError, match="pi_qdefp has 1 entries"):
            serialize_cut(cut, pointer=0, offsets=offsets)

    def test_non_finite_pi_qdefp_names_its_block(self) -> None:
        offsets = CutBlockOffsets(
            rhs=0,
            pi_varm=1,
            pi_qdefp=3,
            pi_gnl=5,
            ncoef=5,
            n_sbm_gnl=0,
            n_estagios=1,
            n_patamares=1,
        )
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

    def test_later_node_gnl_is_refused_without_leaving_files(
        self, tmp_path: Path, offsets_with_gnl: CutBlockOffsets
    ) -> None:
        gnl_size = offsets_with_gnl.ncoef - offsets_with_gnl.pi_gnl
        cuts = [
            [
                CutInput(
                    intercept=1000.0,
                    pi_varm=np.array([1.0, 2.0, 3.0]),
                    pi_qdefp=np.array([]),
                    pi_gnl=pi_gnl,
                )
            ]
            for pi_gnl in (np.zeros(gnl_size), np.ones(gnl_size))
        ]
        path = tmp_path / "cortdeco.rv0"

        with pytest.raises(LayoutError, match="first cut-building node"):
            write_cortdeco(cuts, path, numero_cortes=2, offsets=offsets_with_gnl)

        assert not path.exists()
        assert not path.with_name(path.name + ".partial").exists()

    def test_first_node_replicated_gnl_is_written(
        self, tmp_path: Path, offsets_with_gnl: CutBlockOffsets
    ) -> None:
        gnl_size = offsets_with_gnl.ncoef - offsets_with_gnl.pi_gnl
        first_node_gnl = np.zeros(gnl_size)
        first_node_gnl[:3] = -200.0
        cuts = [
            [
                CutInput(
                    intercept=1000.0,
                    pi_varm=np.array([1.0, 2.0, 3.0]),
                    pi_qdefp=np.array([]),
                    pi_gnl=pi_gnl,
                )
            ]
            for pi_gnl in (first_node_gnl, np.zeros(gnl_size))
        ]
        path = tmp_path / "cortdeco.rv0"

        write_cortdeco(cuts, path, numero_cortes=2, offsets=offsets_with_gnl)

        node_0_record = 1
        start = node_0_record * TAMANHO_CORTE + 4 + 8 * offsets_with_gnl.pi_gnl
        written = np.frombuffer(
            path.read_bytes()[start : start + 8 * gnl_size], dtype="<f8"
        )
        assert written[:3].tolist() == [200.0, 200.0, 200.0]
        assert not written[3:].any()

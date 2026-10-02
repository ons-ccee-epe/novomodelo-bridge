"""Tests for dessem/exporters/fcf/mapcut_writer.py."""

from pathlib import Path

import pytest

from novomodelo_bridge.dessem.exporters.fcf.core.layout import RECORD_SIZE, LayoutError
from novomodelo_bridge.dessem.exporters.fcf.core.mapcut_writer import (
    MapcutHeader,
    write_mapcut,
)


@pytest.fixture
def minimal_header() -> MapcutHeader:
    """A minimal valid mapcut header."""
    return MapcutHeader(
        numero_iteracoes=10,
        numero_cortes=6,
        numero_submercados=5,
        numero_uhes=3,
        numero_cenarios=2,
        numero_estagios=2,
        numero_semanas=2,
        n_utv=0,
        dia=1,
        mes=1,
        ano=2026,
        codigos_uhes=(1, 2, 3),
        codigos_uhes_jusante=(0, 1, 1),
        indice_no_arvore=(1, 1),
        indice_primeiro_no_estagio=(1, 2),
        patamares_por_estagio=(3, 3),
        registro_ultimo_corte_no=(6, 5),
        codigos_submercados_gnl=(1, 2),
        lag_meses_gnl=(2, 2),
        patamares_gnl=(3, 3),
        taxa_desconto=(1.0, 0.998),
        max_lag=0,
    )


class TestMapcutHeader:
    """Test MapcutHeader dataclass."""

    def test_header_is_frozen(self, minimal_header: MapcutHeader) -> None:
        with pytest.raises(AttributeError):
            minimal_header.numero_cortes = 100  # type: ignore[misc]


class TestWriteMapcut:
    """Test write_mapcut function."""

    def test_write_produces_file(
        self, tmp_path: Path, minimal_header: MapcutHeader
    ) -> None:
        path = tmp_path / "mapcut.rv0"
        write_mapcut(minimal_header, path)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_file_size_is_record_multiple(
        self, tmp_path: Path, minimal_header: MapcutHeader
    ) -> None:
        path = tmp_path / "mapcut.rv0"
        write_mapcut(minimal_header, path)

        size = path.stat().st_size
        assert size % RECORD_SIZE == 0

    def test_atomic_write_no_partial(
        self, tmp_path: Path, minimal_header: MapcutHeader
    ) -> None:
        path = tmp_path / "mapcut.rv0"
        write_mapcut(minimal_header, path)

        partial = path.with_name(path.name + ".partial")
        assert not partial.exists()

    def test_creates_parent_directories(
        self, tmp_path: Path, minimal_header: MapcutHeader
    ) -> None:
        path = tmp_path / "subdir" / "nested" / "mapcut.rv0"
        write_mapcut(minimal_header, path)

        assert path.exists()


class TestMapcutValidation:
    """Test header validation in write_mapcut."""

    def test_mismatched_codigos_uhes_raises(self, tmp_path: Path) -> None:
        header = MapcutHeader(
            numero_iteracoes=10,
            numero_cortes=6,
            numero_submercados=5,
            numero_uhes=5,  # Says 5
            numero_cenarios=2,
            numero_estagios=2,
            numero_semanas=2,
            n_utv=0,
            dia=1,
            mes=1,
            ano=2026,
            codigos_uhes=(1, 2, 3),  # But only 3
            codigos_uhes_jusante=(0, 1, 1),
            indice_no_arvore=(1, 1),
            indice_primeiro_no_estagio=(1, 2),
            patamares_por_estagio=(3, 3),
            registro_ultimo_corte_no=(6, 5),
            codigos_submercados_gnl=(1, 2),
            lag_meses_gnl=(2, 2),
            patamares_gnl=(3, 3),
            taxa_desconto=(1.0, 0.998),
        )

        with pytest.raises(LayoutError, match="inconsistent"):
            write_mapcut(header, tmp_path / "mapcut.rv0")

    def test_nonzero_n_utv_raises(
        self, tmp_path: Path, minimal_header: MapcutHeader
    ) -> None:
        # Create header with n_utv > 0 (not supported per P3)
        header = MapcutHeader(
            numero_iteracoes=minimal_header.numero_iteracoes,
            numero_cortes=minimal_header.numero_cortes,
            numero_submercados=minimal_header.numero_submercados,
            numero_uhes=minimal_header.numero_uhes,
            numero_cenarios=minimal_header.numero_cenarios,
            numero_estagios=minimal_header.numero_estagios,
            numero_semanas=minimal_header.numero_semanas,
            n_utv=2,  # Not supported
            dia=minimal_header.dia,
            mes=minimal_header.mes,
            ano=minimal_header.ano,
            codigos_uhes=minimal_header.codigos_uhes,
            codigos_uhes_jusante=minimal_header.codigos_uhes_jusante,
            indice_no_arvore=minimal_header.indice_no_arvore,
            indice_primeiro_no_estagio=minimal_header.indice_primeiro_no_estagio,
            patamares_por_estagio=minimal_header.patamares_por_estagio,
            registro_ultimo_corte_no=minimal_header.registro_ultimo_corte_no,
            codigos_submercados_gnl=minimal_header.codigos_submercados_gnl,
            lag_meses_gnl=minimal_header.lag_meses_gnl,
            patamares_gnl=minimal_header.patamares_gnl,
            taxa_desconto=minimal_header.taxa_desconto,
        )

        with pytest.raises(LayoutError, match="n_utv"):
            write_mapcut(header, tmp_path / "mapcut.rv0")

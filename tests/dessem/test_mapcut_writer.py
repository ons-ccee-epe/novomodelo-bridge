"""Tests for dessem/exporters/fcf/mapcut_writer.py."""

from dataclasses import replace
from pathlib import Path

import pytest

from novomodelo_bridge.dessem.exporters.fcf.core.cross_check import (
    MapcutContents,
    read_mapcut,
)
from novomodelo_bridge.dessem.exporters.fcf.core.layout import RECORD_SIZE, LayoutError
from novomodelo_bridge.dessem.exporters.fcf.core.mapcut_writer import (
    MapcutHeader,
    write_mapcut,
)
from novomodelo_bridge.dessem.exporters.fcf.core.travel_time import TravelTimePlant


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
    )


class TestMapcutHeader:
    """Test MapcutHeader dataclass."""

    def test_header_is_frozen(self, minimal_header: MapcutHeader) -> None:
        with pytest.raises(AttributeError):
            minimal_header.numero_cortes = 100  # ty: ignore[invalid-assignment] intentional mutation to test frozen


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

    def test_travel_time_plant_without_shares_for_every_stage_raises(
        self, tmp_path: Path, minimal_header: MapcutHeader
    ) -> None:
        header = replace(minimal_header, tempo_viagem=(_plant(156, 360, ((1.0,),)),))

        with pytest.raises(LayoutError, match="arrival shares for 1 stages"):
            write_mapcut(header, tmp_path / "mapcut.rv0")


def _plant(
    code: int, hours: int, arrival_shares: tuple[tuple[float, ...], ...]
) -> TravelTimePlant:
    return TravelTimePlant(
        hydro_id=code,
        code=code,
        downstream_id=0,
        hours=hours,
        arrival_shares=arrival_shares,
    )


class TestTravelTimeRecords:
    """Regs 6-8 for two plants whose lags differ, read back from disk."""

    @pytest.fixture
    def written(self, tmp_path: Path, minimal_header: MapcutHeader) -> MapcutContents:
        header = replace(
            minimal_header,
            tempo_viagem=(
                _plant(156, 360, ((0.0, 0.0, 6 / 7, 1 / 7), (4 / 9, 1.0, 1.0, 1 / 7))),
                _plant(162, 100, ((17 / 42, 25 / 42), (137 / 162, 25 / 42))),
            ),
        )
        path = tmp_path / "mapcut.rv0"
        write_mapcut(header, path)
        return read_mapcut(path)

    def test_stage_record_carries_stage_major_lags_and_their_starts(
        self, written: MapcutContents
    ) -> None:
        assert (written.scalars.n_utv, written.scalars.max_lag) == (2, 3)
        assert written.travel_time_lags == (3, 1, 3, 1)
        assert written.travel_time_lag_starts == (1, 7, 4, 8)

    def test_records_run_plant_major_with_code_and_hours_opening_each_plant(
        self, written: MapcutContents
    ) -> None:
        records = written.travel_time_records

        assert [(r.plant, r.stage, r.lag) for r in records] == [
            *((0, 0, lag) for lag in range(4)),
            *((0, 1, lag) for lag in range(4)),
            *((1, 0, lag) for lag in range(2)),
            *((1, 1, lag) for lag in range(2)),
        ]
        assert written.codigos_uhes_tempo_viagem == (156, 162)
        assert [
            (r.record_index, r.numero_horas) for r in records if r.numero_horas
        ] == [
            (20, 360),
            (28, 100),
        ]
        assert [r.share for r in records[8:]] == pytest.approx(
            [17 / 42, 25 / 42, 137 / 162, 25 / 42]
        )

    def test_gnl_records_follow_the_travel_time_records(
        self, written: MapcutContents
    ) -> None:
        assert written.gnl_record_indices[0] == 20 + 12

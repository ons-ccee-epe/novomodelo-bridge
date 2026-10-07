"""Tests for ``NewaveCase`` — the parse-once, cached the source model case object."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from novomodelo_bridge.newave.case import NewaveCase
from tests.conftest import make_case, make_nw_files


def _dger(
    *, start_year: int = 2024, start_month: int = 1, num_anos: int = 5, num_pos: int = 0
) -> MagicMock:
    dger = MagicMock()
    dger.ano_inicio_estudo = start_year
    dger.mes_inicio_estudo = start_month
    dger.num_anos_estudo = num_anos
    dger.num_anos_pos_estudo = num_pos
    return dger


def _confhd(rows: list[dict]) -> MagicMock:
    confhd = MagicMock()
    confhd.usinas = pd.DataFrame(rows)
    return confhd


def test_required_reader_parses_once_and_caches(tmp_path: Path) -> None:
    case = NewaveCase(files=make_nw_files(tmp_path))
    with patch("novomodelo_bridge.newave.case.Dger") as mock_dger:
        sentinel = object()
        mock_dger.read.return_value = sentinel
        first = case.dger
        second = case.dger
    assert first is sentinel
    assert second is sentinel
    mock_dger.read.assert_called_once_with(str(tmp_path / "dger.dat"))


def test_optional_reader_is_none_when_file_absent(tmp_path: Path) -> None:
    case = NewaveCase(files=make_nw_files(tmp_path))  # modif defaults to None
    with patch("novomodelo_bridge.newave.case.Modif") as mock_modif:
        assert case.modif is None
    mock_modif.read.assert_not_called()


def test_polinjus_is_none_when_file_absent(tmp_path: Path) -> None:
    case = NewaveCase(files=make_nw_files(tmp_path))  # polinjus defaults to None
    with patch("novomodelo_bridge.newave.case.UsinasHidreletricas") as mock_uh:
        assert case.polinjus is None
    mock_uh.read.assert_not_called()


def test_polinjus_parses_when_file_present(tmp_path: Path) -> None:
    polinjus_path = tmp_path / "polinjus.csv"
    case = NewaveCase(files=make_nw_files(tmp_path, polinjus=polinjus_path))
    with patch("novomodelo_bridge.newave.case.UsinasHidreletricas") as mock_uh:
        sentinel = object()
        mock_uh.read.return_value = sentinel
        assert case.polinjus is sentinel
        assert case.polinjus is sentinel  # cached
    mock_uh.read.assert_called_once_with(str(polinjus_path))


def test_newave_case_exph_none_when_absent(tmp_path: Path) -> None:
    case = NewaveCase(files=make_nw_files(tmp_path))  # exph defaults to None
    with patch("novomodelo_bridge.newave.case.Exph") as mock_exph:
        assert case.exph is None
    mock_exph.read.assert_not_called()


@pytest.mark.parametrize(
    ("flag", "expected"),
    [(0, True), (1, False), (2, False), (None, False)],
)
def test_fpha_enabled_reflects_dger_flag(
    tmp_path: Path, flag: int | None, expected: bool
) -> None:
    dger = MagicMock()
    dger.funcao_producao_uhe = flag
    case = make_case(tmp_path, dger=dger)
    assert case.fpha_enabled is expected


def test_optional_reader_parses_when_file_present(tmp_path: Path) -> None:
    modif_path = tmp_path / "modif.dat"
    case = NewaveCase(files=make_nw_files(tmp_path, modif=modif_path))
    with patch("novomodelo_bridge.newave.case.Modif") as mock_modif:
        sentinel = object()
        mock_modif.read.return_value = sentinel
        assert case.modif is sentinel
        assert case.modif is sentinel  # cached
    mock_modif.read.assert_called_once_with(str(modif_path))


def test_from_directory_binds_files_without_parsing(tmp_path: Path) -> None:
    files = make_nw_files(tmp_path)
    with patch(
        "novomodelo_bridge.newave.case.NewaveFiles.from_directory", return_value=files
    ) as mock_from_dir:
        case = NewaveCase.from_directory(tmp_path)
    assert case.files is files
    mock_from_dir.assert_called_once_with(tmp_path)


def test_make_case_prefills_slots_without_io(tmp_path: Path) -> None:
    sentinel = object()
    case = make_case(tmp_path, dger=sentinel)
    with patch("novomodelo_bridge.newave.case.Dger") as mock_dger:
        assert case.dger is sentinel
    mock_dger.read.assert_not_called()


def test_make_case_forces_optional_to_none(tmp_path: Path) -> None:
    case = make_case(tmp_path, cvar=None)
    with patch("novomodelo_bridge.newave.case.Cvar") as mock_cvar:
        assert case.cvar is None
    mock_cvar.read.assert_not_called()


def test_vazoes_is_not_cached_on_case(tmp_path: Path) -> None:
    # The large single-use inflow file stays a path on case.files, not a reader.
    case = NewaveCase(files=make_nw_files(tmp_path))
    assert case.files.vazoes == tmp_path / "vazoes.dat"
    assert not hasattr(NewaveCase, "vazoes")
    with pytest.raises(AttributeError):
        _ = case.vazoes  # type: ignore[attr-defined]


# --- Derived accessors -------------------------------------------------------


def test_horizon_derives_from_dger_and_caches(tmp_path: Path) -> None:
    case = make_case(tmp_path, dger=_dger(start_year=2024, start_month=9, num_anos=2))
    horizon = case.horizon
    # study_months = (13 - 9) + (2 - 1) * 12 = 16; no post-study.
    assert horizon.start_year == 2024
    assert horizon.start_month == 9
    assert horizon.total_stages == 16
    assert case.horizon is horizon  # cached (same frozen instance)


def test_active_hydros_excludes_non_existing_and_fictitious(tmp_path: Path) -> None:
    # Code 2 (ρ=0 sharing USINA_A's posto 1) is structurally fictitious; code 3
    # is non-existing (NE). Both are dropped from the active set.
    confhd = _confhd(
        [
            {
                "codigo_usina": 1,
                "nome_usina": "USINA_A",
                "posto": 1,
                "usina_existente": "EX",
            },
            {
                "codigo_usina": 2,
                "nome_usina": "FICT.X",
                "posto": 1,
                "usina_existente": "EX",
            },
            {
                "codigo_usina": 3,
                "nome_usina": "USINA_C",
                "posto": 3,
                "usina_existente": "NE",
            },
            {
                "codigo_usina": 4,
                "nome_usina": "USINA_D",
                "posto": 4,
                "usina_existente": "EX",
            },
        ]
    )
    hidr = MagicMock()
    hidr.cadastro = pd.DataFrame(
        [
            {"codigo_usina": c, "produtibilidade_especifica": r}
            for c, r in {1: 0.01, 2: 0.0, 4: 0.01}.items()
        ]
    ).set_index("codigo_usina")
    case = make_case(tmp_path, confhd=confhd, hidr=hidr)
    assert case.active_hydro_codes == [1, 4]
    assert list(case.active_hydros["codigo_usina"]) == [1, 4]


def test_id_map_built_from_cached_readers(tmp_path: Path) -> None:
    confhd = _confhd(
        [
            {"codigo_usina": 10, "nome_usina": "H1", "usina_existente": "EX"},
            {"codigo_usina": 20, "nome_usina": "H2", "usina_existente": "EX"},
        ]
    )
    conft = MagicMock()
    conft.usinas = pd.DataFrame({"codigo_usina": [7, 8]})
    sistema = MagicMock()
    sistema.custo_deficit = pd.DataFrame({"codigo_submercado": [1, 2]})
    ree = MagicMock()
    ree.rees = None

    case = make_case(tmp_path, confhd=confhd, conft=conft, sistema=sistema, ree=ree)
    id_map = case.id_map
    assert id_map.all_hydro_codes == [10, 20]
    assert id_map.hydro_id(20) == 1
    assert id_map.all_thermal_codes == [7, 8]
    assert id_map.all_bus_ids == [1, 2]
    assert case.id_map is id_map  # cached

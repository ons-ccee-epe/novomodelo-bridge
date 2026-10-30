"""Unit tests for the source model id-map builder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from novomodelo_bridge.newave.files import NewaveFiles
from novomodelo_bridge.newave.id_map import NewaveIdMap
from tests.conftest import make_case
from tests.newave.conftest import (
    _hydro_case,
    _make_confhd_df,
    _make_sistema_mock,
    _make_thermal_dger,
    _thermal_readers,
)


def _make_nw_files(
    tmp_path: Path,
    *,
    modif: Path | None = None,
    ghmin: Path | None = None,
    penalid: Path | None = None,
    vazpast: Path | None = None,
    dsvagua: Path | None = None,
    curva: Path | None = None,
    expt: Path | None = None,
    manutt: Path | None = None,
    c_adic: Path | None = None,
    cvar: Path | None = None,
    agrint: Path | None = None,
    volref_saz: Path | None = None,
    shist: Path | None = None,
) -> NewaveFiles:
    """Construct a ``NewaveFiles`` with sentinel paths under *tmp_path*.

    All required file paths point to ``tmp_path / "<name>.dat"`` regardless
    of whether the files actually exist on disk — the inewave I/O is mocked
    in the tests that use this helper.  Optional paths are passed through as-is
    (default ``None``).
    """
    return NewaveFiles(
        directory=tmp_path,
        dger=tmp_path / "dger.dat",
        confhd=tmp_path / "confhd.dat",
        conft=tmp_path / "conft.dat",
        sistema=tmp_path / "sistema.dat",
        clast=tmp_path / "clast.dat",
        term=tmp_path / "term.dat",
        ree=tmp_path / "ree.dat",
        patamar=tmp_path / "patamar.dat",
        hidr=tmp_path / "hidr.dat",
        vazoes=tmp_path / "vazoes.dat",
        modif=modif,
        ghmin=ghmin,
        penalid=penalid,
        vazpast=vazpast,
        dsvagua=dsvagua,
        curva=curva,
        expt=expt,
        exph=None,
        manutt=manutt,
        c_adic=c_adic,
        cvar=cvar,
        agrint=agrint,
        re_dat=None,
        volref_saz=volref_saz,
        shist=shist,
        adterm=None,
        libs_index=None,
        polinjus=None,
        volumes_referencia=None,
        tratamento_fpha=None,
        restricao_eletrica=None,
        estacao_bombeamento=None,
        estacao_bombeamento_submercado=None,
    )


class TestNewaveIdMap:
    def test_bus_id_remapping(self) -> None:
        id_map = NewaveIdMap(
            subsystem_ids=[3, 1, 4, 2],
            hydro_codes=[],
            thermal_codes=[],
        )
        assert id_map.bus_id(1) == 0
        assert id_map.bus_id(2) == 1
        assert id_map.bus_id(3) == 2
        assert id_map.bus_id(4) == 3

    def test_hydro_id_remapping_sorts_by_code(self) -> None:
        # Hydro Novomodelo IDs follow ascending codigo_usina (hidr.dat registry order),
        # not the order the codes are passed in.
        id_map = NewaveIdMap(
            subsystem_ids=[],
            hydro_codes=[10, 5, 20],
            thermal_codes=[],
        )
        assert id_map.hydro_id(5) == 0
        assert id_map.hydro_id(10) == 1
        assert id_map.hydro_id(20) == 2

    def test_thermal_id_remapping(self) -> None:
        id_map = NewaveIdMap(
            subsystem_ids=[],
            hydro_codes=[],
            thermal_codes=[7, 3, 15],
        )
        assert id_map.thermal_id(3) == 0
        assert id_map.thermal_id(7) == 1
        assert id_map.thermal_id(15) == 2

    def test_unknown_key_raises_key_error(self) -> None:
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[1])
        with pytest.raises(KeyError):
            id_map.bus_id(99)
        with pytest.raises(KeyError):
            id_map.hydro_id(99)
        with pytest.raises(KeyError):
            id_map.thermal_id(99)

    def test_all_hydro_codes_in_novomodelo_id_order(self) -> None:
        # Novomodelo-ID order is ascending codigo_usina, regardless of input order.
        id_map = NewaveIdMap(
            subsystem_ids=[], hydro_codes=[30, 10, 20], thermal_codes=[]
        )
        assert id_map.all_hydro_codes == [10, 20, 30]

    def test_deterministic_regardless_of_input_order(self) -> None:
        a = NewaveIdMap(subsystem_ids=[2, 1, 3], hydro_codes=[], thermal_codes=[])
        b = NewaveIdMap(subsystem_ids=[3, 1, 2], hydro_codes=[], thermal_codes=[])
        assert a.bus_id(1) == b.bus_id(1)
        assert a.bus_id(2) == b.bus_id(2)
        assert a.bus_id(3) == b.bus_id(3)


# ---------------------------------------------------------------------------
# Cross-reference consistency
# ---------------------------------------------------------------------------


class TestCrossReferenceConsistency:
    """Verify bus_id values in hydros and thermals match the buses output."""

    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1, 2],
            hydro_codes=[1, 2],
            thermal_codes=[10, 20, 30],
        )

    def test_all_bus_ids_are_valid(self, tmp_path) -> None:
        hydro_case = _hydro_case(tmp_path)

        conft, clast, term = _thermal_readers()
        nw_files = _make_nw_files(tmp_path)
        thermal_case = make_case(
            nw_files, conft=conft, clast=clast, term=term, dger=_make_thermal_dger()
        )

        from novomodelo_bridge.newave.converters.hydro import convert_hydros
        from novomodelo_bridge.newave.converters.network import convert_buses
        from novomodelo_bridge.newave.converters.thermal import convert_thermals

        # Use a shared id_map that covers both subsystems and all plants.
        id_map = NewaveIdMap(
            subsystem_ids=[1, 2, 99],
            hydro_codes=[1, 2],
            thermal_codes=[10, 20, 30],
        )

        buses_result = convert_buses(
            make_case(nw_files, sistema=_make_sistema_mock()),
            id_map,
            reference_load_mw={0: 2000.0, 1: 1000.0, 2: 0.0},
        )
        hydros_result = convert_hydros(hydro_case, id_map)
        thermals_result = convert_thermals(thermal_case, id_map)

        valid_bus_ids = {b["id"] for b in buses_result["buses"]}

        for h in hydros_result["hydros"]:
            group_bus_id = h["unit_groups"][0]["bus_id"]
            assert group_bus_id in valid_bus_ids, (
                f"Hydro '{h['name']}' has unit_groups[0].bus_id={group_bus_id}"
                " not in buses"
            )

        for t in thermals_result["thermals"]:
            assert t["bus_id"] in valid_bus_ids, (
                f"Thermal '{t['name']}' has bus_id={t['bus_id']} not in buses"
            )

    def test_downstream_ids_are_valid(self, tmp_path) -> None:
        case = _hydro_case(tmp_path)
        from novomodelo_bridge.newave.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        result = convert_hydros(case, id_map)
        valid_hydro_ids = {h["id"] for h in result["hydros"]}

        for h in result["hydros"]:
            ds = h.get("downstream_id")
            if ds is not None:
                assert ds in valid_hydro_ids, (
                    f"Hydro '{h['name']}' has downstream_id={ds} not in hydros"
                )


# ---------------------------------------------------------------------------
# id-map fictitious plant filtering
# ---------------------------------------------------------------------------


def _make_confhd_df_with_fict() -> pd.DataFrame:
    """Four plants: two real, two fictitious accounting twins.

    Each fictitious plant (ρ=0) shares its real twin's posto — the structural
    fictitious signature: ``FICT.SERRA M`` on posto 1 with ``USINA_A`` (ρ>0),
    ``FICT.CAMPO G`` on posto 3 with ``USINA_B`` (ρ>0).
    """
    return pd.DataFrame(
        {
            "codigo_usina": [1, 2, 3, 4],
            "nome_usina": ["USINA_A", "FICT.SERRA M", "USINA_B", "FICT.CAMPO G"],
            "posto": [1, 1, 3, 3],
            "codigo_usina_jusante": [pd.NA, pd.NA, 1, 2],
            "ree": [1, 1, 1, 1],
            "volume_inicial_percentual": [50.0, 60.0, 70.0, 80.0],
            "usina_existente": ["EX", "EX", "EX", "EX"],
            "usina_modificada": [0, 0, 0, 0],
        }
    )


def _fict_cadastro(rho: dict[int, float]) -> pd.DataFrame:
    """Minimal Hidr.cadastro: produtibilidade_especifica per code (for the
    structural fictitious test in id-map builds)."""
    return pd.DataFrame(
        [{"codigo_usina": c, "produtibilidade_especifica": r} for c, r in rho.items()]
    ).set_index("codigo_usina")


def _build_from_frames(confhd_df: pd.DataFrame, cadastro: pd.DataFrame) -> NewaveIdMap:
    """Build the id map from in-memory frames with the structural fictitious set."""
    from novomodelo_bridge.newave.id_map import build_id_map_from_readers
    from novomodelo_bridge.newave.plants import fictitious_codes

    confhd = MagicMock(usinas=confhd_df)
    conft = MagicMock(usinas=pd.DataFrame({"codigo_usina": []}))
    sistema = MagicMock(custo_deficit=None)
    ree_file = MagicMock(rees=None)
    fictitious = fictitious_codes(
        confhd_df, cadastro, pumping_related_codes=frozenset()
    )
    return build_id_map_from_readers(
        confhd, conft, sistema, ree_file, fictitious=fictitious
    )


class TestBuildIdMap:
    """Unit tests for ``build_id_map_from_readers`` fictitious-plant filtering."""

    def test_excludes_fictitious_plants(self) -> None:
        """FICT. plants must be absent from id_map.all_hydro_codes."""
        # Real twins generate; FICT plants have ρ=0 and share their posto →
        # structurally fictitious, so codes 2 and 4 are excluded.
        id_map = _build_from_frames(
            _make_confhd_df_with_fict(),
            _fict_cadastro({1: 0.01, 2: 0.0, 3: 0.01, 4: 0.0}),
        )

        # Only the two non-fictitious plants must appear.
        assert 1 in id_map.all_hydro_codes
        assert 3 in id_map.all_hydro_codes
        assert 2 not in id_map.all_hydro_codes, "FICT.SERRA M must be excluded"
        assert 4 not in id_map.all_hydro_codes, "FICT.CAMPO G must be excluded"
        assert len(id_map.all_hydro_codes) == 2

    def test_count_excludes_fict_plants(self) -> None:
        """15 fictitious twins among 160 existing -> 145 hydro codes in id_map.

        Each fictitious plant (ρ=0) shares a real generator's posto; the real
        plants get unique generating postos.
        """
        n_real, n_fict = 145, 15
        rho: dict[int, float] = {}
        rows = []
        for i in range(1, n_real + n_fict + 1):
            if i > n_real:
                # Fictitious twin: ρ=0, shares the posto of real plant (i - n_real).
                name, posto, r = f"FICT.PLANT_{i}", i - n_real, 0.0
            else:
                name, posto, r = f"PLANT_{i}", i, 0.01
            rho[i] = r
            rows.append(
                {
                    "codigo_usina": i,
                    "nome_usina": name,
                    "posto": posto,
                    "codigo_usina_jusante": pd.NA,
                    "ree": 1,
                    "volume_inicial_percentual": 50.0,
                    "usina_existente": "EX",
                }
            )
        id_map = _build_from_frames(pd.DataFrame(rows), _fict_cadastro(rho))
        assert len(id_map.all_hydro_codes) == n_real

    def test_no_fictitious_plants_proceeds_normally(self) -> None:
        """When no FICT. plants exist, all existing plants are included."""
        # Standard two-plant fixture, no FICT.
        id_map = _build_from_frames(_make_confhd_df(), pd.DataFrame())

        assert 1 in id_map.all_hydro_codes
        assert 2 in id_map.all_hydro_codes

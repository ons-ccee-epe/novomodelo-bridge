"""Hydro/thermal/network entity-conversion builders used only by tests/newave/."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from novomodelo_bridge.newave.id_map import NewaveIdMap
from tests.conftest import _all_converter_patches, make_case, make_nw_files


def _make_confhd_df() -> pd.DataFrame:
    """Two hydros: plant 1 upstream of plant 2, in REE 1 (subsystem 1)."""
    return pd.DataFrame(
        {
            "codigo_usina": [1, 2],
            "nome_usina": ["USINA_A", "USINA_B"],
            "posto": [1, 2],
            "codigo_usina_jusante": [pd.NA, 1],
            "ree": [1, 1],
            "volume_inicial_percentual": [50.0, 75.0],
            "usina_existente": ["EX", "EX"],
            "usina_modificada": [0, 0],
        }
    )


def _make_hidr_cadastro() -> pd.DataFrame:
    """Synthetic Hidr.cadastro for two plants.

    Both plants use ``tipo_regulacao="M"`` with a simple linear polynomial
    ``h(v) = 300 + 0.1*v`` (a0_volume_cota=300, a1_volume_cota=0.1, rest
    zero) and ``canal_fuga_medio=50.0``.  With ``tipo_perda=1`` and
    ``perdas=0.0`` the loss model leaves the net drop unchanged.

    For monthly-regulated plants the height is evaluated at 65% of useful storage
    (``v_65 = vmin + 0.65 * (vmax - vmin)``), matching the source model's
    ``produtibilidade_altura_65`` convention.

    USINA_A: [volume_minimo=100, volume_maximo=1000]
    - v_65 = 100 + 0.65 * 900 = 685.0
    - h(v_65) = 300 + 0.1 * 685.0 = 368.5
    - net_drop = 368.5 - 50.0 = 318.5
    - productivity_A = 0.9 * 318.5 = 286.65

    USINA_B: [volume_minimo=50, volume_maximo=500]
    - v_65 = 50 + 0.65 * 450 = 342.5
    - h(v_65) = 300 + 0.1 * 342.5 = 334.25
    - net_drop = 334.25 - 50.0 = 284.25
    - productivity_B = 0.85 * 284.25 = 241.6125

    Both productivities differ from their raw ``produtibilidade_especifica``
    values (0.9 and 0.85) because ``canal_fuga_medio`` is nonzero.
    """
    months = [
        "JAN",
        "FEV",
        "MAR",
        "ABR",
        "MAI",
        "JUN",
        "JUL",
        "AGO",
        "SET",
        "OUT",
        "NOV",
        "DEZ",
    ]
    base: dict[str, list] = {
        "nome_usina": ["USINA_A", "USINA_B"],
        "posto": [1, 2],
        "submercado": [1, 1],
        "empresa": [1, 1],
        "codigo_usina_jusante": [pd.NA, 1],
        "desvio": [pd.NA, pd.NA],
        "volume_minimo": [100.0, 50.0],
        "volume_maximo": [1000.0, 500.0],
        "volume_referencia": [550.0, 275.0],
        "canal_fuga_medio": [50.0, 50.0],
        "tipo_regulacao": ["M", "M"],
        "tipo_perda": [1, 1],
        "perdas": [0.0, 0.0],
        "a0_volume_cota": [300.0, 300.0],
        "a1_volume_cota": [0.1, 0.1],
        "a2_volume_cota": [0.0, 0.0],
        "a3_volume_cota": [0.0, 0.0],
        "a4_volume_cota": [0.0, 0.0],
        "a0_cota_area": [0.0, 0.0],
        "a1_cota_area": [0.0, 0.0],
        "a2_cota_area": [0.0, 0.0],
        "a3_cota_area": [0.0, 0.0],
        "a4_cota_area": [0.0, 0.0],
        "produtibilidade_especifica": [0.9, 0.85],
        "numero_conjuntos_maquinas": [1, 2],
        "maquinas_conjunto_1": [4, 3],
        "maquinas_conjunto_2": [0, 2],
        "maquinas_conjunto_3": [0, 0],
        "maquinas_conjunto_4": [0, 0],
        "maquinas_conjunto_5": [0, 0],
        "potencia_nominal_conjunto_1": [200.0, 150.0],
        "potencia_nominal_conjunto_2": [0.0, 120.0],
        "potencia_nominal_conjunto_3": [0.0, 0.0],
        "potencia_nominal_conjunto_4": [0.0, 0.0],
        "potencia_nominal_conjunto_5": [0.0, 0.0],
        "vazao_nominal_conjunto_1": [222.2, 176.5],
        "vazao_nominal_conjunto_2": [0.0, 141.2],
        "vazao_nominal_conjunto_3": [0.0, 0.0],
        "vazao_nominal_conjunto_4": [0.0, 0.0],
        "vazao_nominal_conjunto_5": [0.0, 0.0],
        "vazao_minima_historica": [0, 0],
        "teif": [0.0, 0.0],
        "ip": [0.0, 0.0],
        "fator_carga_maximo": [1.0, 1.0],
        "fator_carga_minimo": [0.0, 0.0],
    }
    for m in months:
        base[f"evaporacao_{m}"] = [1.5, 2.0]

    df = pd.DataFrame(base, index=pd.Index([1, 2], name="codigo_usina"))
    return df


def _make_ree_df() -> pd.DataFrame:
    return pd.DataFrame({"codigo": [1], "nome": ["SE"], "submercado": [1]})


def _make_conft_df() -> pd.DataFrame:
    """Three thermals: 2 in subsystem 1, 1 in subsystem 2."""
    return pd.DataFrame(
        {
            "codigo_usina": [10, 20, 30],
            "nome_usina": ["TERMO_A", "TERMO_B", "TERMO_C"],
            "submercado": [1, 1, 2],
            "usina_existente": ["EX", "EX", "EX"],
            "classe": [1, 1, 2],
        }
    )


def _make_clast_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "codigo_usina": [10, 20, 30],
            "nome_usina": ["TERMO_A", "TERMO_B", "TERMO_C"],
            "tipo_combustivel": ["GAS", "GAS", "OLEO"],
            "indice_ano_estudo": [1, 1, 1],
            "valor": [50.0, 80.0, 200.0],
        }
    )


def _make_term_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "codigo_usina": [10, 20, 30],
            "nome_usina": ["TERMO_A", "TERMO_B", "TERMO_C"],
            "potencia_instalada": [100.0, 200.0, 50.0],
            "fator_capacidade_maximo": [90.0, 100.0, 80.0],
            "teif": [0.05, 0.02, 0.10],
            "indisponibilidade_programada": [0.0, 0.0, 0.0],
            "mes": [1, 1, 1],
            "geracao_minima": [10.0, 0.0, 5.0],
        }
    )


def _make_deficit_df(n_patamares: int = 2) -> pd.DataFrame:
    """Deficit costs for subsystems 1 and 2 (non-fictitious) plus fictitious 99."""
    rows = []
    for sub, name, fict in [(1, "SE", 0), (2, "S", 0), (99, "FICT", 1)]:
        for pat in range(1, n_patamares + 1):
            rows.append(
                {
                    "codigo_submercado": sub,
                    "nome_submercado": name,
                    "ficticio": fict,
                    "patamar_deficit": pat,
                    "custo": 500.0 * pat,
                    "corte": 1000.0 if pat < n_patamares else None,
                }
            )
    return pd.DataFrame(rows)


def _make_intercambio_df() -> pd.DataFrame:
    """Three interchange pairs for subsystems 1, 2, 99."""
    import datetime

    d = datetime.datetime(2023, 1, 1)
    rows = [
        # 1 -> 2 direct (sentido=0 means de->para, i.e. 1->2)
        {
            "submercado_de": 1,
            "submercado_para": 2,
            "sentido": 0,
            "data": d,
            "valor": 3000.0,
        },
        # 2 -> 1 reverse (sentido=0 means de->para, i.e. 2->1)
        {
            "submercado_de": 2,
            "submercado_para": 1,
            "sentido": 0,
            "data": d,
            "valor": 2500.0,
        },
        # 1 -> 99 direct
        {
            "submercado_de": 1,
            "submercado_para": 99,
            "sentido": 0,
            "data": d,
            "valor": 4000.0,
        },
        # 99 -> 1 reverse
        {
            "submercado_de": 99,
            "submercado_para": 1,
            "sentido": 0,
            "data": d,
            "valor": 2000.0,
        },
        # 2 -> 99 direct
        {
            "submercado_de": 2,
            "submercado_para": 99,
            "sentido": 0,
            "data": d,
            "valor": 1500.0,
        },
        # 99 -> 2 reverse
        {
            "submercado_de": 99,
            "submercado_para": 2,
            "sentido": 0,
            "data": d,
            "valor": 1200.0,
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EE fixtures: plant 2 in service with machines still to enter (exph.dat).
# ---------------------------------------------------------------------------


def _make_ee_confhd_df() -> pd.DataFrame:
    """The two-plant confhd with plant 2 marked ``EE`` (in service, expanding)."""
    df = _make_confhd_df()
    df["usina_existente"] = ["EX", "EE"]
    return df


def _make_ee_exph_mock(
    *, entry: str = "2024-07-01", code: int = 2, conjunto: int = 2
) -> MagicMock:
    """An ``exph`` reader whose ``expansoes`` carries two machine entries of *code*.

    Mirrors the ``EE`` layout of a real ``exph.dat``: **no** filling row
    (``data_inicio_enchimento`` is ``NaT`` throughout, which is what keeps
    ``filling_hydro_codes`` from claiming the plant) and one row per entering
    machine, both in *conjunto*.
    """
    expansoes = pd.DataFrame(
        {
            "codigo_usina": [code, code],
            "nome_usina": ["", ""],
            "data_inicio_enchimento": [pd.NaT, pd.NaT],
            "duracao_enchimento": [0, 0],
            "volume_morto": [0.0, 0.0],
            "data_entrada_operacao": [pd.Timestamp(entry), pd.Timestamp(entry)],
            "conjunto_maquina_entrada": [conjunto, conjunto],
            "maquina_entrada": [1, 2],
        }
    )
    exph = MagicMock()
    exph.expansoes = expansoes
    return exph


def _ee_expansion_case(tmp_path, *, entry: str = "2024-07-01", **hydro_case_kwargs):
    """A ``NewaveCase`` whose plant 2 is ``EE`` with two machines still to enter.

    Under the default Jan-2024 one-year horizon (12 stages), the Jul-2024 entry
    is stage 6: plant 2 runs on conjunto 1 alone (3 × 150 MW = 450) for stages
    0-5 and reaches its registry configuration (450 + 2 × 120 = 690) from stage
    6. The deck carries no ``modif.dat``, so the study-start configuration comes
    from the ``registry − all entering`` fallback rather than ``NUMMAQ``.
    """
    return _hydro_case(
        tmp_path,
        confhd=_make_ee_confhd_df(),
        exph=_make_ee_exph_mock(entry=entry),
        **hydro_case_kwargs,
    )


# ---------------------------------------------------------------------------
# NE-with-filling fixtures: a JURUENA-shaped run-of-river ('S')
# plant (code 309) admitted into the active set by its exph dead-volume row.
# ---------------------------------------------------------------------------


def _make_ne_confhd_df() -> pd.DataFrame:
    """Two EX plants (1, 2) plus one NE filling plant (309 JURUENA)."""
    return pd.DataFrame(
        {
            "codigo_usina": [1, 2, 309],
            "nome_usina": ["USINA_A", "USINA_B", "JURUENA"],
            "posto": [1, 2, 226],
            "codigo_usina_jusante": [pd.NA, 1, pd.NA],
            "ree": [1, 1, 1],
            "volume_inicial_percentual": [50.0, 75.0, 0.0],
            "usina_existente": ["EX", "EX", "NE"],
            "usina_modificada": [0, 0, 0],
        }
    )


def _make_ne_cadastro() -> pd.DataFrame:
    """Two-plant synthetic cadastro extended with JURUENA (code 309).

    JURUENA is run-of-river (``tipo_regulacao='S'``) with
    ``volume_minimo == volume_maximo == 2.93`` so its reservoir block already
    collapses to the single point 2.93 hm³ via the existing 'S' path.
    """
    base = _make_hidr_cadastro()
    juruena = base.loc[[1]].copy()
    juruena.index = pd.Index([309], name="codigo_usina")
    juruena["nome_usina"] = "JURUENA"
    juruena["posto"] = 226
    juruena["codigo_usina_jusante"] = pd.NA
    juruena["volume_minimo"] = 2.93
    juruena["volume_maximo"] = 2.93
    juruena["volume_referencia"] = 2.93
    juruena["tipo_regulacao"] = "S"
    return pd.concat([base, juruena])


def _make_ne_exph_mock(*, duracao: int = 1, volume_morto: float = 0.0) -> MagicMock:
    """An ``exph`` reader mock whose ``expansoes`` carries JURUENA's filling row.

    Mirrors the real ``exph.dat`` layout (verified on JURUENA): one schedule row
    (non-null ``data_inicio_enchimento``) — what ``filling_hydro_codes`` selects
    and the ``convert_hydros`` filling tests read via ``.iloc[0]`` — then
    one row **per generating unit** with ``data_entrada_operacao`` /
    ``conjunto_maquina_entrada`` / ``maquina_entrada`` populated. JURUENA
    has two machines, both entering Jan 2025 in machine group 1 (so under the
    Sep-2024 horizon their online stage is 4). The schedule row keeps a non-null
    ``data_entrada_operacao`` but a NULL
    ``conjunto_maquina_entrada``, so the ramp branch — which filters unit rows on
    ``conjunto_maquina_entrada`` — never treats it as a generating unit.
    """
    expansoes = pd.DataFrame(
        {
            "codigo_usina": [309, 309, 309],
            "nome_usina": ["JURUENA", "JURUENA", "JURUENA"],
            "data_inicio_enchimento": [
                pd.Timestamp("2024-10-01"),
                pd.NaT,
                pd.NaT,
            ],
            "duracao_enchimento": [duracao, 0, 0],
            "volume_morto": [volume_morto, 0.0, 0.0],
            "data_entrada_operacao": [
                pd.Timestamp("2024-11-01"),
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-01"),
            ],
            "conjunto_maquina_entrada": [pd.NA, 1, 1],
            "maquina_entrada": [pd.NA, 1, 2],
        }
    )
    exph = MagicMock()
    exph.expansoes = expansoes
    return exph


def _ne_filling_case(tmp_path, *, duracao: int = 1, volume_morto: float = 0.0):
    """A ``NewaveCase`` with JURUENA (NE+filling) under a Sep-2024 3-year horizon.

    Study start Sep 2024 ⇒ stage 0 = Sep, stage 1 = Oct, stage 2 = Nov. JURUENA's
    Oct-2024 filling start maps to ``start_sid == 1``; with ``duracao == 1`` the
    entry is ``entry_sid == 2``.
    """
    return _hydro_case(
        tmp_path,
        cadastro=_make_ne_cadastro(),
        confhd=_make_ne_confhd_df(),
        dger=_make_hydro_dger_mock(
            start_year=2024, start_month=9, num_anos=3, num_anos_pos=0
        ),
        exph=_make_ne_exph_mock(duracao=duracao, volume_morto=volume_morto),
    )


def _ne_filling_id_map() -> NewaveIdMap:
    """Id-map including JURUENA (309) so ``hydro_id(309)`` resolves."""
    return NewaveIdMap(
        subsystem_ids=[1],
        hydro_codes=[1, 2, 309],
        thermal_codes=[],
    )


def _make_hydro_dger_mock(
    *,
    start_year: int = 2024,
    start_month: int = 1,
    num_anos: int = 1,
    num_anos_pos: int = 0,
) -> MagicMock:
    """A ``dger`` mock that yields a concrete study horizon for ``convert_hydros``.

    ``convert_hydros`` builds the per-stage date list from ``case.horizon``
    (which reads these four ``dger`` fields) once, unconditionally, before the
    plant loop — so every ``_hydro_case`` needs a ``dger`` that resolves all four
    horizon fields (the module's other ``_make_dger_mock`` leaves
    ``num_anos_pos_estudo`` unset). ``funcao_producao_uhe = 1`` selects the linear
    (constant-productivity) generation model, matching the synthetic plants.
    """
    dger = MagicMock()
    dger.ano_inicio_estudo = start_year
    dger.mes_inicio_estudo = start_month
    dger.num_anos_estudo = num_anos
    dger.num_anos_pos_estudo = num_anos_pos
    # Historical-record start: drives the operational_start_date of always-in-service
    # (EX) plants. Set explicitly so it is a real int, not a MagicMock (whose int()
    # would otherwise resolve to 1).
    dger.ano_inicial_historico = 1931
    dger.funcao_producao_uhe = 1
    return dger


def _hydro_case(
    tmp_path,
    *,
    cadastro: pd.DataFrame | None = None,
    confhd: pd.DataFrame | None = None,
    rees: pd.DataFrame | None = None,
    modif=None,
    volref_volumes: pd.DataFrame | None = None,
    ghmin=None,
    penalid=None,
    dsvagua=None,
    dger=None,
    exph=None,
    **file_overrides,
):
    """Build a ``NewaveCase`` with mock hydro readers pre-cached.

    The three required hydro files (hidr/confhd/ree) default to the shared
    synthetic fixtures; pass ``cadastro`` / ``confhd`` / ``rees`` DataFrames to
    override. Optional readers (modif, volref_saz, ghmin, penalid, dsvagua) are
    passed as already-built mock reader objects; for those guarded behind a
    ``case.files.X`` path check, set the matching path via ``file_overrides``
    (e.g. ``volref_saz=tmp_path / "volref_saz.dat"``).

    ``dger`` defaults to :func:`_make_hydro_dger_mock` (a Jan-2024 one-year horizon) so
    ``case.horizon`` always resolves; pass a custom ``dger`` mock to drive a
    different horizon. ``exph`` is the dead-volume filling reader mock (``None``
    by default — EX-only cases admit no filling plant).
    """
    mock_hidr = MagicMock()
    mock_hidr.cadastro = _make_hidr_cadastro() if cadastro is None else cadastro

    mock_confhd = MagicMock()
    mock_confhd.usinas = _make_confhd_df() if confhd is None else confhd

    mock_ree = MagicMock()
    mock_ree.rees = _make_ree_df() if rees is None else rees

    parsed: dict = {
        "hidr": mock_hidr,
        "confhd": mock_confhd,
        "ree": mock_ree,
        "dger": _make_hydro_dger_mock() if dger is None else dger,
        "exph": exph,
    }

    if volref_volumes is not None:
        mock_volref = MagicMock()
        mock_volref.volumes = volref_volumes
        parsed["volref_saz"] = mock_volref
    if modif is not None:
        parsed["modif"] = modif
    if ghmin is not None:
        parsed["ghmin"] = ghmin
    if penalid is not None:
        parsed["penalid"] = penalid
    if dsvagua is not None:
        parsed["dsvagua"] = dsvagua

    files = make_nw_files(tmp_path, **file_overrides)
    return make_case(files, **parsed)


def _make_prod_model_dger_mock(
    *,
    ano_inicio: int = 2025,
    mes_inicio: int = 1,
    num_anos: int = 5,
    num_anos_pos: int = 0,
) -> MagicMock:
    """Return a mock Dger object for use in production model tests."""
    m = MagicMock()
    m.ano_inicio_estudo = ano_inicio
    m.mes_inicio_estudo = mes_inicio
    m.num_anos_estudo = num_anos
    m.num_anos_pos_estudo = num_anos_pos
    return m


def _make_cfuga_rec(month: int, year: int, nivel: float) -> MagicMock:
    import datetime

    r = MagicMock()
    type(r).__name__ = "CFUGA"
    r.data_inicio = datetime.datetime(year, month, 1)
    r.nivel = nivel
    return r


def _thermal_readers():
    conft = MagicMock()
    conft.usinas = _make_conft_df()
    clast = MagicMock()
    clast.usinas = _make_clast_df()
    clast.modificacoes = None
    term = MagicMock()
    term.usinas = _make_term_df()
    return conft, clast, term


def _make_thermal_dger(mes_inicio: int = 1) -> MagicMock:
    """The ``dger`` fields the thermal converters read: horizon plus the GNL switch.

    Two study years from January 2023, no post-study tail, and one maintenance
    year — so stages 0-11 are inside it and 12-23 after it, which is what the
    IP and minimum-generation regimes switch on.
    """
    dger = MagicMock()
    dger.ano_inicio_estudo = 2023
    dger.mes_inicio_estudo = mes_inicio
    dger.num_anos_estudo = 2
    dger.num_anos_pos_estudo = 0
    dger.num_anos_manutencao_utes = 1
    dger.despacho_antecipado_gnl = 0
    return dger


def _make_sistema_mock() -> MagicMock:
    """Build the ``Sistema`` reader mock shared by the network tests."""
    mock_sistema = MagicMock()
    mock_sistema.custo_deficit = _make_deficit_df(n_patamares=2)
    mock_sistema.limites_intercambio = _make_intercambio_df()
    return mock_sistema


def _make_penalid_df() -> pd.DataFrame:
    """Synthetic PENALID.DAT penalties for two REEs and several variables.

    REE 1 has DESVIO=8300.0, VAZMIN=3179.35, GHMIN=4500.0 at patamar 1.
    REE 2 has DESVIO=9100.0, VAZMIN=2800.0 at patamar 1.
    Both REEs have patamar 2 rows with NaN values (unbounded tier).
    TURBMX is included to verify the "no mapping" skip path.
    """
    return pd.DataFrame(
        {
            "variavel": [
                "DESVIO",
                "DESVIO",
                "VAZMIN",
                "VAZMIN",
                "GHMIN",
                "GHMIN",
                "TURBMX",
                "TURBMX",
                "DESVIO",
                "DESVIO",
                "VAZMIN",
                "VAZMIN",
            ],
            "codigo_ree_submercado": [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            "patamar_penalidade": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "patamar_carga": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "valor_R$_MWh": [
                8300.0,
                math.nan,
                3179.35,
                math.nan,
                4500.0,
                math.nan,
                999.0,  # TURBMX — should be skipped (no mapping)
                math.nan,
                9100.0,
                math.nan,
                2800.0,
                math.nan,
            ],
            "valor_R$_hm3": [0.0] * 12,
        }
    )


def _make_dger_mock(start_year: int, start_month: int, num_anos: int) -> MagicMock:
    """Build a MagicMock mimicking the Dger object."""
    mock = MagicMock()
    mock.ano_inicio_estudo = start_year
    mock.mes_inicio_estudo = start_month
    mock.num_anos_estudo = num_anos
    return mock


def _run_with_all_mocks(src: Path, dst: Path) -> object:
    """Run convert_newave_case with all converters replaced by canned fakes."""
    import contextlib

    from novomodelo_bridge.newave.pipeline import convert_newave_case

    fake_id_map = MagicMock()
    with contextlib.ExitStack() as stack:
        for p in _all_converter_patches(fake_id_map):
            stack.enter_context(p)
        return convert_newave_case(src, dst)

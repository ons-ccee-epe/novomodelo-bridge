"""Tests for the DECOMP output readers.

Synthetic tables for the parsing contracts, plus small real-format result
file excerpts under ``tests/fixtures/decomp_results/`` for the file layout.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from idecomp.decomp import (
    DecDesvFpha,
    DecEstatFpha,
    Decomptim,
    DecOperEvap,
    DecOperGnl,
    DecOperRee,
    DecOperRheSoft,
    EcoFpha,
    Relato,
)

from novomodelo_bridge.comparators.decomp.readers import (
    _read_dec_oper,
    _read_relato_table,
    _resolve_relato,
    _resolve_relato2,
    _resolve_result_file,
    _resolve_revisioned_file,
    read_dec_desvfpha,
    read_dec_estatfpha,
    read_dec_oper_evap,
    read_dec_oper_gnl,
    read_dec_oper_ree,
    read_dec_oper_rhesoft,
    read_decomp_tim,
    read_eco_fpha,
    read_relato2_costs,
    read_relato_balance,
    read_relato_convergence,
    read_relato_costs,
    read_relato_expected_cost,
    read_relato_membership,
)
from novomodelo_bridge.comparators.decomp.results import reconcile_kdollars_to_reais

# Real-format DECOMP result files (a few KB each) vendored from the
# sintetizador-decomp project's test mocks, so the readers are exercised
# against the model's own file layout without any deck outside the repo.
_RESULTS_EXCERPT = Path(__file__).resolve().parents[1] / "fixtures" / "decomp_results"


class _StubFile:
    def __init__(self, table: pd.DataFrame | None) -> None:
        self.tabela = table


class _StubTimFile:
    """Stub for a `Decomptim.read(...)` result, carrying `tempos_etapas`."""

    def __init__(self, table: pd.DataFrame | None) -> None:
        self.tempos_etapas = table


def _stub_reader(table: pd.DataFrame | None) -> type:
    class _Reader:
        @staticmethod
        def read(path: str) -> _StubFile:  # noqa: ARG004
            return _StubFile(table)

    return _Reader


class _StubRelatoFile:
    """Stub for a `Relato.read(...)` result, carrying named pandas tables
    (e.g. ``convergencia=...``, ``balanco_energetico=...``)."""

    def __init__(self, **tables: pd.DataFrame | None) -> None:
        for name, table in tables.items():
            setattr(self, name, table)


class TestResolveResultFile:
    """`_resolve_result_file`: root-only, case-insensitive discovery."""

    def test_saidas_only_is_not_found(self, tmp_path: Path) -> None:
        """A file present only under `saidas/` is a miss -- the clean break
        retiring the saidas-first union."""
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "dec_oper_ree.csv").touch()
        assert _resolve_result_file(tmp_path, "dec_oper_ree.csv") is None

    def test_root_resolves(self, tmp_path: Path) -> None:
        target = tmp_path / "dec_oper_sist.csv"
        target.touch()
        assert _resolve_result_file(tmp_path, "dec_oper_sist.csv") == target

    def test_root_resolves_even_when_saidas_present(self, tmp_path: Path) -> None:
        """`saidas/` is never consulted, even when a copy sits there too."""
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "dec_oper_sist.csv").touch()
        root_copy = tmp_path / "dec_oper_sist.csv"
        root_copy.touch()
        assert _resolve_result_file(tmp_path, "dec_oper_sist.csv") == root_copy

    def test_case_insensitive_in_root(self, tmp_path: Path) -> None:
        target = tmp_path / "DEC_OPER_SIST.CSV"
        target.touch()
        assert _resolve_result_file(tmp_path, "dec_oper_sist.csv") == target

    def test_absent_from_root_returns_none(self, tmp_path: Path) -> None:
        assert _resolve_result_file(tmp_path, "dec_oper_ree.csv") is None


class TestResolveRelato:
    """`_resolve_relato`: same root-only resolution for relato.rvN."""

    def test_saidas_only_is_not_found(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "relato.rv2").touch()
        assert _resolve_relato(tmp_path) is None

    def test_root_resolves(self, tmp_path: Path) -> None:
        target = tmp_path / "relato.rv2"
        target.touch()
        assert _resolve_relato(tmp_path) == target

    def test_absent_from_root_returns_none(self, tmp_path: Path) -> None:
        assert _resolve_relato(tmp_path) is None


class TestResolveRevisionedFile:
    """`_resolve_revisioned_file`: the shared root-only ``<stem>.rvN``
    resolver behind `_resolve_relato` and the FPHA readers."""

    def test_saidas_only_is_not_found(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "dec_estatfpha.rv2").touch()
        assert _resolve_revisioned_file(tmp_path, "dec_estatfpha") is None

    def test_root_resolves(self, tmp_path: Path) -> None:
        target = tmp_path / "dec_desvfpha.rv3"
        target.touch()
        assert _resolve_revisioned_file(tmp_path, "dec_desvfpha") == target

    def test_root_resolves_even_when_saidas_present(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "eco_fpha.rv2").touch()
        root_copy = tmp_path / "eco_fpha.rv2"
        root_copy.touch()
        assert _resolve_revisioned_file(tmp_path, "eco_fpha") == root_copy

    def test_absent_from_root_returns_none(self, tmp_path: Path) -> None:
        assert _resolve_revisioned_file(tmp_path, "eco_fpha") is None

    def test_does_not_match_a_different_stem(self, tmp_path: Path) -> None:
        (tmp_path / "dec_estatfpha.rv2").touch()
        assert _resolve_revisioned_file(tmp_path, "dec_desvfpha") is None


class _StubDecEstatFphaFile:
    """Stub for a `DecEstatFpha.read(...)` result, carrying
    `estatisticas_desvios`."""

    def __init__(self, table: pd.DataFrame | None) -> None:
        self.estatisticas_desvios = table


class TestReadDecDesvfpha:
    """`read_dec_desvfpha`: per-hydro FPHA deviation table."""

    def test_finds_root_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "dec_desvfpha.rv2").touch()
        stub_table = pd.DataFrame(
            {
                "codigo_usina": [1],
                "estagio": [1],
                "volume_total_hm3": [500.0],
                "vazao_turbinada_m3s": [80.0],
                "vazao_vertida_m3s": [0.0],
                "geracao_hidraulica_fph": [37.68],
                "geracao_hidraulica_fpha": [37.61],
                "desvio_absoluto_MW": [-0.07],
                "desvio_percentual": [-0.19],
            }
        )
        monkeypatch.setattr(
            DecDesvFpha,
            "read",
            staticmethod(lambda _path: _StubFile(stub_table)),
        )
        result = read_dec_desvfpha(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert "geracao_hidraulica_fpha" in result.columns

    def test_saidas_only_is_not_found(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "dec_desvfpha.rv2").touch()
        with pytest.raises(FileNotFoundError, match="dec_desvfpha") as exc_info:
            read_dec_desvfpha(tmp_path)
        message = str(exc_info.value)
        assert str(tmp_path) in message
        assert "saidas/" not in message

    def test_missing_file_raises_naming_case_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="dec_desvfpha") as exc_info:
            read_dec_desvfpha(tmp_path)
        message = str(exc_info.value)
        assert str(tmp_path) in message
        assert "saidas/" not in message

    def test_empty_parse_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "dec_desvfpha.rv2").touch()
        monkeypatch.setattr(
            DecDesvFpha,
            "read",
            staticmethod(lambda _path: _StubFile(pd.DataFrame())),
        )
        with pytest.raises(ValueError, match="parsed empty"):
            read_dec_desvfpha(tmp_path)


class TestReadEcoFpha:
    """`read_eco_fpha`: per-hydro/stage FPHA fitting-grid echo."""

    def test_finds_root_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "eco_fpha.rv2").touch()
        stub_table = pd.DataFrame(
            {
                "codigo_usina": [1],
                "estagio": [1],
                "numero_pontos_volume_armazenado": [5],
                "volume_armazenado_minimo": [387.3],
                "volume_armazenado_maximo": [656.1],
                "numero_pontos_vazao_turbinada": [5],
                "vazao_turbinada_minima": [0.0],
                "vazao_turbinada_maxima": [210.7],
            }
        )
        monkeypatch.setattr(
            EcoFpha,
            "read",
            staticmethod(lambda _path: _StubFile(stub_table)),
        )
        result = read_eco_fpha(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert "numero_pontos_volume_armazenado" in result.columns

    def test_saidas_only_is_not_found(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "eco_fpha.rv2").touch()
        with pytest.raises(FileNotFoundError, match="eco_fpha") as exc_info:
            read_eco_fpha(tmp_path)
        message = str(exc_info.value)
        assert str(tmp_path) in message
        assert "saidas/" not in message

    def test_missing_file_raises_naming_case_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="eco_fpha") as exc_info:
            read_eco_fpha(tmp_path)
        message = str(exc_info.value)
        assert str(tmp_path) in message
        assert "saidas/" not in message

    def test_empty_parse_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "eco_fpha.rv2").touch()
        monkeypatch.setattr(
            EcoFpha,
            "read",
            staticmethod(lambda _path: _StubFile(None)),
        )
        with pytest.raises(ValueError, match="parsed empty"):
            read_eco_fpha(tmp_path)


class TestReadDecEstatfpha:
    """`read_dec_estatfpha`: deck-wide FPHA deviation summary."""

    def test_finds_root_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "dec_estatfpha.rv2").touch()
        stub_table = pd.DataFrame({"variavel": ["DESVIO MEDIO (MW)"], "valor": [3.83]})
        monkeypatch.setattr(
            DecEstatFpha,
            "read",
            staticmethod(lambda _path: _StubDecEstatFphaFile(stub_table)),
        )
        result = read_dec_estatfpha(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert result["variavel"][0] == "DESVIO MEDIO (MW)"

    def test_saidas_only_is_not_found(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "dec_estatfpha.rv2").touch()
        with pytest.raises(FileNotFoundError, match="dec_estatfpha") as exc_info:
            read_dec_estatfpha(tmp_path)
        message = str(exc_info.value)
        assert str(tmp_path) in message
        assert "saidas/" not in message

    def test_missing_file_raises_naming_case_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="dec_estatfpha") as exc_info:
            read_dec_estatfpha(tmp_path)
        message = str(exc_info.value)
        assert str(tmp_path) in message
        assert "saidas/" not in message

    def test_none_table_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "dec_estatfpha.rv2").touch()
        monkeypatch.setattr(
            DecEstatFpha,
            "read",
            staticmethod(lambda _path: _StubDecEstatFphaFile(None)),
        )
        with pytest.raises(ValueError, match="parsed empty"):
            read_dec_estatfpha(tmp_path)


class TestReadDecOperCore:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="dec_oper_sist.csv"):
            _read_dec_oper(tmp_path, "dec_oper_sist.csv", _stub_reader(None))

    def test_missing_file_names_case_dir_only(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError) as exc_info:
            _read_dec_oper(tmp_path, "dec_oper_ree.csv", _stub_reader(None))
        message = str(exc_info.value)
        assert str(tmp_path) in message
        assert "saidas/" not in message

    def test_saidas_only_is_not_found(self, tmp_path: Path) -> None:
        """A file present only under `saidas/` is a miss: `_resolve_result_file`
        returns `None` and `_read_dec_oper` raises `FileNotFoundError` naming
        only `case_dir`."""
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "dec_oper_sist.csv").touch()
        with pytest.raises(FileNotFoundError) as exc_info:
            _read_dec_oper(tmp_path, "dec_oper_sist.csv", _stub_reader(None))
        message = str(exc_info.value)
        assert str(tmp_path) in message
        assert "saidas/" not in message

    def test_found_in_root_only(self, tmp_path: Path) -> None:
        (tmp_path / "dec_oper_sist.csv").touch()
        table = pd.DataFrame({"estagio": [1, 2], "cmo": [10.0, 12.0]})
        result = _read_dec_oper(tmp_path, "dec_oper_sist.csv", _stub_reader(table))
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 2)

    def test_empty_parse_raises(self, tmp_path: Path) -> None:
        (tmp_path / "dec_oper_sist.csv").touch()
        with pytest.raises(ValueError, match="parsed empty"):
            _read_dec_oper(tmp_path, "dec_oper_sist.csv", _stub_reader(None))
        with pytest.raises(ValueError, match="parsed empty"):
            _read_dec_oper(tmp_path, "dec_oper_sist.csv", _stub_reader(pd.DataFrame()))

    def test_returns_polars_frame(self, tmp_path: Path) -> None:
        (tmp_path / "dec_oper_sist.csv").touch()
        table = pd.DataFrame({"estagio": [1, 2], "cmo": [10.0, 12.0]})
        result = _read_dec_oper(tmp_path, "dec_oper_sist.csv", _stub_reader(table))
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 2)


class TestRealDeckReaders:
    def test_relato_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="relato"):
            read_relato_convergence(tmp_path)


class TestReadRelatoConvergenceDiscovery:
    """Tier-1: `read_relato_convergence` resolves a root-only report; a
    `saidas/`-only report is a miss."""

    def test_finds_root_relato(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        stub_table = pd.DataFrame(
            {
                "iteracao": [1],
                "zinf": [1.0],
                "zsup": [1.0],
                "gap_percentual": [0.0],
            }
        )
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(lambda _path: _StubRelatoFile(convergencia=stub_table)),
        )
        result = read_relato_convergence(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1

    def test_saidas_only_relato_is_not_found(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "relato.rv2").touch()
        with pytest.raises(FileNotFoundError, match="relato") as exc_info:
            read_relato_convergence(tmp_path)
        assert "saidas/" not in str(exc_info.value)


class TestReadRelatoTable:
    """`_read_relato_table`: shared helper behind every relato reader."""

    def test_missing_relato_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="relato"):
            _read_relato_table(tmp_path, "convergencia")

    def test_none_table_raises_naming_attr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(lambda _path: _StubRelatoFile(balanco_energetico=None)),
        )
        with pytest.raises(ValueError, match="balanco_energetico"):
            _read_relato_table(tmp_path, "balanco_energetico")

    def test_empty_table_raises_naming_attr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(
                lambda _path: _StubRelatoFile(balanco_energetico=pd.DataFrame())
            ),
        )
        with pytest.raises(ValueError, match="balanco_energetico"):
            _read_relato_table(tmp_path, "balanco_energetico")

    def test_finds_root_relato(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        stub_table = pd.DataFrame({"nome_submercado": ["SUDESTE"]})
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(lambda _path: _StubRelatoFile(balanco_energetico=stub_table)),
        )
        result = _read_relato_table(tmp_path, "balanco_energetico")
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1

    def test_saidas_only_relato_is_not_found(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "relato.rv2").touch()
        with pytest.raises(FileNotFoundError, match="relato") as exc_info:
            _read_relato_table(tmp_path, "balanco_energetico")
        assert "saidas/" not in str(exc_info.value)


class TestReadRelatoBalance:
    """`read_relato_balance`: per-submarket energy balance table."""

    def test_returns_expected_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        stub_table = pd.DataFrame(
            {
                "estagio": [1],
                "cenario": [1],
                "nome_submercado": ["SUDESTE"],
                "geracao_hidraulica": [1000.0],
                "geracao_termica": [200.0],
                "compra": [0.0],
                "venda": [0.0],
                "geracao_itaipu_50hz": [50.0],
                "geracao_itaipu_60hz": [60.0],
            }
        )
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(lambda _path: _StubRelatoFile(balanco_energetico=stub_table)),
        )
        result = read_relato_balance(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        for column in (
            "nome_submercado",
            "geracao_hidraulica",
            "geracao_termica",
            "compra",
            "venda",
            "geracao_itaipu_50hz",
            "geracao_itaipu_60hz",
        ):
            assert column in result.columns

    def test_none_table_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(lambda _path: _StubRelatoFile(balanco_energetico=None)),
        )
        with pytest.raises(ValueError, match="balanco_energetico"):
            read_relato_balance(tmp_path)

    def test_missing_relato_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="relato"):
            read_relato_balance(tmp_path)


class TestReadRelatoCosts:
    """`read_relato_costs`: per-(stage, scenario) operating cost table."""

    def test_returns_expected_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        stub_table = pd.DataFrame(
            {
                "estagio": [1],
                "cenario": [1],
                "probabilidade": [1.0],
                "custo_futuro": [1234.5],
                "custo_presente": [678.9],
                "geracao_termica": [50.0],
                "violacao_desvio": [0.0],
                "penalidade_vertimento_reservatorio": [0.0],
                "penalidade_vertimento_fio": [0.0],
                "violacao_turbinamento_reservatorio": [0.0],
                "violacao_turbinamento_fio": [0.0],
                "penalidade_intercambio": [0.0],
                "cmo_SE": [100.0],
            }
        )
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(
                lambda _path: _StubRelatoFile(relatorio_operacao_custos=stub_table)
            ),
        )
        result = read_relato_costs(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        for column in (
            "custo_presente",
            "custo_futuro",
            "geracao_termica",
            "penalidade_intercambio",
            "penalidade_vertimento_reservatorio",
        ):
            assert column in result.columns

    def test_none_table_raises_naming_attr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(lambda _path: _StubRelatoFile(relatorio_operacao_custos=None)),
        )
        with pytest.raises(ValueError, match="relatorio_operacao_custos"):
            read_relato_costs(tmp_path)

    def test_missing_relato_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="relato"):
            read_relato_costs(tmp_path)


class TestResolveRelato2:
    """`_resolve_relato2`: root-only resolution for relato2.rvN, distinct from
    relato.rvN (the relato resolver must not pick up relato2 or vice versa)."""

    def test_root_resolves(self, tmp_path: Path) -> None:
        target = tmp_path / "relato2.rv2"
        target.touch()
        assert _resolve_relato2(tmp_path) == target

    def test_does_not_match_relato(self, tmp_path: Path) -> None:
        (tmp_path / "relato.rv2").touch()
        assert _resolve_relato2(tmp_path) is None

    def test_absent_returns_none(self, tmp_path: Path) -> None:
        assert _resolve_relato2(tmp_path) is None


class TestReadRelato2Costs:
    """`read_relato2_costs`: the fan-stage per-(stage, scenario) cost table
    from relato2.rvN, or an empty frame when relato2 is absent (optional)."""

    def test_absent_relato2_returns_empty(self, tmp_path: Path) -> None:
        result = read_relato2_costs(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()

    def test_reads_fan_stage_costs_with_probabilities(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato2.rv2").touch()
        stub_table = pd.DataFrame(
            {
                "estagio": [4, 4],
                "cenario": [1, 2],
                "probabilidade": [0.504438, 0.495562],
                "custo_futuro": [0.0, 0.0],
                "custo_presente": [1300.0, 300.0],
                "geracao_termica": [1300.0, 300.0],
                "penalidade_intercambio": [0.0, 0.0],
            }
        )
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(
                lambda _path: _StubRelatoFile(relatorio_operacao_custos=stub_table)
            ),
        )
        result = read_relato2_costs(tmp_path)
        assert result.height == 2
        assert set(result["estagio"].to_list()) == {4}
        assert "probabilidade" in result.columns

    def test_empty_table_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato2.rv2").touch()
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(lambda _path: _StubRelatoFile(relatorio_operacao_custos=None)),
        )
        assert read_relato2_costs(tmp_path).is_empty()


class TestReadRelatoExpectedCost:
    """`read_relato_expected_cost`: per-parcela expected cost, wide by stage."""

    def test_returns_estagio_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        stub_table = pd.DataFrame(
            {
                "parcela": ["GERACAO TERMICA"],
                "estagio_1": [10.0],
                "estagio_2": [20.0],
                "estagio_3": [30.0],
            }
        )
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(
                lambda _path: _StubRelatoFile(custo_operacao_valor_esperado=stub_table)
            ),
        )
        result = read_relato_expected_cost(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        for column in ("parcela", "estagio_1", "estagio_2", "estagio_3"):
            assert column in result.columns

    def test_none_table_raises_naming_attr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(
                lambda _path: _StubRelatoFile(custo_operacao_valor_esperado=None)
            ),
        )
        with pytest.raises(ValueError, match="custo_operacao_valor_esperado"):
            read_relato_expected_cost(tmp_path)

    def test_missing_relato_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="relato"):
            read_relato_expected_cost(tmp_path)


class TestReadRelatoMembership:
    """`read_relato_membership`: hydro -> REE -> submarket membership table."""

    def test_returns_the_documented_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        stub_table = pd.DataFrame(
            {
                "codigo_usina": [1, 2],
                "nome_usina": ["CAMARGOS", "FURNAS"],
                "codigo_ree": [10, 10],
                "nome_ree": ["SUDESTE", "SUDESTE"],
                "codigo_submercado": [1, 1],
                "nome_submercado": ["SE", "SE"],
                "nome_submercado_newave": ["SUDESTE", "SUDESTE"],
            }
        )
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(
                lambda _path: _StubRelatoFile(uhes_rees_submercados=stub_table)
            ),
        )
        result = read_relato_membership(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 2
        for column in (
            "codigo_usina",
            "nome_usina",
            "codigo_ree",
            "nome_ree",
            "codigo_submercado",
            "nome_submercado",
            "nome_submercado_newave",
        ):
            assert column in result.columns

    def test_none_table_raises_naming_attr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(lambda _path: _StubRelatoFile(uhes_rees_submercados=None)),
        )
        with pytest.raises(ValueError, match="uhes_rees_submercados"):
            read_relato_membership(tmp_path)

    def test_missing_relato_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="relato"):
            read_relato_membership(tmp_path)


class TestReadDecOperGnl:
    """`read_dec_oper_gnl`: anticipated-thermal operation, root-only file."""

    def test_finds_root_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "dec_oper_gnl.csv").touch()
        stub_table = pd.DataFrame(
            {
                "estagio": [1],
                "codigo_usina": [1],
                "custo_incremental": [5.0],
                "geracao_MW": [100.0],
                "custo_geracao": [42.0],
            }
        )
        monkeypatch.setattr(
            DecOperGnl,
            "read",
            staticmethod(lambda _path: _StubFile(stub_table)),
        )
        result = read_dec_oper_gnl(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert "custo_geracao" in result.columns
        assert "geracao_MW" in result.columns

    def test_saidas_only_is_not_found(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "dec_oper_gnl.csv").touch()
        with pytest.raises(FileNotFoundError, match="dec_oper_gnl.csv") as exc_info:
            read_dec_oper_gnl(tmp_path)
        assert "saidas/" not in str(exc_info.value)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="dec_oper_gnl.csv"):
            read_dec_oper_gnl(tmp_path)

    def test_empty_parse_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "dec_oper_gnl.csv").touch()
        monkeypatch.setattr(
            DecOperGnl,
            "read",
            staticmethod(lambda _path: _StubFile(None)),
        )
        with pytest.raises(ValueError, match="parsed empty"):
            read_dec_oper_gnl(tmp_path)

    def test_result_excerpt_parses(self) -> None:
        df = read_dec_oper_gnl(_RESULTS_EXCERPT)
        assert df.height > 0


class TestReadDecOperRee:
    """`read_dec_oper_ree`: per-REE energy operation (ENA / EARM)."""

    def test_finds_root_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "dec_oper_ree.csv").touch()
        stub_table = pd.DataFrame(
            {
                "estagio": [1],
                "no": [1],
                "cenario": [1],
                "codigo_ree": [10],
                "nome_ree": ["SUDESTE"],
                "codigo_submercado": [1],
                "nome_submercado": ["SE"],
                "ena_MWmes": [1000.0],
                "earm_inicial_MWmes": [5000.0],
                "earm_inicial_percentual": [50.0],
                "earm_final_MWmes": [4800.0],
                "earm_final_percentual": [48.0],
                "earm_maximo_MWmes": [10000.0],
            }
        )
        monkeypatch.setattr(
            DecOperRee,
            "read",
            staticmethod(lambda _path: _StubFile(stub_table)),
        )
        result = read_dec_oper_ree(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        for column in ("codigo_ree", "nome_ree", "ena_MWmes", "earm_final_MWmes"):
            assert column in result.columns

    def test_saidas_only_is_not_found(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "dec_oper_ree.csv").touch()
        with pytest.raises(FileNotFoundError, match="dec_oper_ree.csv") as exc_info:
            read_dec_oper_ree(tmp_path)
        assert "saidas/" not in str(exc_info.value)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="dec_oper_ree.csv"):
            read_dec_oper_ree(tmp_path)

    def test_empty_parse_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "dec_oper_ree.csv").touch()
        monkeypatch.setattr(
            DecOperRee,
            "read",
            staticmethod(lambda _path: _StubFile(None)),
        )
        with pytest.raises(ValueError, match="parsed empty"):
            read_dec_oper_ree(tmp_path)

    def test_result_excerpt_parses(self) -> None:
        df = read_dec_oper_ree(_RESULTS_EXCERPT)
        assert df.height > 0
        assert "patamar" not in df.columns


class TestReadDecOperEvap:
    """`read_dec_oper_evap`: per-hydro/stage reservoir evaporation.
    Verified columns against idecomp 1.14.2's
    ``DecOperEvap.tabela``."""

    def test_finds_root_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "dec_oper_evap.csv").touch()
        stub_table = pd.DataFrame(
            {
                "estagio": [1],
                "no": [1],
                "cenario": [1],
                "codigo_usina": [10],
                "nome_usina": ["A"],
                "codigo_submercado": [1],
                "codigo_ree": [100],
                "volume_util_inicial_hm3": [500.0],
                "volume_util_inicial_percentual": [70.0],
                "volume_util_final_hm3": [490.0],
                "volume_util_final_percentual": [68.0],
                "evaporacao_modelo_hm3": [10.5],
                "evaporacao_calculada_hm3": [10.0],
                "desvio_absoluto_hm3": [-0.5],
                "desvio_percentual": [-4.76],
            }
        )
        monkeypatch.setattr(
            DecOperEvap,
            "read",
            staticmethod(lambda _path: _StubFile(stub_table)),
        )
        result = read_dec_oper_evap(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        for column in (
            "estagio",
            "no",
            "cenario",
            "codigo_usina",
            "nome_usina",
            "codigo_submercado",
            "codigo_ree",
            "volume_util_inicial_hm3",
            "volume_util_inicial_percentual",
            "volume_util_final_hm3",
            "volume_util_final_percentual",
            "evaporacao_modelo_hm3",
            "evaporacao_calculada_hm3",
            "desvio_absoluto_hm3",
            "desvio_percentual",
        ):
            assert column in result.columns

    def test_saidas_only_is_not_found(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "dec_oper_evap.csv").touch()
        with pytest.raises(FileNotFoundError, match="dec_oper_evap.csv") as exc_info:
            read_dec_oper_evap(tmp_path)
        assert "saidas/" not in str(exc_info.value)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="dec_oper_evap.csv"):
            read_dec_oper_evap(tmp_path)

    def test_empty_parse_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "dec_oper_evap.csv").touch()
        monkeypatch.setattr(
            DecOperEvap,
            "read",
            staticmethod(lambda _path: _StubFile(None)),
        )
        with pytest.raises(ValueError, match="parsed empty"):
            read_dec_oper_evap(tmp_path)


class TestReadDecOperRheSoft:
    """`read_dec_oper_rhesoft`: RHE soft-constraint achieved LHS vs limit."""

    def test_finds_root_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "dec_oper_rhesoft.csv").touch()
        stub_table = pd.DataFrame(
            {
                "estagio": [4],
                "no": [4],
                "cenario": [1],
                "codigo_restricao": [101],
                "limite_MW": [10191.52],
                "valor_MW": [34550.64],
                "violacao_absoluta_MW": [0.0],
                "violacao_percentual": [0.0],
            }
        )
        monkeypatch.setattr(
            DecOperRheSoft,
            "read",
            staticmethod(lambda _path: _StubFile(stub_table)),
        )
        result = read_dec_oper_rhesoft(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        for column in (
            "estagio",
            "no",
            "cenario",
            "codigo_restricao",
            "limite_MW",
            "valor_MW",
            "violacao_absoluta_MW",
            "violacao_percentual",
        ):
            assert column in result.columns

    def test_saidas_only_is_not_found(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "dec_oper_rhesoft.csv").touch()
        with pytest.raises(FileNotFoundError, match="dec_oper_rhesoft.csv") as exc_info:
            read_dec_oper_rhesoft(tmp_path)
        assert "saidas/" not in str(exc_info.value)

    def test_missing_file_raises_naming_case_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="dec_oper_rhesoft.csv") as exc_info:
            read_dec_oper_rhesoft(tmp_path)
        message = str(exc_info.value)
        assert str(tmp_path) in message
        assert "saidas/" not in message

    def test_empty_parse_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "dec_oper_rhesoft.csv").touch()
        monkeypatch.setattr(
            DecOperRheSoft,
            "read",
            staticmethod(lambda _path: _StubFile(None)),
        )
        with pytest.raises(ValueError, match="parsed empty"):
            read_dec_oper_rhesoft(tmp_path)


class TestReadDecompTim:
    """`read_decomp_tim`: wall-clock timing table from ``decomp.tim``."""

    def test_finds_root_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "decomp.tim").touch()
        stub_table = pd.DataFrame(
            {
                "Etapa": ["Leitura de Dados", "Convergencia", "Tempo Total"],
                "Tempo": [
                    timedelta(seconds=1),
                    timedelta(minutes=5),
                    timedelta(minutes=6, seconds=1),
                ],
            }
        )
        monkeypatch.setattr(
            Decomptim,
            "read",
            staticmethod(lambda _path: _StubTimFile(stub_table)),
        )
        result = read_decomp_tim(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["Etapa", "Tempo"]
        assert result.height == 3

    def test_saidas_only_is_not_found(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "decomp.tim").touch()
        with pytest.raises(FileNotFoundError, match="decomp.tim") as exc_info:
            read_decomp_tim(tmp_path)
        assert "saidas/" not in str(exc_info.value)

    def test_missing_file_raises_naming_case_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="decomp.tim") as exc_info:
            read_decomp_tim(tmp_path)
        message = str(exc_info.value)
        assert str(tmp_path) in message
        assert "saidas/" not in message

    def test_none_table_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "decomp.tim").touch()
        monkeypatch.setattr(
            Decomptim,
            "read",
            staticmethod(lambda _path: _StubTimFile(None)),
        )
        with pytest.raises(ValueError, match="parsed empty"):
            read_decomp_tim(tmp_path)

    def test_empty_table_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "decomp.tim").touch()
        monkeypatch.setattr(
            Decomptim,
            "read",
            staticmethod(lambda _path: _StubTimFile(pd.DataFrame())),
        )
        with pytest.raises(ValueError, match="parsed empty"):
            read_decomp_tim(tmp_path)

    def test_result_excerpt_parses(self) -> None:
        df = read_decomp_tim(_RESULTS_EXCERPT)
        assert df.height > 0
        assert "total" in " ".join(df["Etapa"].to_list()).lower()


class TestReconcileKdollarsToReais:
    """`reconcile_kdollars_to_reais`: the single k$ -> R$ conversion site."""

    @pytest.mark.parametrize(
        ("kdollars", "reais"),
        [(1.0, 1000.0), (2.5, 2500.0), (0.0, 0.0)],
    )
    def test_applies_thousand_factor(self, kdollars: float, reais: float) -> None:
        assert reconcile_kdollars_to_reais(kdollars) == reais

    def test_docstring_names_kdollar_provenance(self) -> None:
        docstring = reconcile_kdollars_to_reais.__doc__
        assert docstring is not None
        assert "project_decomp_fcf_unit_conversion_bug" in docstring
        assert "k$" in docstring

"""Tests for ``DecompFiles``, the resolved input files of one deck revision."""

from __future__ import annotations

from pathlib import Path

import pytest

from novomodelo_bridge.core.errors import SourceFileError
from novomodelo_bridge.decomp.files import (
    discover_decomp_files,
    unindexed_libs_diagnostic,
)
from tests.conftest import make_decomp_files


def test_fixed_column_files_are_the_register_files_present(tmp_path: Path) -> None:
    without_gnl = make_decomp_files(tmp_path, renovaveis=tmp_path / "renovaveis.rv0")
    with_gnl = make_decomp_files(tmp_path, dadgnl=tmp_path / "dadgnl.rv0")

    assert without_gnl.fixed_column_files == [tmp_path / "dadger"]
    assert with_gnl.fixed_column_files == [tmp_path / "dadger", tmp_path / "dadgnl.rv0"]


# ---------------------------------------------------------------------------
# LIBS files: the index FA names, then the records it lists
# ---------------------------------------------------------------------------

_INDEX = """\
&**** ID ; Descricao ; Caminho
 HIDRELETRICA-CADASTRO-RESERVATORIO-CURVAJUSANTE ;pol ;polinjus.csv
 PARQUE-EOLICO-EQUIVALENTE-CADASTRO ;renovaveis ;renovaveis.csv
 PARQUE-EOLICO-EQUIVALENTE-SUBMERCADO ;renovaveis ;renovaveis.csv
 RESTRICAO-ELETRICA-ESPECIAL ;eletricas ;lib_restricao-eletrica-especial.csv
"""

_LIBS_FILES = ("polinjus.csv", "renovaveis.csv", "lib_restricao-eletrica-especial.csv")


def _deck(tmp_path: Path, dadger_lines: str, **texts: str) -> Path:
    """A minimal deck whose ``dadger`` carries *dadger_lines*, plus files named
    by *texts*' keys (``.`` spelled ``_``)."""
    (tmp_path / "caso.dat").write_text("rv0", encoding="latin-1")
    (tmp_path / "dadger.rv0").write_text(
        f"TE  deck\n&FA\n{dadger_lines}", encoding="latin-1"
    )
    for name in ("vazoes.rv0", "hidr.dat", *_LIBS_FILES):
        (tmp_path / name).write_text("", encoding="latin-1")
    for key, text in texts.items():
        (tmp_path / key.replace("_", ".", 1)).write_text(text, encoding="latin-1")
    return tmp_path


def test_libs_files_come_from_the_index_the_fa_register_names(tmp_path: Path) -> None:
    deck = _deck(tmp_path, "FA  lista.csv\n", lista_csv=_INDEX)

    files = discover_decomp_files(deck)

    assert files.libs_index == deck / "lista.csv"
    assert files.polinjus == deck / "polinjus.csv"
    assert files.renovaveis == deck / "renovaveis.csv"
    assert files.libs_restricao_eletrica == deck / "lib_restricao-eletrica-especial.csv"
    assert unindexed_libs_diagnostic(files) is None


def test_without_fa_the_index_and_its_files_are_ignored_and_reported(
    tmp_path: Path,
) -> None:
    deck = _deck(tmp_path, "", indices_csv=_INDEX)

    files = discover_decomp_files(deck)

    assert files.libs_index is None
    assert files.polinjus is None
    assert files.renovaveis is None
    assert files.libs_restricao_eletrica is None
    diagnostic = unindexed_libs_diagnostic(files)
    assert diagnostic is not None and diagnostic.table is not None
    assert {row[0] for row in diagnostic.table.rows} == {"indices.csv", *_LIBS_FILES}


def test_a_listed_file_the_index_does_not_declare_is_not_found_by_name(
    tmp_path: Path,
) -> None:
    index = "".join(
        line
        for line in _INDEX.splitlines(keepends=True)
        if "RESTRICAO-ELETRICA" not in line
    )
    deck = _deck(tmp_path, "FA  indices.csv\n", indices_csv=index)

    files = discover_decomp_files(deck)

    assert files.libs_restricao_eletrica is None
    diagnostic = unindexed_libs_diagnostic(files)
    assert diagnostic is not None and diagnostic.table is not None
    assert diagnostic.table.rows == [
        ["lib_restricao-eletrica-especial.csv", "RESTRICAO-ELETRICA-ESPECIAL"]
    ]


def test_fa_naming_a_missing_index_is_an_error(tmp_path: Path) -> None:
    deck = _deck(tmp_path, "FA  indices.csv\n")

    with pytest.raises(SourceFileError) as err:
        discover_decomp_files(deck)
    assert err.value.field == "FA"


def test_fa_registers_naming_different_files_are_an_error(tmp_path: Path) -> None:
    deck = _deck(
        tmp_path, "FA  indices.csv\nFA  outro.csv\n", indices_csv=_INDEX, outro_csv=""
    )

    with pytest.raises(SourceFileError, match="FA registers naming different files"):
        discover_decomp_files(deck)


def test_fj_and_the_index_agreeing_on_polinjus_is_accepted(tmp_path: Path) -> None:
    deck = _deck(tmp_path, "FA  indices.csv\nFJ  polinjus.csv\n", indices_csv=_INDEX)

    assert discover_decomp_files(deck).polinjus == deck / "polinjus.csv"


def test_fj_alone_declares_polinjus(tmp_path: Path) -> None:
    deck = _deck(tmp_path, "FJ  polinjus.csv\n")

    assert discover_decomp_files(deck).polinjus == deck / "polinjus.csv"


def test_fj_and_the_index_naming_different_polinjus_files_are_an_error(
    tmp_path: Path,
) -> None:
    deck = _deck(
        tmp_path, "FA  indices.csv\nFJ  outro.csv\n", indices_csv=_INDEX, outro_csv=""
    )

    with pytest.raises(SourceFileError) as err:
        discover_decomp_files(deck)
    assert err.value.field == "FJ"

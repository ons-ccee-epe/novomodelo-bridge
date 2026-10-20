"""Tests for ``NewaveFiles`` discovery: required-file failures as typed errors,
and LIBS files taken only from the records ``indices.csv`` declares.

The discovery chain (``caso.dat`` → ``Arquivos`` → required files) now raises
:class:`~novomodelo_bridge.core.errors.SourceFileError` (a :class:`FileNotFoundError`
subclass) carrying ``path``/``field`` location detail, instead of a bare
``FileNotFoundError``. These tests pin the type, the instance-check that keeps
``except FileNotFoundError`` handlers working, and the location attributes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from novomodelo_bridge.core.errors import SourceFileError
from novomodelo_bridge.core.libs_index import (
    HIDRELETRICA_CURVAJUSANTE,
    HIDRELETRICA_FPHA_REDUCAO_CORTES,
)
from novomodelo_bridge.newave.files import (
    NewaveFiles,
    _resolve_required,
    unindexed_libs_diagnostic,
)
from tests.conftest import make_nw_files


def test_missing_caso_raises_source_file_error(tmp_path: Path) -> None:
    with pytest.raises(SourceFileError) as exc_info:
        NewaveFiles.from_directory(tmp_path)
    assert isinstance(exc_info.value, FileNotFoundError)


def test_source_file_error_carries_path_and_field(tmp_path: Path) -> None:
    with pytest.raises(SourceFileError) as exc_info:
        NewaveFiles.from_directory(tmp_path)
    exc = exc_info.value
    assert exc.path == str(tmp_path)
    assert exc.field == "caso.dat"


def test_resolve_required_missing_carries_path_and_field(tmp_path: Path) -> None:
    with pytest.raises(SourceFileError) as exc_info:
        _resolve_required(tmp_path, "hidr.dat")
    exc = exc_info.value
    assert isinstance(exc, FileNotFoundError)
    assert exc.path == str(tmp_path)
    assert exc.field == "hidr.dat"


def test_missing_caso_message_is_preserved(tmp_path: Path) -> None:
    with pytest.raises(SourceFileError, match=f"caso.dat not found in {tmp_path}"):
        NewaveFiles.from_directory(tmp_path)


def test_fixed_column_files_leave_out_binary_delimited_and_absent_inputs(
    tmp_path: Path,
) -> None:
    files = make_nw_files(
        tmp_path,
        cvar=tmp_path / "cvar.dat",
        libs_index=tmp_path / "indices.csv",
        polinjus=tmp_path / "polinjus.csv",
        tratamento_fpha=tmp_path / "tratamento-fpha.csv",
        restricao_eletrica=tmp_path / "restricao-eletrica.csv",
    )

    names = {path.name for path in files.fixed_column_files}

    assert names == {
        "dger.dat",
        "confhd.dat",
        "conft.dat",
        "sistema.dat",
        "clast.dat",
        "term.dat",
        "ree.dat",
        "patamar.dat",
        "cvar.dat",
    }


# ---------------------------------------------------------------------------
# LIBS files: declared by indices.csv, never found by name
# ---------------------------------------------------------------------------

_MINI_DECK = Path(__file__).parents[1] / "decks" / "newave_mini"


def _mini_deck(tmp_path: Path, index: str | None, *files: str) -> Path:
    """A copy of the committed mini deck whose ``indices.csv`` is *index*
    (absent when ``None``), plus empty *files*."""
    deck = tmp_path / "deck"
    shutil.copytree(_MINI_DECK, deck, ignore=shutil.ignore_patterns("indices.csv"))
    if index is not None:
        (deck / "indices.csv").write_text(index, encoding="utf-8")
    for name in files:
        (deck / name).write_text("", encoding="utf-8")
    return deck


def test_libs_files_come_from_the_records_indices_csv_lists(tmp_path: Path) -> None:
    deck = _mini_deck(
        tmp_path,
        "HIDRELETRICA-CADASTRO-RESERVATORIO-CURVAJUSANTE ; ; polinjus.csv\n"
        "RESTRICAO-ELETRICA-ESPECIAL; ; restricao-eletrica.csv\n",
        "restricao-eletrica.csv",
    )

    files = NewaveFiles.from_directory(deck)

    assert files.libs_index == deck / "indices.csv"
    assert files.polinjus == deck / "polinjus.csv"
    assert files.restricao_eletrica == deck / "restricao-eletrica.csv"
    assert files.tratamento_fpha is None
    assert unindexed_libs_diagnostic(files) is None


def test_commented_out_treatment_record_leaves_its_file_unread_and_reported(
    tmp_path: Path,
) -> None:
    deck = _mini_deck(
        tmp_path,
        "HIDRELETRICA-CADASTRO-RESERVATORIO-CURVAJUSANTE ; ; polinjus.csv\n"
        "&HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES; ; tratamento-fpha.csv\n",
        "tratamento-fpha.csv",
    )

    files = NewaveFiles.from_directory(deck)

    assert files.tratamento_fpha is None
    diagnostic = unindexed_libs_diagnostic(files)
    assert diagnostic is not None and diagnostic.table is not None
    assert diagnostic.table.rows == [
        ["tratamento-fpha.csv", HIDRELETRICA_FPHA_REDUCAO_CORTES]
    ]


def test_without_indices_csv_the_deck_has_no_libs_files(tmp_path: Path) -> None:
    files = NewaveFiles.from_directory(_mini_deck(tmp_path, None))

    assert files.libs_index is None
    assert files.polinjus is None
    diagnostic = unindexed_libs_diagnostic(files)
    assert diagnostic is not None and diagnostic.table is not None
    assert diagnostic.table.rows == [["polinjus.csv", HIDRELETRICA_CURVAJUSANTE]]


def test_a_listed_libs_file_missing_from_the_deck_is_an_error(tmp_path: Path) -> None:
    deck = _mini_deck(
        tmp_path, "RESTRICAO-ELETRICA-ESPECIAL; ; restricao-eletrica.csv\n"
    )

    with pytest.raises(SourceFileError) as err:
        NewaveFiles.from_directory(deck)
    assert err.value.field == "RESTRICAO-ELETRICA-ESPECIAL"


# ---------------------------------------------------------------------------
# Pumping station LIBS file discovery
# ---------------------------------------------------------------------------


def test_pumping_files_resolved_from_indices(tmp_path: Path) -> None:
    """indices.csv listing pumping keys resolves both fields."""
    from novomodelo_bridge.core.libs_index import (
        ESTACAO_BOMBEAMENTO,
        ESTACAO_BOMBEAMENTO_SUBMERCADO,
    )

    deck = _mini_deck(
        tmp_path,
        f"{ESTACAO_BOMBEAMENTO} ; ; estacao-bombeamento.csv\n"
        f"{ESTACAO_BOMBEAMENTO_SUBMERCADO} ; ; estacao-bombeamento-submercado.csv\n",
        "estacao-bombeamento.csv",
        "estacao-bombeamento-submercado.csv",
    )

    files = NewaveFiles.from_directory(deck)

    assert files.estacao_bombeamento == deck / "estacao-bombeamento.csv"
    assert (
        files.estacao_bombeamento_submercado
        == deck / "estacao-bombeamento-submercado.csv"
    )


def test_pumping_files_none_when_not_in_index(tmp_path: Path) -> None:
    """Pumping files not in indices.csv are None, not discovered by name."""
    deck = _mini_deck(
        tmp_path,
        "",  # Empty indices.csv — no records listed
        "estacao-bombeamento.csv",  # File present but not indexed
    )

    files = NewaveFiles.from_directory(deck)

    assert files.estacao_bombeamento is None
    assert files.estacao_bombeamento_submercado is None


def test_pumping_files_unindexed_warning(tmp_path: Path) -> None:
    """Unlisted pumping file produces libs-unindexed-file warning."""
    deck = _mini_deck(
        tmp_path,
        "",  # Empty indices.csv
        "estacao-bombeamento.csv",  # Present but not listed
    )

    files = NewaveFiles.from_directory(deck)
    diagnostic = unindexed_libs_diagnostic(files)

    assert diagnostic is not None
    assert diagnostic.table is not None
    # Should report the pumping file as unindexed
    file_names = [row[0] for row in diagnostic.table.rows]
    assert "estacao-bombeamento.csv" in file_names

"""Tests for the LIBS index reader and the undeclared-files warning.

The index layouts mirror the two real styles: the DECOMP one opens with a block
of ``&`` comment lines and indents its keys, the NEWAVE one is bare padded
lines.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.errors import SourceFileError
from novomodelo_bridge.core.libs_index import (
    HIDRELETRICA_CURVAJUSANTE,
    HIDRELETRICA_FPHA_REDUCAO_CORTES,
    PARQUE_EOLICO_EQUIVALENTE_PREFIX,
    RESTRICAO_ELETRICA_ESPECIAL,
    UNINDEXED_FILES_CODE,
    ConventionalFile,
    ignored_files,
    read_libs_index,
    unindexed_files_diagnostic,
)

_DECOMP_STYLE = """\
&**************************************************;;
& ID: Id dos cards de indice;;
& ID ;Descricao ;Caminho
& ; ;
 HIDRELETRICA-CADASTRO-RESERVATORIO-CURVAJUSANTE ;pol ;polinjus.csv
 PARQUE-EOLICO-EQUIVALENTE-CADASTRO ;Dados de usinas renovaveis ;renovaveis.csv
 PARQUE-EOLICO-EQUIVALENTE-CONFIGURACAO ;Dados de usinas renovaveis ;renovaveis.csv
"""

_NEWAVE_STYLE = """\
RESTRICAO-ELETRICA-ESPECIAL; ; restricao-eletrica.csv
HIDRELETRICA-CADASTRO-RESERVATORIO-CURVAJUSANTE ; ; polinjus.csv
&HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES; ; tratamento-fpha.csv

"""


def _deck(tmp_path: Path, index: str, *files: str) -> Path:
    (tmp_path / "indices.csv").write_text(index, encoding="utf-8")
    for name in files:
        (tmp_path / name).write_text("", encoding="utf-8")
    return tmp_path / "indices.csv"


def test_read_libs_index_skips_the_comment_header_and_strips_padded_fields(
    tmp_path: Path,
) -> None:
    index = read_libs_index(tmp_path, _deck(tmp_path, _DECOMP_STYLE))

    assert dict(index.entries) == {
        HIDRELETRICA_CURVAJUSANTE: "polinjus.csv",
        "PARQUE-EOLICO-EQUIVALENTE-CADASTRO": "renovaveis.csv",
        "PARQUE-EOLICO-EQUIVALENTE-CONFIGURACAO": "renovaveis.csv",
    }


def test_read_libs_index_ignores_a_commented_out_entry(tmp_path: Path) -> None:
    index = read_libs_index(
        tmp_path, _deck(tmp_path, _NEWAVE_STYLE, "tratamento-fpha.csv")
    )

    assert HIDRELETRICA_FPHA_REDUCAO_CORTES not in index.entries
    assert index.path(HIDRELETRICA_FPHA_REDUCAO_CORTES) is None


def test_read_libs_index_without_an_index_file_is_empty(tmp_path: Path) -> None:
    index = read_libs_index(tmp_path, None)

    assert index.source is None
    assert index.path(HIDRELETRICA_CURVAJUSANTE) is None
    assert index.path_for_family(PARQUE_EOLICO_EQUIVALENTE_PREFIX) is None


def test_read_libs_index_reads_a_latin1_index(tmp_path: Path) -> None:
    (tmp_path / "indices.csv").write_bytes(
        "& Descrição\nRESTRICAO-ELETRICA-ESPECIAL; ; re.csv\n".encode("latin-1")
    )

    index = read_libs_index(tmp_path, tmp_path / "indices.csv")

    assert dict(index.entries) == {RESTRICAO_ELETRICA_ESPECIAL: "re.csv"}


def test_read_libs_index_accepts_a_key_repeated_with_the_same_file(
    tmp_path: Path,
) -> None:
    path = _deck(
        tmp_path,
        "RESTRICAO-ELETRICA-ESPECIAL; ; re.csv\nRESTRICAO-ELETRICA-ESPECIAL; ; RE.csv\n",
    )

    assert read_libs_index(tmp_path, path).entries[RESTRICAO_ELETRICA_ESPECIAL] == (
        "re.csv"
    )


def test_read_libs_index_rejects_a_key_listed_with_two_files(tmp_path: Path) -> None:
    path = _deck(
        tmp_path,
        "RESTRICAO-ELETRICA-ESPECIAL; ; a.csv\nRESTRICAO-ELETRICA-ESPECIAL; ; b.csv\n",
    )

    with pytest.raises(SourceFileError, match="twice, in a.csv and in b.csv") as err:
        read_libs_index(tmp_path, path)
    assert err.value.field == RESTRICAO_ELETRICA_ESPECIAL


def test_read_libs_index_rejects_an_entry_without_a_file_name(tmp_path: Path) -> None:
    path = _deck(tmp_path, "RESTRICAO-ELETRICA-ESPECIAL; ; \n")

    with pytest.raises(SourceFileError, match="without a file name"):
        read_libs_index(tmp_path, path)


def test_path_resolves_the_listed_file_ignoring_case(tmp_path: Path) -> None:
    index = read_libs_index(tmp_path, _deck(tmp_path, _NEWAVE_STYLE, "POLINJUS.CSV"))

    assert index.path(HIDRELETRICA_CURVAJUSANTE) == tmp_path / "POLINJUS.CSV"


def test_path_raises_when_the_listed_file_is_missing(tmp_path: Path) -> None:
    index = read_libs_index(tmp_path, _deck(tmp_path, _NEWAVE_STYLE))

    with pytest.raises(SourceFileError, match="lists RESTRICAO-ELETRICA-ESPECIAL in"):
        index.path(RESTRICAO_ELETRICA_ESPECIAL)


def test_path_for_family_returns_the_one_shared_file(tmp_path: Path) -> None:
    index = read_libs_index(tmp_path, _deck(tmp_path, _DECOMP_STYLE, "renovaveis.csv"))

    assert index.path_for_family(PARQUE_EOLICO_EQUIVALENTE_PREFIX) == (
        tmp_path / "renovaveis.csv"
    )


def test_path_for_family_rejects_records_spread_over_two_files(
    tmp_path: Path,
) -> None:
    split = _DECOMP_STYLE.replace(
        "CONFIGURACAO ;Dados de usinas renovaveis ;renovaveis.csv",
        "CONFIGURACAO ;Dados de usinas renovaveis ;outro.csv",
    )
    index = read_libs_index(tmp_path, _deck(tmp_path, split, "renovaveis.csv"))

    with pytest.raises(SourceFileError, match="several files"):
        index.path_for_family(PARQUE_EOLICO_EQUIVALENTE_PREFIX)


def test_ignored_files_lists_conventional_files_the_index_did_not_resolve(
    tmp_path: Path,
) -> None:
    for name in ("polinjus.csv", "Tratamento-FPHA.csv", "other.csv"):
        (tmp_path / name).write_text("", encoding="utf-8")
    conventional = [
        ConventionalFile(
            "polinjus.csv", HIDRELETRICA_CURVAJUSANTE, tmp_path / "polinjus.csv"
        ),
        ConventionalFile(
            "tratamento[-_]fpha.*", HIDRELETRICA_FPHA_REDUCAO_CORTES, None
        ),
    ]

    assert ignored_files(tmp_path, conventional) == [
        ("Tratamento-FPHA.csv", HIDRELETRICA_FPHA_REDUCAO_CORTES)
    ]


def test_unindexed_files_diagnostic_is_a_warning_with_one_row_per_file() -> None:
    diagnostic = unindexed_files_diagnostic(
        [("tratamento-fpha.csv", HIDRELETRICA_FPHA_REDUCAO_CORTES)],
        remediation="List the record in indices.csv.",
    )

    assert diagnostic is not None
    assert diagnostic.code == UNINDEXED_FILES_CODE
    assert diagnostic.severity is Severity.WARNING
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [
        ["tratamento-fpha.csv", HIDRELETRICA_FPHA_REDUCAO_CORTES]
    ]
    text = f"{diagnostic.summary} {diagnostic.remediation}"
    for leak in ("docs/", "plans/", "~/git", "ticket-", "epic-", ".py"):
        assert leak not in text


def test_unindexed_files_diagnostic_is_none_without_ignored_files() -> None:
    assert unindexed_files_diagnostic([], remediation="unused") is None

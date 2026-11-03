"""Tests for the LIBS index reader and the LIBS file warnings.

The index layouts mirror the two real styles: the DECOMP one opens with a block
of ``&`` comment lines and indents its keys, the NEWAVE one is bare padded
lines. Each index test runs with both tracks' ``Indices`` readers.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from idecomp.libs import Indices as DecompIndices
from idecomp.libs.restricoes import Restricoes as DecompRestricoes
from inewave.libs import Indices as NewaveIndices
from inewave.libs.restricoes import Restricoes as NewaveRestricoes

from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.errors import SourceFileError
from novomodelo_bridge.core.libs_index import (
    HIDRELETRICA_CURVAJUSANTE,
    HIDRELETRICA_FPHA_REDUCAO_CORTES,
    PARQUE_EOLICO_EQUIVALENTE_PREFIX,
    RESTRICAO_ELETRICA_ESPECIAL,
    UNINDEXED_FILES_CODE,
    UNREAD_LINES_CODE,
    ConventionalFile,
    IndexFile,
    ignored_files,
    read_libs_index,
    unindexed_files_diagnostic,
    unread_lines_diagnostic,
)

ReadIndex = Callable[[str], IndexFile]

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


@pytest.fixture(
    params=[NewaveIndices.read, DecompIndices.read], ids=["inewave", "idecomp"]
)
def read(request: pytest.FixtureRequest) -> ReadIndex:
    return request.param


def test_read_libs_index_skips_the_comment_header_and_strips_padded_fields(
    tmp_path: Path, read: ReadIndex
) -> None:
    index = read_libs_index(tmp_path, _deck(tmp_path, _DECOMP_STYLE), read)

    assert dict(index.entries) == {
        HIDRELETRICA_CURVAJUSANTE: "polinjus.csv",
        "PARQUE-EOLICO-EQUIVALENTE-CADASTRO": "renovaveis.csv",
        "PARQUE-EOLICO-EQUIVALENTE-CONFIGURACAO": "renovaveis.csv",
    }


def test_read_libs_index_ignores_a_commented_out_entry(
    tmp_path: Path, read: ReadIndex
) -> None:
    index = read_libs_index(
        tmp_path, _deck(tmp_path, _NEWAVE_STYLE, "tratamento-fpha.csv"), read
    )

    assert HIDRELETRICA_FPHA_REDUCAO_CORTES not in index.entries
    assert index.path(HIDRELETRICA_FPHA_REDUCAO_CORTES) is None


def test_read_libs_index_without_an_index_file_is_empty(
    tmp_path: Path, read: ReadIndex
) -> None:
    index = read_libs_index(tmp_path, None, read)

    assert index.source is None
    assert index.path(HIDRELETRICA_CURVAJUSANTE) is None
    assert index.path_for_family(PARQUE_EOLICO_EQUIVALENTE_PREFIX) is None


def test_read_libs_index_reads_a_latin1_index(tmp_path: Path, read: ReadIndex) -> None:
    (tmp_path / "indices.csv").write_bytes(
        "& Descrição\nRESTRICAO-ELETRICA-ESPECIAL; ; re.csv\n".encode("latin-1")
    )

    index = read_libs_index(tmp_path, tmp_path / "indices.csv", read)

    assert dict(index.entries) == {RESTRICAO_ELETRICA_ESPECIAL: "re.csv"}


def test_read_libs_index_reads_an_index_that_opens_with_a_utf8_bom(
    tmp_path: Path, read: ReadIndex
) -> None:
    (tmp_path / "indices.csv").write_bytes(
        b"\xef\xbb\xbfRESTRICAO-ELETRICA-ESPECIAL; ; re.csv\n"
    )

    index = read_libs_index(tmp_path, tmp_path / "indices.csv", read)

    assert dict(index.entries) == {RESTRICAO_ELETRICA_ESPECIAL: "re.csv"}


def test_read_libs_index_accepts_a_key_repeated_with_the_same_file(
    tmp_path: Path, read: ReadIndex
) -> None:
    path = _deck(
        tmp_path,
        "RESTRICAO-ELETRICA-ESPECIAL; ; re.csv\nRESTRICAO-ELETRICA-ESPECIAL; ; RE.csv\n",
    )

    index = read_libs_index(tmp_path, path, read)

    assert index.entries[RESTRICAO_ELETRICA_ESPECIAL] == "re.csv"


def test_read_libs_index_rejects_a_key_listed_with_two_files(
    tmp_path: Path, read: ReadIndex
) -> None:
    path = _deck(
        tmp_path,
        "RESTRICAO-ELETRICA-ESPECIAL; ; a.csv\nRESTRICAO-ELETRICA-ESPECIAL; ; b.csv\n",
    )

    with pytest.raises(SourceFileError, match="twice, in a.csv and in b.csv") as err:
        read_libs_index(tmp_path, path, read)
    assert err.value.field == RESTRICAO_ELETRICA_ESPECIAL


@pytest.mark.parametrize(
    "line",
    ["RESTRICAO-ELETRICA-ESPECIAL; ; ", "RESTRICAO-ELETRICA-ESPECIAL; Restricoes"],
    ids=["blank-file", "no-file-field"],
)
def test_read_libs_index_rejects_an_entry_without_a_file_name(
    tmp_path: Path, read: ReadIndex, line: str
) -> None:
    path = _deck(tmp_path, f"{line}\n")

    with pytest.raises(SourceFileError, match="without a file name"):
        read_libs_index(tmp_path, path, read)


def test_read_libs_index_does_not_give_a_short_line_the_file_above_it(
    tmp_path: Path, read: ReadIndex
) -> None:
    path = _deck(
        tmp_path,
        "HIDRELETRICA-CADASTRO-RESERVATORIO-CURVAJUSANTE; ; polinjus.csv\n"
        "RESTRICAO-ELETRICA-ESPECIAL; Restricoes\n",
    )

    with pytest.raises(SourceFileError, match="RESTRICAO-ELETRICA-ESPECIAL without"):
        read_libs_index(tmp_path, path, read)


def test_path_resolves_the_listed_file_ignoring_case(
    tmp_path: Path, read: ReadIndex
) -> None:
    index = read_libs_index(
        tmp_path, _deck(tmp_path, _NEWAVE_STYLE, "POLINJUS.CSV"), read
    )

    assert index.path(HIDRELETRICA_CURVAJUSANTE) == tmp_path / "POLINJUS.CSV"


def test_path_raises_when_the_listed_file_is_missing(
    tmp_path: Path, read: ReadIndex
) -> None:
    index = read_libs_index(tmp_path, _deck(tmp_path, _NEWAVE_STYLE), read)

    with pytest.raises(SourceFileError, match="lists RESTRICAO-ELETRICA-ESPECIAL in"):
        index.path(RESTRICAO_ELETRICA_ESPECIAL)


def test_path_for_family_returns_the_one_shared_file(
    tmp_path: Path, read: ReadIndex
) -> None:
    index = read_libs_index(
        tmp_path, _deck(tmp_path, _DECOMP_STYLE, "renovaveis.csv"), read
    )

    assert index.path_for_family(PARQUE_EOLICO_EQUIVALENTE_PREFIX) == (
        tmp_path / "renovaveis.csv"
    )


def test_path_for_family_rejects_records_spread_over_two_files(
    tmp_path: Path, read: ReadIndex
) -> None:
    split = _DECOMP_STYLE.replace(
        "CONFIGURACAO ;Dados de usinas renovaveis ;renovaveis.csv",
        "CONFIGURACAO ;Dados de usinas renovaveis ;outro.csv",
    )
    index = read_libs_index(tmp_path, _deck(tmp_path, split, "renovaveis.csv"), read)

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


_RESTRICTIONS = """\
& comentario
RE;1;ger_usih(1)

re;2;ger_usih(2)
RESTRICAO-ELETRICA-FORMULA;3;ger_usih(3)
RE-LIM-FOR-PER-PAT;1;1;6;1;;4100
"""


@pytest.mark.parametrize(
    ("read_restrictions", "unread"),
    [
        pytest.param(
            NewaveRestricoes.read,
            [
                "re;2;ger_usih(2)",
                "RESTRICAO-ELETRICA-FORMULA;3;ger_usih(3)",
                "RE-LIM-FOR-PER-PAT;1;1;6;1;;4100",
            ],
            id="inewave",
        ),
        pytest.param(
            DecompRestricoes.read,
            ["re;2;ger_usih(2)", "RE-LIM-FOR-PER-PAT;1;1;6;1;;4100"],
            id="idecomp",
        ),
    ],
)
def test_unread_lines_diagnostic_lists_the_lines_kept_as_text(
    tmp_path: Path,
    read_restrictions: Callable[[str], NewaveRestricoes | DecompRestricoes],
    unread: list[str],
) -> None:
    path = tmp_path / "restricao-eletrica.csv"
    path.write_text(_RESTRICTIONS, encoding="utf-8")

    diagnostic = unread_lines_diagnostic(path, read_restrictions(str(path)).data)

    assert diagnostic is not None
    assert diagnostic.code == UNREAD_LINES_CODE
    assert diagnostic.severity is Severity.WARNING
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [[line] for line in unread]
    text = f"{diagnostic.summary} {diagnostic.remediation}"
    for leak in ("docs/", "plans/", "~/git", "ticket-", "epic-", ".py"):
        assert leak not in text


def test_unread_lines_diagnostic_is_none_when_every_line_is_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "restricao-eletrica.csv"
    path.write_text("& comentario\n\nRE;1;ger_usih(1)\n", encoding="utf-8")

    assert unread_lines_diagnostic(path, NewaveRestricoes.read(str(path)).data) is None

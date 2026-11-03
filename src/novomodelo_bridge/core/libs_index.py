"""The LIBS index: which deck file holds each LIBS record.

A LIBS record is part of a deck only when the deck's index lists it; the source
model ignores a file the index does not name, whatever the file is called. The
index sits in the deck folder and names files in that same folder. Each line is
``RECORD-KEY ; description ; file``, and ``&`` opens a comment line.

Each track reads the index with its own library's ``Indices`` reader
(``inewave.libs`` or ``idecomp.libs``, which share the API); this module holds
the checks both tracks apply on top.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol

import pandas as pd
from cfinterface.components.defaultregister import DefaultRegister

from novomodelo_bridge.core.diagnostics import Diagnostic, DiagnosticTable, Severity
from novomodelo_bridge.core.errors import SourceFileError
from novomodelo_bridge.core.paths import find_case_insensitive

#: The name the source model reads its index by when nothing in the deck names
#: the index file.
INDEX_FILENAME = "indices.csv"

RESTRICAO_ELETRICA_ESPECIAL = "RESTRICAO-ELETRICA-ESPECIAL"
HIDRELETRICA_CADASTRO_RESERVATORIO = "HIDRELETRICA-CADASTRO-RESERVATORIO"
HIDRELETRICA_CURVAJUSANTE = "HIDRELETRICA-CADASTRO-RESERVATORIO-CURVAJUSANTE"
HIDRELETRICA_FPHA_REDUCAO_CORTES = "HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES"
PARQUE_EOLICO_EQUIVALENTE_PREFIX = "PARQUE-EOLICO-EQUIVALENTE-"

# Pumping station records (NEWAVE only; DECOMP uses dadger.ue())
ESTACAO_BOMBEAMENTO = "ESTACAO-BOMBEAMENTO"
ESTACAO_BOMBEAMENTO_SUBMERCADO = "ESTACAO-BOMBEAMENTO-SUBMERCADO"

UNINDEXED_FILES_CODE = "libs-unindexed-file"
UNREAD_LINES_CODE = "libs-unread-lines"

_COMMENT_PREFIX = "&"


class IndexFile(Protocol):
    """The ``Indices`` reader of ``inewave.libs`` and ``idecomp.libs``."""

    def indices(self, *, df: Literal[True]) -> pd.DataFrame: ...


@dataclass(frozen=True)
class LibsIndex:
    """A deck's parsed LIBS index; empty when the deck has none."""

    deck_dir: Path
    #: The index file read, or ``None`` when the deck has no index.
    source: Path | None
    #: Upper-cased record key -> file name as the index writes it.
    entries: Mapping[str, str]

    def path(self, key: str) -> Path | None:
        """The file holding *key*'s records, or ``None`` when the index does not
        list *key*.

        Raises:
            SourceFileError: the index lists *key* but its file is not in the
                deck folder.
        """
        name = self.entries.get(key)
        return None if name is None else self._resolve(key, name)

    def path_for_family(self, prefix: str) -> Path | None:
        """The one file every key starting with *prefix* points at, or ``None``
        when the index lists none of them.

        Raises:
            SourceFileError: the keys point at different files, or the file is
                not in the deck folder.
        """
        family = sorted(
            (key, name) for key, name in self.entries.items() if key.startswith(prefix)
        )
        if not family:
            return None
        names = sorted({name.lower() for _, name in family})
        if len(names) > 1:
            raise SourceFileError(
                f"{self._index_name} spreads the {prefix}* records over several "
                f"files ({', '.join(names)}); they must all be in one file",
                path=str(self.source),
                field=prefix,
            )
        return self._resolve(*family[0])

    @property
    def _index_name(self) -> str:
        return self.source.name if self.source is not None else INDEX_FILENAME

    def _resolve(self, key: str, name: str) -> Path:
        path = find_case_insensitive(self.deck_dir, name)
        if path is None:
            raise SourceFileError(
                f"{self._index_name} lists {key} in {name}, but {name} is not in "
                f"{self.deck_dir}",
                path=str(self.deck_dir / name),
                field=key,
            )
        return path


def read_libs_index(
    deck_dir: Path, index_file: Path | None, read: Callable[[str], IndexFile]
) -> LibsIndex:
    """Parse *index_file*, the index of the deck in *deck_dir*.

    Args:
        index_file: located by the caller, since each source model has its own
            rule for it; ``None`` gives an empty index.
        read: the track's ``Indices.read``.

    Raises:
        SourceFileError: a key is listed with no file, or twice with different
            files.
    """
    if index_file is None:
        return LibsIndex(deck_dir, None, MappingProxyType({}))
    entries: dict[str, str] = {}
    for raw_key, raw_name in _index_rows(read(str(index_file))):
        if not raw_key:
            continue
        key, name = raw_key.upper(), raw_name or ""
        if not name:
            raise SourceFileError(
                f"{index_file.name} lists {key} without a file name",
                path=str(index_file),
                field=key,
            )
        listed = entries.setdefault(key, name)
        if listed.lower() != name.lower():
            raise SourceFileError(
                f"{index_file.name} lists {key} twice, in {listed} and in {name}",
                path=str(index_file),
                field=key,
            )
    return LibsIndex(deck_dir, index_file, MappingProxyType(entries))


def _index_rows(index: IndexFile) -> Iterable[tuple[str | None, str | None]]:
    """``(key, file)`` per index line; a field missing at the line's end is
    ``None``."""
    frame = index.indices(df=True)
    if frame.empty:
        return []
    # The readers decode plain UTF-8, which leaves a byte-order mark on the
    # first key.
    return [
        (
            None if pd.isna(key) else str(key).removeprefix("\ufeff"),
            None if pd.isna(name) else str(name),
        )
        for key, name in zip(frame["identificador"], frame["arquivo"], strict=True)
    ]


@dataclass(frozen=True)
class ConventionalFile:
    """A name a LIBS file usually has, and the record it would hold."""

    #: ``fnmatch`` pattern, matched against lower-cased file names.
    pattern: str
    record: str
    #: The file the index resolved for *record*; never reported as ignored.
    resolved: Path | None


def ignored_files(
    deck_dir: Path, conventional: Sequence[ConventionalFile]
) -> list[tuple[str, str]]:
    """``(file name, record)`` for each file in *deck_dir* that has a
    conventional LIBS name but is not the file the index resolved."""
    try:
        names = sorted(entry.name for entry in deck_dir.iterdir() if entry.is_file())
    except OSError:
        return []
    found: list[tuple[str, str]] = []
    for file in conventional:
        for name in names:
            if not fnmatch.fnmatchcase(name.lower(), file.pattern):
                continue
            if file.resolved is not None and file.resolved.name == name:
                continue
            found.append((name, file.record))
    return found


def unindexed_files_diagnostic(
    ignored: Sequence[tuple[str, str]], *, remediation: str
) -> Diagnostic | None:
    """WARNING naming the deck files the source model ignores because nothing
    declares them, or ``None`` when there are none."""
    if not ignored:
        return None
    return Diagnostic(
        code=UNINDEXED_FILES_CODE,
        severity=Severity.WARNING,
        category="Source files",
        title="Deck files not declared in the LIBS index",
        summary=(
            f"{len(ignored)} file(s) in the deck folder look like LIBS input files, "
            "but the deck does not declare them, so the source model ignores them "
            "and so does the conversion."
        ),
        table=DiagnosticTable(
            columns=["file", "record"], rows=[list(row) for row in ignored]
        ),
        remediation=remediation,
    )


def unread_lines_diagnostic(
    file: Path, registers: Iterable[object]
) -> Diagnostic | None:
    """WARNING naming the lines of LIBS *file* that its reader kept as plain
    text, other than blank and ``&`` comment lines; ``None`` when there are none."""
    lines = [
        line
        for line in (
            str(register.data).strip()
            for register in registers
            if isinstance(register, DefaultRegister)
        )
        if line and not line.startswith(_COMMENT_PREFIX)
    ]
    if not lines:
        return None
    return Diagnostic(
        code=UNREAD_LINES_CODE,
        severity=Severity.WARNING,
        category="Source files",
        title=f"Lines of {file.name} that are not read",
        summary=(
            f"{len(lines)} line(s) of {file.name} are not records the conversion "
            "knows, so it ignores them. Record names are read only as the manual "
            "writes them, in full and in upper case."
        ),
        table=DiagnosticTable(columns=["line"], rows=[[line] for line in lines]),
        remediation=(
            "Write each record name exactly as the manual does, in upper case, or "
            "delete the line if it is not a record."
        ),
    )

"""Deck file discovery via caso.dat -> the revision index file.

LIBS files come only from the LIBS index that ``dadger``'s ``FA`` register
names (:mod:`novomodelo_bridge.core.libs_index`); a deck without ``FA`` has none.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from novomodelo_bridge.core.diagnostics import Diagnostic
from novomodelo_bridge.core.errors import SourceFileError
from novomodelo_bridge.core.libs_index import (
    HIDRELETRICA_CURVAJUSANTE,
    INDEX_FILENAME,
    PARQUE_EOLICO_EQUIVALENTE_PREFIX,
    RESTRICAO_ELETRICA_ESPECIAL,
    ConventionalFile,
    ignored_files,
    read_libs_index,
    unindexed_files_diagnostic,
)
from novomodelo_bridge.core.paths import find_case_insensitive

#: Opens a comment line in the deck's register and index files.
REGISTER_COMMENT_PREFIX = "&"


@dataclass(frozen=True)
class DecompFiles:
    """Resolved input files of one deck revision."""

    revision: str
    dadger: Path
    vazoes: Path
    hidr: Path
    dadgnl: Path | None
    renovaveis: Path | None
    polinjus: Path | None
    #: The deck's LIBs-era electrical-constraint file, from the LIBS index;
    #: ``None`` when the index lists none. Defaults to ``None`` so every
    #: pre-existing ``DecompFiles(...)`` call site keeps constructing without it.
    libs_restricao_eletrica: Path | None = None
    #: The deck's boundary-FCF header file (``cortesh.dat``), resolved via
    #: :func:`_resolve_fc_record_path` (the deck's ``FC NEWV21`` record) or,
    #: failing that, the ``cortesh*`` glob idiom; ``None`` when the deck
    #: carries no boundary FCF. Defaults to ``None`` for the
    #: same back-compat reason as ``libs_restricao_eletrica`` above.
    cortesh: Path | None = None
    #: The deck's boundary-FCF cut-record file: a single-stage partition
    #: export (``cortes-<estagio>.dat``, preferred) or the consolidated
    #: archive (``cortes.dat``), resolved via :func:`_resolve_fc_record_path`
    #: (the deck's ``FC NEWCUT`` record) or the glob idiom; ``None`` when
    #: absent.
    cortes: Path | None = None
    #: The LIBS index file ``FA`` names; ``None`` when ``dadger`` has no ``FA``.
    libs_index: Path | None = None

    @property
    def fixed_column_files(self) -> list[Path]:
        """The resolved inputs whose reader slices fixed columns."""
        return [path for path in (self.dadger, self.dadgnl) if path is not None]


#: ``FC`` register ``tipo`` mnemonics (``idecomp.decomp.Dadger.fc()``) naming
#: the boundary-FCF header and cut-record files respectively.
_FC_TIPO_CORTESH = "NEWV21"
_FC_TIPO_CORTES = "NEWCUT"


def _resolve_fc_record_path(dadger: Path, deck_dir: Path, *, tipo: str) -> Path | None:
    """Resolve one boundary-FCF file named by the deck's ``FC`` register.

    A lightweight fixed-width text scan of *dadger*, as :func:`_libs_registers`
    does for ``FA``/``FJ``, rather than a full :class:`idecomp.decomp.Dadger`
    parse (the caller re-parses *dadger* structurally right after discovery returns;
    duplicating that heavier, structured parse here would be wasted work and
    would need to guard against whatever exception surface a malformed
    dadger raises through it). The ``FC`` register
    (``idecomp.decomp.modelos.dadger.FC``) is fixed-width: identifier at
    columns 0:4, ``tipo`` mnemonic at columns 4:10, ``caminho`` at columns
    14:214 — confirmed against ``example/decomp-mar-26-rv2/dadger.rv2``'s own
    ``FC  NEWV21    cortesh.dat`` / ``FC  NEWCUT    cortes-004.dat`` lines.
    Resolves ``caminho`` (which may be a relative path, including
    parent-directory references, e.g. ``../../cortesh.dat``) against
    *deck_dir*.

    Never raises: returns ``None`` — falling through to the glob idiom —
    when *dadger* is unreadable, carries no ``FC`` record for *tipo*, or the
    named path does not resolve to an existing file. Plain string slicing
    cannot itself raise, so no malformed-content exception handling is
    needed beyond the ``OSError`` guard on the read itself.
    """
    try:
        text = dadger.read_text(encoding="latin-1")
    except OSError:
        return None
    for line in text.splitlines():
        if line[:4] != "FC  " or line[4:10].strip().upper() != tipo:
            continue
        caminho = line[14:214].strip()
        if not caminho:
            continue
        # `.resolve()` collapses any `..` in `caminho` (e.g. `../../cortesh.dat`)
        # so the returned path is a plain, normalized filesystem path rather
        # than one carrying the FC record's own relative-path spelling.
        candidate = (deck_dir / caminho).resolve()
        if candidate.is_file():
            return candidate
    return None


#: ``FA``/``FJ`` registers (``idecomp.decomp.modelos.dadger.FA``/``FJ``):
#: identifier at columns 0:4, file name at columns 4:54 and 4:16.
_FA_IDENTIFIER, _FA_NAME = "FA  ", slice(4, 54)
_FJ_IDENTIFIER, _FJ_NAME = "FJ  ", slice(4, 16)


def _libs_registers(dadger: Path) -> tuple[list[str], list[str]]:
    """The file names *dadger*'s ``FA`` and ``FJ`` lines give, in deck order.

    Empty when *dadger* cannot be read; the full parse right after discovery
    reports that.
    """
    try:
        text = dadger.read_text(encoding="latin-1")
    except OSError:
        return [], []
    fa: list[str] = []
    fj: list[str] = []
    for line in text.splitlines():
        if line[:4] == _FA_IDENTIFIER and (name := line[_FA_NAME].strip()):
            fa.append(name)
        elif line[:4] == _FJ_IDENTIFIER and (name := line[_FJ_NAME].strip()):
            fj.append(name)
    return fa, fj


def _register_file(names: list[str], register: str, src: Path) -> Path | None:
    """The deck file *register* names, or ``None`` when *dadger* has none.

    Raises:
        SourceFileError: the register names two different files, or a file not
            in the deck folder.
    """
    distinct = sorted({name.lower() for name in names})
    if not distinct:
        return None
    if len(distinct) > 1:
        raise SourceFileError(
            f"dadger has {register} registers naming different files: "
            f"{', '.join(distinct)}",
            path=str(src),
            field=register,
        )
    path = find_case_insensitive(src, names[0])
    if path is None:
        raise SourceFileError(
            f"dadger's {register} register names {names[0]}, which is not in {src}",
            path=str(src / names[0]),
            field=register,
        )
    return path


def _polinjus(fj: Path | None, indexed: Path | None) -> Path | None:
    """The polinjus file ``FJ`` and the LIBS index declare.

    Raises:
        SourceFileError: they name different files.
    """
    if fj is not None and indexed is not None and fj != indexed:
        raise SourceFileError(
            f"dadger's FJ register names {fj.name}, but the LIBS index lists "
            f"{HIDRELETRICA_CURVAJUSANTE} in {indexed.name}",
            path=str(fj),
            field="FJ",
        )
    return fj if fj is not None else indexed


def discover_decomp_files(src: Path) -> DecompFiles:
    """Resolve the deck files via ``caso.dat`` → the revision index file."""
    caso = src / "caso.dat"
    if not caso.is_file():
        raise SourceFileError(
            f"{caso} not found; not a deck directory",
            path=str(src),
            field="caso.dat",
        )
    revision = caso.read_text(encoding="latin-1").split()[0]

    names: list[str] = []
    index = src / revision
    if index.is_file():
        names = [
            stripped
            for line in index.read_text(encoding="latin-1").splitlines()
            if (stripped := line.strip())
            and not stripped.startswith(REGISTER_COMMENT_PREFIX)
        ]

    def find(prefix: str, required: bool, *, exclude: str | None = None) -> Path | None:
        def is_candidate(name: str) -> bool:
            lname = name.lower()
            return lname.startswith(prefix) and (
                exclude is None or not lname.startswith(exclude)
            )

        for name in names:
            if is_candidate(name):
                path = src / name
                if path.is_file():
                    return path
        matches = sorted(p for p in src.glob(f"{prefix}*") if is_candidate(p.name))
        if matches:
            return matches[0]
        if required:
            raise SourceFileError(
                f"no {prefix}* file found in {src}",
                path=str(src),
                field=prefix,
            )
        return None

    dadger = find("dadger", required=True)
    vazoes = find("vazoes", required=True)
    hidr = find("hidr", required=True)
    dadgnl = find("dadgnl", required=False)
    assert dadger is not None and vazoes is not None and hidr is not None
    fa_names, fj_names = _libs_registers(dadger)
    libs_index = _register_file(fa_names, "FA", src)
    libs = read_libs_index(src, libs_index)
    # Prefer the deck's own FC record over the glob: its caminho may point
    # outside `src` (e.g. `../../cortesh.dat`), which the glob can never find.
    # The `cortes-` prefix is tried before the broader `cortes` so a
    # single-stage partition export wins over the consolidated `cortes.dat`
    # archive when both are present (matching `fcf/cortes.py`'s trailer-based
    # shape detection); `exclude="cortesh"` keeps the `cortes` glob from
    # mistaking the header file for the record file.
    cortesh = _resolve_fc_record_path(dadger, src, tipo=_FC_TIPO_CORTESH) or find(
        "cortesh", required=False
    )
    cortes = (
        _resolve_fc_record_path(dadger, src, tipo=_FC_TIPO_CORTES)
        or find("cortes-", required=False)
        or find("cortes", required=False, exclude="cortesh")
    )
    return DecompFiles(
        revision=revision,
        dadger=dadger,
        vazoes=vazoes,
        hidr=hidr,
        dadgnl=dadgnl,
        renovaveis=libs.path_for_family(PARQUE_EOLICO_EQUIVALENTE_PREFIX),
        polinjus=_polinjus(
            _register_file(fj_names, "FJ", src), libs.path(HIDRELETRICA_CURVAJUSANTE)
        ),
        libs_restricao_eletrica=libs.path(RESTRICAO_ELETRICA_ESPECIAL),
        cortesh=cortesh,
        cortes=cortes,
        libs_index=libs_index,
    )


def unindexed_libs_diagnostic(files: DecompFiles) -> Diagnostic | None:
    """WARNING naming the deck files that look like LIBS inputs but that the
    deck does not declare, or ``None`` when there are none."""
    conventional = [
        ConventionalFile("polinjus*", HIDRELETRICA_CURVAJUSANTE, files.polinjus),
        ConventionalFile(
            "renovaveis*", f"{PARQUE_EOLICO_EQUIVALENTE_PREFIX}*", files.renovaveis
        ),
        ConventionalFile(
            "lib_restricao-eletrica-especial*.csv",
            RESTRICAO_ELETRICA_ESPECIAL,
            files.libs_restricao_eletrica,
        ),
        ConventionalFile(INDEX_FILENAME, "FA register (dadger)", files.libs_index),
    ]
    return unindexed_files_diagnostic(
        ignored_files(files.dadger.parent, conventional),
        remediation=(
            "To have a file read, list its record in the LIBS index, and name the "
            "index itself in a dadger FA register; otherwise this is informational."
        ),
    )

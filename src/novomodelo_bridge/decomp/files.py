"""Deck file discovery via caso.dat -> the revision index file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cobre_bridge.core.errors import SourceFileError
from cobre_bridge.decomp import constraint_registers


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
    #: The deck's LIBs-era electrical-constraint file, resolved
    #: via :func:`constraint_registers.resolve_libs_electrical_path`; ``None``
    #: when the deck carries none. Defaults to ``None`` so every pre-existing
    #: ``DecompFiles(...)`` call site keeps constructing without it.
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


#: ``FC`` register ``tipo`` mnemonics (``idecomp.decomp.Dadger.fc()``) naming
#: the boundary-FCF header and cut-record files respectively.
_FC_TIPO_CORTESH = "NEWV21"
_FC_TIPO_CORTES = "NEWCUT"


def _resolve_fc_record_path(dadger: Path, deck_dir: Path, *, tipo: str) -> Path | None:
    """Resolve one boundary-FCF file named by the deck's ``FC`` register.

    A lightweight fixed-width text scan of *dadger* — mirroring
    :func:`~cobre_bridge.decomp.constraint_registers.resolve_libs_electrical_path`'s
    own text-scan idiom for a deck-relative file named by an index entry —
    rather than a full :class:`idecomp.decomp.Dadger` parse (the caller
    re-parses *dadger* structurally right after discovery returns;
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
            if (stripped := line.strip()) and not stripped.startswith("&")
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
    renovaveis = find("renovaveis", required=False)
    polinjus = find("polinjus", required=False)
    assert dadger is not None and vazoes is not None and hidr is not None
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
        renovaveis=renovaveis,
        polinjus=polinjus,
        libs_restricao_eletrica=constraint_registers.resolve_libs_electrical_path(src),
        cortesh=cortesh,
        cortes=cortes,
    )

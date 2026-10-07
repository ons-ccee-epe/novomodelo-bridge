"""Dynamic the source model file discovery via caso.dat -> Arquivos.

Provides ``NewaveFiles``, a frozen dataclass whose ``from_directory`` constructor
resolves all the source model input file paths by reading the two fixed entry points —
``caso.dat`` (case-insensitive) and the ``Arquivos`` file it references.  Binary files
that are not listed in Arquivos (``hidr.dat`` and ``vazoes.dat``) are discovered via a
case-insensitive directory scan. Optional files (``modif.dat``, ``ghmin.dat``,
``penalid.dat``, ``vazpast.dat``, ``dsvagua.dat``, ``expt.dat``, ``manutt.dat``,
``c_adic.dat``, ``cvar.dat``, ``agrint.dat``, ``volref_saz.dat``) are returned as ``Path
| None``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from cobre_bridge.core.errors import SourceFileError

_LOG = logging.getLogger(__name__)


def _find_file_case_insensitive(directory: Path, filename: str) -> Path | None:
    """Return the path for *filename* in *directory*, ignoring case.

    Scans *directory* for any file whose name matches *filename* case-
    insensitively.  Returns ``None`` if no match is found.  If multiple
    entries match (unusual on case-sensitive file systems with differently-
    cased variants), the first match in iteration order is returned.
    """
    lower_target = filename.lower()
    try:
        for entry in directory.iterdir():
            if entry.is_file() and entry.name.lower() == lower_target:
                return entry
    except OSError:
        pass
    return None


def _find_first(directory: Path, filenames: tuple[str, ...]) -> Path | None:
    """Return the first of *filenames* that exists in *directory* (case-insensitive).

    Used for fixed-name "libs" cadastro files (``polinjus``, ``tratamento-fpha``) that
    source-model does not list in ``arquivos.dat``; we probe the conventional names
    directly. Returns ``None`` when none are present.
    """
    for filename in filenames:
        path = _find_file_case_insensitive(directory, filename)
        if path is not None:
            return path
    return None


def _resolve_required(directory: Path, filename: str) -> Path:
    """Return the case-insensitive path for a required file, or raise.

    Parameters
    ----------
    directory:
        The source model case directory to search in.
    filename:
        The expected filename (case-insensitive match).

    Raises
    ------
    FileNotFoundError
        If no file matching *filename* exists in *directory*.
    """
    path = _find_file_case_insensitive(directory, filename)
    if path is None:
        raise SourceFileError(
            f"Required NEWAVE file not found in {directory}: {filename}",
            path=str(directory),
            field=filename,
        )
    return path


@dataclass(frozen=True)
class NewaveFiles:
    """Resolved file paths for a source-model case, discovered via caso.dat -> Arquivos.

    All required file paths are validated at construction time — a
    ``FileNotFoundError`` is raised immediately if any required file is
    absent.  Optional files are ``None`` when absent.

    Construct via ``NewaveFiles.from_directory(directory)`` in production
    code.  In tests, instantiate directly by passing explicit paths to avoid
    any file-system dependency.
    """

    directory: Path

    # Required files
    dger: Path
    confhd: Path
    conft: Path
    sistema: Path
    clast: Path
    term: Path
    ree: Path
    patamar: Path
    hidr: Path  # binary — not listed in Arquivos; case-insensitive lookup
    vazoes: Path  # binary — not listed in Arquivos; case-insensitive lookup

    # Optional files (None when absent)
    modif: Path | None
    ghmin: Path | None
    penalid: Path | None
    vazpast: Path | None
    dsvagua: Path | None
    curva: Path | None
    expt: Path | None
    exph: Path | None
    manutt: Path | None
    c_adic: Path | None
    cvar: Path | None
    agrint: Path | None
    re_dat: Path | None
    volref_saz: Path | None
    shist: Path | None
    adterm: Path | None
    # FPHA libs files — fixed names, not listed in Arquivos (None when absent)
    polinjus: Path | None
    tratamento_fpha: Path | None

    @classmethod
    def from_directory(cls, directory: Path) -> NewaveFiles:
        """Discover all the source model files from caso.dat -> Arquivos.

        Parameters
        ----------
        directory:
            Path to the source model case directory.  Must exist and contain
            ``caso.dat`` (case-insensitive).

        Returns
        -------
        NewaveFiles
            Fully resolved file paths.

        Raises
        ------
        FileNotFoundError
            If ``caso.dat`` is missing, the Arquivos file it references is
            missing, or any required file is absent.
        """
        from inewave.newave import Arquivos, Caso

        # --- Step 1: read caso.dat (case-insensitive) --------------------------
        caso_path = _find_file_case_insensitive(directory, "caso.dat")
        if caso_path is None:
            raise SourceFileError(
                f"caso.dat not found in {directory}",
                path=str(directory),
                field="caso.dat",
            )

        caso = Caso.read(str(caso_path))
        arq_filename: str = caso.arquivos
        _LOG.debug("caso.dat -> Arquivos file: %s", arq_filename)

        # --- Step 2: read the Arquivos file ------------------------------------
        arq_path = _find_file_case_insensitive(directory, arq_filename)
        if arq_path is None:
            raise SourceFileError(
                f"Arquivos file '{arq_filename}' referenced by caso.dat "
                f"not found in {directory}",
                path=str(directory),
                field=arq_filename,
            )

        arq = Arquivos.read(str(arq_path))

        # --- Step 3: resolve required files from Arquivos ----------------------
        def _req(attr: str) -> Path:
            """Resolve a required Arquivos attribute to an existing Path."""
            fname: str = getattr(arq, attr)
            return _resolve_required(directory, fname)

        dger = _req("dger")
        confhd = _req("confhd")
        conft = _req("conft")
        sistema = _req("sistema")
        clast = _req("clast")
        term = _req("term")
        ree = _req("ree")
        patamar = _req("patamar")

        # --- Step 4: binary files (not in Arquivos) ----------------------------
        hidr = _resolve_required(directory, "hidr.dat")
        vazoes = _resolve_required(directory, "vazoes.dat")

        # --- Step 4b: FPHA libs files (fixed names, not in Arquivos, optional) --
        polinjus = _find_first(directory, ("polinjus.csv", "polinjus.dat"))
        tratamento_fpha = _find_first(
            directory,
            ("tratamento-fpha.csv", "tratamento_fpha.csv", "tratamento-fpha.dat"),
        )

        # --- Step 5: optional files from Arquivos ------------------------------
        def _opt(attr: str) -> Path | None:
            """Resolve an optional Arquivos attribute, returning None if absent."""
            try:
                fname: str | None = getattr(arq, attr, None)
            except Exception:  # noqa: BLE001
                return None
            if not fname:
                return None
            return _find_file_case_insensitive(directory, fname)

        modif = _opt("modif")
        ghmin = _opt("ghmin")
        penalid = _opt("penalid")
        vazpast = _opt("vazpast")
        dsvagua = _opt("dsvagua")
        curva = _opt("curva")
        expt = _opt("expt")
        exph = _opt("exph")
        manutt = _opt("manutt")
        c_adic = _opt("c_adic")
        cvar = _opt("cvar")
        agrint = _opt("agrint")
        re_dat = _opt("re")
        volref_saz = _opt("volume_referencia_sazonal")
        shist = _opt("shist")
        adterm = _opt("adterm")

        return cls(
            directory=directory,
            dger=dger,
            confhd=confhd,
            conft=conft,
            sistema=sistema,
            clast=clast,
            term=term,
            ree=ree,
            patamar=patamar,
            hidr=hidr,
            vazoes=vazoes,
            modif=modif,
            ghmin=ghmin,
            penalid=penalid,
            vazpast=vazpast,
            dsvagua=dsvagua,
            curva=curva,
            expt=expt,
            exph=exph,
            manutt=manutt,
            c_adic=c_adic,
            cvar=cvar,
            agrint=agrint,
            re_dat=re_dat,
            volref_saz=volref_saz,
            shist=shist,
            adterm=adterm,
            polinjus=polinjus,
            tratamento_fpha=tratamento_fpha,
        )

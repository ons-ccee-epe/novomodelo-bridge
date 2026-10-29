"""Dynamic the source model file discovery via caso.dat -> Arquivos.

Provides ``NewaveFiles``, a frozen dataclass whose ``from_directory`` constructor
resolves all the source model input file paths by reading the two fixed entry points —
``caso.dat`` (case-insensitive) and the ``Arquivos`` file it references.  Binary files
that are not listed in Arquivos (``hidr.dat`` and ``vazoes.dat``) are discovered via a
case-insensitive directory scan. LIBS files come only from the deck's ``indices.csv``
(:mod:`novomodelo_bridge.core.libs_index`). Optional files are returned as ``Path | None``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path

from novomodelo_bridge.core.diagnostics import Diagnostic
from novomodelo_bridge.core.errors import SourceFileError
from novomodelo_bridge.core.libs_index import (
    ESTACAO_BOMBEAMENTO,
    ESTACAO_BOMBEAMENTO_SUBMERCADO,
    HIDRELETRICA_CADASTRO_RESERVATORIO,
    HIDRELETRICA_CURVAJUSANTE,
    HIDRELETRICA_FPHA_REDUCAO_CORTES,
    INDEX_FILENAME,
    RESTRICAO_ELETRICA_ESPECIAL,
    ConventionalFile,
    ignored_files,
    read_libs_index,
    unindexed_files_diagnostic,
)
from novomodelo_bridge.core.paths import find_case_insensitive

_LOG = logging.getLogger(__name__)

#: ``hidr``/``vazoes`` are binary and the LIBS files are delimited; every other
#: discovered input is a fixed-column text file.
_NOT_FIXED_COLUMN = frozenset(
    {
        "directory",
        "hidr",
        "vazoes",
        "libs_index",
        "polinjus",
        "volumes_referencia",
        "tratamento_fpha",
        "restricao_eletrica",
        "estacao_bombeamento",
        "estacao_bombeamento_submercado",
    }
)


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
    path = find_case_insensitive(directory, filename)
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
    # LIBS files, declared by ``indices.csv`` (None when the index lists none)
    libs_index: Path | None
    polinjus: Path | None
    volumes_referencia: Path | None
    tratamento_fpha: Path | None
    restricao_eletrica: Path | None
    estacao_bombeamento: Path | None
    estacao_bombeamento_submercado: Path | None

    @property
    def fixed_column_files(self) -> list[Path]:
        """The resolved inputs whose reader slices fixed columns."""
        return [
            value
            for f in fields(self)
            if f.name not in _NOT_FIXED_COLUMN
            and isinstance(value := getattr(self, f.name), Path)
        ]

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
        caso_path = find_case_insensitive(directory, "caso.dat")
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
        arq_path = find_case_insensitive(directory, arq_filename)
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

        # --- Step 4b: LIBS files, only those indices.csv declares ---------------
        libs_index = find_case_insensitive(directory, INDEX_FILENAME)
        libs = read_libs_index(directory, libs_index)
        polinjus = libs.path(HIDRELETRICA_CURVAJUSANTE)
        volumes_referencia = libs.path(HIDRELETRICA_CADASTRO_RESERVATORIO)
        tratamento_fpha = libs.path(HIDRELETRICA_FPHA_REDUCAO_CORTES)
        restricao_eletrica = libs.path(RESTRICAO_ELETRICA_ESPECIAL)
        estacao_bombeamento = libs.path(ESTACAO_BOMBEAMENTO)
        estacao_bombeamento_submercado = libs.path(ESTACAO_BOMBEAMENTO_SUBMERCADO)

        # --- Step 5: optional files from Arquivos ------------------------------
        def _opt(attr: str) -> Path | None:
            """Resolve an optional Arquivos attribute, returning None if absent."""
            try:
                fname: str | None = getattr(arq, attr, None)
            except Exception:
                return None
            if not fname:
                return None
            return find_case_insensitive(directory, fname)

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
            libs_index=libs_index,
            polinjus=polinjus,
            volumes_referencia=volumes_referencia,
            tratamento_fpha=tratamento_fpha,
            restricao_eletrica=restricao_eletrica,
            estacao_bombeamento=estacao_bombeamento,
            estacao_bombeamento_submercado=estacao_bombeamento_submercado,
        )


def unindexed_libs_diagnostic(files: NewaveFiles) -> Diagnostic | None:
    """WARNING naming the deck files that look like LIBS inputs but that
    ``indices.csv`` does not declare, or ``None`` when there are none."""
    conventional = [
        ConventionalFile("polinjus.csv", HIDRELETRICA_CURVAJUSANTE, files.polinjus),
        ConventionalFile("polinjus.dat", HIDRELETRICA_CURVAJUSANTE, files.polinjus),
        ConventionalFile(
            "volumes-referencia.csv",
            HIDRELETRICA_CADASTRO_RESERVATORIO,
            files.volumes_referencia,
        ),
        ConventionalFile(
            "tratamento[-_]fpha.*",
            HIDRELETRICA_FPHA_REDUCAO_CORTES,
            files.tratamento_fpha,
        ),
        ConventionalFile(
            "restricao-eletrica.csv",
            RESTRICAO_ELETRICA_ESPECIAL,
            files.restricao_eletrica,
        ),
        ConventionalFile(
            "estacao-bombeamento.csv",
            ESTACAO_BOMBEAMENTO,
            files.estacao_bombeamento,
        ),
        ConventionalFile(
            "estacao-bombeamento-submercado.csv",
            ESTACAO_BOMBEAMENTO_SUBMERCADO,
            files.estacao_bombeamento_submercado,
        ),
    ]
    return unindexed_files_diagnostic(
        ignored_files(files.directory, conventional),
        remediation=(
            "To have a file read, list its record in indices.csv; otherwise this "
            "is informational."
        ),
    )

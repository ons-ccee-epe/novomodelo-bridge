"""The source deck a case was converted from: its revision and hydro plants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from novomodelo_bridge.core.errors import FieldParseError
from novomodelo_bridge.dessem.exporters.fcf.core.mapper import SourceHydro


@dataclass(frozen=True)
class SourceDeck:
    """A deck revision and its operated hydro plants, keyed by novomodelo hydro id."""

    directory: Path
    revision: str
    hydros: Mapping[int, SourceHydro]


def read_source_deck(directory: Path) -> SourceDeck:
    """Read the revision named by *directory*'s `caso.dat` and its operated plants.

    Hydro ids come from `DecompIdMap.hydro_id` and names from `hidr.dat`, the
    same sources `convert decomp` writes `hydros.json` from.

    Raises:
        SourceFileError: *directory* is not a deck directory.
        FieldParseError: an operated plant is missing from `hidr.dat`.
    """
    from novomodelo_bridge.decomp.case import DecompCase

    case = DecompCase.from_directory(directory)
    hidr = case.hidr
    hydros: dict[int, SourceHydro] = {}
    for code in case.id_map.hydro_codes:
        if code not in hidr.index:
            raise FieldParseError(
                f"operated plant {code} is not in the hydro registry",
                field="UH register",
            )
        name = str(hidr.loc[code, "nome_usina"]).strip()
        hydros[case.id_map.hydro_id(code)] = SourceHydro(code=code, name=name)
    return SourceDeck(directory=directory, revision=case.files.revision, hydros=hydros)

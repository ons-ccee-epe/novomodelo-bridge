"""Whether a converted case and a source deck hold the same hydro plants.

`convert decomp` writes each hydro's id and name from the deck, so a deck the case
was not converted from shows as a different plant set or a renamed plant. Every
command that reads a deck and a case together checks this one rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from novomodelo_bridge.core.errors import CaseIdentityError, FieldParseError

if TYPE_CHECKING:
    from novomodelo_bridge.decomp.case import DecompCase


@dataclass(frozen=True)
class SourceHydro:
    """A hydro plant as the source deck names it: `hidr.dat` code and name."""

    code: int
    name: str


def require_registered_hydros(case: DecompCase) -> None:
    """Refuse a deck whose `UH` records give a converted plant `hidr.dat` lacks.

    Raises:
        FieldParseError: `hidr.dat` misses a converted plant.
    """
    hidr = case.hidr
    for code in case.id_map.hydro_codes:
        if code not in hidr.index:
            raise FieldParseError(
                f"converted plant {code} is not in the hydro registry",
                field="UH register",
            )


def deck_hydros(case: DecompCase) -> dict[int, SourceHydro]:
    """The deck's converted hydros keyed by novomodelo hydro id, in ascending code order.

    Ids come from `DecompIdMap.hydro_id` and names from `hidr.dat`, the same
    sources `convert decomp` writes `hydros.json` from.

    Raises:
        FieldParseError: `hidr.dat` misses a converted plant.
    """
    require_registered_hydros(case)
    hidr = case.hidr
    hydros: dict[int, SourceHydro] = {}
    for code in case.id_map.hydro_codes:
        name = str(hidr.loc[code, "nome_usina"]).strip()
        hydros[case.id_map.hydro_id(code)] = SourceHydro(code=code, name=name)
    return hydros


def match_case_hydros(
    case_names: Mapping[int, str], deck: Mapping[int, SourceHydro]
) -> None:
    """Refuse a deck other than the one the case was converted from.

    *case_names* maps each of the case's hydro ids to its name; *deck* is keyed by
    novomodelo hydro id. The deck must convert exactly the case's hydros, each
    under the name the case gives it.

    Raises:
        CaseIdentityError: the plant sets or any plant name differ.
    """
    if case_names.keys() != deck.keys():
        raise CaseIdentityError(
            f"the deck converts {len(deck)} hydro plants but the case holds "
            f"{len(case_names)}"
        )
    renamed = [
        (hydro_id, case_names[hydro_id], deck[hydro_id])
        for hydro_id in sorted(case_names)
        if case_names[hydro_id] != deck[hydro_id].name
    ]
    if renamed:
        hydro_id, case_name, plant = renamed[0]
        raise CaseIdentityError(
            f"{len(renamed)} hydro plant names differ between the case and the "
            f"deck, first hydro {hydro_id}: {case_name!r} in the case, "
            f"{plant.name!r} (plant {plant.code}) in the deck"
        )

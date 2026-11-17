"""The source deck a case was converted from: its revision, hydro and GNL plants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from novomodelo_bridge.decomp.case_identity import SourceHydro, deck_hydros
from novomodelo_bridge.dessem.exporters.fcf.core.mapper import SourceGnlThermal


@dataclass(frozen=True)
class SourceDeck:
    """A deck revision, its converted hydros keyed by novomodelo hydro id, and its GNL plants.

    *gnl_thermals* ascend by deck code and are empty when the deck commits no
    GNL dispatch. *travel_time_codes* are the plant codes of the deck's `VI`
    records, in record order, the order DECOMP numbers travel-time plants in.
    *dormant_codes* ascend and are the codes of the `UH` records marked `NW`.
    """

    directory: Path
    revision: str
    hydros: Mapping[int, SourceHydro]
    gnl_thermals: tuple[SourceGnlThermal, ...]
    travel_time_codes: tuple[int, ...] = ()
    dormant_codes: tuple[int, ...] = ()


def read_source_deck(directory: Path) -> SourceDeck:
    """Read the revision named by *directory*'s `caso.dat`, its hydro and GNL plants.

    Hydro ids come from `DecompIdMap.hydro_id` and names from `hidr.dat`, the
    same sources `convert decomp` writes `hydros.json` from.

    Raises:
        SourceFileError: *directory* is not a deck directory.
        FieldParseError: `hidr.dat` holds no plant or misses a converted one,
            or `dadgnl` has a malformed record.
    """
    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.converters.anticipated import read_gnl_model
    from novomodelo_bridge.decomp.converters.travel_time import read_travel_times

    case = DecompCase.from_directory(directory)
    hydros = deck_hydros(case)
    dadgnl = case.dadgnl
    gnl_model = None if dadgnl is None else read_gnl_model(dadgnl)
    gnl_thermals = (
        ()
        if gnl_model is None
        else tuple(
            SourceGnlThermal(
                code=thermal.code,
                name=thermal.name,
                lag_months=gnl_model.nl_lag_months.get(thermal.code),
            )
            for thermal in gnl_model.thermals
        )
    )
    return SourceDeck(
        directory=directory,
        revision=case.files.revision,
        hydros=hydros,
        gnl_thermals=gnl_thermals,
        travel_time_codes=tuple(read_travel_times(case.dadger)),
        dormant_codes=case.id_map.dormant_hydro_codes,
    )

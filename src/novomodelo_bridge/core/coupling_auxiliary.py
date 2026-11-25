"""Ids and names of the auxiliary coupling plants, the hydros a converted case
adds so the boundary cuts read each plant's inflow on its coupling posto.

Every reader of a converted case recognizes one by :func:`is_coupling_auxiliary`,
never by its name.
"""

from __future__ import annotations

from novomodelo_bridge.core.errors import UnsupportedSourceError

#: An auxiliary's hydro id is this offset plus its plant's deck code, so no
#: plant with a code at or above it can have one.
COUPLING_AUXILIARY_ID_OFFSET = 1000


def coupling_auxiliary_id(plant_code: int) -> int:
    """The hydro id of *plant_code*'s auxiliary coupling plant.

    Raises
    ------
    UnsupportedSourceError
        The code is not below :data:`COUPLING_AUXILIARY_ID_OFFSET`.
    """
    if not 0 < plant_code < COUPLING_AUXILIARY_ID_OFFSET:
        raise UnsupportedSourceError(
            f"plant {plant_code} reads the boundary cuts on a coupling posto "
            "that differs from its own inflow; a plant code outside "
            f"1..{COUPLING_AUXILIARY_ID_OFFSET - 1} with such a posto is not "
            "converted yet, so convert the deck with --no-fcf"
        )
    return COUPLING_AUXILIARY_ID_OFFSET + plant_code


def coupling_auxiliary_name(plant_code: int, plant_name: str) -> str:
    """The auxiliary's name: ``ACOPL-<code> <plant name>``."""
    return f"ACOPL-{plant_code} {plant_name.strip()}".rstrip()


def is_coupling_auxiliary(hydro_id: int) -> bool:
    """Whether *hydro_id* is an auxiliary coupling plant."""
    return hydro_id >= COUPLING_AUXILIARY_ID_OFFSET

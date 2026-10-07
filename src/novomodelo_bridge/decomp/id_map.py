"""Deterministic entity-id mapping for DECOMP-like decks.

Buses: the deck's declared subsystems (``SB`` records — real subsystems
plus the fictitious accounting one) sorted by code and mapped to dense
0-based ids, followed by one converter-created transhipment bus (the
implicit exchange node the deck references only by name in ``IA``
records). Other entity families join this map as their converters land.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cobre_bridge.core.errors import FieldParseError

if TYPE_CHECKING:
    from idecomp.decomp import Dadger

TRANSHIPMENT_BUS_NAME = "IV"


@dataclass(frozen=True)
class DecompIdMap:
    """Deterministic mapping from deck codes to dense 0-based Cobre ids.

    Hydro codes are the ``UH``-activated *operated* plants (rows carrying an
    initial volume); thermal codes are the ``CT``-declared plants. Both are
    sorted ascending and mapped to dense 0-based ids.
    """

    bus_codes: tuple[int, ...]
    bus_names: tuple[str, ...]
    hydro_codes: tuple[int, ...] = ()
    thermal_codes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.bus_codes) != len(self.bus_names):
            raise ValueError(
                f"{len(self.bus_codes)} bus codes for {len(self.bus_names)} names"
            )
        for label, codes in (
            ("bus", self.bus_codes),
            ("hydro", self.hydro_codes),
            ("thermal", self.thermal_codes),
        ):
            if list(codes) != sorted(set(codes)):
                raise ValueError(
                    f"{label} codes must be strictly ascending; got {codes}"
                )
        if TRANSHIPMENT_BUS_NAME in {n.strip() for n in self.bus_names}:
            raise ValueError(
                f"'{TRANSHIPMENT_BUS_NAME}' is reserved for the converter-created "
                "transhipment bus but appears among the declared subsystems"
            )

    @classmethod
    def from_dadger(cls, dadger: Dadger) -> DecompIdMap:
        """Build the map from a deck's ``SB`` + ``UH`` + ``CT`` records."""
        sb = dadger.sb(df=True)
        if sb is None or sb.empty:
            raise FieldParseError(
                "the deck has no SB records; cannot build the id map",
                field="SB register",
            )
        rows = sorted(
            (int(r["codigo_submercado"]), str(r["nome_submercado"]).strip())
            for _, r in sb.iterrows()
        )

        uh = dadger.uh(df=True)
        hydro_codes: tuple[int, ...] = ()
        if uh is not None and not uh.empty:
            operated = uh[uh["volume_inicial"].notna()]
            hydro_codes = tuple(sorted(int(c) for c in operated["codigo_usina"]))

        ct = dadger.ct(df=True)
        thermal_codes: tuple[int, ...] = ()
        if ct is not None and not ct.empty:
            thermal_codes = tuple(sorted({int(c) for c in ct["codigo_usina"]}))

        return cls(
            bus_codes=tuple(code for code, _ in rows),
            bus_names=tuple(name for _, name in rows),
            hydro_codes=hydro_codes,
            thermal_codes=thermal_codes,
        )

    def hydro_id(self, code: int) -> int:
        """Map an operated hydro code to its 0-based id."""
        try:
            return self.hydro_codes.index(code)
        except ValueError:
            raise KeyError(f"unknown hydro code {code}") from None

    def thermal_id(self, code: int) -> int:
        """Map a thermal code to its 0-based id."""
        try:
            return self.thermal_codes.index(code)
        except ValueError:
            raise KeyError(f"unknown thermal code {code}") from None

    @property
    def n_buses(self) -> int:
        """Total bus count, including the transhipment bus."""
        return len(self.bus_codes) + 1

    @property
    def transhipment_bus_id(self) -> int:
        """Id of the converter-created transhipment bus (always last)."""
        return len(self.bus_codes)

    def bus_id(self, code: int) -> int:
        """Map a declared subsystem code to its 0-based bus id."""
        try:
            return self.bus_codes.index(code)
        except ValueError:
            raise KeyError(f"unknown subsystem code {code}") from None

    def bus_id_by_name(self, name: str) -> int:
        """Map a subsystem name (or the transhipment name) to its bus id."""
        stripped = name.strip()
        if stripped == TRANSHIPMENT_BUS_NAME:
            return self.transhipment_bus_id
        try:
            return self.bus_names.index(stripped)
        except ValueError:
            raise KeyError(f"unknown subsystem name {name!r}") from None

    def bus_name(self, bus_id: int) -> str:
        """Return the display name of a bus id (declared or transhipment)."""
        if bus_id == self.transhipment_bus_id:
            return TRANSHIPMENT_BUS_NAME
        if 0 <= bus_id < len(self.bus_names):
            return self.bus_names[bus_id]
        raise KeyError(f"unknown bus id {bus_id}")

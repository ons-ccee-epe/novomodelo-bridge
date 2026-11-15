"""Deterministic entity-id mapping for DECOMP-like decks.

Buses: the deck's declared subsystems (``SB`` records — real subsystems
plus the fictitious accounting one) sorted by code and mapped to dense
0-based ids, followed by one converter-created transhipment bus (the
implicit exchange node the deck references only by name in ``IA``
records). Other entity families join this map as their converters land.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from novomodelo_bridge.core.errors import FieldParseError, UnsupportedSourceError
from novomodelo_bridge.core.pandas_utils import as_frame
from novomodelo_bridge.decomp import plants

TRANSHIPMENT_BUS_NAME = "IV"


class _IdMapRegisters(Protocol):
    def sb(self, *, df: bool) -> object: ...

    def uh(self, *, df: bool) -> object: ...

    def ct(self, *, df: bool) -> object: ...


@dataclass(frozen=True)
class DecompIdMap:
    """Deterministic mapping from deck codes to dense 0-based Novomodelo ids.

    Hydro codes are the *converted* ``UH`` plants (see
    :mod:`novomodelo_bridge.decomp.plants`): the operated rows carrying an
    initial volume, and the dormant rows marked ``NW``, which DECOMP does not
    operate but whose cut terms the case prices. Thermal codes are the
    ``CT``-declared plants. Both are sorted ascending and mapped to dense
    0-based ids, so a dormant code takes an id among the operated ones.
    ``dormant_hydro_codes`` are the dormant plants, a subset of the hydro
    codes. Coupling-only codes are the ``UH`` rows without an initial volume
    that are not dormant, which DECOMP keeps only for the coupling with
    NEWAVE; they get no id.
    """

    bus_codes: tuple[int, ...]
    bus_names: tuple[str, ...]
    hydro_codes: tuple[int, ...] = ()
    thermal_codes: tuple[int, ...] = ()
    coupling_only_codes: tuple[int, ...] = ()
    dormant_hydro_codes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.bus_codes) != len(self.bus_names):
            raise ValueError(
                f"{len(self.bus_codes)} bus codes for {len(self.bus_names)} names"
            )
        for label, codes in (
            ("bus", self.bus_codes),
            ("hydro", self.hydro_codes),
            ("thermal", self.thermal_codes),
            ("coupling-only", self.coupling_only_codes),
            ("dormant", self.dormant_hydro_codes),
        ):
            if list(codes) != sorted(set(codes)):
                raise ValueError(
                    f"{label} codes must be strictly ascending; got {codes}"
                )
        if not set(self.dormant_hydro_codes) <= set(self.hydro_codes):
            raise ValueError(
                f"dormant codes {self.dormant_hydro_codes} are not all hydro "
                f"codes {self.hydro_codes}"
            )
        if set(self.dormant_hydro_codes) & set(self.coupling_only_codes):
            raise ValueError(
                f"dormant codes {self.dormant_hydro_codes} overlap the "
                f"coupling-only codes {self.coupling_only_codes}"
            )
        if TRANSHIPMENT_BUS_NAME in {n.strip() for n in self.bus_names}:
            raise UnsupportedSourceError(
                f"dadger.rvN declares a subsystem named {TRANSHIPMENT_BUS_NAME} in "
                "SB, the name the conversion gives the transhipment node it adds. "
                "Rename that subsystem in SB and in the records that name it.",
                path="dadger.rvN",
                field="SB nome_submercado",
            )

    @classmethod
    def from_dadger(cls, dadger: _IdMapRegisters) -> DecompIdMap:
        """Build the map from a deck's ``SB`` + ``UH`` + ``CT`` records."""
        sb = as_frame(dadger.sb(df=True))
        if sb is None or sb.empty:
            raise FieldParseError(
                "the deck has no SB records; cannot build the id map",
                field="SB register",
            )
        rows = sorted(
            (int(r["codigo_submercado"]), str(r["nome_submercado"]).strip())
            for _, r in sb.iterrows()
        )

        uh = as_frame(dadger.uh(df=True))
        hydro_codes: tuple[int, ...] = ()
        coupling_only_codes: tuple[int, ...] = ()
        dormant_hydro_codes: tuple[int, ...] = ()
        if uh is not None and not uh.empty:
            hydro_codes = plants.converted_codes(uh)
            coupling_only_codes = plants.coupling_only_codes(uh)
            dormant_hydro_codes = plants.dormant_codes(uh)

        ct = as_frame(dadger.ct(df=True))
        thermal_codes: tuple[int, ...] = ()
        if ct is not None and not ct.empty:
            thermal_codes = tuple(sorted({int(c) for c in ct["codigo_usina"]}))

        return cls(
            bus_codes=tuple(code for code, _ in rows),
            bus_names=tuple(name for _, name in rows),
            hydro_codes=hydro_codes,
            thermal_codes=thermal_codes,
            coupling_only_codes=coupling_only_codes,
            dormant_hydro_codes=dormant_hydro_codes,
        )

    @property
    def operated_hydro_codes(self) -> tuple[int, ...]:
        """The hydro codes DECOMP operates: every hydro code but the dormant ones."""
        dormant = set(self.dormant_hydro_codes)
        return tuple(code for code in self.hydro_codes if code not in dormant)

    @property
    def idle_hydro_codes(self) -> tuple[int, ...]:
        """The plants DECOMP does not operate: the coupling-only and the dormant ones."""
        return tuple(sorted({*self.coupling_only_codes, *self.dormant_hydro_codes}))

    def hydro_id(self, code: int) -> int:
        """Map a converted hydro code, operated or dormant, to its 0-based id."""
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

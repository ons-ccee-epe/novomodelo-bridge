"""NewaveIdMap: deterministic 1-based-to-0-based entity ID remapping.

Accepts sorted lists of the source model entity IDs and provides 0-based lookups that
are consistent across all entity files.  Both conversion tracks import this class
to share the same mapping produced during entity conversion.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cobre_bridge.core.diagnostics import Diagnostic, DiagnosticTable, Severity, emit
from cobre_bridge.newave.plants import active_hydro_codes, fictitious_codes

if TYPE_CHECKING:
    from inewave.newave import Confhd, Conft, Exph, Hidr, Ree, Sistema

    from cobre_bridge.newave.files import NewaveFiles

_LOG = logging.getLogger(__name__)


class NewaveIdMap:
    """Bidirectional ID map from the source model 1-based codes to Cobre 0-based IDs.

    Subsystems, thermals, and hydros are all remapped deterministically by sorting
    the source model IDs (``codigo_usina``) ascending and assigning Cobre IDs
    0, 1, 2, … in that order.  Sorting hydros by code — rather than by the
    ``confhd.dat`` declaration order — makes the Cobre hydro id follow the
    ``hidr.dat`` registry order and keeps the mapping independent of how a
    particular deck happens to order ``confhd.dat``.

    Parameters
    ----------
    subsystem_ids:
        Iterable of the source model subsystem (submercado) codes to register, including
        fictitious ones.  Each unique code maps to one bus ID.
    hydro_codes:
        The source model hydro plant codes (``codigo_usina`` from ``confhd.dat``).
        Cobre hydro IDs are assigned in ascending code order (the codes are sorted
        here), so the caller need not pre-sort them.
    thermal_codes:
        Iterable of the source model thermal plant codes (``codigo_usina`` from
        ``conft.dat``).
    """

    def __init__(
        self,
        subsystem_ids: list[int],
        hydro_codes: list[int],
        thermal_codes: list[int],
    ) -> None:
        self._bus: dict[int, int] = {
            newave_id: cobre_id
            for cobre_id, newave_id in enumerate(sorted(subsystem_ids))
        }
        self._hydro: dict[int, int] = {
            newave_id: cobre_id
            for cobre_id, newave_id in enumerate(sorted(hydro_codes))
        }
        self._thermal: dict[int, int] = {
            newave_id: cobre_id
            for cobre_id, newave_id in enumerate(sorted(thermal_codes))
        }

    def bus_id(self, newave_subsystem_id: int) -> int:
        """Return the 0-based Cobre bus ID for a source-model subsystem code.

        Raises
        ------
        KeyError
            If ``newave_subsystem_id`` was not registered at construction.
        """
        return self._bus[newave_subsystem_id]

    def hydro_id(self, newave_hydro_code: int) -> int:
        """Return the 0-based Cobre hydro ID for a source-model plant code.

        Raises
        ------
        KeyError
            If ``newave_hydro_code`` was not registered at construction.
        """
        return self._hydro[newave_hydro_code]

    def thermal_id(self, newave_thermal_code: int) -> int:
        """Return the 0-based Cobre thermal ID for a source-model plant code.

        Raises
        ------
        KeyError
            If ``newave_thermal_code`` was not registered at construction.
        """
        return self._thermal[newave_thermal_code]

    @property
    def all_bus_ids(self) -> list[int]:
        """Sorted list of registered the source model subsystem codes."""
        return sorted(self._bus)

    @property
    def all_hydro_codes(self) -> list[int]:
        """The source model hydro codes in Cobre-ID order (ascending ``codigo_usina``,
        matching ``hidr.dat`` registry order)."""
        return list(self._hydro)

    @property
    def all_thermal_codes(self) -> list[int]:
        """Sorted list of registered the source model thermal codes."""
        return sorted(self._thermal)


def build_id_map(nw_files: NewaveFiles) -> NewaveIdMap:
    """Build the canonical :class:`NewaveIdMap` from a case's the source model inputs.

    Reads ``confhd.dat`` (hydros, existing non-fictitious only), ``conft.dat``
    (thermals), ``sistema.dat`` + ``ree.dat`` (subsystems). This is the single public
    entry point shared by the conversion pipeline and the comparators, so both derive
    the source model→Cobre mapping the same way.

    This path-only entry point intentionally does **not** thread ``exph``, so it
    forwards ``exph=None`` to :func:`build_id_map_from_readers`: the comparators and
    any other caller that only holds paths keep the in-service-only enumeration
    (they run against cases whose plants are already in service). The conversion
    pipeline admits ``NE``-with-filling plants through :attr:`NewaveCase.id_map`,
    which passes the case's ``exph`` reader.
    """
    from inewave.newave import Confhd, Conft, Hidr, Ree, Sistema

    confhd = Confhd.read(str(nw_files.confhd))
    conft = Conft.read(str(nw_files.conft))
    sistema = Sistema.read(str(nw_files.sistema))
    ree_file = Ree.read(str(nw_files.ree))
    hidr = Hidr.read(str(nw_files.hidr))
    return build_id_map_from_readers(confhd, conft, sistema, ree_file, hidr)


def build_id_map_from_readers(
    confhd: Confhd,
    conft: Conft,
    sistema: Sistema,
    ree_file: Ree,
    hidr: Hidr,
    *,
    exph: Exph | None = None,
) -> NewaveIdMap:
    """Build the canonical :class:`NewaveIdMap` from already-parsed readers.

    The core of :func:`build_id_map`, factored out so :attr:`NewaveCase.id_map`
    can reuse the case's cached readers instead of re-parsing the files. ``hidr``
    supplies the productivity used to identify fictitious accounting plants
    structurally (see :func:`plants.fictitious_codes`).

    When *exph* (the case's ``Exph`` reader) is supplied, its ``expansoes`` table is
    threaded into :func:`plants.active_hydro_codes`, admitting the ``NE`` plants that
    carry a dead-volume filling row (each at its confhd declaration position). With
    ``exph is None`` — the default, used by every path-only caller (``build_id_map``,
    the comparators) — enumeration is byte-identical to the in-service set: no ``NE``
    plant is admitted.
    """
    # Hydro codes from confhd — existing plants minus the fictitious accounting
    # nodes (zero productivity sharing a generating plant's posto).
    confhd_df = confhd.usinas
    cadastro = hidr.cadastro
    fict = fictitious_codes(confhd_df, cadastro)
    if fict:
        rows = sorted(
            (
                (int(r["codigo_usina"]), str(r["nome_usina"]).strip())
                for _, r in confhd_df.iterrows()
                if int(r["codigo_usina"]) in fict
            ),
            key=lambda pair: pair[0],
        )
        emit(
            Diagnostic(
                code="fictitious-plants-excluded",
                severity=Severity.INFO,
                category="Entity exclusion",
                title=f"Fictitious plants excluded ({len(rows)})",
                summary=(
                    f"Excluded {len(rows)} fictitious accounting plant(s) from the "
                    "id map (zero-productivity nodes sharing a real plant's posto)."
                ),
                table=DiagnosticTable(
                    columns=["Code", "Name"],
                    rows=[[code, name] for code, name in rows],
                    justify=["right", "left"],
                ),
            ),
            logger=_LOG,
        )
    hydro_codes = active_hydro_codes(
        confhd_df, cadastro, exph_df=exph.expansoes if exph else None
    )

    # Thermal codes from conft.
    conft_df = conft.usinas
    thermal_codes = [int(r["codigo_usina"]) for _, r in conft_df.iterrows()]

    # Subsystem codes from sistema deficit table.
    deficit_df = sistema.custo_deficit
    if deficit_df is not None:
        subsystem_ids = sorted(
            {int(r["codigo_submercado"]) for _, r in deficit_df.iterrows()}
        )
    else:
        subsystem_ids = []

    # Also include subsystem codes referenced in ree.dat (for completeness).
    ree_df = ree_file.rees
    if ree_df is not None:
        for _, row in ree_df.iterrows():
            code = int(row["submercado"])
            if code not in subsystem_ids:
                subsystem_ids.append(code)

    return NewaveIdMap(
        subsystem_ids=subsystem_ids,
        hydro_codes=hydro_codes,
        thermal_codes=thermal_codes,
    )

"""Parsed deck case — read each input file once, cache the result.

``DecompFiles`` (:mod:`cobre_bridge.decomp.files`) resolves the *paths* of
a deck revision; this module adds the parsing layer on top. Converters
historically took a raw ``dadger`` plus several derived params
(``id_map``, ``calendar``, ``hidr``) computed ad hoc by the pipeline and
hand-threaded positionally, so a conversion's own orchestrator re-derived
``id_map``/``calendar`` from the same ``dadger`` parse at each call site.

:class:`DecompCase` wraps a :class:`~cobre_bridge.decomp.files.DecompFiles`
and exposes one :func:`functools.cached_property` per input file, mirroring
:class:`~cobre_bridge.newave.case.NewaveCase` on the source-model track. The first
access parses via the matching reader; every later access is free. Required
files return the reader (or, for ``hidr``, the registry ``DataFrame``);
optional files return ``Reader | None`` (``None`` when the file is absent),
mirroring the ``Path | None`` shape on ``DecompFiles``. ``id_map`` and
``calendar`` derive from the cached ``dadger`` parse rather than re-reading
the file.

The underlying paths stay reachable via :attr:`DecompCase.files`. The large,
single-use ``vazoes`` inflow file is deliberately *not* cached here — its one
consumer reads it straight off ``case.files.vazoes``, mirroring
``NewaveCase``'s exclusion of its own inflow file, so the full inflow history
need not be held for the whole conversion.

In tests, build a case and pre-fill the cached slots so no I/O happens (see
the ``make_decomp_case`` factory in ``tests/conftest.py``); ``cached_property``
reads from the instance ``__dict__`` first, so a pre-set value shadows the
parse. For this reason ``DecompCase`` must stay a plain (non-frozen,
non-slotted) dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from idecomp.decomp import Dadger, Dadgnl
from idecomp.libs import Renovaveis
from idecomp.libs.restricoes import Restricoes

from cobre_bridge.decomp.files import discover_decomp_files

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd
    from idecomp.libs import UsinasHidreletricas

    from cobre_bridge.decomp.files import DecompFiles
    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage


@dataclass
class DecompCase:
    """A deck revision with each input file parsed once and cached.

    Construct via :meth:`from_directory` in production code. Each property
    below parses the matching file on first access through its reader and
    caches the reader (or derived) object on the instance.
    """

    files: DecompFiles

    @classmethod
    def from_directory(cls, directory: Path) -> DecompCase:
        """Discover and bind the deck files in *directory* (no parsing yet)."""
        return cls(files=discover_decomp_files(directory))

    # --- Required files --------------------------------------------------------

    @cached_property
    def dadger(self) -> Dadger:
        return Dadger.read(str(self.files.dadger))

    @cached_property
    def hidr(self) -> pd.DataFrame:
        from cobre_bridge.decomp.converters.hydro import read_hidr

        return read_hidr(self.files.hidr)

    @cached_property
    def id_map(self) -> DecompIdMap:
        """The deck's :class:`~cobre_bridge.decomp.id_map.DecompIdMap`."""
        from cobre_bridge.decomp.id_map import DecompIdMap

        return DecompIdMap.from_dadger(self.dadger)

    @cached_property
    def calendar(self) -> list[OperativeStage]:
        """The deck's operative calendar."""
        from cobre_bridge.decomp.temporal import operative_calendar_from_dadger

        return operative_calendar_from_dadger(self.dadger)

    # --- Optional files (None when absent) -------------------------------------

    @cached_property
    def dadgnl(self) -> Dadgnl | None:
        path = self.files.dadgnl
        return Dadgnl.read(str(path)) if path is not None else None

    @cached_property
    def renovaveis(self) -> Renovaveis | None:
        path = self.files.renovaveis
        return Renovaveis.read(str(path)) if path is not None else None

    @cached_property
    def polinjus(self) -> UsinasHidreletricas | None:
        from cobre_bridge.decomp.converters.fpha import read_polinjus

        path = self.files.polinjus
        return read_polinjus(path) if path is not None else None

    @cached_property
    def libs_restricao_eletrica(self) -> Restricoes | None:
        path = self.files.libs_restricao_eletrica
        return Restricoes.read(str(path)) if path is not None else None

    # --- Derived state -----------------------------------------------------------

    @property
    def start_date(self) -> date:
        """The deck's operative-calendar start date."""
        return self.calendar[0].start_date

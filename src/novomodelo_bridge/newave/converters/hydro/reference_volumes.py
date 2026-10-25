"""The reference volumes the source model reads each plant at.

The source model keeps two independent per-plant references, and this module
is the only reader of their files:

- the FPHA reference (:func:`fpha_reference_volumes`), one volume per study
  stage: the forebay level of the plant's own turbined fit range and the
  downstream level its upstream plant's tailrace families are read at;
- the seasonal operating reference (:func:`seasonal_useful_volumes`), one
  useful volume per calendar month from ``volref_saz.dat``: the evaporation
  reference (:func:`evaporation_reference_volumes`) and the head of the
  constant-productivity ρ_eq and turbine limit.

The modif.dat temporal records move neither.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from novomodelo_bridge.core.diagnostics import Diagnostic, Severity, emit
from novomodelo_bridge.core.errors import SourceFileError
from novomodelo_bridge.core.libs_index import HIDRELETRICA_CADASTRO_RESERVATORIO
from novomodelo_bridge.core.pandas_utils import is_na
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.converters.hydro.geometry import (
    held_volume,
    operating_storage_range,
)
from novomodelo_bridge.newave.converters.hydro.overrides import percent_of_useful_volume
from novomodelo_bridge.newave.horizon import stage_dates_for

_LOG = logging.getLogger(__name__)

#: ``VOLUME-REFERENCIAL-TIPO-PADRAO`` values: 0 ignores the per-period rows, 1
#: reads them. The source model reads a file without the record as 0.
_PERIODIC_OFF = 0
_PERIODIC_ON = 1


def initial_storage_hm3(hreg: pd.Series, percent: float) -> float:
    """A reservoir's initial storage in hm³: *percent* (confhd
    ``volume_inicial_percentual``, clamped into ``[0, 100]``) of the useful
    volume of *hreg*, the permanent-override-corrected registry row."""
    return percent_of_useful_volume(
        min(100.0, max(0.0, percent)),
        float(hreg["volume_minimo"]),
        float(hreg["volume_maximo"]),
    )


def fpha_reference_volumes(
    case: NewaveCase, cadastro: pd.DataFrame
) -> dict[int, list[float]]:
    """``{plant code: FPHA reference volume (hm³) of each study stage}`` for
    every active plant of an FPHA case; empty otherwise.

    - A daily or run-of-river plant reads its
      :func:`~novomodelo_bridge.newave.converters.hydro.geometry.held_volume`.
    - A monthly reservoir reads its per-period rows of
      ``volumes-referencia.csv`` when ``VOLUME-REFERENCIAL-TIPO-PADRAO`` is 1,
      or its hidr ``volume_referencia`` when it has no row.
    - Otherwise a monthly reservoir reads its initial storage in every stage.

    A value may lie outside ``[volume_minimo, volume_maximo]``;
    :func:`~novomodelo_bridge.newave.converters.hydro.geometry.generate_hydro_geometry`
    extends the geometry to cover it.

    Raises:
        SourceFileError: a plant's rows leave a study stage uncovered, which
            the source model also rejects.
    """
    if not case.fpha_enabled:
        return {}
    stage_months = [(d.year, d.month) for d in stage_dates_for(case.horizon)]
    periodic = _periodic_references(case)
    if periodic is None:
        emit(_initial_storage_diagnostic(case), logger=_LOG)

    references: dict[int, list[float]] = {}
    for _, row in case.active_hydros.iterrows():
        code = int(row["codigo_usina"])
        if code not in cadastro.index:
            continue
        hreg = cadastro.loc[code]
        rows = None if periodic is None else periodic.get(code)
        listed = (
            None if rows is None else _stage_references(case, code, rows, stage_months)
        )
        held = held_volume(hreg)
        if held is not None:
            references[code] = [held] * len(stage_months)
        elif periodic is None:
            storage = initial_storage_hm3(hreg, float(row["volume_inicial_percentual"]))
            references[code] = [storage] * len(stage_months)
        elif listed is not None:
            references[code] = listed
        elif not is_na(hreg.get("volume_referencia")):
            references[code] = [float(hreg["volume_referencia"])] * len(stage_months)
    return references


def _stage_references(
    case: NewaveCase,
    code: int,
    by_month: dict[tuple[int, int], float],
    stage_months: list[tuple[int, int]],
) -> list[float]:
    missing = [ym for ym in stage_months if ym not in by_month]
    if missing:
        year, month = missing[0]
        path = case.files.volumes_referencia
        raise SourceFileError(
            f"{_file_name(case)} gives plant {code} "
            f"reference volumes but none for {year:04d}/{month:02d} "
            f"({len(missing)} study stage(s) uncovered); the source model "
            "requires a value in every period",
            path=str(path),
            field=f"{HIDRELETRICA_CADASTRO_RESERVATORIO} {code}",
        )
    return [by_month[ym] for ym in stage_months]


def _periodic_references(
    case: NewaveCase,
) -> dict[int, dict[tuple[int, int], float]] | None:
    """``{plant code: {(year, month): hm³}}`` from ``volumes-referencia.csv``,
    or ``None`` when the deck does not declare it or switches its rows off.

    Raises:
        SourceFileError: an unknown ``VOLUME-REFERENCIAL-TIPO-PADRAO`` value,
            or two rows covering the same month of one plant.
    """
    reader = case.volumes_referencia
    if reader is None:
        return None
    path = case.files.volumes_referencia
    tipo_df = reader.volume_referencial_tipo_padrao(df=True)
    tipos = (
        set()
        if tipo_df is None or tipo_df.empty
        else {int(v) for v in tipo_df["tipo_referencia"]}
    )
    if not tipos <= {_PERIODIC_OFF, _PERIODIC_ON} or len(tipos) > 1:
        raise SourceFileError(
            f"{_file_name(case)} sets "
            f"VOLUME-REFERENCIAL-TIPO-PADRAO to {sorted(tipos)}; the source model "
            f"accepts one record of {_PERIODIC_OFF} or {_PERIODIC_ON}",
            path=str(path),
            field="VOLUME-REFERENCIAL-TIPO-PADRAO",
        )
    if tipos != {_PERIODIC_ON}:
        return None

    rows = reader.volume_referencial_periodo(df=True)
    by_plant: dict[int, dict[tuple[int, int], float]] = {}
    if rows is None or rows.empty:
        return by_plant
    for _, row in rows.iterrows():
        code = int(row["codigo_usina"])
        volume = float(row["volume_referencia"])
        plant = by_plant.setdefault(code, {})
        for ym in _months(row["data_inicio"], row["data_fim"]):
            if ym in plant:
                raise SourceFileError(
                    f"{_file_name(case)} gives plant {code} more than one reference "
                    f"volume for {ym[0]:04d}/{ym[1]:02d}, which the source model "
                    "rejects",
                    path=str(path),
                    field=f"{HIDRELETRICA_CADASTRO_RESERVATORIO} {code}",
                )
            plant[ym] = volume
    return by_plant


def _file_name(case: NewaveCase) -> str:
    path = case.files.volumes_referencia
    return path.name if path is not None else "volumes-referencia.csv"


def _months(start: date, end: date) -> list[tuple[int, int]]:
    """``(year, month)`` from *start*'s month through *end*'s, both included."""
    first = start.year * 12 + start.month - 1
    last = end.year * 12 + end.month - 1
    return [(index // 12, index % 12 + 1) for index in range(first, last + 1)]


def _initial_storage_diagnostic(case: NewaveCase) -> Diagnostic:
    declared = case.volumes_referencia is not None
    reason = (
        "turns its per-period reference volumes off"
        if declared
        else f"does not declare {HIDRELETRICA_CADASTRO_RESERVATORIO}"
    )
    return Diagnostic(
        code="fpha-reference-initial-storage",
        severity=Severity.INFO,
        category="Production model",
        title="FPHA reference volumes at the initial storage",
        summary=(
            f"The deck {reason}, so each monthly reservoir's FPHA reference "
            "volume is its initial storage in every stage, as in the source model."
        ),
    )


def seasonal_useful_volumes(case: NewaveCase) -> dict[int, dict[int, float]]:
    """``volref_saz.dat`` as ``{plant code: {calendar month: useful hm³}}``.

    A row of all zeros means the plant has no seasonal reference, so it is
    left out. In any other row a zero month means ``volume_minimo`` and is
    kept. Empty when the file is absent.
    """
    vs = case.volref_saz
    if vs is None:
        return {}
    df = vs.volumes
    if df is None or df.empty:
        return {}

    by_plant: dict[int, dict[int, float]] = {}
    for _, row in df.iterrows():
        by_plant.setdefault(int(row["codigo_usina"]), {})[int(row["mes"])] = float(
            row["valor"]
        )
    return {
        code: months
        for code, months in by_plant.items()
        if any(v > 0.0 for v in months.values())
    }


def evaporation_reference_volumes(
    case: NewaveCase, cadastro: pd.DataFrame
) -> dict[int, list[float]]:
    """``{plant code: twelve evaporation reference volumes (hm³), January
    first}`` for each plant with a :func:`seasonal_useful_volumes` row.

    Each month is the low end of the plant's
    :func:`~novomodelo_bridge.newave.converters.hydro.geometry.operating_storage_range`
    plus its useful volume, clamped into that range, so a daily or
    run-of-river plant reads its held volume.
    """
    references: dict[int, list[float]] = {}
    for code, monthly_useful in seasonal_useful_volumes(case).items():
        if code not in cadastro.index:
            continue
        vol_min, vol_max = operating_storage_range(cadastro.loc[code])
        references[code] = [
            max(vol_min, min(vol_max, vol_min + monthly_useful.get(month, 0.0)))
            for month in range(1, 13)
        ]
    return references

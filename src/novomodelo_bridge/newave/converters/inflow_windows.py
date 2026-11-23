"""Windowed inflow emission.

Builds the two windowed inflow inputs of the unified representation:

- the realized-inflow history Parquet
  (``{hydro_id, start_date, end_date, value_m3s}``, one row per hydro per
  calendar month of the historical record), and
- the conditioning windows for ``hydros[].recent_inflows`` of
  ``boundary_conditions.json`` (the source model's hydrological tendency, as
  the 12 calendar months preceding the study start).

Values are incremental inflows in m³/s. The module also holds the readers of
``vazoes.dat`` and ``vazpast.dat`` behind them: the posto width, the inflow
cascade and the natural-to-incremental subtraction.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
from inewave.newave import Dger, Vazoes

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    Severity,
    emit,
    unreadable_file_diagnostic,
)
from novomodelo_bridge.core.errors import FieldParseError
from novomodelo_bridge.core.incremental_inflow import (
    ARTIFICIAL_GAUGE,
    artificial_gauge_data_ignored_diagnostic,
    artificial_gauge_plants_diagnostic,
    artificial_pass_through,
    artificial_pass_through_diagnostic,
    incremental_by_plant,
    is_artificial_gauge,
)
from novomodelo_bridge.core.inflow_windows import (
    format_observation_windows,
    month_window,
    previous_months,
)
from novomodelo_bridge.core.pandas_utils import as_int
from novomodelo_bridge.newave import plants
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.horizon import history_start_year, study_start
from novomodelo_bridge.newave.id_map import NewaveIdMap

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

logger = logging.getLogger(__name__)

# Parquet schema for the windowed past-inflow history.
INFLOW_HISTORY_WINDOW_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32()),
        pa.field("start_date", pa.date32()),
        pa.field("end_date", pa.date32()),
        pa.field("value_m3s", pa.float64()),
    ]
)


# ``vazoes.dat`` is a headerless int32 matrix (months x postos) whose width the
# deck never declares anywhere, so it is derived from the file size.
_POSTO_COUNTS: tuple[int, int] = (320, 600)
_DEFAULT_POSTO_COUNT = 320


def _posto_count(path: Path, *, hist_start_year: int, study_start_year: int) -> int:
    """Return the number of postos in ``vazoes.dat``, derived from its size.

    A width qualifies when it divides the matrix into whole years, since the
    historical record always ends in December, and when the record it implies
    ends no later than the study start year, the last year the history spans.
    Every 600-posto history whose length is a multiple of 8 years is also a
    whole number of years at 320, so the year bound is what tells the two
    apart; reading a deck at the wrong width silently reinterprets the whole
    series. Should both widths still qualify, the size resolves to the far
    more common width and says so. A size no width fits fails here, naming
    the file, where reading on surfaces as an opaque reshape error from
    inside the reader.
    """
    size = path.stat().st_size
    if size % 4:
        raise FieldParseError(
            f"vazoes.dat is {size} bytes, which is not a whole number of "
            "4-byte values -- the file is truncated, or it is not a "
            "historical inflow record.",
            path=str(path),
            field="vazoes.dat",
        )
    num_values = size // 4
    if num_values == 0:
        # An empty file implies nothing about its width, and the callers below
        # already report the emptiness itself.
        return _DEFAULT_POSTO_COUNT

    years_by_count = {
        count: num_values // (12 * count)
        for count in _POSTO_COUNTS
        if num_values % (12 * count) == 0
    }
    if not years_by_count:
        raise FieldParseError(
            f"vazoes.dat holds {num_values} values, which is not a whole "
            f"number of years at {' or '.join(str(c) for c in _POSTO_COUNTS)} "
            "postos -- the file is truncated, or it is not a historical "
            "inflow record.",
            path=str(path),
            field="vazoes.dat",
        )

    candidates = [
        count
        for count, years in years_by_count.items()
        if hist_start_year + years - 1 <= study_start_year
    ]
    if not candidates:
        overruns = " and ".join(
            f"to {hist_start_year + years - 1} at {count} postos"
            for count, years in years_by_count.items()
        )
        raise FieldParseError(
            f"vazoes.dat holds {num_values} values, a history from "
            f"{hist_start_year} running {overruns} -- past {study_start_year}, "
            "the study start year. The file does not match the deck's "
            "historical period, or it is not a historical inflow record.",
            path=str(path),
            field="vazoes.dat",
        )
    if len(candidates) == 1:
        return candidates[0]

    emit(
        Diagnostic(
            code="vazoes-posto-count-ambiguous",
            severity=Severity.WARNING,
            category="Historical inflows",
            title="Ambiguous vazoes.dat width",
            summary=(
                f"vazoes.dat holds {num_values} values, a whole number of "
                f"years ending by {study_start_year} at "
                f"{' and at '.join(str(c) for c in candidates)} postos; it is "
                f"read at {_DEFAULT_POSTO_COUNT}, the width almost every deck "
                "uses. A deck written at the other width is read as a "
                "different number of months of history, with no further sign "
                "that anything is wrong."
            ),
            remediation=(
                "Check the deck's posto count if the historical inflow "
                "series look implausible."
            ),
        ),
        logger=logger,
    )
    return _DEFAULT_POSTO_COUNT


def _read_vazoes(path: Path, dger: Dger) -> Vazoes:
    """Read ``vazoes.dat`` at the width its size and the deck's years imply."""
    count = _posto_count(
        path,
        hist_start_year=history_start_year(dger),
        study_start_year=study_start(dger)[0],
    )
    return Vazoes.read(path, postos=count)


class _InflowCascade(NamedTuple):
    """The cascade's inflow nodes: in-service and admitted filling plants."""

    posto_of: dict[int, int]
    upstream_of: dict[int, list[int]]


def _inflow_cascade(
    confhd_df: pd.DataFrame,
    *,
    filling_codes: AbstractSet[int] = frozenset(),
) -> _InflowCascade:
    """Return each inflow node's station and directly upstream inflow nodes.

    A plant out of service (``NE``/``NC``, unless admitted for filling) is
    not an inflow node, but its downstream link is still the cascade's
    topology: the walk steps through it to the next inflow node, or the
    upstream plant's water would never reach the plant below. An admitted
    filling plant receives inflow, so it is a node and is not stepped past.
    A node whose ``posto`` is NaN is left out.
    """
    row_by_code: dict[int, pd.Series] = {}
    posto_of: dict[int, int] = {}
    for _, row in confhd_df.iterrows():
        code = int(row["codigo_usina"])
        row_by_code[code] = row
        status = str(row["usina_existente"]).strip()
        if status in plants.IN_SERVICE_STATUSES or code in filling_codes:
            if pd.isna(row["posto"]):
                continue
            posto_of[code] = int(row["posto"])

    def _downstream_code(code: int) -> int:
        raw = row_by_code[code].get("codigo_usina_jusante")
        return int(raw) if raw is not None and not pd.isna(raw) else 0

    def _next_node(start_code: int) -> int | None:
        current = start_code
        seen: set[int] = set()
        while current != 0 and current not in seen:
            seen.add(current)
            if current in posto_of:
                return current
            if current not in row_by_code:
                return None
            current = _downstream_code(current)
        return None

    upstream_of: dict[int, list[int]] = {}
    for code in posto_of:
        downstream = _next_node(_downstream_code(code))
        if downstream is not None:
            upstream_of.setdefault(downstream, []).append(code)
    return _InflowCascade(posto_of, upstream_of)


def _case_cascade(case: NewaveCase) -> _InflowCascade:
    """The inflow cascade of *case*, with its admitted filling plants."""
    filling = plants.filling_hydro_codes(
        case.hydro_plants,
        case.exph.expansoes if case.exph is not None else None,
    )
    return _inflow_cascade(case.hydro_plants, filling_codes=filling)


def _incremental_by_hydro(
    cascade: _InflowCascade,
    id_map: NewaveIdMap,
    natural_by_posto: Mapping[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Incremental series keyed by Novomodelo hydro id, one per converted plant."""
    codes = [code for code in id_map.all_hydro_codes if code in cascade.posto_of]
    by_code = incremental_by_plant(
        natural_by_posto, cascade.posto_of, cascade.upstream_of, codes
    )
    return {id_map.hydro_id(code): series for code, series in by_code.items()}


def _report_artificial_gauge(
    case: NewaveCase,
    cascade: _InflowCascade,
    id_map: NewaveIdMap,
    natural_by_posto: Mapping[int, np.ndarray],
) -> None:
    """Report the plants on the artificial gauge and the pairs that may double
    count upstream inflow."""
    codes = [code for code in id_map.all_hydro_codes if code in cascade.posto_of]
    artificial = [code for code in codes if is_artificial_gauge(cascade.posto_of[code])]
    if not artificial:
        return
    name_of = {
        int(row["codigo_usina"]): str(row["nome_usina"]).strip()
        for _, row in case.hydro_plants.iterrows()
    }
    emit(
        artificial_gauge_plants_diagnostic(
            [(code, name_of.get(code, "")) for code in artificial]
        ),
        logger=logger,
    )
    stored = natural_by_posto.get(ARTIFICIAL_GAUGE)
    if stored is not None and np.any(stored != 0.0):
        emit(artificial_gauge_data_ignored_diagnostic("vazoes.dat"), logger=logger)
    pairs = artificial_pass_through(cascade.posto_of, cascade.upstream_of, codes)
    if pairs:
        emit(
            artificial_pass_through_diagnostic(
                [
                    (code, name_of.get(code, ""), up, name_of.get(up, ""))
                    for code, up in pairs
                ]
            ),
            logger=logger,
        )


def _vazpast_incremental(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> dict[int, dict[int, float]]:
    """Read the hydrological-tendency file and return incremental inflows.

    Returns ``{novomodelo_hydro_id: {calendar_month: value_m3s}}`` after the
    posto -> plant mapping and the natural -> incremental subtraction.
    Empty when the file is absent, unreadable, or carries no tendency data.
    """
    if case.files.vazpast is None:
        logger.debug("vazpast.dat not found; no recent inflow lags.")
        return {}

    try:
        vazpast_obj = case.vazpast
    except Exception as exc:
        emit(
            unreadable_file_diagnostic(
                code="vazpast-unreadable",
                file_name="vazpast.dat",
                category="Inflow scenarios",
                effect=(
                    "the case has none of the recent inflows its inflow model "
                    "starts from, and the first stages' inflow scenarios differ "
                    "from NEWAVE's."
                ),
                exc=exc,
            ),
            logger=logger,
        )
        return {}

    df_tend: pd.DataFrame | None = (
        vazpast_obj.tendencia if vazpast_obj is not None else None
    )
    if df_tend is None or df_tend.empty:
        return {}

    months = list(range(1, 13))
    natural: dict[int, np.ndarray] = {}
    present: dict[int, set[int]] = {}
    for posto, posto_df in df_tend.groupby("codigo_posto"):
        values = dict(zip(posto_df["mes"].astype(int), posto_df["valor"].astype(float)))
        natural[as_int(posto)] = np.array([values.get(m, 0.0) for m in months])
        present[as_int(posto)] = set(values)

    cascade = _case_cascade(case)
    posto_by_hydro = {
        id_map.hydro_id(code): cascade.posto_of[code]
        for code in id_map.all_hydro_codes
        if code in cascade.posto_of
    }
    all_months = set(months)
    return {
        hydro_id: {
            m: float(series[m - 1])
            for m in months
            if m in present.get(posto_by_hydro[hydro_id], all_months)
        }
        for hydro_id, series in _incremental_by_hydro(cascade, id_map, natural).items()
    }


def _incremental_history(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> tuple[int, int, dict[int, np.ndarray]]:
    """Read the historical record and return incremental series per hydro.

    Returns ``(hist_start_year, n_rows, series_by_hydro)``: monthly
    incremental inflow arrays (m³/s) keyed by Novomodelo hydro id, one entry per
    month from January of ``hist_start_year``, truncated at the month
    before the study start.

    Raises
    ------
    FileNotFoundError
        If the vazoes.dat DataFrame is absent or empty.
    """
    hist_start_year, n_rows, natural_by_posto = _natural_history(case)
    cascade = _case_cascade(case)
    _report_artificial_gauge(case, cascade, id_map, natural_by_posto)
    incremental = _incremental_by_hydro(cascade, id_map, natural_by_posto)
    return hist_start_year, n_rows, incremental


def natural_monthly_means(case: NewaveCase) -> dict[int, list[float]]:
    """Per posto, the mean natural inflow (m³/s) of each calendar month,
    January first, over the historical record :func:`_incremental_history`
    reads.

    Raises
    ------
    FileNotFoundError
        If the vazoes.dat DataFrame is absent or empty.
    """
    _, _, natural_by_posto = _natural_history(case)
    return {
        posto: [float(series[month::12].mean()) for month in range(12)]
        for posto, series in natural_by_posto.items()
        if len(series) >= 12
    }


def _natural_history(case: NewaveCase) -> tuple[int, int, dict[int, np.ndarray]]:
    """``(hist_start_year, n_rows, natural series per posto)`` of
    ``vazoes.dat``, one entry per month from January of ``hist_start_year``,
    truncated at the month before the study start."""
    # vazoes.dat is large and read only here, so it stays uncached on case.files.
    dger = case.dger
    vazoes_obj = _read_vazoes(case.files.vazoes, dger)
    df_vazoes: pd.DataFrame | None = vazoes_obj.vazoes
    if df_vazoes is None or df_vazoes.empty:
        raise FileNotFoundError("vazoes.dat not found or empty")

    hist_start_year = history_start_year(dger)
    study_start_year, study_start_month = study_start(dger)

    # History is truncated at the month BEFORE the study start.
    # e.g., study starts March 2026 → history ends February 2026.
    n_total_rows = len(df_vazoes)
    cutoff_months = (study_start_year - hist_start_year) * 12 + (study_start_month - 1)
    n_rows = min(cutoff_months, n_total_rows)

    return hist_start_year, n_rows, _natural_by_posto(df_vazoes, n_rows)


def _natural_by_posto(df_vazoes: pd.DataFrame, n_rows: int) -> dict[int, np.ndarray]:
    """The first ``n_rows`` months of every posto column of ``vazoes.dat``."""
    natural: dict[int, np.ndarray] = {}
    for col in df_vazoes.columns:
        if col == "data":
            continue
        try:
            posto = int(col)
        except (ValueError, TypeError):
            continue
        natural[posto] = df_vazoes[col].to_numpy(dtype=float)[:n_rows]
    return natural


def convert_inflow_history_windows(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> pa.Table:
    """Convert the full historical inflow record to windowed Parquet rows.

    One row per (hydro, calendar month) from January of the record's first
    year through the month before the study start, with explicit
    ``[start_date, end_date)`` windows. Values are incremental m³/s.

    Raises
    ------
    FileNotFoundError
        If the historical-record DataFrame is absent or empty.
    """
    hist_start_year, n_rows, incremental = _incremental_history(case, id_map)

    rows_hydro_id: list[int] = []
    rows_start: list[date] = []
    rows_end: list[date] = []
    rows_value: list[float] = []

    for novomodelo_id in sorted(incremental):
        values = incremental[novomodelo_id]
        for i in range(n_rows):
            y = hist_start_year + (i // 12)
            m = (i % 12) + 1
            start, end = month_window(y, m)
            rows_hydro_id.append(novomodelo_id)
            rows_start.append(start)
            rows_end.append(end)
            rows_value.append(float(values[i]))

    return pa.table(
        {
            "hydro_id": pa.array(rows_hydro_id, type=pa.int32()),
            "start_date": pa.array(rows_start, type=pa.date32()),
            "end_date": pa.array(rows_end, type=pa.date32()),
            "value_m3s": pa.array(rows_value, type=pa.float64()),
        },
        schema=INFLOW_HISTORY_WINDOW_SCHEMA,
    )


def convert_recent_observation_windows(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> list[dict]:
    """Convert the hydrological tendency to conditioning windows.

    Emits, per hydro, the 12 calendar-month windows immediately preceding
    the study start (oldest first), shaped for the
    ``hydros[].recent_inflows`` field of ``boundary_conditions.json``. Values are
    incremental m³/s. A calendar month absent from the tendency data is
    omitted rather than zero-filled — downstream coverage validation
    surfaces genuine gaps, and a fabricated zero-inflow window would be
    worse than an absent one.

    Returns an empty list when the tendency file is absent or unreadable.
    """
    incremental = _vazpast_incremental(case, id_map)
    if not incremental:
        return []

    months = previous_months(*study_start(case.dger), 12)

    rows: list[tuple[int, date, date, float]] = []
    for hydro_id in sorted(incremental):
        month_values = incremental[hydro_id]
        for y, m in months:
            if m not in month_values:
                continue
            start, end = month_window(y, m)
            rows.append((hydro_id, start, end, month_values[m]))

    return format_observation_windows(rows)

"""Windowed inflow emission (Cobre >= 0.13 input shapes).

Builds the two windowed inflow inputs of the unified representation:

- the realized-inflow history Parquet
  (``{hydro_id, start_date, end_date, value_m3s}``, one row per hydro per
  calendar month of the historical record), and
- the conditioning windows for ``initial_conditions.recent_observations``
  (the source model's hydrological tendency, as the 12 calendar months
  preceding the study start).

Values are incremental m³/s, produced by the same posto-mapping and
upstream-subtraction helpers the point-dated emitters use. The pipeline
switches to these emitters when the Cobre dependency pin moves to the
windowed schema; until then the module is additive and unused by
``convert``.
"""

from __future__ import annotations

from datetime import date

import pyarrow as pa

from cobre_bridge.core.inflow_windows import (
    format_observation_windows,
    month_window,
    previous_months,
)
from cobre_bridge.newave.case import NewaveCase
from cobre_bridge.newave.converters.stochastic import (
    _incremental_history,
    _vazpast_incremental,
)
from cobre_bridge.newave.id_map import NewaveIdMap

# Parquet schema for the windowed past-inflow history (Cobre >= 0.13).
INFLOW_HISTORY_WINDOW_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32()),
        pa.field("start_date", pa.date32()),
        pa.field("end_date", pa.date32()),
        pa.field("value_m3s", pa.float64()),
    ]
)


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

    for cobre_id in sorted(incremental):
        values = incremental[cobre_id]
        for i in range(n_rows):
            y = hist_start_year + (i // 12)
            m = (i % 12) + 1
            start, end = month_window(y, m)
            rows_hydro_id.append(cobre_id)
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
    ``initial_conditions.recent_observations`` field. Values are
    incremental m³/s. A calendar month absent from the tendency data is
    omitted rather than zero-filled — downstream coverage validation
    surfaces genuine gaps, and a fabricated zero-inflow window would be
    worse than an absent one.

    Returns an empty list when the tendency file is absent or unreadable.
    """
    incremental = _vazpast_incremental(case, id_map)
    if not incremental:
        return []

    dger = case.dger
    months = previous_months(dger.ano_inicio_estudo, dger.mes_inicio_estudo, 12)

    rows: list[tuple[int, date, date, float]] = []
    for hydro_id in sorted(incremental):
        month_values = incremental[hydro_id]
        for y, m in months:
            if m not in month_values:
                continue
            start, end = month_window(y, m)
            rows.append((hydro_id, start, end, month_values[m]))

    return format_observation_windows(rows)

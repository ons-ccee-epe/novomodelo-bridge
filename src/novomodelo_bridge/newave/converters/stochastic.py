"""Stochastic data converter: maps the source model inflow and load data to Novomodelo
Parquet.

Converts ``vazoes.dat`` (historical inflow series), ``vazpast.dat`` (recent
past inflows), ``sistema.dat`` (load demand), ``patamar.dat`` (load block
factors), and ``c_adic.dat`` (additional generation added to load) into
PyArrow Tables and dicts that are written as Parquet files and JSON files in
the ``scenarios/`` directory of a Novomodelo case.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
from inewave.newave import Cadic, Dger, Vazoes

from novomodelo_bridge.core.diagnostics import Diagnostic, Severity, emit
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
from novomodelo_bridge.core.pandas_utils import as_int
from novomodelo_bridge.newave import plants
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.horizon import POST_STUDY_YEAR, study_horizon
from novomodelo_bridge.newave.id_map import NewaveIdMap
from novomodelo_bridge.newave.switches import switch_off_diagnostic
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

logger = logging.getLogger(__name__)


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
        hist_start_year=int(dger.ano_inicial_historico),
        study_start_year=int(dger.ano_inicio_estudo),
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
        case.confhd.usinas,
        case.exph.expansoes if case.exph is not None else None,
    )
    return _inflow_cascade(case.confhd.usinas, filling_codes=filling)


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
        for _, row in case.confhd.usinas.iterrows()
    }
    emit(
        artificial_gauge_plants_diagnostic(
            [
                (code, name_of.get(code, ""), id_map.hydro_id(code))
                for code in artificial
            ]
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


# Parquet schema for inflow seasonal statistics.
_INFLOW_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32()),
        pa.field("stage_id", pa.int32()),
        pa.field("mean_m3s", pa.float64()),
        pa.field("std_m3s", pa.float64()),
    ]
)

# Parquet schema for load seasonal statistics.
_LOAD_SCHEMA = pa.schema(
    [
        pa.field("bus_id", pa.int32()),
        pa.field("stage_id", pa.int32()),
        pa.field("mean_mw", pa.float64()),
        pa.field("std_mw", pa.float64()),
    ]
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
    except Exception:  # noqa: BLE001
        logger.warning("vazpast.dat could not be parsed; skipping recent lags.")
        return {}

    df_tend: pd.DataFrame | None = (
        vazpast_obj.tendencia if vazpast_obj is not None else None
    )
    if df_tend is None or df_tend.empty:
        return {}

    # The tendency's "codigo_usina" column holds the posto, as in vazoes.dat.
    months = list(range(1, 13))
    natural: dict[int, np.ndarray] = {}
    present: dict[int, set[int]] = {}
    for posto, posto_df in df_tend.groupby("codigo_usina"):
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

    hist_start_year: int = dger.ano_inicial_historico
    study_start_year: int = dger.ano_inicio_estudo
    study_start_month: int = dger.mes_inicio_estudo

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


def convert_inflow_stats(case: NewaveCase, id_map: NewaveIdMap) -> pa.Table:
    """Convert the source model historical inflow data to Novomodelo inflow seasonal
    statistics.

    For each hydro plant and each study stage (calendar month), computes the
    mean and standard deviation of historical monthly inflows across all
    available years in ``vazoes.dat``.  The station-to-plant mapping comes
    from ``confhd.dat``.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Entity ID map produced during entity conversion.  Used to resolve the source
        model hydro codes to 0-based Novomodelo hydro IDs.

    Returns
    -------
    pyarrow.Table
        Table with columns ``hydro_id`` (int32), ``stage_id`` (int32),
        ``mean_m3s`` (float64), ``std_m3s`` (float64).  One row per
        (hydro, stage) pair.  The number of stages equals the number of
        months in the study horizon registered in *id_map*.

    Raises
    ------
    FileNotFoundError
        If ``vazoes.dat`` DataFrame is empty.
    """
    dger = case.dger
    vazoes_obj = _read_vazoes(case.files.vazoes, dger)
    df_vazoes: pd.DataFrame | None = vazoes_obj.vazoes

    if df_vazoes is None or df_vazoes.empty:
        raise FileNotFoundError("vazoes.dat not found or empty")

    # Truncate to months before the study start (same window as inflow_history).
    hist_start_year: int = dger.ano_inicial_historico
    study_start_year: int = dger.ano_inicio_estudo
    study_start_month: int = dger.mes_inicio_estudo
    n_total_rows = len(df_vazoes)
    cutoff_months = (study_start_year - hist_start_year) * 12 + (study_start_month - 1)
    n_rows = min(cutoff_months, n_total_rows)

    if "data" in df_vazoes.columns:
        cal_months: np.ndarray = df_vazoes["data"].dt.month.to_numpy()[:n_rows]
    else:
        cal_months = np.array([(i % 12) + 1 for i in range(n_rows)])

    incremental = _incremental_by_hydro(
        _case_cascade(case), id_map, _natural_by_posto(df_vazoes, n_rows)
    )

    hydro_codes = id_map.all_hydro_codes
    study_months = _derive_study_stage_months(dger)

    rows_hydro_id: list[int] = []
    rows_stage_id: list[int] = []
    rows_mean: list[float] = []
    rows_std: list[float] = []

    for hydro_code in hydro_codes:
        novomodelo_hydro_id = id_map.hydro_id(hydro_code)
        inc_series = incremental.get(novomodelo_hydro_id)

        seasonal_mean: dict[int, float] = {}
        seasonal_std: dict[int, float] = {}

        for cal_month in range(1, 13):
            mask = cal_months == cal_month
            if inc_series is not None and mask.any():
                vals = inc_series[mask]
                seasonal_mean[cal_month] = float(np.nanmean(vals))
                seasonal_std[cal_month] = float(
                    np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0
                )
            else:
                seasonal_mean[cal_month] = 0.0
                seasonal_std[cal_month] = 0.0

        for stage_id, cm in enumerate(study_months):
            rows_hydro_id.append(novomodelo_hydro_id)
            rows_stage_id.append(stage_id)
            rows_mean.append(seasonal_mean[cm])
            rows_std.append(seasonal_std[cm])

    return pa.table(
        {
            "hydro_id": pa.array(rows_hydro_id, type=pa.int32()),
            "stage_id": pa.array(rows_stage_id, type=pa.int32()),
            "mean_m3s": pa.array(rows_mean, type=pa.float64()),
            "std_m3s": pa.array(rows_std, type=pa.float64()),
        }
    )


def convert_load_factors(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> dict:
    """Convert the source model patamar load factors to a Novomodelo ``load_factors.json``
    dict.

    Reads ``patamar.dat::carga_patamares`` and ``dger.dat`` to produce one
    entry per (bus, stage) pair containing a ``block_factors`` list with one
    entry per patamar.

    For post-study stages (beyond the study horizon), the last available
    year's factors are repeated seasonally (calendar-month match).

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Entity ID map.  Used to resolve the source model subsystem codes to 0-based
        Novomodelo bus IDs.

    Returns
    -------
    dict
        JSON-serializable dict with key ``"load_factors"`` containing a list
        of ``{"bus_id": int, "stage_id": int, "block_factors": [...]}``
        entries.
    """
    patamar = case.patamar
    df_carga: pd.DataFrame | None = patamar.carga_patamares

    horizon = case.horizon
    start_month = horizon.start_month
    start_year = horizon.start_year
    study_months = horizon.study_months
    total_stages = horizon.total_stages

    # Study end boundary: first month *after* the study horizon.
    study_end_year = start_year + (start_month - 1 + study_months) // 12
    study_end_month = ((start_month - 1 + study_months) % 12) + 1

    if df_carga is None or df_carga.empty:
        # A single block's load factor is 1 by definition, which an empty list
        # already gives; only a multi-block deck is missing data.
        num_blocks = patamar.numero_patamares or 1
        if num_blocks > 1:
            logger.warning(
                "%s declares %d load blocks but no block load factors; "
                "load_factors.json will be empty.",
                case.files.patamar.name,
                num_blocks,
            )
        return {
            "$schema": novomodelo_schemas.schema_url_for("scenarios/load_factors.json"),
            "load_factors": [],
        }

    # ``carga_patamares`` numbers its ``patamar`` field as a GLOBAL running index
    # across submarkets (submarket 1 -> patamares 1..P, submarket 2 -> P+1..2P,
    # ...), exactly like ``usinas_nao_simuladas``.  Normalize it to a per-submarket
    # 1-based block via ``(patamar - 1) % P + 1`` so every submarket's load profile
    # lands on blocks 1..P (a no-op for decks that reset the index per submarket).
    # Without this, only the first submarket's patamares fall in range and every
    # other submarket's per-block load profile is silently flattened to 1.0.
    num_patamares: int = patamar.numero_patamares or 1

    # Columns: codigo_submercado, data (datetime), patamar (global), valor (float)
    # Build lookup: {(subsystem_code, year, cal_month, block) -> factor}
    # Also track {(subsystem_code, cal_month, block) -> factor} for post-study.
    study_lookup: dict[tuple[int, int, int, int], float] = {}

    for _, row in df_carga.iterrows():
        sub = int(row["codigo_submercado"])
        dt = row["data"]
        cal_month = int(dt.month)
        yr = int(dt.year)
        block = (int(row["patamar"]) - 1) % num_patamares + 1
        val = float(row["valor"])
        study_lookup[(sub, yr, cal_month, block)] = val

    # Build last-year lookup: for each (sub, cal_month, patamar), take the value
    # from the last study year that has data for that calendar month.
    # Group by (sub, cal_month, pat) and pick the max year.
    last_year_per_key: dict[tuple[int, int, int], tuple[int, float]] = {}
    for (sub, yr, cal_month, pat), val in study_lookup.items():
        key = (sub, cal_month, pat)
        existing = last_year_per_key.get(key)
        if existing is None or yr > existing[0]:
            last_year_per_key[key] = (yr, val)

    last_year_lookup = {k: v for k, (_, v) in last_year_per_key.items()}

    # Determine the unique subsystem codes (num_patamares computed above).
    subsystem_codes = sorted(df_carga["codigo_submercado"].unique())

    load_factors: list[dict] = []

    for sub_code in subsystem_codes:
        try:
            bus_id = id_map.bus_id(int(sub_code))
        except KeyError:
            logger.warning(
                "Subsystem code %d from carga_patamares not in id_map; skipping",
                sub_code,
            )
            continue

        y, m = start_year, start_month
        for stage_id in range(total_stages):
            is_post_study = (y > study_end_year) or (
                y == study_end_year and m >= study_end_month
            )

            block_factors: list[dict] = []
            for pat_idx in range(1, num_patamares + 1):
                if is_post_study:
                    factor = last_year_lookup.get((int(sub_code), m, pat_idx))
                    if factor is None:
                        factor = 1.0
                else:
                    factor = study_lookup.get((int(sub_code), y, m, pat_idx))
                    if factor is None:
                        factor = 1.0

                block_factors.append({"block_id": pat_idx - 1, "factor": factor})

            load_factors.append(
                {
                    "bus_id": bus_id,
                    "stage_id": stage_id,
                    "block_factors": block_factors,
                }
            )

            m += 1
            if m > 12:
                m = 1
                y += 1

    return {
        "$schema": novomodelo_schemas.schema_url_for("scenarios/load_factors.json"),
        "load_factors": load_factors,
    }


def _derive_study_stage_months(dger: Dger) -> list[int]:
    """Return the ordered sequence of calendar months (1-12) for each study stage.

    Parameters
    ----------
    dger:
        A ``Dger`` object with ``mes_inicio_estudo``, ``ano_inicio_estudo``,
        and ``num_anos_estudo`` attributes.

    Returns
    -------
    list[int]
        Calendar month (1-12) for each stage.  Length equals
        :attr:`~novomodelo_bridge.newave.horizon.StudyHorizon.total_stages`.
    """
    horizon = study_horizon(dger)
    start_month = horizon.start_month
    total_stages = horizon.total_stages
    return [((start_month - 1 + i) % 12) + 1 for i in range(total_stages)]


def parse_cadical(path: Path) -> dict[tuple[int, int, int], float]:
    """Parse a C_ADIC.DAT file into a lookup of added load values.

    Public, stable parsing seam: both this converter and the results
    comparator (:mod:`novomodelo_bridge.comparators.newave.readers`) reconstruct the source
    model load from C_ADIC via this function, so its signature and the
    ``(subsystem_code, year, cal_month) -> total_mw`` return shape are part of the
    shared contract — change them in lockstep with both callers.

    Delegates the fixed-width parsing to inewave's :class:`~inewave.newave.Cadic`
    reader. C_ADIC.DAT contains must-take energy (in average MW) that source-model adds
    to the bus load, broken down by *razão* (reason) per subsystem per month. All razões
    for the same ``(subsystem_code, year, cal_month)`` are summed so the caller receives
    a single additive contribution.

    Post-study ("POS") rows carry inewave's sentinel year ``9999`` (PRE rows use
    ``1``), consistent with the convention in :func:`convert_load_stats`; PRE
    years simply never match a study/post-study stage and are ignored downstream.

    Parameters
    ----------
    path:
        Path to the C_ADIC.DAT file.

    Returns
    -------
    dict[tuple[int, int, int], float]
        Mapping of ``(subsystem_code, year, cal_month_1_based) -> total_mw``.
    """
    cargas = Cadic.read(str(path)).cargas
    result: dict[tuple[int, int, int], float] = {}
    if cargas is None:
        return result

    for _, row in cargas.iterrows():
        value = row["valor"]
        if pd.isna(value):
            continue
        dt = row["data"]
        key = (int(row["codigo_submercado"]), int(dt.year), int(dt.month))
        result[key] = result.get(key, 0.0) + float(value)

    return result


def convert_load_stats(case: NewaveCase, id_map: NewaveIdMap) -> pa.Table:
    """Convert the source model subsystem load data to Novomodelo load seasonal statistics.

    Reads ``sistema.dat`` and converts the ``mercado_energia`` DataFrame
    (load demand per subsystem per month) into a PyArrow Table.  When a
    ``c_adic.dat`` file is present its must-take energy values are added to
    the load on a per-(subsystem, stage) basis.

    The ``mercado_energia`` values from inewave are in average MW (the
    inewave library already converts from MWmonth to average MW), so no
    unit conversion is applied.  C_ADIC values are likewise in average MW.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Entity ID map.  Used to resolve the source model subsystem codes to 0-based
        Novomodelo bus IDs.

    Returns
    -------
    pyarrow.Table
        Table with columns ``bus_id`` (int32), ``stage_id`` (int32),
        ``mean_mw`` (float64), ``std_mw`` (float64).  One row per
        (bus, stage) pair.
    """
    sistema_obj = case.sistema
    df_load: pd.DataFrame = sistema_obj.mercado_energia

    horizon = case.horizon
    start_month = horizon.start_month
    start_year = horizon.start_year
    num_anos = horizon.num_anos
    study_months = horizon.study_months
    total_stages = horizon.total_stages

    # Load optional C_ADIC additions: {(sub_code, year_or_9999, cal_month) -> mw}.
    cadical_lookup: dict[tuple[int, int, int], float] = {}
    if case.files.c_adic is not None and not case.switches.c_adic.on:
        emit(switch_off_diagnostic(case.switches.c_adic), logger=logger)
    elif case.files.c_adic is not None:
        try:
            cadical_lookup = parse_cadical(case.files.c_adic)
            logger.debug(
                "Loaded %d C_ADIC entries from %s",
                len(cadical_lookup),
                case.files.c_adic,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "c_adic.dat could not be parsed; no additional load will be added.",
                exc_info=True,
            )

    rows_bus_id: list[int] = []
    rows_stage_id: list[int] = []
    rows_mean: list[float] = []
    rows_std: list[float] = []

    subsystem_codes_processed: set[int] = set()

    for subsystem_code, group in df_load.groupby("codigo_submercado", sort=True):
        sub_int = as_int(subsystem_code)
        try:
            bus_id = id_map.bus_id(sub_int)
        except KeyError:
            logger.warning(
                "Subsystem code %d from mercado_energia not in id_map; skipping",
                subsystem_code,
            )
            continue
        subsystem_codes_processed.add(sub_int)

        # Build a month-keyed lookup: study-period months by (year, month),
        # post-study months (year=9999) by calendar month for seasonal repeat.
        study_values: dict[tuple[int, int], float] = {}
        pos_values: dict[int, float] = {}  # cal_month -> value
        for _, row in group.iterrows():
            dt = row["data"]
            val = row["valor"]
            if pd.isna(val):
                continue
            y, m = dt.year, dt.month
            if y == POST_STUDY_YEAR:
                pos_values[m] = float(val)
            else:
                study_values[(y, m)] = float(val)

        # Emit one row per stage.
        y, m = start_year, start_month
        for stage_id in range(total_stages):
            is_post_study = stage_id >= study_months
            if is_post_study:
                val = pos_values.get(m)
                if val is None:
                    val = study_values.get((start_year + num_anos - 1, m), 0.0)
                # C_ADIC post-study: use the post-study sentinel year.
                val = (val or 0.0) + cadical_lookup.get(
                    (sub_int, POST_STUDY_YEAR, m), 0.0
                )
            else:
                val = study_values.get((y, m), 0.0)
                val += cadical_lookup.get((sub_int, y, m), 0.0)

            rows_bus_id.append(bus_id)
            rows_stage_id.append(stage_id)
            rows_mean.append(val)
            rows_std.append(0.0)

            m += 1
            if m > 12:
                m = 1
                y += 1

    # Emit zero-load rows for buses that exist in id_map but have no
    # mercado_energia data (e.g., fictitious subsystems like NOFICT1).
    emitted_buses = set(rows_bus_id)
    for bus_code in id_map.all_bus_ids:
        bus_id = id_map.bus_id(bus_code)
        if bus_id not in emitted_buses:
            for stage_id in range(total_stages):
                rows_bus_id.append(bus_id)
                rows_stage_id.append(stage_id)
                rows_mean.append(0.0)
                rows_std.append(0.0)

    return pa.table(
        {
            "bus_id": pa.array(rows_bus_id, type=pa.int32()),
            "stage_id": pa.array(rows_stage_id, type=pa.int32()),
            "mean_mw": pa.array(rows_mean, type=pa.float64()),
            "std_mw": pa.array(rows_std, type=pa.float64()),
        }
    )

"""Load converter: maps the source model load data to Novomodelo Parquet and JSON.

Converts ``sistema.dat`` (load demand), ``patamar.dat`` (load block factors)
and ``c_adic.dat`` (additional generation added to load) into the tables and
dicts written in the ``scenarios/`` directory of a Novomodelo case.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa
from inewave.newave import Cadic

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
    unreadable_file_diagnostic,
)
from novomodelo_bridge.core.pandas_utils import as_int, require_frame
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.horizon import POST_STUDY_YEAR
from novomodelo_bridge.newave.id_map import NewaveIdMap
from novomodelo_bridge.newave.switches import switch_off_diagnostic
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


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
            patamar_name = case.files.patamar.name.lower()
            emit(
                Diagnostic(
                    code="patamar-load-factors-missing",
                    severity=Severity.WARNING,
                    category="Load blocks",
                    title="Load blocks without load factors",
                    summary=(
                        f"{patamar_name} declares {num_blocks} load blocks but "
                        "no load-block factors, so the case gives every block "
                        "the stage's mean load, and the load of each block "
                        "differs from NEWAVE's wherever the blocks should differ."
                    ),
                    remediation=f"Declare the load-block factors in {patamar_name}.",
                ),
                logger=logger,
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
    undeclared: list[int] = []

    for sub_code in subsystem_codes:
        try:
            bus_id = id_map.bus_id(int(sub_code))
        except KeyError:
            undeclared.append(int(sub_code))
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

    if undeclared:
        emit(
            Diagnostic(
                code="load-factor-subsystem-undeclared",
                severity=Severity.INFO,
                category="Network",
                title=f"Load-block factors of undeclared subsystems ({len(undeclared)})",
                summary=(
                    f"{case.files.patamar.name.lower()} gives load-block factors "
                    "for the listed subsystems, which sistema.dat does not "
                    "declare, so neither model has a bus for them and the case "
                    "leaves those factors out."
                ),
                table=DiagnosticTable(
                    columns=["Subsystem code"], rows=[[c] for c in undeclared]
                ),
                remediation=(
                    "Declare each listed subsystem in sistema.dat, or remove its "
                    "load-block factors."
                ),
            ),
            logger=logger,
        )
    return {
        "$schema": novomodelo_schemas.schema_url_for("scenarios/load_factors.json"),
        "load_factors": load_factors,
    }


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
    df_load = require_frame(
        sistema_obj.mercado_energia, "energy market table in sistema.dat"
    )

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
        except Exception as exc:
            emit(
                unreadable_file_diagnostic(
                    code="c-adic-unreadable",
                    file_name="c_adic.dat",
                    category="Load",
                    effect=(
                        "the case has none of the additional loads it gives, and "
                        "the load is lower than NEWAVE's."
                    ),
                    exc=exc,
                ),
                logger=logger,
            )

    rows_bus_id: list[int] = []
    rows_stage_id: list[int] = []
    rows_mean: list[float] = []
    rows_std: list[float] = []

    undeclared: list[int] = []

    for subsystem_code, group in df_load.groupby("codigo_submercado", sort=True):
        sub_int = as_int(subsystem_code)
        try:
            bus_id = id_map.bus_id(sub_int)
        except KeyError:
            undeclared.append(sub_int)
            continue

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

    if undeclared:
        emit(
            Diagnostic(
                code="load-subsystem-undeclared",
                severity=Severity.INFO,
                category="Network",
                title=f"Energy market of undeclared subsystems ({len(undeclared)})",
                summary=(
                    "sistema.dat gives an energy market for the listed "
                    "subsystems but declares no such subsystem, so neither "
                    "model has a bus for them and the case leaves that load out."
                ),
                table=DiagnosticTable(
                    columns=["Subsystem code"], rows=[[c] for c in undeclared]
                ),
                remediation=(
                    "Declare each listed subsystem in sistema.dat, or remove its "
                    "energy market."
                ),
            ),
            logger=logger,
        )

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


def block_loads(
    load_stats: pa.Table, load_factors: Mapping[str, Any], num_blocks: int
) -> dict[tuple[int, int], list[float]]:
    """Each ``(bus_id, stage_id)``'s load in MW on each of its *num_blocks*
    blocks, as the case gives it: the mean load of *load_stats* times the
    block's factor in *load_factors* (:func:`convert_load_factors`), which is
    1 where the case gives none."""
    factors = {
        (entry["bus_id"], entry["stage_id"]): {
            block["block_id"]: block["factor"] for block in entry["block_factors"]
        }
        for entry in load_factors["load_factors"]
    }
    loads: dict[tuple[int, int], list[float]] = {}
    for bus_id, stage_id, mean_mw in zip(
        load_stats.column("bus_id").to_pylist(),
        load_stats.column("stage_id").to_pylist(),
        load_stats.column("mean_mw").to_pylist(),
        strict=True,
    ):
        block_factor = factors.get((bus_id, stage_id), {})
        loads[(bus_id, stage_id)] = [
            mean_mw * block_factor.get(block_id, 1.0) for block_id in range(num_blocks)
        ]
    return loads

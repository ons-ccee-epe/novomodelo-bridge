"""Pumping station converter for the NEWAVE track."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd
import pyarrow as pa

from novomodelo_bridge.core.bound_schemas import PUMPING_BOUNDS_SCHEMA
from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from novomodelo_bridge.core.errors import SourceFileError
from novomodelo_bridge.newave.horizon import stage_dates_for
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas

if TYPE_CHECKING:
    from novomodelo_bridge.newave.case import NewaveCase
    from novomodelo_bridge.newave.id_map import NewaveIdMap


class PumpingInputs(NamedTuple):
    """Validated pumping input data ready for conversion."""

    stations: pd.DataFrame
    limits: pd.DataFrame


def read_pumping_inputs(case: NewaveCase) -> PumpingInputs | None:
    """Read and validate pumping data from LIBS files.

    Returns ``None`` when the switch is off or the estacao_bombeamento file is
    absent. Raises :class:`SourceFileError` on validation failures.
    """
    if not case.switches.bombeamento.on:
        return None
    reader = case.estacao_bombeamento
    if reader is None:
        return None

    stations_df = reader.estacao_bombeamento(df=True)
    if stations_df is None or stations_df.empty:
        return None

    # Validate submercado mapping
    submercado_reader = case.estacao_bombeamento_submercado
    if submercado_reader is None:
        raise SourceFileError(
            "Pumping stations defined but submercado file missing",
            path="indices.csv",
            field="ESTACAO-BOMBEAMENTO-SUBMERCADO",
        )
    submercado_df = submercado_reader.estacao_bombeamento_submercado(df=True)
    if submercado_df is None or submercado_df.empty:
        raise SourceFileError(
            "Pumping submercado file empty",
            path="estacao-bombeamento-submercado.csv",
            field="codigo_estacao",
        )

    # Each station must have exactly one submercado row
    for code in stations_df["codigo_estacao"]:
        rows = submercado_df[submercado_df["codigo_estacao"] == code]
        if len(rows) == 0:
            raise SourceFileError(
                f"Station {code} has no submercado assignment",
                path="estacao-bombeamento-submercado.csv",
                field="codigo_estacao",
            )
        if len(rows) > 1:
            raise SourceFileError(
                f"Station {code} has multiple submercado assignments",
                path="estacao-bombeamento-submercado.csv",
                field="codigo_estacao",
            )

    # Validate hydro codes exist in confhd
    confhd_codes = set(case.confhd.usinas["codigo_usina"])
    for col in ("codigo_usina_origem", "codigo_usina_destino"):
        for code in stations_df[col]:
            if int(code) not in confhd_codes:
                raise SourceFileError(
                    f"Station references unknown hydro {code}",
                    path="estacao-bombeamento.csv",
                    field=col,
                )

    # Join submercado
    merged = stations_df.merge(
        submercado_df[["codigo_estacao", "codigo_submercado"]],
        on="codigo_estacao",
    )

    # Get limits
    limits_df = reader.estacao_bombeamento_limites_periodo_patamar(df=True)
    if limits_df is None:
        limits_df = pd.DataFrame()

    return PumpingInputs(stations=merged, limits=limits_df)


def pumping_station_id_map(inputs: PumpingInputs) -> dict[int, int]:
    """{codigo_estacao: 0-based id}, sorted by code."""
    codes = sorted(int(c) for c in inputs.stations["codigo_estacao"])
    return {code: station_id for station_id, code in enumerate(codes)}


def convert_pumping_stations(
    case: NewaveCase,
    id_map: NewaveIdMap,
    inputs: PumpingInputs | None,
    station_ids: dict[int, int],
) -> dict:
    """Generate system/pumping_stations.json dict.

    Returns empty stations list when inputs is None.
    """
    if inputs is None:
        return {
            "$schema": novomodelo_schemas.schema_url_for(
                "system/pumping_stations.json"
            ),
            "pumping_stations": [],
        }

    op_date = date(case.horizon.start_year, case.horizon.start_month, 1).isoformat()
    stations: list[dict] = []

    for _, row in inputs.stations.sort_values("codigo_estacao").iterrows():
        code = int(row["codigo_estacao"])
        name = str(row["nome_estacao"]).strip()
        source = int(row["codigo_usina_origem"])
        destination = int(row["codigo_usina_destino"])
        stations.append(
            {
                "id": station_ids[code],
                "name": name,
                "operational_start_date": op_date,
                "bus_id": id_map.bus_id(int(row["codigo_submercado"])),
                "source_hydro_id": id_map.hydro_id(source),
                "destination_hydro_id": id_map.hydro_id(destination),
                "consumption_mw_per_m3s": float(row["consumo_estacao"]),
                "flow": {
                    "min_m3s": 0.0,
                    "max_m3s": float(row["bombeamento_maximo"]),
                },
            }
        )

    # Emit INFO diagnostic
    code_by_id = {v: k for k, v in station_ids.items()}
    exempted = sorted(case.pumping_related_codes)
    emit(
        Diagnostic(
            code="pumping-stations-converted",
            severity=Severity.INFO,
            category="Pumped storage",
            title=f"Pumping stations converted ({len(stations)})",
            summary=(
                f"Converted {len(stations)} station(s). "
                f"Hydros {exempted} exempted from fictitious filter."
            ),
            table=DiagnosticTable(
                columns=["ID", "Code", "Name", "Source", "Dest", "Bus"],
                rows=[
                    [
                        s["id"],
                        code_by_id[s["id"]],
                        s["name"],
                        s["source_hydro_id"],
                        s["destination_hydro_id"],
                        s["bus_id"],
                    ]
                    for s in stations
                ],
            ),
        )
    )

    return {
        "$schema": novomodelo_schemas.schema_url_for("system/pumping_stations.json"),
        "pumping_stations": stations,
    }


def convert_pumping_bounds(
    case: NewaveCase,
    inputs: PumpingInputs | None,
    station_ids: dict[int, int],
) -> pa.Table | None:
    """Generate constraints/pumping_bounds.parquet table.

    Returns None when inputs is None or no limit rows.
    """
    if inputs is None or inputs.limits.empty:
        return None

    stage_dates: list[date] = stage_dates_for(case.horizon)
    n_blocks = case.patamar.numero_patamares or 1

    rows: list[dict] = []
    for _, limit_row in inputs.limits.iterrows():
        code = int(limit_row["codigo_estacao"])
        if code not in station_ids:
            continue

        station_id = station_ids[code]
        patamar = int(limit_row["patamar"])

        # Validate patamar range
        if patamar < 1 or patamar > n_blocks:
            raise SourceFileError(
                f"Station {code} limit has invalid patamar {patamar} "
                f"(expected 1..{n_blocks})",
                path="estacao-bombeamento.csv",
                field="patamar",
            )
        block_id = patamar - 1

        start_dt = limit_row["data_inicio"]
        end_dt = limit_row["data_fim"]

        # Convert Timestamps to date for comparison with stage_date
        start_date = start_dt.date() if hasattr(start_dt, "date") else start_dt
        end_date = end_dt.date() if hasattr(end_dt, "date") else end_dt

        min_m3s = float(limit_row["limite_inferior"])
        max_m3s = float(limit_row["limite_superior"])

        for stage_id, stage_date in enumerate(stage_dates):
            if stage_date < start_date:
                continue
            if end_date is not None and not pd.isna(end_date) and stage_date > end_date:
                break
            rows.append(
                {
                    "pumping_station_id": station_id,
                    "stage_id": stage_id,
                    "block_id": block_id,
                    "min_m3s": min_m3s,
                    "max_m3s": max_m3s,
                }
            )

    if not rows:
        return None

    rows.sort(key=lambda r: (r["pumping_station_id"], r["stage_id"], r["block_id"]))
    return pa.Table.from_pylist(rows, schema=PUMPING_BOUNDS_SCHEMA)

"""The auxiliary coupling plants' entries in the converted case's files.

Each auxiliary (:class:`~novomodelo_bridge.decomp.fcf.coupling.CouplingAuxiliary`)
is a hydro with no storage, turbining, generation or productivity, no plant
downstream and free spillage, so whatever flows in spills. Its inflow is 0
before the last stage and its planned inflow at that stage, and its
recent observations are 0, so its lag state at the boundary is the last
stage's share of that inflow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from novomodelo_bridge.core import diagnostics as dx

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from novomodelo_bridge.decomp.fcf.coupling import CouplingAuxiliary
    from novomodelo_bridge.decomp.id_map import DecompIdMap


def with_auxiliary_hydros(
    hydros_doc: dict,
    auxiliaries: Sequence[CouplingAuxiliary],
    *,
    id_map: DecompIdMap,
    submercados: Mapping[int, int],
    operational_start_date: str,
) -> dict:
    """``hydros_doc`` with each auxiliary appended, leaving ``hydros_doc`` as is.

    An auxiliary sits on its plant's bus; for a cut plant not in the case,
    on the bus of its ``hidr.dat`` subsystem, else the first bus. It
    generates nothing, so the bus does not change the results.
    """
    if not auxiliaries:
        return hydros_doc
    bus_of_hydro = {
        hydro["id"]: hydro["unit_groups"][0]["bus_id"]
        for hydro in hydros_doc["hydros"]
        if hydro.get("unit_groups")
    }
    converted = set(id_map.hydro_codes)

    def bus_id(code: int) -> int:
        if code in converted:
            return bus_of_hydro.get(id_map.hydro_id(code), 0)
        submercado = submercados.get(code)
        return (
            id_map.bus_id(submercado)
            if submercado is not None and submercado in id_map.bus_codes
            else 0
        )

    entries = [
        {
            "id": auxiliary.hydro_id,
            "name": auxiliary.name,
            "operational_start_date": operational_start_date,
            "downstream_id": None,
            "reservoir": {"min_storage_hm3": 0.0, "max_storage_hm3": 0.0},
            "outflow": {"min_outflow_m3s": 0.0, "max_outflow_m3s": None},
            "generation": {
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 0.0,
                "min_generation_mw": 0.0,
                "max_generation_mw": 0.0,
            },
            "unit_groups": [
                {
                    "id": 0,
                    "name": auxiliary.name,
                    "bus_id": bus_id(auxiliary.plant_code),
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 0.0,
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 0.0,
                }
            ],
            "penalties": {"spillage_cost": 0.0},
        }
        for auxiliary in auxiliaries
    ]
    return {**hydros_doc, "hydros": [*hydros_doc["hydros"], *entries]}


def auxiliary_production_models(
    auxiliaries: Sequence[CouplingAuxiliary],
) -> list[dict]:
    """``hydro_production_models.json`` entries: constant productivity throughout."""
    return [
        {
            "hydro_id": auxiliary.hydro_id,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
                {
                    "start_stage_id": 0,
                    "end_stage_id": None,
                    "model": "constant_productivity",
                }
            ],
        }
        for auxiliary in auxiliaries
    ]


def with_auxiliary_productivity(
    table: pa.Table, auxiliaries: Sequence[CouplingAuxiliary]
) -> pa.Table:
    """``hydro_energy_productivity.parquet`` with a zero productivity per auxiliary."""
    rows = [
        {
            "hydro_id": auxiliary.hydro_id,
            "stage_id": None,
            "equivalent_productivity_mw_per_m3s": 0.0,
            "reference_outflow_m3s": None,
            "specific_productivity_mw_per_m3s_per_m": None,
            "integrated_equivalent_productivity_mw_per_m3s": 0.0,
        }
        for auxiliary in auxiliaries
    ]
    return _appended(table, rows)


def with_auxiliary_inflow_stats(
    table: pa.Table, auxiliaries: Sequence[CouplingAuxiliary]
) -> pa.Table:
    """``inflow_seasonal_stats.parquet`` with the identity row (mean 0, std 1)
    of every stage the table has, per auxiliary."""
    stage_ids = sorted(set(table.column("stage_id").to_pylist()))
    rows = [
        {
            "hydro_id": auxiliary.hydro_id,
            "stage_id": stage_id,
            "mean_m3s": 0.0,
            "std_m3s": 1.0,
        }
        for auxiliary in auxiliaries
        for stage_id in stage_ids
    ]
    return _appended(table, rows)


def with_auxiliary_external_inflows(
    table: pa.Table, auxiliaries: Sequence[CouplingAuxiliary], n_stages: int
) -> pa.Table:
    """``external_inflow_scenarios.parquet`` with each auxiliary's inflow: 0 on
    every stage before the last, its planned inflow per scenario at the last."""
    if not auxiliaries:
        return table
    terminal = n_stages - 1
    rows = [
        {
            "stage_id": stage_id,
            "scenario_id": 0,
            "hydro_id": auxiliary.hydro_id,
            "value_m3s": 0.0,
        }
        for auxiliary in auxiliaries
        for stage_id in range(terminal)
    ] + [
        {
            "stage_id": terminal,
            "scenario_id": scenario_id,
            "hydro_id": auxiliary.hydro_id,
            "value_m3s": value,
        }
        for auxiliary in auxiliaries
        for scenario_id, value in enumerate(auxiliary.terminal_inflows_m3s)
    ]
    return _appended(table, rows).sort_by(
        [
            ("stage_id", "ascending"),
            ("scenario_id", "ascending"),
            ("hydro_id", "ascending"),
        ]
    )


def auxiliary_initial_storage(
    auxiliaries: Sequence[CouplingAuxiliary],
) -> list[dict]:
    """``initial_conditions.json`` storage entries: every auxiliary starts empty."""
    return [
        {"hydro_id": auxiliary.hydro_id, "value_hm3": 0.0} for auxiliary in auxiliaries
    ]


def auxiliary_observation_windows(
    windows: Sequence[dict], auxiliaries: Sequence[CouplingAuxiliary]
) -> list[dict]:
    """Zero ``recent_observations`` for each auxiliary, on the windows the
    converted hydros' observations use."""
    spans = sorted({(window["start_date"], window["end_date"]) for window in windows})
    return [
        {
            "hydro_id": auxiliary.hydro_id,
            "start_date": start,
            "end_date": end,
            "value_m3s": 0.0,
        }
        for auxiliary in auxiliaries
        for start, end in spans
    ]


def coupling_auxiliary_diagnostic(
    auxiliaries: Sequence[CouplingAuxiliary], names: Mapping[int, str]
) -> dx.Diagnostic:
    """The note listing the plants that take an auxiliary coupling plant."""
    return dx.Diagnostic(
        code="boundary-fcf-coupling-auxiliary-plants",
        severity=dx.Severity.INFO,
        category="Boundary FCF",
        title=(
            "Auxiliary plants carry the boundary cuts' coupling inflow "
            f"({len(auxiliaries)})"
        ),
        summary=(
            "DECOMP evaluates the boundary cuts' inflow terms of the listed "
            "plants on a coupling posto whose last-stage inflow differs from "
            "the inflow the converted case carries. Each plant gets an "
            "auxiliary plant in system/hydros.json, with no storage or "
            "generation and free spillage, whose last-stage inflow carries that "
            "difference into the cuts, so the boundary cost is DECOMP's. Reports "
            "leave the auxiliary plants out."
        ),
        table=dx.DiagnosticTable(
            columns=[
                "Code",
                "Name",
                "Coupling posto",
                "Operation posto",
                "Auxiliary ID",
                "Max gap (m³/s)",
            ],
            rows=[
                [
                    auxiliary.plant_code,
                    names.get(auxiliary.plant_code, ""),
                    auxiliary.coupling_posto,
                    "-"
                    if auxiliary.operation_posto is None
                    else auxiliary.operation_posto,
                    auxiliary.hydro_id,
                    round(auxiliary.max_gap_m3s, 3),
                ]
                for auxiliary in auxiliaries
            ],
            justify=["right", "left", "right", "right", "right", "right"],
        ),
    )


def _appended(table: pa.Table, rows: list[dict]) -> pa.Table:
    if not rows:
        return table
    return pa.concat_tables([table, pa.Table.from_pylist(rows, schema=table.schema)])

"""Synthetic Novomodelo output writers for tests.

Each helper writes a small, hand-built slice of a Novomodelo case (simulation
partitions, ``stages.json``, ``system/`` entity lists, corrupt parquet files)
under a caller-supplied directory. :func:`build_solved_case` writes a whole
case-level slice from one :class:`SolvedCaseSpec`, and :func:`drop_partition`,
:func:`drop_column` and :func:`corrupt_file` damage the output it built.
Nothing here reads ``example/``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from novomodelo_bridge.novomodelo.cost_categories import (
    NOVOMODELO_COST_COMPONENT_COLUMNS,
)
from tests.conftest import hydro_with_group

GARBAGE_BYTES = b"this is definitely not a valid parquet file -- no PAR1 footer"


def write_corrupt_sim_parquet(output_dir: Path, entity: str) -> Path:
    """Create ``output/simulation/<entity>/scenario_id=0000/data.parquet``
    containing non-parquet garbage and return its path.
    """
    p = output_dir / "simulation" / entity / "scenario_id=0000" / "data.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(GARBAGE_BYTES)
    return p


def write_hydro_bus_generation_partition(
    output_dir: Path,
    scenario_id: int,
    rows: list[dict[str, object]],
) -> Path:
    """Write a synthetic ``hydro_bus_generation`` scenario partition with pyarrow.

    *rows* are dicts with keys ``stage_id``, ``block_id`` (``None`` allowed
    for a stage-level row), ``hydro_id``, ``bus_id``, ``turbined_m3s``,
    ``generation_mw``, ``generation_mwh``. Returns the written
    ``data.parquet`` path.
    """
    part_dir = (
        output_dir
        / "simulation"
        / "hydro_bus_generation"
        / f"scenario_id={scenario_id:04d}"
    )
    part_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "stage_id": pa.array([r["stage_id"] for r in rows], type=pa.int32()),
            "block_id": pa.array([r["block_id"] for r in rows], type=pa.int32()),
            "hydro_id": pa.array([r["hydro_id"] for r in rows], type=pa.int32()),
            "bus_id": pa.array([r["bus_id"] for r in rows], type=pa.int32()),
            "turbined_m3s": pa.array(
                [r["turbined_m3s"] for r in rows], type=pa.float64()
            ),
            "generation_mw": pa.array(
                [r["generation_mw"] for r in rows], type=pa.float64()
            ),
            "generation_mwh": pa.array(
                [r["generation_mwh"] for r in rows], type=pa.float64()
            ),
        }
    )
    path = part_dir / "data.parquet"
    pq.write_table(table, path)
    return path


def write_sim_entity_partition(
    output_dir: Path,
    entity: str,
    scenario_id: int,
    rows: list[dict[str, object]],
) -> Path:
    """Write a synthetic hive-partitioned ``simulation/<entity>`` parquet.

    Column dtypes are inferred by pyarrow from the Python values in *rows*
    (unlike :func:`write_hydro_bus_generation_partition`, which pins the
    hydro_bus_generation partition's exact writer dtypes) -- fine here since these
    fixtures exercise the unchanged thermal/NCS/bus map-and-join branches,
    not the dtype-sensitive new hydro path.
    """
    part_dir = output_dir / "simulation" / entity / f"scenario_id={scenario_id:04d}"
    part_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    path = part_dir / "data.parquet"
    pq.write_table(table, path)
    return path


def write_stages_json(
    case_dir: Path, stage_blocks: dict[int, list[tuple[int, float]]]
) -> Path:
    """Write a minimal ``stages.json`` at the case-dir level.

    *stage_blocks* maps ``stage_id -> [(block_id, hours), ...]``.
    """
    data = {
        "stages": [
            {
                "id": stage_id,
                "blocks": [{"id": bid, "hours": hours} for bid, hours in blocks],
            }
            for stage_id, blocks in stage_blocks.items()
        ]
    }
    path = case_dir / "stages.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_entity_bus_json(
    case_dir: Path, entity: str, entries: list[dict[str, int]]
) -> Path:
    """Write a minimal ``system/<entity>.json`` with ``id``/``bus_id`` entries."""
    path = case_dir / "system" / f"{entity}.json"
    _write_json(path, {entity: entries})
    return path


def write_minimal_hydros_json(case_dir: Path, hydros: list[dict[str, object]]) -> Path:
    """Write a minimal ``system/hydros.json`` (physics fields only, no bus)."""
    path = case_dir / "system" / "hydros.json"
    _write_json(path, {"hydros": hydros})
    return path


def write_corrupt_parquet(path: Path) -> Path:
    """Create a corrupt parquet at *path* (parents made) and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(GARBAGE_BYTES)
    return path


@dataclass(frozen=True)
class SolvedCaseSpec:
    """What :func:`build_solved_case` writes.

    Entity ids are ``id_offset`` plus the entity's position in its tuple, so two
    specs with equal names and different offsets disagree on every id. Every
    hydro sits on the first bus and every thermal on the last. *lines* holds
    ``(line name, source bus name, target bus name)``. ``"pre_0_14"`` selects
    the convergence columns of novomodelo before 0.14. *scale* multiplies the
    convergence bounds and their standard deviation, and every simulated
    flow, price and cost. A simulated value depends on the entity's position,
    never its id. *scenarios* is the number of simulated scenarios, all of
    equal weight. *fpha_planes* writes two production hyperplanes per hydro
    and stage.
    *load_stats* writes the case's seasonal load statistics, one row per bus
    and stage, and *non_controllables* one source per name on the first bus
    with its simulated generation; both are off by default.
    """

    hydros: tuple[str, ...] = ("FURNAS", "ITAIPU")
    thermals: tuple[str, ...] = ("GAS_A",)
    buses: tuple[str, ...] = ("SUDESTE", "SUL")
    lines: tuple[tuple[str, str, str], ...] = (("SE-S", "SUDESTE", "SUL"),)
    id_offset: int = 0
    stages: int = 2
    iterations: int = 3
    scenarios: int = 2
    scale: float = 1.0
    fpha_planes: bool = True
    convergence_schema: Literal["current", "pre_0_14"] = "current"
    upper_bound_kind: str = "statistical"
    software_version: str | None = "0.21.0"
    version_key: str = "software_version"
    risk_measure: str | dict[str, Any] = "expectation"
    load_stats: bool = False
    non_controllables: tuple[str, ...] = ()


_VERSION_KEYS = ("software_version", "cobre_version")
_CONVERGENCE_SCHEMAS = ("current", "pre_0_14")
_BLOCK_HOURS = 744.0


def build_solved_case(case_dir: Path, spec: SolvedCaseSpec = SolvedCaseSpec()) -> Path:
    """Write the case-level files, training files, simulation partitions and
    FPHA hyperplanes of a solved case under *case_dir* and return
    ``case_dir / "output"``.

    Raises :class:`ValueError` before writing when *spec* names a line bus
    outside ``spec.buses`` or a ``convergence_schema`` or ``version_key`` this
    builder does not know. An entity list that is empty leaves its simulation
    partition out, as novomodelo does.
    """
    _check_spec(spec)
    output = case_dir / "output"
    _write_stages(case_dir, spec)
    _write_system(case_dir, spec)
    _write_training(output / "training", spec)
    _write_simulation(output, spec)
    if spec.fpha_planes:
        _write_fpha(output / "hydro_models", spec)
    if spec.load_stats:
        _write_load_stats(case_dir / "scenarios", spec)
    return output


def _check_spec(spec: SolvedCaseSpec) -> None:
    if spec.convergence_schema not in _CONVERGENCE_SCHEMAS:
        raise ValueError(
            f"convergence_schema must be one of {_CONVERGENCE_SCHEMAS}, "
            f"got {spec.convergence_schema!r}"
        )
    if spec.version_key not in _VERSION_KEYS:
        raise ValueError(
            f"version_key must be one of {_VERSION_KEYS}, got {spec.version_key!r}"
        )
    for name, source, target in spec.lines:
        if source not in spec.buses or target not in spec.buses:
            raise ValueError(
                f"line {name!r} joins {source!r} to {target!r}, "
                f"not both in spec.buses {spec.buses}"
            )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_stages(case_dir: Path, spec: SolvedCaseSpec) -> None:
    _write_json(
        case_dir / "stages.json",
        {
            "stages": [
                {
                    "id": stage_id,
                    "blocks": [{"id": 0, "hours": _BLOCK_HOURS}],
                    "risk_measure": spec.risk_measure,
                }
                for stage_id in range(spec.stages)
            ]
        },
    )


def _write_system(case_dir: Path, spec: SolvedCaseSpec) -> None:
    bus_ids = {name: spec.id_offset + i for i, name in enumerate(spec.buses)}
    first_bus, last_bus = bus_ids[spec.buses[0]], bus_ids[spec.buses[-1]]
    hydros = []
    for i, name in enumerate(spec.hydros):
        hydro = hydro_with_group(
            spec.id_offset + i,
            first_bus,
            name=name,
            max_generation_mw=1000.0,
            max_turbined_m3s=12000.0,
            reservoir={"min_storage_hm3": 100.0, "max_storage_hm3": 5000.0},
        )
        hydro["generation"]["productivity_mw_per_m3s"] = 0.08
        hydros.append(hydro)
    thermals = [
        {
            "id": spec.id_offset + i,
            "name": name,
            "bus_id": last_bus,
            "cost_per_mwh": 200.0,
            "generation": {"min_mw": 0.0, "max_mw": 300.0},
        }
        for i, name in enumerate(spec.thermals)
    ]
    lines = [
        {
            "id": spec.id_offset + i,
            "name": name,
            "source_bus_id": bus_ids[source],
            "target_bus_id": bus_ids[target],
        }
        for i, (name, source, target) in enumerate(spec.lines)
    ]
    system = case_dir / "system"
    _write_json(system / "hydros.json", {"hydros": hydros})
    _write_json(
        system / "buses.json",
        {"buses": [{"id": bus_ids[name], "name": name} for name in spec.buses]},
    )
    _write_json(system / "thermals.json", {"thermals": thermals})
    _write_json(system / "lines.json", {"lines": lines})
    if spec.non_controllables:
        _write_json(
            system / "non_controllable_sources.json",
            {
                "non_controllable_sources": [
                    {"id": spec.id_offset + i, "name": name, "bus_id": first_bus}
                    for i, name in enumerate(spec.non_controllables)
                ]
            },
        )


def _write_training(training_dir: Path, spec: SolvedCaseSpec) -> None:
    metadata: dict[str, Any] = {"duration_seconds": 0.3 * spec.iterations}
    if spec.software_version is not None:
        metadata[spec.version_key] = spec.software_version
    _write_json(training_dir / "metadata.json", metadata)
    pq.write_table(_convergence_table(spec), training_dir / "convergence.parquet")


def _convergence_table(spec: SolvedCaseSpec) -> pa.Table:
    n = spec.iterations
    iterations = range(1, n + 1)
    lower = [spec.scale * 1.0e9 * (1 - 0.5**k) for k in iterations]
    upper = [spec.scale * 1.0e9 * (1 + 0.5**k) for k in iterations]
    std = None if spec.upper_bound_kind == "exact" else spec.scale * 1.0e7
    current = spec.convergence_schema == "current"
    columns: dict[str, pa.Array] = {
        "iteration": pa.array(iterations, type=pa.int32()),
        "lower_bound": pa.array(lower, type=pa.float64()),
        ("upper_bound" if current else "upper_bound_mean"): pa.array(
            upper, type=pa.float64()
        ),
        "upper_bound_std": pa.array([std] * n, type=pa.float64()),
    }
    if current:
        columns["upper_bound_kind"] = pa.array(
            [spec.upper_bound_kind] * n, type=pa.string()
        )
    columns["gap_percent"] = pa.array(
        [100 * (hi - lo) / lo for lo, hi in zip(lower, upper, strict=True)],
        type=pa.float64(),
    )
    columns["time_forward_ms"] = pa.array([100] * n, type=pa.int64())
    columns["time_backward_ms"] = pa.array([200] * n, type=pa.int64())
    columns["time_total_ms"] = pa.array([300] * n, type=pa.int64())
    columns["forward_passes"] = pa.array([5] * n, type=pa.int32())
    return pa.table(columns)


#: Each ``_*_BASES`` maps a column to its value at position 0, stage 0,
#: scenario 0 and scale 1; see :func:`_pattern`.
_HYDRO_BASES: dict[str, float] = {
    "generation_mw": 400.0,
    "generation_mwh": 400.0 * _BLOCK_HOURS,
    "turbined_m3s": 5000.0,
    "spillage_m3s": 50.0,
    "evaporation_m3s": 5.0,
    "outflow_m3s": 5050.0,
    "inflow_m3s": 5100.0,
    "incremental_inflow_m3s": 600.0,
    "storage_initial_hm3": 2000.0,
    "storage_final_hm3": 2100.0,
    "water_value_per_hm3": 1.5,
    "stored_energy_initial_mwh": 3.0e6,
    "stored_energy_final_mwh": 3.1e6,
    "incremental_inflow_energy_mw": 50.0,
    "evaporation_violation_pos_m3s": 0.0,
    "evaporation_violation_neg_m3s": 0.0,
    "water_withdrawal_violation_pos_m3s": 0.0,
    "water_withdrawal_violation_neg_m3s": 0.0,
    "inflow_nonnegativity_slack_m3s": 0.0,
}
_HYDRO_BUS_COLUMNS = (
    "stage_id",
    "block_id",
    "hydro_id",
    "turbined_m3s",
    "generation_mw",
    "generation_mwh",
)
_THERMAL_BASES: dict[str, float] = {"generation_mw": 150.0}
_BUS_BASES: dict[str, float] = {
    "spot_price": 100.0,
    "deficit_mw": 0.0,
    "excess_mw": 0.0,
    "load_mw": 1000.0,
}
_EXCHANGE_BASES: dict[str, float] = {"net_flow_mw": 300.0}
_NON_CONTROLLABLE_BASES: dict[str, float] = {"generation_mw": 100.0}
_COST_BASES: dict[str, float] = {
    **dict.fromkeys(NOVOMODELO_COST_COMPONENT_COLUMNS, 1.0e3),
    "thermal_cost": 5.0e6,
}
_FUTURE_COST = 4.0e9
#: Coefficients stay out of the ``_*_BASES``: unlike a flow, price or cost,
#: they do not follow ``SolvedCaseSpec.scale``.
_EQUIVALENT_PRODUCTIVITY_MW_PER_M3S = 0.08
_ACCUMULATED_PRODUCTIVITY_MW_PER_M3S = 0.08
_DISCOUNT_PER_STAGE = 0.99
_FPHA_PLANES_PER_HYDRO_STAGE = 2
_FPHA_SCHEMA = pa.schema(
    [
        ("hydro_id", pa.int32()),
        ("stage_id", pa.int32()),
        ("plane_id", pa.int32()),
        ("gamma_0", pa.float64()),
        ("gamma_v", pa.float64()),
        ("gamma_q", pa.float64()),
        ("gamma_s", pa.float64()),
        ("kappa", pa.float64()),
    ]
)


def _pattern(
    base: float, position: int, stage: int, scenario: int, scale: float
) -> float:
    return base * (1 + position) * (1 + 0.1 * stage) * (1 + 0.05 * scenario) * scale


def _rows(
    spec: SolvedCaseSpec,
    scenario: int,
    count: int,
    id_column: str,
    bases: dict[str, float],
) -> list[dict[str, object]]:
    return [
        {
            "stage_id": stage,
            "block_id": 0,
            id_column: spec.id_offset + position,
            **{
                column: _pattern(base, position, stage, scenario, spec.scale)
                for column, base in bases.items()
            },
        }
        for stage in range(spec.stages)
        for position in range(count)
    ]


def _hydro_rows(spec: SolvedCaseSpec, scenario: int) -> list[dict[str, object]]:
    rows = _rows(spec, scenario, len(spec.hydros), "hydro_id", _HYDRO_BASES)
    return [
        {
            **row,
            "equivalent_productivity_mw_per_m3s": _EQUIVALENT_PRODUCTIVITY_MW_PER_M3S,
            "accumulated_productivity_mw_per_m3s": (
                _ACCUMULATED_PRODUCTIVITY_MW_PER_M3S
            ),
        }
        for row in rows
    ]


def _hydro_bus_generation_rows(
    spec: SolvedCaseSpec, hydro_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    return [
        {**{c: row[c] for c in _HYDRO_BUS_COLUMNS}, "bus_id": spec.id_offset}
        for row in hydro_rows
    ]


def _cost_rows(spec: SolvedCaseSpec, scenario: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stage in range(spec.stages):
        components = {
            column: _pattern(base, 0, stage, scenario, spec.scale)
            for column, base in _COST_BASES.items()
        }
        immediate = sum(components.values())
        future = _pattern(_FUTURE_COST, 0, stage, scenario, spec.scale)
        rows.append(
            {
                "stage_id": stage,
                "block_id": 0,
                **components,
                "immediate_cost": immediate,
                "future_cost": future,
                "total_cost": immediate + future,
                "discount_factor": _DISCOUNT_PER_STAGE**stage,
            }
        )
    return rows


def _write_simulation(output: Path, spec: SolvedCaseSpec) -> None:
    for scenario in range(spec.scenarios):
        hydros = _hydro_rows(spec, scenario)
        if hydros:
            write_hydro_bus_generation_partition(
                output, scenario, _hydro_bus_generation_rows(spec, hydros)
            )
        partitions = {
            "hydros": hydros,
            "thermals": _rows(
                spec, scenario, len(spec.thermals), "thermal_id", _THERMAL_BASES
            ),
            "buses": _rows(spec, scenario, len(spec.buses), "bus_id", _BUS_BASES),
            "exchanges": _rows(
                spec, scenario, len(spec.lines), "line_id", _EXCHANGE_BASES
            ),
            "non_controllables": _rows(
                spec,
                scenario,
                len(spec.non_controllables),
                "non_controllable_id",
                _NON_CONTROLLABLE_BASES,
            ),
            "costs": _cost_rows(spec, scenario),
        }
        for entity, rows in partitions.items():
            if rows:
                write_sim_entity_partition(output, entity, scenario, rows)
    pq.write_table(
        pa.table(
            {
                "scenario_id": pa.array(range(spec.scenarios), type=pa.int32()),
                "probability": pa.array([None] * spec.scenarios, type=pa.float64()),
            }
        ),
        output / "simulation" / "scenario_summary.parquet",
    )


def _write_fpha(hydro_models_dir: Path, spec: SolvedCaseSpec) -> None:
    rows = [
        {
            "hydro_id": spec.id_offset + position,
            "stage_id": stage,
            "plane_id": plane,
            "gamma_0": 10.0 * (1 + position) * (1 + 0.1 * stage) * (1 + plane),
            "gamma_v": 1.0e-3,
            "gamma_q": 0.08 * (1 + plane),
            "gamma_s": -0.01,
            "kappa": 1.0,
        }
        for stage in range(spec.stages)
        for position in range(len(spec.hydros))
        for plane in range(_FPHA_PLANES_PER_HYDRO_STAGE)
    ]
    hydro_models_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=_FPHA_SCHEMA),
        hydro_models_dir / "fpha_hyperplanes.parquet",
    )


def _write_load_stats(scenarios_dir: Path, spec: SolvedCaseSpec) -> None:
    stats = [
        (
            spec.id_offset + position,
            stage,
            _pattern(_BUS_BASES["load_mw"], position, stage, 0, spec.scale),
        )
        for stage in range(spec.stages)
        for position in range(len(spec.buses))
    ]
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "bus_id": pa.array([s[0] for s in stats], type=pa.int32()),
                "stage_id": pa.array([s[1] for s in stats], type=pa.int32()),
                "mean_mw": pa.array([s[2] for s in stats], type=pa.float64()),
                "std_mw": pa.array([0.0] * len(stats), type=pa.float64()),
            }
        ),
        scenarios_dir / "load_seasonal_stats.parquet",
    )


def _existing(output_dir: Path, relative: str) -> Path:
    root = output_dir.resolve()
    target = (output_dir / relative).resolve()
    if root not in target.parents or not target.exists():
        raise FileNotFoundError(f"no {relative!r} under {output_dir}")
    return target


def _parquet_files(target: Path) -> list[Path]:
    files = [target] if target.is_file() else sorted(target.rglob("data.parquet"))
    if not files:
        raise FileNotFoundError(f"no data.parquet below {target}")
    return files


def drop_partition(output_dir: Path, relative: str) -> None:
    """Delete the directory or file *relative* to *output_dir*, such as
    ``simulation/thermals`` or ``training/convergence.parquet``.

    Raises :class:`FileNotFoundError` when *relative* does not name a path
    below *output_dir*.
    """
    target = _existing(output_dir, relative)
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def drop_column(output_dir: Path, relative: str, column: str) -> None:
    """Rewrite without *column* the parquet file *relative* to *output_dir*, or
    every ``data.parquet`` below that directory.

    Raises :class:`FileNotFoundError` when *relative* does not name a path
    below *output_dir*, and :class:`KeyError`, before any file is rewritten,
    when a file lacks *column*.
    """
    tables = {
        path: pq.read_table(path, partitioning=None)
        for path in _parquet_files(_existing(output_dir, relative))
    }
    for path, table in tables.items():
        if column not in table.column_names:
            raise KeyError(f"{column!r} is not a column of {path}")
    for path, table in tables.items():
        pq.write_table(table.drop_columns([column]), path)


def corrupt_file(output_dir: Path, relative: str) -> None:
    """Overwrite with :data:`GARBAGE_BYTES` the parquet file *relative* to
    *output_dir*, or the first ``data.parquet`` below that directory.

    Raises :class:`FileNotFoundError` when *relative* does not name a path
    below *output_dir*.
    """
    _parquet_files(_existing(output_dir, relative))[0].write_bytes(GARBAGE_BYTES)

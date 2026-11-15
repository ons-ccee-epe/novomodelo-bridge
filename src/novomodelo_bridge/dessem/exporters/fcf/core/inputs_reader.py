"""Readers for the Novomodelo JSON inputs the conversion depends on.

Only JSON is read. Because premise P1 zero-fills mapcut record indices 4-17,
no Novomodelo Parquet artifact is needed. Stage-level cost overrides in
constraints/thermal_bounds.parquet are deliberately not read, so a thermal is
priced at its thermals.json cost_per_mwh at every stage.

Stage durations are NOT uniform in the reference deck: stages 0-5 span 168
hours each while stage 6 spans 600. Anything weighted by time must read the
per-stage block hours rather than assume a week.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

_STAGES_FILE = "stages.json"
_CONFIG_FILE = "config.json"
_INITIAL_CONDITIONS_FILE = "initial_conditions.json"
_SYSTEM_DIR = "system"
_HYDROS_FILE = "hydros.json"
_THERMALS_FILE = "thermals.json"
_BUSES_FILE = "buses.json"


class InputReadError(Exception):
    """Raised when a Novomodelo input file is missing, malformed or incomplete."""


@dataclass(frozen=True)
class BlockInfo:
    """One load block of a stage."""

    id: int
    name: str
    hours: float


@dataclass(frozen=True)
class StageInfo:
    """One study stage, its load blocks and which state variables it carries."""

    id: int
    start_date: str
    end_date: str
    blocks: tuple[BlockInfo, ...]
    storage_state: bool
    inflow_lags_state: bool


@dataclass(frozen=True)
class HydroInfo:
    """One hydro plant.

    `None` marks a terminal plant in `downstream_id`, in
    `travel_time_hours` a plant whose defluence reaches its downstream plant at
    once, and in `entry_stage_id` a plant in service from the first stage.
    """

    id: int
    name: str
    downstream_id: int | None
    min_storage_hm3: float
    max_storage_hm3: float
    travel_time_hours: float | None = None
    entry_stage_id: int | None = None


@dataclass(frozen=True)
class ThermalInfo:
    """One thermal plant. `lead_time_hours` is `None` when not GNL-capable.

    `cost_per_mwh` is in R$/MWh.
    """

    id: int
    name: str
    bus_id: int
    lead_time_hours: float | None
    cost_per_mwh: float


@dataclass(frozen=True)
class CommitmentWindow:
    """One `past_anticipated_commitments` window, `[start_date, end_date)` with the end excluded."""

    thermal_id: int
    start_date: date
    end_date: date
    value_mw: float


@dataclass(frozen=True)
class BusInfo:
    """One electrical bus."""

    id: int
    name: str


@dataclass(frozen=True)
class CaseInputs:
    """Everything the conversion reads from the case's JSON inputs.

    *initial_storage_hm3* maps a hydro id to its initial storage and is empty
    when `initial_conditions.json` declares none.
    """

    stages: tuple[StageInfo, ...]
    hydros: tuple[HydroInfo, ...]
    thermals: tuple[ThermalInfo, ...]
    buses: tuple[BusInfo, ...]
    annual_discount_rate: float
    past_anticipated_commitments: tuple[CommitmentWindow, ...]
    initial_storage_hm3: Mapping[int, float] = field(default_factory=dict)


def _load_json(path: Path) -> Any:
    """Load JSON from a file with proper error handling."""
    if not path.is_file():
        raise InputReadError(f"Novomodelo input file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputReadError(f"{path} is not valid JSON: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise InputReadError(f"{path} is not valid UTF-8: {exc}") from exc


def _require(obj: Any, key: str, dotted: str) -> Any:
    """Require a key in a dict, with path for error message."""
    if not isinstance(obj, dict):
        raise InputReadError(f"{dotted} must be an object, got {type(obj).__name__}")
    if key not in obj:
        raise InputReadError(f"missing required key {dotted}")
    return obj[key]


def _require_list(obj: Any, key: str, dotted: str) -> list[Any]:
    """Require a list value."""
    value = _require(obj, key, dotted)
    if not isinstance(value, list):
        raise InputReadError(f"{dotted} must be a list, got {type(value).__name__}")
    return value


def _require_str(obj: Any, key: str, dotted: str) -> str:
    """Require a string value."""
    value = _require(obj, key, dotted)
    if not isinstance(value, str):
        raise InputReadError(f"{dotted} must be str, got {type(value).__name__}")
    return value


def _require_int(obj: Any, key: str, dotted: str) -> int:
    """Require an integer value (not bool)."""
    value = _require(obj, key, dotted)
    if type(value) is not int:
        raise InputReadError(f"{dotted} must be int, got {type(value).__name__}")
    return value


def _require_float(obj: Any, key: str, dotted: str) -> float:
    """Require a finite numeric value."""
    value = _require(obj, key, dotted)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputReadError(f"{dotted} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise InputReadError(f"{dotted} must be a finite number, got {value!r}")
    return number


def _require_iso_date(obj: Any, key: str, dotted: str) -> date:
    """Require an ISO-8601 date string."""
    value = _require_str(obj, key, dotted)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise InputReadError(
            f"{dotted} must be an ISO-8601 date, got {value!r}"
        ) from exc


def _require_bool(obj: Any, key: str, dotted: str) -> bool:
    """Require a boolean value."""
    value = _require(obj, key, dotted)
    if type(value) is not bool:
        raise InputReadError(f"{dotted} must be bool, got {type(value).__name__}")
    return value


def _read_blocks(raw_stage: Any, stage_dotted: str) -> tuple[BlockInfo, ...]:
    """Read load blocks from a stage object."""
    blocks = _require_list(raw_stage, "blocks", f"{stage_dotted}.blocks")
    return tuple(
        BlockInfo(
            id=_require_int(block, "id", f"{stage_dotted}.blocks[{pos}].id"),
            name=_require_str(block, "name", f"{stage_dotted}.blocks[{pos}].name"),
            hours=_require_float(block, "hours", f"{stage_dotted}.blocks[{pos}].hours"),
        )
        for pos, block in enumerate(blocks)
    )


def _read_stages(path: Path) -> tuple[tuple[StageInfo, ...], float]:
    """Read stages.json and extract stages and discount rate."""
    raw = _load_json(path)
    policy_graph = _require(raw, "policy_graph", "policy_graph")
    discount_rate = _require_float(
        policy_graph, "annual_discount_rate", "policy_graph.annual_discount_rate"
    )

    stages = []
    for pos, raw_stage in enumerate(_require_list(raw, "stages", "stages")):
        dotted = f"stages[{pos}]"
        state_variables = _require(
            raw_stage, "state_variables", f"{dotted}.state_variables"
        )
        stages.append(
            StageInfo(
                id=_require_int(raw_stage, "id", f"{dotted}.id"),
                start_date=_require_str(
                    raw_stage, "start_date", f"{dotted}.start_date"
                ),
                end_date=_require_str(raw_stage, "end_date", f"{dotted}.end_date"),
                blocks=_read_blocks(raw_stage, dotted),
                storage_state=_require_bool(
                    state_variables, "storage", f"{dotted}.state_variables.storage"
                ),
                inflow_lags_state=_require_bool(
                    state_variables,
                    "inflow_lags",
                    f"{dotted}.state_variables.inflow_lags",
                ),
            )
        )
    return tuple(stages), discount_rate


def _read_hydros(path: Path) -> tuple[HydroInfo, ...]:
    """Read hydros.json."""
    raw = _load_json(path)
    hydros = []
    for pos, entry in enumerate(_require_list(raw, "hydros", "hydros")):
        dotted = f"hydros[{pos}]"
        reservoir = _require(entry, "reservoir", f"{dotted}.reservoir")
        downstream = entry.get("downstream_id")
        if downstream is not None and type(downstream) is not int:
            raise InputReadError(
                f"{dotted}.downstream_id must be int or null, "
                f"got {type(downstream).__name__}"
            )
        travel_time = (
            _require_float(entry, "travel_time_hours", f"{dotted}.travel_time_hours")
            if entry.get("travel_time_hours") is not None
            else None
        )
        if travel_time is not None and travel_time < 0.0:
            raise InputReadError(
                f"{dotted}.travel_time_hours must be non-negative, got {travel_time}"
            )
        entry_stage = (
            _require_int(entry, "entry_stage_id", f"{dotted}.entry_stage_id")
            if entry.get("entry_stage_id") is not None
            else None
        )
        if entry_stage is not None and entry_stage < 0:
            raise InputReadError(
                f"{dotted}.entry_stage_id must be non-negative, got {entry_stage}"
            )
        hydros.append(
            HydroInfo(
                id=_require_int(entry, "id", f"{dotted}.id"),
                name=_require_str(entry, "name", f"{dotted}.name"),
                downstream_id=downstream,
                min_storage_hm3=_require_float(
                    reservoir,
                    "min_storage_hm3",
                    f"{dotted}.reservoir.min_storage_hm3",
                ),
                max_storage_hm3=_require_float(
                    reservoir,
                    "max_storage_hm3",
                    f"{dotted}.reservoir.max_storage_hm3",
                ),
                travel_time_hours=travel_time,
                entry_stage_id=entry_stage,
            )
        )
    return tuple(hydros)


def _read_thermals(path: Path) -> tuple[ThermalInfo, ...]:
    """Read thermals.json."""
    if not path.exists():
        return ()
    raw = _load_json(path)
    thermals = []
    for pos, entry in enumerate(_require_list(raw, "thermals", "thermals")):
        dotted = f"thermals[{pos}]"
        anticipated = (
            entry.get("anticipated_config") if isinstance(entry, dict) else None
        )
        lead_time = (
            None
            if anticipated is None
            else _require_float(
                anticipated,
                "lead_time_hours",
                f"{dotted}.anticipated_config.lead_time_hours",
            )
        )
        cost = _require_float(entry, "cost_per_mwh", f"{dotted}.cost_per_mwh")
        if cost < 0.0:
            raise InputReadError(
                f"{dotted}.cost_per_mwh must be non-negative, got {cost}"
            )
        thermals.append(
            ThermalInfo(
                id=_require_int(entry, "id", f"{dotted}.id"),
                name=_require_str(entry, "name", f"{dotted}.name"),
                bus_id=_require_int(entry, "bus_id", f"{dotted}.bus_id"),
                lead_time_hours=lead_time,
                cost_per_mwh=cost,
            )
        )
    return tuple(thermals)


def _read_buses(path: Path) -> tuple[BusInfo, ...]:
    """Read buses.json."""
    raw = _load_json(path)
    return tuple(
        BusInfo(
            id=_require_int(entry, "id", f"buses[{pos}].id"),
            name=_require_str(entry, "name", f"buses[{pos}].name"),
        )
        for pos, entry in enumerate(_require_list(raw, "buses", "buses"))
    )


def _read_initial_conditions(
    path: Path,
) -> tuple[tuple[CommitmentWindow, ...], dict[int, float]]:
    """Read the past_anticipated_commitments windows and the storage of initial_conditions.json."""
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise InputReadError(
            f"{path} must hold a JSON object, got {type(raw).__name__}"
        )
    return _read_commitment_windows(raw), _read_initial_storage(raw)


def _read_commitment_windows(raw: dict[str, Any]) -> tuple[CommitmentWindow, ...]:
    if "past_anticipated_commitments" not in raw:
        return ()
    windows = []
    for pos, entry in enumerate(
        _require_list(
            raw, "past_anticipated_commitments", "past_anticipated_commitments"
        )
    ):
        dotted = f"past_anticipated_commitments[{pos}]"
        thermal_id = _require_int(entry, "thermal_id", f"{dotted}.thermal_id")
        start = _require_iso_date(entry, "start_date", f"{dotted}.start_date")
        end = _require_iso_date(entry, "end_date", f"{dotted}.end_date")
        value_mw = _require_float(entry, "value_mw", f"{dotted}.value_mw")
        if end <= start:
            raise InputReadError(
                f"{dotted}.end_date {end} is not after start_date {start}"
            )
        windows.append(
            CommitmentWindow(
                thermal_id=thermal_id,
                start_date=start,
                end_date=end,
                value_mw=value_mw,
            )
        )
    return tuple(windows)


def _read_initial_storage(raw: dict[str, Any]) -> dict[int, float]:
    if "storage" not in raw:
        return {}
    storage: dict[int, float] = {}
    for pos, entry in enumerate(_require_list(raw, "storage", "storage")):
        dotted = f"storage[{pos}]"
        hydro_id = _require_int(entry, "hydro_id", f"{dotted}.hydro_id")
        value_hm3 = _require_float(entry, "value_hm3", f"{dotted}.value_hm3")
        if hydro_id in storage:
            raise InputReadError(
                f"{dotted}.hydro_id {hydro_id} repeats an earlier storage entry"
            )
        if value_hm3 < 0.0:
            raise InputReadError(
                f"{dotted}.value_hm3 must be non-negative, got {value_hm3}"
            )
        storage[hydro_id] = value_hm3
    return storage


def anticipated_thermals(inputs: CaseInputs) -> tuple[ThermalInfo, ...]:
    """Return the GNL-capable thermals, ascending by id."""
    return tuple(
        sorted(
            (t for t in inputs.thermals if t.lead_time_hours is not None),
            key=lambda t: t.id,
        )
    )


def read_case_inputs(case_dir: Path) -> CaseInputs:
    """Read stages.json, initial_conditions.json and the system/ inputs the conversion needs."""
    case_dir = case_dir.resolve()
    system_dir = case_dir / _SYSTEM_DIR
    stages, annual_discount_rate = _read_stages(case_dir / _STAGES_FILE)
    commitments, initial_storage = _read_initial_conditions(
        case_dir / _INITIAL_CONDITIONS_FILE
    )
    inputs = CaseInputs(
        stages=stages,
        hydros=_read_hydros(system_dir / _HYDROS_FILE),
        thermals=_read_thermals(system_dir / _THERMALS_FILE),
        buses=_read_buses(system_dir / _BUSES_FILE),
        annual_discount_rate=annual_discount_rate,
        past_anticipated_commitments=commitments,
        initial_storage_hm3=initial_storage,
    )

    _LOG.info(
        "read case inputs %s: stages=%d hydros=%d thermals=%d buses=%d "
        "anticipated_thermals=%d annual_discount_rate=%s commitment_windows=%d",
        case_dir,
        len(inputs.stages),
        len(inputs.hydros),
        len(inputs.thermals),
        len(inputs.buses),
        len(anticipated_thermals(inputs)),
        inputs.annual_discount_rate,
        len(inputs.past_anticipated_commitments),
    )
    terminal = sum(1 for h in inputs.hydros if h.downstream_id is None)
    _LOG.info("hydro topology: %d terminal plants with no downstream", terminal)
    durations = {sum(b.hours for b in stage.blocks) for stage in inputs.stages}
    if len(durations) > 1:
        _LOG.info(
            "stage durations are not uniform: %s hours",
            sorted(durations),
        )
    return inputs

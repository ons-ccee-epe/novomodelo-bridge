"""Readers for the Novomodelo JSON inputs the conversion depends on.

Only JSON is read. Because premise P1 zero-fills mapcut record indices 4-17,
no Novomodelo Parquet artifact is needed.

Stage durations are NOT uniform in the reference deck: stages 0-5 span 168
hours each while stage 6 spans 600. Anything weighted by time must read the
per-stage block hours rather than assume a week.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

_STAGES_FILE = "stages.json"
_CONFIG_FILE = "config.json"
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

    `None` marks a terminal plant in `downstream_id`, and in
    `travel_time_hours` a plant whose defluence reaches its downstream plant at
    once.
    """

    id: int
    name: str
    downstream_id: int | None
    min_storage_hm3: float
    max_storage_hm3: float
    travel_time_hours: float | None = None


@dataclass(frozen=True)
class ThermalInfo:
    """One thermal plant. `lead_time_hours` is `None` when not GNL-capable."""

    id: int
    name: str
    bus_id: int
    lead_time_hours: float | None


@dataclass(frozen=True)
class BusInfo:
    """One electrical bus."""

    id: int
    name: str


@dataclass(frozen=True)
class CaseInputs:
    """Everything the conversion reads from the case's JSON inputs."""

    stages: tuple[StageInfo, ...]
    hydros: tuple[HydroInfo, ...]
    thermals: tuple[ThermalInfo, ...]
    buses: tuple[BusInfo, ...]
    annual_discount_rate: float


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
        downstream = entry.get("downstream_id") if isinstance(entry, dict) else None
        if downstream is not None and type(downstream) is not int:
            raise InputReadError(
                f"{dotted}.downstream_id must be int or null, "
                f"got {type(downstream).__name__}"
            )
        travel_time = (
            _require_float(entry, "travel_time_hours", f"{dotted}.travel_time_hours")
            if isinstance(entry, dict) and entry.get("travel_time_hours") is not None
            else None
        )
        if travel_time is not None and travel_time < 0.0:
            raise InputReadError(
                f"{dotted}.travel_time_hours must be non-negative, got {travel_time}"
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
            )
        )
    return tuple(hydros)


def _read_thermals(path: Path) -> tuple[ThermalInfo, ...]:
    """Read thermals.json."""
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
        thermals.append(
            ThermalInfo(
                id=_require_int(entry, "id", f"{dotted}.id"),
                name=_require_str(entry, "name", f"{dotted}.name"),
                bus_id=_require_int(entry, "bus_id", f"{dotted}.bus_id"),
                lead_time_hours=lead_time,
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


def anticipated_thermals(inputs: CaseInputs) -> tuple[ThermalInfo, ...]:
    """Return the GNL-capable thermals, ascending by id."""
    return tuple(
        sorted(
            (t for t in inputs.thermals if t.lead_time_hours is not None),
            key=lambda t: t.id,
        )
    )


def read_case_inputs(case_dir: Path) -> CaseInputs:
    """Read stages.json and the system/ inputs the conversion needs."""
    case_dir = case_dir.resolve()
    system_dir = case_dir / _SYSTEM_DIR
    stages, annual_discount_rate = _read_stages(case_dir / _STAGES_FILE)
    inputs = CaseInputs(
        stages=stages,
        hydros=_read_hydros(system_dir / _HYDROS_FILE),
        thermals=_read_thermals(system_dir / _THERMALS_FILE),
        buses=_read_buses(system_dir / _BUSES_FILE),
        annual_discount_rate=annual_discount_rate,
    )

    _LOG.info(
        "read case inputs %s: stages=%d hydros=%d thermals=%d buses=%d "
        "anticipated_thermals=%d annual_discount_rate=%s",
        case_dir,
        len(inputs.stages),
        len(inputs.hydros),
        len(inputs.thermals),
        len(inputs.buses),
        len(anticipated_thermals(inputs)),
        inputs.annual_discount_rate,
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

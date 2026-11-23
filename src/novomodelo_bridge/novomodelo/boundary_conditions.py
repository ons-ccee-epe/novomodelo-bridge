"""Plant-keyed ``boundary_conditions.json`` and the post-study calendars of ``stages.json``.

Reshapes the per-kind dicts the conversion tracks build (initial conditions and,
for DECOMP, the post-study calendars) into ``boundary_conditions.json`` and the
``post_study_thermal_stages`` and ``post_study_water_stages`` lists of
``stages.json``. Pure: it writes no file. Plants are sorted by id and the rows of
a plant keep their input order, so the result does not depend on how the caller
ordered its rows.

A violated precondition is a bridge bug, not a deck problem, so every refusal is
a bare ``ValueError`` naming the offending key, id or row.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from novomodelo_bridge.novomodelo.schemas import schema_url_for

_INITIAL_KEYS = frozenset(
    {
        "$schema",
        "storage",
        "filling_storage",
        "recent_observations",
        "past_defluences",
        "past_anticipated_commitments",
    }
)
_POST_STUDY_KEYS = frozenset(
    {"thermal_stages", "thermal_bounds", "water_bucket_stages"}
)
_HOURS_PER_DAY = 24
_FLOW_FIELDS = ("start_date", "end_date", "value_m3s")
_COMMITMENT_FIELDS = ("start_date", "end_date", "value_mw")
_HYDRO_FIELDS = ("storage_hm3", "recent_inflows", "past_defluences")


@dataclass(frozen=True)
class BoundaryFiles:
    boundary_conditions: dict[str, Any]
    stages: dict[str, Any]


def _refuse_unknown_keys(
    name: str, doc: Mapping[str, Any], allowed: frozenset[str]
) -> None:
    if unknown := sorted(doc.keys() - allowed):
        raise ValueError(f"{name}: unexpected keys {unknown}")


def _group(
    rows: Iterable[Mapping[str, Any]], id_field: str, fields: Sequence[str]
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row[id_field], []).append(
            {name: row[name] for name in fields}
        )
    return grouped


def _entries(
    id_field: str, sources: Mapping[str, Mapping[int, Any]]
) -> list[dict[str, Any]]:
    """One entry per plant id found in any source, ascending; a key is omitted without data."""
    ids = sorted(set[int]().union(*sources.values()))
    return [
        {
            id_field: plant_id,
            **{key: src[plant_id] for key, src in sources.items() if plant_id in src},
        }
        for plant_id in ids
    ]


def _storage_seeds(initial_conditions: Mapping[str, Any]) -> dict[int, float]:
    seeds: dict[int, float] = {}
    for array in ("storage", "filling_storage"):
        for row in initial_conditions.get(array, []):
            if row["hydro_id"] in seeds:
                raise ValueError(
                    f"initial_conditions: hydro {row['hydro_id']} has more than one "
                    "storage row across `storage` and `filling_storage`"
                )
            seeds[row["hydro_id"]] = row["value_hm3"]
    return seeds


def _first_post_study_id(stages: Mapping[str, Any]) -> int:
    return 1 + max(
        (
            stage["id"]
            for stage in stages["stages"]
            if isinstance(stage.get("id"), int) and stage["id"] >= 0
        ),
        default=-1,
    )


def _calendar(
    rows: Sequence[Mapping[str, Any]], key: str, first_id: int
) -> list[dict[str, Any]]:
    calendar: list[dict[str, Any]] = []
    previous_start: date | None = None
    for k, row in enumerate(rows):
        start = date.fromisoformat(row["start_date"])
        hours = row["duration_hours"]
        if hours <= 0 or hours % _HOURS_PER_DAY != 0:
            raise ValueError(
                f"post_study: `{key}[{k}]` starting {start} lasts {hours:g} h, which "
                f"is not a positive multiple of {_HOURS_PER_DAY} h"
            )
        if previous_start is not None and start <= previous_start:
            raise ValueError(
                f"post_study: `{key}[{k}]` starts {start}, not after the previous "
                f"stage's {previous_start}"
            )
        previous_start = start
        end = start + timedelta(days=int(hours) // _HOURS_PER_DAY)
        calendar.append(
            {
                "id": first_id + k,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            }
        )
    return calendar


def _post_study_bounds(
    rows: Sequence[Mapping[str, Any]], stage_count: int, first_id: int
) -> dict[int, list[dict[str, Any]]]:
    for row in rows:
        index = row["post_study_stage_index"]
        if not 0 <= index < stage_count:
            raise ValueError(
                f"post_study: `thermal_bounds` of thermal {row['thermal_id']} has "
                f"`post_study_stage_index` {index}, outside `thermal_stages` of "
                f"length {stage_count}"
            )
    return _group(
        sorted(
            (
                {**row, "stage_id": first_id + row["post_study_stage_index"]}
                for row in rows
            ),
            key=lambda row: row["stage_id"],
        ),
        "thermal_id",
        ("stage_id", "cost_per_mwh", "min_mw", "max_mw"),
    )


def plant_keyed_boundary(
    initial_conditions: Mapping[str, Any],
    post_study: Mapping[str, Any] | None,
    stages: Mapping[str, Any],
) -> BoundaryFiles:
    """Reshape the per-kind dicts into ``boundary_conditions.json`` and ``stages.json``.

    ``post_study`` is ``None`` for a track without post-study calendars. Neither
    input is mutated; the returned ``stages`` is ``stages`` plus the post-study
    lists the calendars declare, their stage ids continuing the study's.
    """
    post_study = post_study or {}
    _refuse_unknown_keys("initial_conditions", initial_conditions, _INITIAL_KEYS)
    _refuse_unknown_keys("post_study", post_study, _POST_STUDY_KEYS)
    for declared, missing in (
        ("thermal_stages", "thermal_bounds"),
        ("thermal_bounds", "thermal_stages"),
    ):
        if declared in post_study and missing not in post_study:
            raise ValueError(
                f"post_study: `{declared}` is declared without `{missing}`"
            )

    first_id = _first_post_study_id(stages)
    thermal_rows = post_study.get("thermal_stages", [])
    lists: dict[str, list[dict[str, Any]]] = {}
    if "thermal_stages" in post_study:
        lists["post_study_thermal_stages"] = _calendar(
            thermal_rows, "thermal_stages", first_id
        )
    if "water_bucket_stages" in post_study:
        lists["post_study_water_stages"] = _calendar(
            post_study["water_bucket_stages"],
            "water_bucket_stages",
            first_id + len(thermal_rows),
        )

    hydros = _entries(
        "hydro_id",
        {
            "storage_hm3": _storage_seeds(initial_conditions),
            "recent_inflows": _group(
                initial_conditions.get("recent_observations", []),
                "hydro_id",
                _FLOW_FIELDS,
            ),
            "past_defluences": _group(
                initial_conditions.get("past_defluences", []),
                "hydro_id",
                _FLOW_FIELDS,
            ),
        },
    )
    thermals = _entries(
        "thermal_id",
        {
            "commitments": _group(
                initial_conditions.get("past_anticipated_commitments", []),
                "thermal_id",
                _COMMITMENT_FIELDS,
            ),
            "post_study_bounds": _post_study_bounds(
                post_study.get("thermal_bounds", []), len(thermal_rows), first_id
            ),
        },
    )
    return BoundaryFiles(
        boundary_conditions={
            "$schema": schema_url_for("boundary_conditions.json"),
            "hydros": hydros,
            "thermals": thermals,
        },
        stages={**stages, **lists},
    )


def add_recent_inflows(
    boundary_conditions: dict[str, Any], windows: Sequence[Mapping[str, Any]]
) -> None:
    """Add ``recent_inflows`` to ``boundary_conditions["hydros"]`` in place.

    ``windows`` rows are ``{hydro_id, start_date, end_date, value_m3s}``. A hydro
    absent so far is inserted at its ascending-id position; the result equals
    :func:`plant_keyed_boundary` fed the same rows as ``recent_observations``.
    """
    sources: dict[str, dict[int, Any]] = {
        field: {
            hydro["hydro_id"]: hydro[field]
            for hydro in boundary_conditions["hydros"]
            if field in hydro
        }
        for field in _HYDRO_FIELDS
    }
    for hydro_id, rows in _group(windows, "hydro_id", _FLOW_FIELDS).items():
        if hydro_id in sources["recent_inflows"]:
            raise ValueError(f"hydro {hydro_id} already has `recent_inflows`")
        sources["recent_inflows"][hydro_id] = rows
    boundary_conditions["hydros"] = _entries("hydro_id", sources)

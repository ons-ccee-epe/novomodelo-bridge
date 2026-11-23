"""Tier-1 tests for the plant-keyed boundary writer (``novomodelo/boundary_conditions.py``)."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import pytest

from novomodelo_bridge.novomodelo.boundary_conditions import (
    BoundaryFiles,
    add_recent_inflows,
    plant_keyed_boundary,
)
from novomodelo_bridge.novomodelo.schemas import schema_url_for

_STAGES = {"$schema": "stages-url", "stages": [{"id": 0}, {"id": 3}]}


def _flow(hydro_id: int, start: str, end: str, value: float) -> dict[str, Any]:
    return {
        "hydro_id": hydro_id,
        "start_date": start,
        "end_date": end,
        "value_m3s": value,
    }


def _bound(thermal_id: int, index: int, cost: float) -> dict[str, Any]:
    return {
        "thermal_id": thermal_id,
        "post_study_stage_index": index,
        "cost_per_mwh": cost,
        "min_mw": 1.0,
        "max_mw": 2.0,
    }


def _stage(start: str, hours: float) -> dict[str, Any]:
    return {"start_date": start, "duration_hours": hours}


def _gnl_post_study() -> dict[str, Any]:
    return {
        "thermal_stages": [_stage("2026-05-01", 168.0), _stage("2026-05-08", 336.0)],
        "thermal_bounds": [_bound(2, 1, 20.0), _bound(2, 0, 10.0), _bound(1, 0, 5.0)],
        "water_bucket_stages": [_stage("2026-05-22", 744.0)],
    }


def test_worked_example() -> None:
    result = plant_keyed_boundary(
        {
            "storage": [{"hydro_id": 1, "value_hm3": 5.0}],
            "filling_storage": [{"hydro_id": 0, "value_hm3": 2.0}],
        },
        {"water_bucket_stages": [_stage("2026-05-01", 744.0)]},
        {"stages": [{"id": 0}, {"id": 3}]},
    )

    assert isinstance(result, BoundaryFiles)
    assert result.boundary_conditions["hydros"] == [
        {"hydro_id": 0, "storage_hm3": 2.0},
        {"hydro_id": 1, "storage_hm3": 5.0},
    ]
    assert result.boundary_conditions["thermals"] == []
    assert result.stages["post_study_water_stages"] == [
        {"id": 4, "start_date": "2026-05-01", "end_date": "2026-06-01"}
    ]
    assert list(json.loads(json.dumps(result.boundary_conditions))) == [
        "$schema",
        "hydros",
        "thermals",
    ]


def test_schema_url_is_the_registered_one_and_incoming_schema_is_ignored() -> None:
    result = plant_keyed_boundary(
        {"$schema": "legacy-url", "storage": []}, None, _STAGES
    )

    assert result.boundary_conditions == {
        "$schema": schema_url_for("boundary_conditions.json"),
        "hydros": [],
        "thermals": [],
    }


def test_hydro_entry_keys_follow_the_fixed_order_and_omit_missing_data() -> None:
    result = plant_keyed_boundary(
        {
            "storage": [{"hydro_id": 7, "value_hm3": 1.5}],
            "recent_observations": [
                _flow(7, "2026-04-01", "2026-04-08", 3.0),
                _flow(9, "2026-04-01", "2026-04-08", 4.0),
            ],
            "past_defluences": [_flow(7, "2026-03-01", "2026-03-08", 2.0)],
        },
        None,
        _STAGES,
    )

    hydros = result.boundary_conditions["hydros"]
    assert [list(h) for h in hydros] == [
        ["hydro_id", "storage_hm3", "recent_inflows", "past_defluences"],
        ["hydro_id", "recent_inflows"],
    ]
    assert hydros[0]["recent_inflows"] == [
        {"start_date": "2026-04-01", "end_date": "2026-04-08", "value_m3s": 3.0}
    ]


def test_filling_storage_and_storage_share_one_storage_hm3_key() -> None:
    result = plant_keyed_boundary(
        {
            "storage": [{"hydro_id": 3, "value_hm3": 3.0}],
            "filling_storage": [{"hydro_id": 1, "value_hm3": 1.0}],
        },
        None,
        _STAGES,
    )

    assert result.boundary_conditions["hydros"] == [
        {"hydro_id": 1, "storage_hm3": 1.0},
        {"hydro_id": 3, "storage_hm3": 3.0},
    ]


def test_rows_of_a_plant_keep_input_order_while_plants_are_sorted() -> None:
    result = plant_keyed_boundary(
        {
            "recent_observations": [
                _flow(5, "2026-04-15", "2026-04-22", 1.0),
                _flow(2, "2026-04-01", "2026-04-08", 2.0),
                _flow(5, "2026-04-01", "2026-04-08", 3.0),
            ],
        },
        None,
        _STAGES,
    )

    hydros = result.boundary_conditions["hydros"]
    assert [h["hydro_id"] for h in hydros] == [2, 5]
    assert [r["start_date"] for r in hydros[1]["recent_inflows"]] == [
        "2026-04-15",
        "2026-04-01",
    ]


def test_gnl_case_writes_commitments_bounds_and_both_calendars() -> None:
    stages = {"stages": [{"id": i} for i in range(4)]}
    result = plant_keyed_boundary(
        {
            "past_anticipated_commitments": [
                {
                    "thermal_id": 3,
                    "start_date": "2026-04-01",
                    "end_date": "2026-04-08",
                    "value_mw": 50.0,
                },
                {
                    "thermal_id": 2,
                    "start_date": "2026-04-01",
                    "end_date": "2026-04-08",
                    "value_mw": 40.0,
                },
            ],
        },
        _gnl_post_study(),
        stages,
    )

    assert result.boundary_conditions["thermals"] == [
        {
            "thermal_id": 1,
            "post_study_bounds": [
                {"stage_id": 4, "cost_per_mwh": 5.0, "min_mw": 1.0, "max_mw": 2.0}
            ],
        },
        {
            "thermal_id": 2,
            "commitments": [
                {"start_date": "2026-04-01", "end_date": "2026-04-08", "value_mw": 40.0}
            ],
            "post_study_bounds": [
                {"stage_id": 4, "cost_per_mwh": 10.0, "min_mw": 1.0, "max_mw": 2.0},
                {"stage_id": 5, "cost_per_mwh": 20.0, "min_mw": 1.0, "max_mw": 2.0},
            ],
        },
        {
            "thermal_id": 3,
            "commitments": [
                {"start_date": "2026-04-01", "end_date": "2026-04-08", "value_mw": 50.0}
            ],
        },
    ]
    assert result.stages["post_study_thermal_stages"] == [
        {"id": 4, "start_date": "2026-05-01", "end_date": "2026-05-08"},
        {"id": 5, "start_date": "2026-05-08", "end_date": "2026-05-22"},
    ]
    assert result.stages["post_study_water_stages"] == [
        {"id": 6, "start_date": "2026-05-22", "end_date": "2026-06-22"}
    ]
    assert list(result.stages) == [
        "stages",
        "post_study_thermal_stages",
        "post_study_water_stages",
    ]


def test_newave_case_without_post_study_returns_stages_equal() -> None:
    result = plant_keyed_boundary({"storage": []}, None, _STAGES)

    assert result.stages == _STAGES


def test_post_study_without_a_calendar_returns_stages_equal() -> None:
    assert plant_keyed_boundary({}, {}, _STAGES).stages == _STAGES


def test_empty_thermal_stages_still_write_an_empty_list_and_shift_no_ids() -> None:
    result = plant_keyed_boundary(
        {},
        {
            "thermal_stages": [],
            "thermal_bounds": [],
            "water_bucket_stages": [_stage("2026-05-01", 24.0)],
        },
        _STAGES,
    )

    assert result.stages["post_study_thermal_stages"] == []
    assert result.stages["post_study_water_stages"] == [
        {"id": 4, "start_date": "2026-05-01", "end_date": "2026-05-02"}
    ]


def test_first_id_is_zero_without_a_non_negative_integer_stage_id() -> None:
    result = plant_keyed_boundary(
        {},
        {"water_bucket_stages": [_stage("2026-05-01", 24.0)]},
        {"stages": [{"id": -1}, {"name": "unnamed"}]},
    )

    assert result.stages["post_study_water_stages"][0]["id"] == 0


def test_water_ids_continue_after_the_thermal_stages() -> None:
    result = plant_keyed_boundary({}, _gnl_post_study(), _STAGES)

    assert [s["id"] for s in result.stages["post_study_thermal_stages"]] == [4, 5]
    assert [s["id"] for s in result.stages["post_study_water_stages"]] == [6]


def test_inputs_are_not_mutated() -> None:
    initial = {
        "storage": [{"hydro_id": 1, "value_hm3": 5.0}],
        "recent_observations": [_flow(1, "2026-04-01", "2026-04-08", 3.0)],
    }
    post_study = _gnl_post_study()
    stages = copy.deepcopy(_STAGES)
    before = copy.deepcopy((initial, post_study, stages))

    plant_keyed_boundary(initial, post_study, stages)

    assert (initial, post_study, stages) == before


def test_add_recent_inflows_equals_the_recent_observations_path() -> None:
    initial = {
        "storage": [{"hydro_id": 4, "value_hm3": 1.0}],
        "past_defluences": [_flow(4, "2026-03-01", "2026-03-08", 2.0)],
    }
    windows = [
        _flow(9, "2026-04-01", "2026-04-08", 3.0),
        _flow(4, "2026-04-08", "2026-04-15", 4.0),
        _flow(4, "2026-04-01", "2026-04-08", 5.0),
        _flow(2, "2026-04-01", "2026-04-08", 6.0),
    ]
    boundary = plant_keyed_boundary(initial, None, _STAGES).boundary_conditions
    hydros = boundary["hydros"]

    add_recent_inflows(boundary, windows)

    expected = plant_keyed_boundary(
        {**initial, "recent_observations": windows}, None, _STAGES
    ).boundary_conditions
    assert boundary == expected
    assert boundary["hydros"] is not hydros
    assert [h["hydro_id"] for h in boundary["hydros"]] == [2, 4, 9]
    assert list(boundary["hydros"][1]) == [
        "hydro_id",
        "storage_hm3",
        "recent_inflows",
        "past_defluences",
    ]


def test_add_recent_inflows_keeps_the_inflows_of_hydros_outside_the_windows() -> None:
    initial = {"recent_observations": [_flow(1, "2026-04-01", "2026-04-08", 3.0)]}
    boundary = plant_keyed_boundary(initial, None, _STAGES).boundary_conditions

    add_recent_inflows(boundary, [_flow(2, "2026-04-01", "2026-04-08", 4.0)])

    assert [h["hydro_id"] for h in boundary["hydros"]] == [1, 2]
    assert "recent_inflows" in boundary["hydros"][0]


def test_add_recent_inflows_does_not_mutate_the_windows() -> None:
    boundary = plant_keyed_boundary({}, None, _STAGES).boundary_conditions
    windows = [_flow(1, "2026-04-01", "2026-04-08", 3.0)]
    before = copy.deepcopy(windows)

    add_recent_inflows(boundary, windows)

    assert windows == before


def _plant_keyed(
    initial: dict[str, Any], post_study: dict[str, Any] | None = None
) -> Callable[[], object]:
    return lambda: plant_keyed_boundary(initial, post_study, _STAGES)


def _add_duplicate() -> None:
    boundary = plant_keyed_boundary(
        {"recent_observations": [_flow(8, "2026-04-01", "2026-04-08", 3.0)]},
        None,
        _STAGES,
    ).boundary_conditions
    add_recent_inflows(boundary, [_flow(8, "2026-04-08", "2026-04-15", 4.0)])


@pytest.mark.parametrize(
    ("call", "match"),
    [
        pytest.param(_plant_keyed({"surprise": []}), "surprise", id="initial-key"),
        pytest.param(
            _plant_keyed({}, {"stages": [_stage("2026-05-01", 24.0)]}),
            "stages",
            id="legacy-post-study-key",
        ),
        pytest.param(
            _plant_keyed({}, {"thermal_stages": [_stage("2026-05-01", 24.0)]}),
            "thermal_bounds",
            id="thermal-stages-without-bounds",
        ),
        pytest.param(
            _plant_keyed({}, {"thermal_bounds": [_bound(1, 0, 1.0)]}),
            "thermal_stages",
            id="bounds-without-thermal-stages",
        ),
        pytest.param(
            _plant_keyed(
                {
                    "storage": [{"hydro_id": 6, "value_hm3": 1.0}],
                    "filling_storage": [{"hydro_id": 6, "value_hm3": 2.0}],
                }
            ),
            "hydro 6",
            id="two-storage-rows",
        ),
        pytest.param(
            _plant_keyed(
                {},
                {
                    "thermal_stages": [_stage("2026-05-01", 100.0)],
                    "thermal_bounds": [],
                },
            ),
            "100",
            id="duration-not-whole-days",
        ),
        pytest.param(
            _plant_keyed({}, {"water_bucket_stages": [_stage("2026-05-01", 0.0)]}),
            "water_bucket_stages",
            id="duration-not-positive",
        ),
        pytest.param(
            _plant_keyed(
                {},
                {
                    "thermal_stages": [
                        _stage("2026-05-08", 24.0),
                        _stage("2026-05-01", 24.0),
                    ],
                    "thermal_bounds": [],
                },
            ),
            "2026-05-01",
            id="thermal-calendar-not-ascending",
        ),
        pytest.param(
            _plant_keyed(
                {},
                {
                    "water_bucket_stages": [
                        _stage("2026-05-01", 24.0),
                        _stage("2026-05-01", 24.0),
                    ]
                },
            ),
            "water_bucket_stages",
            id="water-calendar-repeats-a-start",
        ),
        pytest.param(
            _plant_keyed(
                {},
                {
                    "thermal_stages": [_stage("2026-05-01", 24.0)],
                    "thermal_bounds": [_bound(4, 1, 1.0)],
                },
            ),
            "thermal 4",
            id="bound-index-past-the-calendar",
        ),
        pytest.param(
            _plant_keyed(
                {},
                {
                    "thermal_stages": [_stage("2026-05-01", 24.0)],
                    "thermal_bounds": [_bound(4, -1, 1.0)],
                },
            ),
            "thermal 4",
            id="bound-index-negative",
        ),
        pytest.param(_add_duplicate, "hydro 8", id="recent-inflows-already-present"),
    ],
)
def test_internal_invariant_violations_raise_value_error_naming_the_offender(
    call: Callable[[], object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        call()

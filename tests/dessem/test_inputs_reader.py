"""Tests for dessem/exporters/fcf/inputs_reader.py."""

import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from novomodelo_bridge.dessem.exporters.fcf.core.inputs_reader import (
    CaseInputs,
    CommitmentWindow,
    InputReadError,
    anticipated_thermals,
    read_case_inputs,
)
from tests.dessem.conftest import weekly_windows, write_commitments


def _declare_storage(case_dir: Path, storage: Any) -> None:
    path = case_dir / "initial_conditions.json"
    conditions = json.loads(path.read_text(encoding="utf-8"))
    conditions["storage"] = storage
    path.write_text(json.dumps(conditions), encoding="utf-8")


def _write_boundary_conditions(case_dir: Path, document: Any) -> None:
    (case_dir / "initial_conditions.json").unlink(missing_ok=True)
    (case_dir / "boundary_conditions.json").write_text(
        json.dumps(document), encoding="utf-8"
    )


def _declare_filling(case_dir: Path, pos: int) -> None:
    path = case_dir / "system" / "hydros.json"
    hydros = json.loads(path.read_text(encoding="utf-8"))
    hydros["hydros"][pos]["filling"] = {"start_stage_id": 1}
    path.write_text(json.dumps(hydros), encoding="utf-8")


def _boundary_document(
    windows: Sequence[CommitmentWindow] = (),
    storage_hm3: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    by_thermal: dict[int, list[dict[str, Any]]] = {}
    for w in windows:
        by_thermal.setdefault(w.thermal_id, []).append(
            {
                "start_date": w.start_date.isoformat(),
                "end_date": w.end_date.isoformat(),
                "value_mw": w.value_mw,
            }
        )
    return {
        "hydros": [
            {"hydro_id": hydro_id, "storage_hm3": value}
            for hydro_id, value in (storage_hm3 or {}).items()
        ],
        "thermals": [
            {"thermal_id": thermal_id, "commitments": commitments}
            for thermal_id, commitments in by_thermal.items()
        ],
    }


def _declare_entry_stage_id(case_dir: Path, pos: int, value: Any) -> None:
    path = case_dir / "system" / "hydros.json"
    hydros = json.loads(path.read_text(encoding="utf-8"))
    hydros["hydros"][pos]["entry_stage_id"] = value
    path.write_text(json.dumps(hydros), encoding="utf-8")


class TestReadCaseInputs:
    """Test reading Novomodelo case JSON files."""

    def test_read_synthetic_case(self, synthetic_case_dir: Path) -> None:
        inputs = read_case_inputs(synthetic_case_dir)

        assert len(inputs.stages) == 3
        assert len(inputs.hydros) == 3
        assert len(inputs.thermals) == 2
        assert len(inputs.buses) == 6
        assert inputs.annual_discount_rate == pytest.approx(0.12)

    def test_stages_have_blocks(self, synthetic_case_dir: Path) -> None:
        inputs = read_case_inputs(synthetic_case_dir)

        for stage in inputs.stages:
            assert len(stage.blocks) == 3
            assert sum(b.hours for b in stage.blocks) == 168.0

    def test_hydros_have_reservoirs(self, synthetic_case_dir: Path) -> None:
        inputs = read_case_inputs(synthetic_case_dir)

        for hydro in inputs.hydros:
            assert hydro.max_storage_hm3 >= hydro.min_storage_hm3

    def test_travel_time_is_read_where_the_case_declares_it(
        self, travel_time_case_dir: Path
    ) -> None:
        inputs = read_case_inputs(travel_time_case_dir)

        assert [hydro.travel_time_hours for hydro in inputs.hydros] == [
            None,
            240.0,
            None,
        ]

    def test_coupling_auxiliary_plants_are_left_out(
        self, synthetic_case_dir: Path
    ) -> None:
        hydros_path = synthetic_case_dir / "system" / "hydros.json"
        hydros = json.loads(hydros_path.read_text(encoding="utf-8"))
        real_ids = [hydro["id"] for hydro in hydros["hydros"]]
        hydros["hydros"].append({**hydros["hydros"][0], "id": 1288, "name": "ACOPL"})
        hydros_path.write_text(json.dumps(hydros), encoding="utf-8")

        inputs = read_case_inputs(synthetic_case_dir)

        assert [hydro.id for hydro in inputs.hydros] == real_ids

    def test_negative_travel_time_raises(self, synthetic_case_dir: Path) -> None:
        hydros_path = synthetic_case_dir / "system" / "hydros.json"
        hydros = json.loads(hydros_path.read_text(encoding="utf-8"))
        hydros["hydros"][0]["travel_time_hours"] = -1.0
        hydros_path.write_text(json.dumps(hydros), encoding="utf-8")

        with pytest.raises(InputReadError, match="must be non-negative"):
            read_case_inputs(synthetic_case_dir)

    def test_entry_stage_id_is_read_where_the_case_declares_it(
        self, synthetic_case_dir: Path
    ) -> None:
        _declare_entry_stage_id(synthetic_case_dir, 1, 3)

        inputs = read_case_inputs(synthetic_case_dir)

        assert [hydro.entry_stage_id for hydro in inputs.hydros] == [None, 3, None]

    def test_malformed_entry_stage_id_raises(self, synthetic_case_dir: Path) -> None:
        cases: list[tuple[Any, str]] = [
            (-1, r"hydros\[1\]\.entry_stage_id must be non-negative, got -1"),
            (True, r"hydros\[1\]\.entry_stage_id must be int, got bool"),
            (1.5, r"hydros\[1\]\.entry_stage_id must be int, got float"),
            ("3", r"hydros\[1\]\.entry_stage_id must be int, got str"),
        ]
        for value, message in cases:
            _declare_entry_stage_id(synthetic_case_dir, 1, value)

            with pytest.raises(InputReadError, match=message):
                read_case_inputs(synthetic_case_dir)

    def test_initial_storage_is_read_by_hydro_id(
        self, synthetic_case_dir: Path
    ) -> None:
        assert read_case_inputs(synthetic_case_dir).initial_storage_hm3 == {}
        _declare_storage(
            synthetic_case_dir,
            [
                {"hydro_id": 2, "value_hm3": 0.0},
                {"hydro_id": 1, "value_hm3": 10.0},
            ],
        )

        inputs = read_case_inputs(synthetic_case_dir)

        assert inputs.initial_storage_hm3 == {1: 10.0, 2: 0.0}
        assert inputs.past_anticipated_commitments == weekly_windows(100)

    def test_malformed_initial_storage_raises(self, synthetic_case_dir: Path) -> None:
        cases: list[tuple[Any, str]] = [
            ({"hydro_id": 1, "value_hm3": 1.0}, r"storage must be a list, got dict"),
            ([{"value_hm3": 1.0}], r"missing required key storage\[0\]\.hydro_id"),
            (
                [{"hydro_id": 1.5, "value_hm3": 1.0}],
                r"storage\[0\]\.hydro_id must be int, got float",
            ),
            ([{"hydro_id": 1}], r"missing required key storage\[0\]\.value_hm3"),
            (
                [{"hydro_id": 1, "value_hm3": "10"}],
                r"storage\[0\]\.value_hm3 must be a number, got str",
            ),
            (
                [
                    {"hydro_id": 1, "value_hm3": 1.0},
                    {"hydro_id": 1, "value_hm3": 2.0},
                ],
                r"storage\[1\]\.hydro_id 1 repeats an earlier storage entry",
            ),
            (
                [{"hydro_id": 1, "value_hm3": -1.0}],
                r"storage\[0\]\.value_hm3 must be non-negative, got -1\.0",
            ),
        ]
        for storage, message in cases:
            _declare_storage(synthetic_case_dir, storage)

            with pytest.raises(InputReadError, match=message):
                read_case_inputs(synthetic_case_dir)

    def test_absent_thermals_file_reads_as_no_thermals(
        self, synthetic_case_dir: Path
    ) -> None:
        (synthetic_case_dir / "system" / "thermals.json").unlink()

        inputs = read_case_inputs(synthetic_case_dir)

        assert inputs.thermals == ()
        assert anticipated_thermals(inputs) == ()

    def test_missing_stages_file(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "empty_case"
        case_dir.mkdir()

        with pytest.raises(InputReadError, match="not found"):
            read_case_inputs(case_dir)

    def test_missing_cost_per_mwh_raises(self, synthetic_case_dir: Path) -> None:
        thermals_path = synthetic_case_dir / "system" / "thermals.json"
        thermals = json.loads(thermals_path.read_text(encoding="utf-8"))
        del thermals["thermals"][1]["cost_per_mwh"]
        thermals_path.write_text(json.dumps(thermals), encoding="utf-8")

        with pytest.raises(
            InputReadError, match=r"missing required key thermals\[1\]\.cost_per_mwh"
        ):
            read_case_inputs(synthetic_case_dir)

    def test_negative_cost_per_mwh_raises(self, synthetic_case_dir: Path) -> None:
        thermals_path = synthetic_case_dir / "system" / "thermals.json"
        thermals = json.loads(thermals_path.read_text(encoding="utf-8"))
        thermals["thermals"][1]["cost_per_mwh"] = -1.0
        thermals_path.write_text(json.dumps(thermals), encoding="utf-8")

        with pytest.raises(
            InputReadError,
            match=r"thermals\[1\]\.cost_per_mwh must be non-negative, got -1\.0",
        ):
            read_case_inputs(synthetic_case_dir)


class TestReadInitialConditions:
    """Test reading the past_anticipated_commitments windows."""

    def test_builder_case_windows_and_costs(self, synthetic_case_dir: Path) -> None:
        inputs = read_case_inputs(synthetic_case_dir)

        assert inputs.past_anticipated_commitments == weekly_windows(100)
        assert [w.start_date for w in inputs.past_anticipated_commitments] == [
            date(2026, 1, 1),
            date(2026, 1, 8),
            date(2026, 1, 15),
        ]
        assert [t.cost_per_mwh for t in inputs.thermals] == [200.0, 150.0]

    def test_absent_key_reads_as_no_windows(self, synthetic_case_dir: Path) -> None:
        (synthetic_case_dir / "initial_conditions.json").write_text(
            "{}", encoding="utf-8"
        )

        inputs = read_case_inputs(synthetic_case_dir)

        assert inputs.past_anticipated_commitments == ()

    def test_missing_file_raises(self, synthetic_case_dir: Path) -> None:
        (synthetic_case_dir / "initial_conditions.json").unlink()

        with pytest.raises(
            InputReadError,
            match=r"Novomodelo input file not found: .*initial_conditions\.json",
        ):
            read_case_inputs(synthetic_case_dir)

    def test_unknown_keys_are_ignored(self, synthetic_case_dir: Path) -> None:
        path = synthetic_case_dir / "initial_conditions.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["note"] = "x"
        document["past_anticipated_commitments"][0]["note"] = "x"
        path.write_text(json.dumps(document), encoding="utf-8")

        inputs = read_case_inputs(synthetic_case_dir)

        assert inputs.past_anticipated_commitments == weekly_windows(100)

    @pytest.mark.parametrize(
        ("document", "message"),
        [
            ([], r"initial_conditions\.json must hold a JSON object, got list"),
            (
                {"past_anticipated_commitments": {}},
                r"past_anticipated_commitments must be a list, got dict",
            ),
            (
                {"past_anticipated_commitments": [3]},
                r"past_anticipated_commitments\[0\]\.thermal_id must be an object, got int",
            ),
        ],
        ids=["top-level-not-object", "key-not-list", "entry-not-object"],
    )
    def test_malformed_document_raises(
        self, synthetic_case_dir: Path, document: Any, message: str
    ) -> None:
        (synthetic_case_dir / "initial_conditions.json").write_text(
            json.dumps(document), encoding="utf-8"
        )

        with pytest.raises(InputReadError, match=message):
            read_case_inputs(synthetic_case_dir)

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            (
                {"start_date": "2026-13-01"},
                r"past_anticipated_commitments\[0\]\.start_date must be an ISO-8601 date, got '2026-13-01'",
            ),
            (
                {"end_date": "2026-01-01"},
                r"past_anticipated_commitments\[0\]\.end_date 2026-01-01 is not after start_date 2026-01-01",
            ),
            (
                {"thermal_id": 100.0},
                r"past_anticipated_commitments\[0\]\.thermal_id must be int",
            ),
            (
                {"thermal_id": True},
                r"past_anticipated_commitments\[0\]\.thermal_id must be int, got bool",
            ),
            (
                {"value_mw": "10"},
                r"past_anticipated_commitments\[0\]\.value_mw must be a number",
            ),
            (
                {"value_mw": float("nan")},
                r"past_anticipated_commitments\[0\]\.value_mw must be a finite number",
            ),
        ],
        ids=[
            "bad-start-date",
            "end-equals-start",
            "float-thermal-id",
            "bool-thermal-id",
            "string-value",
            "nan-value",
        ],
    )
    def test_malformed_window_raises(
        self, synthetic_case_dir: Path, override: dict[str, Any], message: str
    ) -> None:
        path = synthetic_case_dir / "initial_conditions.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["past_anticipated_commitments"][0].update(override)
        path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(InputReadError, match=message):
            read_case_inputs(synthetic_case_dir)


class TestReadBoundaryConditions:
    """Test reading boundary_conditions.json, the plant-keyed boundary file."""

    def test_boundary_commitments_follow_file_order(
        self, synthetic_case_dir: Path
    ) -> None:
        windows = (*weekly_windows(101, (5.0, 6.0)), *weekly_windows(100, (7.0,)))
        _write_boundary_conditions(synthetic_case_dir, _boundary_document(windows))

        inputs = read_case_inputs(synthetic_case_dir)

        assert inputs.past_anticipated_commitments == windows
        assert inputs.initial_storage_hm3 == {}

    def test_boundary_storage_is_read_by_hydro_id(
        self, synthetic_case_dir: Path
    ) -> None:
        document = _boundary_document(storage_hm3={2: 0.0, 1: 10.0})
        document["hydros"].append({"hydro_id": 3})
        document["hydros"].append({"hydro_id": 4, "storage_hm3": None})
        _write_boundary_conditions(synthetic_case_dir, document)

        inputs = read_case_inputs(synthetic_case_dir)

        assert inputs.initial_storage_hm3 == {1: 10.0, 2: 0.0}
        assert inputs.past_anticipated_commitments == ()

    def test_boundary_storage_of_a_filling_hydro_is_not_read(
        self, synthetic_case_dir: Path
    ) -> None:
        _declare_filling(synthetic_case_dir, 1)
        _write_boundary_conditions(
            synthetic_case_dir, _boundary_document(storage_hm3={1: 10.0, 2: 20.0})
        )

        inputs = read_case_inputs(synthetic_case_dir)

        assert [hydro.has_filling for hydro in inputs.hydros] == [False, True, False]
        assert inputs.initial_storage_hm3 == {1: 10.0}

    def test_boundary_absent_arrays_and_unknown_keys_read_as_empty(
        self, synthetic_case_dir: Path
    ) -> None:
        _write_boundary_conditions(
            synthetic_case_dir,
            {
                "$schema": "x",
                "note": "x",
                "thermals": [{"thermal_id": 100, "note": "x"}],
                "hydros": [{"hydro_id": 1, "recent_inflows": [], "note": "x"}],
            },
        )

        inputs = read_case_inputs(synthetic_case_dir)

        assert inputs.past_anticipated_commitments == ()
        assert inputs.initial_storage_hm3 == {}

        _write_boundary_conditions(synthetic_case_dir, {})

        assert read_case_inputs(synthetic_case_dir).past_anticipated_commitments == ()

    def test_boundary_and_deprecated_file_together_raise(
        self, synthetic_case_dir: Path
    ) -> None:
        (synthetic_case_dir / "boundary_conditions.json").write_text(
            "{}", encoding="utf-8"
        )

        with pytest.raises(
            InputReadError,
            match=r"holds both boundary_conditions\.json and the deprecated "
            r"initial_conditions\.json; keep one file set",
        ):
            read_case_inputs(synthetic_case_dir)

    def test_boundary_file_missing_names_both_files(
        self, synthetic_case_dir: Path
    ) -> None:
        (synthetic_case_dir / "initial_conditions.json").unlink()

        with pytest.raises(
            InputReadError,
            match=r"Novomodelo input file not found: .*boundary_conditions\.json "
            r"\(or the deprecated initial_conditions\.json\)",
        ):
            read_case_inputs(synthetic_case_dir)

    def test_boundary_top_level_not_an_object_raises(
        self, synthetic_case_dir: Path
    ) -> None:
        _write_boundary_conditions(synthetic_case_dir, [])

        with pytest.raises(
            InputReadError,
            match=r"boundary_conditions\.json must hold a JSON object, got list",
        ):
            read_case_inputs(synthetic_case_dir)

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            (
                {"start_date": "2026-13-01"},
                r"thermals\[1\]\.commitments\[1\]\.start_date must be an ISO-8601 date, got '2026-13-01'",
            ),
            (
                {"end_date": "2026-01-08"},
                r"thermals\[1\]\.commitments\[1\]\.end_date 2026-01-08 is not after start_date 2026-01-08",
            ),
            (
                {"value_mw": "10"},
                r"thermals\[1\]\.commitments\[1\]\.value_mw must be a number",
            ),
            (
                {"value_mw": float("nan")},
                r"thermals\[1\]\.commitments\[1\]\.value_mw must be a finite number",
            ),
        ],
        ids=["bad-start-date", "end-equals-start", "string-value", "nan-value"],
    )
    def test_boundary_malformed_commitment_raises(
        self, synthetic_case_dir: Path, override: dict[str, Any], message: str
    ) -> None:
        windows = (*weekly_windows(101, (5.0,)), *weekly_windows(100, (7.0, 8.0)))
        document = _boundary_document(windows)
        document["thermals"][1]["commitments"][1].update(override)
        _write_boundary_conditions(synthetic_case_dir, document)

        with pytest.raises(InputReadError, match=message):
            read_case_inputs(synthetic_case_dir)

    def test_boundary_malformed_document_raises(self, synthetic_case_dir: Path) -> None:
        cases: list[tuple[Any, str]] = [
            ({"thermals": {}}, r"thermals must be a list, got dict"),
            ({"hydros": {}}, r"hydros must be a list, got dict"),
            (
                {"thermals": [{"commitments": []}]},
                r"missing required key thermals\[0\]\.thermal_id",
            ),
            (
                {"thermals": [{"thermal_id": 100.0}]},
                r"thermals\[0\]\.thermal_id must be int, got float",
            ),
            (
                {"thermals": [{"thermal_id": 100, "commitments": {}}]},
                r"thermals\[0\]\.commitments must be a list, got dict",
            ),
            (
                {"thermals": [{"thermal_id": 100, "commitments": [3]}]},
                r"thermals\[0\]\.commitments\[0\]\.start_date must be an object, got int",
            ),
            (
                {"hydros": [{"storage_hm3": 1.0}]},
                r"missing required key hydros\[0\]\.hydro_id",
            ),
            (
                {"hydros": [{"hydro_id": 1, "storage_hm3": "10"}]},
                r"hydros\[0\]\.storage_hm3 must be a number, got str",
            ),
            (
                {
                    "hydros": [
                        {"hydro_id": 1, "storage_hm3": 1.0},
                        {"hydro_id": 1, "storage_hm3": 2.0},
                    ]
                },
                r"hydros\[1\]\.hydro_id 1 repeats an earlier hydros entry",
            ),
            (
                {"hydros": [{"hydro_id": 1, "storage_hm3": -1.0}]},
                r"hydros\[0\]\.storage_hm3 must be non-negative, got -1\.0",
            ),
        ]
        for document, message in cases:
            _write_boundary_conditions(synthetic_case_dir, document)

            with pytest.raises(InputReadError, match=message):
                read_case_inputs(synthetic_case_dir)

    def test_boundary_and_deprecated_formats_read_equal(
        self, synthetic_case_dir: Path, tmp_path: Path
    ) -> None:
        legacy_dir = tmp_path / "legacy_case"
        shutil.copytree(synthetic_case_dir, legacy_dir)
        for case_dir in (synthetic_case_dir, legacy_dir):
            _declare_filling(case_dir, 2)
        windows = (*weekly_windows(100, (7.0, 8.0)), *weekly_windows(101, (5.0,)))
        write_commitments(legacy_dir, windows)
        _declare_storage(legacy_dir, [{"hydro_id": 2, "value_hm3": 20.0}])
        _write_boundary_conditions(
            synthetic_case_dir, _boundary_document(windows, {2: 20.0, 3: 30.0})
        )

        from_boundary = read_case_inputs(synthetic_case_dir)
        from_legacy = read_case_inputs(legacy_dir)

        assert from_boundary == from_legacy
        assert from_boundary.initial_storage_hm3 == {2: 20.0}
        assert from_boundary.past_anticipated_commitments == windows


class TestAnticipatedThermals:
    """Test extraction of GNL-capable thermals."""

    def test_filters_gnl_thermals(self, synthetic_case_dir: Path) -> None:
        inputs = read_case_inputs(synthetic_case_dir)
        gnl = anticipated_thermals(inputs)

        assert len(gnl) == 1
        assert gnl[0].name == "GNL_A"
        assert gnl[0].lead_time_hours == 1512.0

    def test_non_gnl_excluded(self, synthetic_inputs: CaseInputs) -> None:
        gnl = anticipated_thermals(synthetic_inputs)

        for thermal in gnl:
            assert thermal.lead_time_hours is not None

    def test_sorted_by_id(self, synthetic_inputs: CaseInputs) -> None:
        gnl = anticipated_thermals(synthetic_inputs)

        ids = [t.id for t in gnl]
        assert ids == sorted(ids)

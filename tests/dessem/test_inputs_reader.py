"""Tests for dessem/exporters/fcf/inputs_reader.py."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from novomodelo_bridge.dessem.exporters.fcf.core.inputs_reader import (
    CaseInputs,
    InputReadError,
    anticipated_thermals,
    read_case_inputs,
)
from tests.dessem.conftest import weekly_windows


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

    def test_negative_travel_time_raises(self, synthetic_case_dir: Path) -> None:
        hydros_path = synthetic_case_dir / "system" / "hydros.json"
        hydros = json.loads(hydros_path.read_text(encoding="utf-8"))
        hydros["hydros"][0]["travel_time_hours"] = -1.0
        hydros_path.write_text(json.dumps(hydros), encoding="utf-8")

        with pytest.raises(InputReadError, match="must be non-negative"):
            read_case_inputs(synthetic_case_dir)

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

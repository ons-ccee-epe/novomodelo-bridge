"""Tests for dessem/exporters/fcf/inputs_reader.py."""

from pathlib import Path

import pytest

from novomodelo_bridge.dessem.exporters.fcf.core.inputs_reader import (
    CaseInputs,
    InputReadError,
    anticipated_thermals,
    read_case_inputs,
)


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

    def test_missing_stages_file(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "empty_case"
        case_dir.mkdir()

        with pytest.raises(InputReadError, match="not found"):
            read_case_inputs(case_dir)


class TestAnticipatedThermals:
    """Test extraction of GNL-capable thermals."""

    def test_filters_gnl_thermals(self, synthetic_case_dir: Path) -> None:
        inputs = read_case_inputs(synthetic_case_dir)
        gnl = anticipated_thermals(inputs)

        assert len(gnl) == 1
        assert gnl[0].name == "GNL_A"
        assert gnl[0].lead_time_hours == 1608.0

    def test_non_gnl_excluded(self, synthetic_inputs: CaseInputs) -> None:
        gnl = anticipated_thermals(synthetic_inputs)

        for thermal in gnl:
            assert thermal.lead_time_hours is not None

    def test_sorted_by_id(self, synthetic_inputs: CaseInputs) -> None:
        gnl = anticipated_thermals(synthetic_inputs)

        ids = [t.id for t in gnl]
        assert ids == sorted(ids)

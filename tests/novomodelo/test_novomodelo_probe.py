"""Tests of the output partition probe over ``build_solved_case`` trees."""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.novomodelo.probe import (
    PARQUET_FILES,
    SIMULATION_ENTITIES,
    is_novomodelo_output,
    probe_output,
    probe_parquet_file,
    probe_simulation_entity,
)
from novomodelo_bridge.novomodelo.readers import NovomodeloReadError
from tests.novomodelo_output_builders import (
    build_solved_case,
    corrupt_file,
    drop_column,
    drop_partition,
    write_sim_entity_partition,
)

_BUILT = {
    "simulation/hydros",
    "simulation/thermals",
    "simulation/buses",
    "simulation/exchanges",
    "simulation/costs",
    "simulation/hydro_bus_generation",
    "training/convergence.parquet",
    "simulation/scenario_summary.parquet",
    "hydro_models/fpha_hyperplanes.parquet",
}


def _write_columns(path: Path, columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({column: [1] for column in columns}), path)


def test_probe_output_keys_follow_the_constants(tmp_path: Path) -> None:
    found = probe_output(build_solved_case(tmp_path / "case"))

    assert set(found) == {f"simulation/{e}" for e in SIMULATION_ENTITIES} | set(
        PARQUET_FILES
    )


def test_probe_output_reports_the_columns_of_a_solved_case(tmp_path: Path) -> None:
    found = probe_output(build_solved_case(tmp_path / "case"))

    assert {key for key, columns in found.items() if columns is not None} == _BUILT
    assert found["simulation/thermals"] is not None
    assert {"thermal_id", "generation_mw", "stage_id", "scenario_id"} <= (
        found["simulation/thermals"]
    )
    assert found["training/dictionaries/bounds.parquet"] is None


def test_dropped_partition_is_absent(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")
    drop_partition(output, "simulation/thermals")

    assert probe_simulation_entity(output, "thermals") is None
    assert probe_output(output)["simulation/thermals"] is None


def test_dropped_column_is_missing_from_the_columns(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")
    drop_column(output, "simulation/buses", "spot_price")

    columns = probe_simulation_entity(output, "buses")

    assert columns is not None
    assert "spot_price" not in columns
    assert {"bus_id", "load_mw", "scenario_id"} <= columns


def test_corrupt_partition_raises_naming_the_directory(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")
    corrupt_file(output, "simulation/hydros")

    with pytest.raises(
        NovomodeloReadError, match=re.escape(str(output / "simulation" / "hydros"))
    ):
        probe_simulation_entity(output, "hydros")


def test_corrupt_file_raises_naming_the_file(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")
    corrupt_file(output, "training/convergence.parquet")
    path = output / "training" / "convergence.parquet"

    with pytest.raises(NovomodeloReadError, match=re.escape(str(path))):
        probe_parquet_file(path)


@pytest.mark.parametrize(
    "corrupt", ["simulation/hydros", "training/convergence.parquet"]
)
def test_probe_output_propagates_an_unreadable_file(
    tmp_path: Path, corrupt: str
) -> None:
    output = build_solved_case(tmp_path / "case")
    corrupt_file(output, corrupt)

    with pytest.raises(NovomodeloReadError):
        probe_output(output)


@pytest.mark.parametrize("hive_subdir", [False, True])
def test_simulation_directory_without_a_parquet_file_is_absent(
    tmp_path: Path, hive_subdir: bool
) -> None:
    output = build_solved_case(tmp_path / "case")
    entity_dir = output / "simulation" / "non_controllables"
    entity_dir.mkdir()
    if hive_subdir:
        (entity_dir / "scenario_id=0000").mkdir()

    assert probe_simulation_entity(output, "non_controllables") is None


def test_output_directory_that_does_not_exist_probes_as_absent(
    tmp_path: Path,
) -> None:
    found = probe_output(tmp_path / "case" / "output")

    assert set(found.values()) == {None}


def test_entity_outside_the_constant_is_probed(tmp_path: Path) -> None:
    output = tmp_path / "case" / "output"
    write_sim_entity_partition(output, "storage_banks", 0, [{"bank_id": 1}])

    assert probe_simulation_entity(output, "storage_banks") == {
        "bank_id",
        "scenario_id",
    }
    assert "simulation/storage_banks" not in probe_output(output)


def test_case_relative_files_resolve_under_the_case_directory(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")
    columns = ("hydro_id", "stage_id", "min_generation_mw")
    _write_columns(tmp_path / "case" / "constraints" / "hydro_bounds.parquet", columns)
    _write_columns(output / "constraints" / "line_bounds.parquet", columns)

    found = probe_output(output)

    assert found["constraints/hydro_bounds.parquet"] == frozenset(columns)
    assert found["constraints/line_bounds.parquet"] is None


def test_probes_emit_no_diagnostic(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")
    drop_partition(output, "simulation/thermals")
    drop_column(output, "simulation/buses", "spot_price")

    with dx.collect() as found:
        probe_output(output)

    assert found == []


@pytest.mark.parametrize(
    ("directories", "files", "expected"),
    [
        ((), (), False),
        (("training",), (), True),
        (("simulation",), (), True),
        (("policy", "hydro_models"), ("training",), False),
    ],
    ids=["empty", "training_only", "simulation_only", "no_directory_named_for_a_phase"],
)
def test_is_novomodelo_output_needs_a_phase_directory(
    tmp_path: Path,
    directories: tuple[str, ...],
    files: tuple[str, ...],
    expected: bool,
) -> None:
    for name in directories:
        (tmp_path / name).mkdir()
    for name in files:
        (tmp_path / name).write_bytes(b"")

    assert is_novomodelo_output(tmp_path) is expected

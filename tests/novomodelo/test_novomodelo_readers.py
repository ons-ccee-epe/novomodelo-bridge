"""Tests for the fail-loud contract of the Novomodelo output readers.

ROBUST-01: a read/parse/aggregation failure on an **already-existing**
Novomodelo output file must raise :class:`NovomodeloReadError` (so the comparison
engine never reports a false "no divergence" on unreadable data), while a
genuinely **absent** optional output must still yield an empty frame.  The
CLI maps :class:`NovomodeloReadError` to exit code 2 (distinct from exit 1 =
"mismatches found").
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import typer

from novomodelo_bridge.cli.args import CompareArgs
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.errors import (
    FieldParseError,
    NovomodeloPartitionMissingError,
    SourceFileError,
    diagnostic_from_exception,
)
from novomodelo_bridge.core.pandas_utils import as_float, as_int
from novomodelo_bridge.novomodelo.readers import (
    NovomodeloReadError,
    _load_entity_bus_map,
    hydro_home_bus,
    novomodelo_software_version,
    read_novomodelo_bus_aggregates,
    read_novomodelo_bus_means,
    read_novomodelo_bus_metadata,
    read_novomodelo_bus_net_import,
    read_novomodelo_bus_percentiles,
    read_novomodelo_buses,
    read_novomodelo_convergence,
    read_novomodelo_convergence_bounds,
    read_novomodelo_coupling_auxiliary_count,
    read_novomodelo_default_exchange_cost,
    read_novomodelo_fpha_fit_config,
    read_novomodelo_generic_constraints,
    read_novomodelo_gnl_delivered_cost,
    read_novomodelo_hydro_block_operation,
    read_novomodelo_hydro_bus_generation,
    read_novomodelo_hydro_bus_labels,
    read_novomodelo_hydro_means,
    read_novomodelo_hydro_metadata,
    read_novomodelo_hydro_per_stage_bounds,
    read_novomodelo_hydro_total_flows,
    read_novomodelo_hydro_withdrawal,
    read_novomodelo_hydros,
    read_novomodelo_inflow_non_negativity_method,
    read_novomodelo_line_bounds,
    read_novomodelo_line_means,
    read_novomodelo_lines,
    read_novomodelo_load_bus_ids,
    read_novomodelo_lp_max_generation,
    read_novomodelo_network_lines,
    read_novomodelo_non_controllable_sources,
    read_novomodelo_productivity_detail,
    read_novomodelo_required_entities,
    read_novomodelo_required_generic_constraints,
    read_novomodelo_scenario_weights,
    read_novomodelo_spillage_energy,
    read_novomodelo_stage_line_bounds,
    read_novomodelo_stage_risk_aversion,
    read_novomodelo_thermal_means,
    read_novomodelo_thermal_metadata,
    read_novomodelo_thermals,
    read_novomodelo_training_metadata,
    scan_simulation_entity,
    split_plant_row,
)
from tests.conftest import hydro_with_group
from tests.novomodelo_output_builders import (
    write_corrupt_parquet,
    write_corrupt_sim_parquet,
    write_entity_bus_json,
    write_hydro_bus_generation_partition,
    write_minimal_hydros_json,
    write_sim_entity_partition,
    write_stages_json,
)

# ---------------------------------------------------------------------------
# Corrupt *existing* file -> NovomodeloReadError.
# ---------------------------------------------------------------------------


class TestCorruptExistingRaises:
    def test_hydro_means_corrupt_parquet_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        write_corrupt_sim_parquet(out, "hydros")
        with pytest.raises(NovomodeloReadError, match="scan parquets"):
            read_novomodelo_hydro_means(out)

    def test_thermal_means_corrupt_parquet_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        write_corrupt_sim_parquet(out, "thermals")
        with pytest.raises(NovomodeloReadError):
            read_novomodelo_thermal_means(out)

    def test_bus_means_corrupt_parquet_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        write_corrupt_sim_parquet(out, "buses")
        with pytest.raises(NovomodeloReadError):
            read_novomodelo_bus_means(out)

    def test_line_means_corrupt_parquet_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        write_corrupt_sim_parquet(out, "exchanges")
        with pytest.raises(NovomodeloReadError):
            read_novomodelo_line_means(out)

    def test_lp_max_generation_corrupt_bounds_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        write_corrupt_parquet(out / "training" / "dictionaries" / "bounds.parquet")
        with pytest.raises(NovomodeloReadError, match="bounds.parquet"):
            read_novomodelo_lp_max_generation(out)

    def test_hydro_withdrawal_corrupt_constraints_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        write_corrupt_parquet(tmp_path / "constraints" / "hydro_bounds.parquet")
        with pytest.raises(NovomodeloReadError, match="hydro_bounds.parquet"):
            read_novomodelo_hydro_withdrawal(out)

    def test_hydro_per_stage_bounds_corrupt_constraints_raises(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        write_corrupt_parquet(tmp_path / "constraints" / "hydro_bounds.parquet")
        with pytest.raises(NovomodeloReadError, match="hydro_bounds.parquet"):
            read_novomodelo_hydro_per_stage_bounds(out)


# ---------------------------------------------------------------------------
# Absent file -> empty DataFrame (no raise).  PRESERVED behavior.
# ---------------------------------------------------------------------------


class TestAbsentReturnsEmpty:
    def test_hydro_means_absent_dir_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_novomodelo_hydro_means(out)
        assert df.is_empty()

    def test_thermal_means_absent_dir_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_novomodelo_thermal_means(out)
        assert df.is_empty()

    def test_bus_means_absent_dir_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_novomodelo_bus_means(out)
        assert df.is_empty()

    def test_line_means_absent_dir_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_novomodelo_line_means(out)
        assert df.is_empty()

    def test_lp_max_generation_absent_file_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_novomodelo_lp_max_generation(out)
        assert df.is_empty()

    def test_hydro_withdrawal_absent_file_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_novomodelo_hydro_withdrawal(out)
        assert df.is_empty()

    def test_hydro_per_stage_bounds_absent_file_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_novomodelo_hydro_per_stage_bounds(out)
        assert df.is_empty()


# ---------------------------------------------------------------------------
# Soft (metadata) reader stays soft: locks in the judgment call.
# ---------------------------------------------------------------------------


class TestSoftMetadataReadersStaySoft:
    def test_thermal_metadata_corrupt_json_returns_empty(self, tmp_path: Path) -> None:
        # thermals.json is metadata (entity names only), not correctness-
        # bearing comparison data — a malformed file degrades to {}.
        out = tmp_path / "output"
        sysdir = tmp_path / "system"
        sysdir.mkdir(parents=True)
        (sysdir / "thermals.json").write_text(
            "{ this is not valid json ", encoding="utf-8"
        )
        out.mkdir()
        assert read_novomodelo_thermal_metadata(out) == {}


# ---------------------------------------------------------------------------
# CLI maps NovomodeloReadError -> exit code 2 (distinct from 1 = mismatches).
# ---------------------------------------------------------------------------


class TestCliExitCodeTwoOnNovomodeloReadError:
    def test_results_cli_exits_2_when_reader_raises(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from novomodelo_bridge.cli.compare import _run_newave_comparison

        args = CompareArgs(
            source_dir=tmp_path / "newave",
            novomodelo_output_dir=tmp_path / "output",
            tolerance=1e-2,
            format=None,
            out_dir=None,
            json_output=False,
            verbose=0,
            log_file=None,
            no_color=False,
            quiet=False,
        )

        def _raise(**_kwargs: object) -> object:
            raise NovomodeloReadError(
                "Failed to aggregate hydro simulation data: /x/hydros"
            )

        with (
            patch(
                "novomodelo_bridge.novomodelo.readers.read_novomodelo_lines",
                return_value=[],
            ),
            patch(
                "novomodelo_bridge.newave.case.NewaveCase.from_directory",
                return_value=MagicMock(),
            ),
            patch(
                "novomodelo_bridge.comparators.newave.alignment.build_entity_alignment",
                return_value=object(),
            ),
            patch(
                "novomodelo_bridge.comparators.newave.results.compare_results",
                side_effect=_raise,
            ),
        ):
            with pytest.raises(typer.Exit) as excinfo:
                _run_newave_comparison(args)

        assert excinfo.value.exit_code == 2
        err = capsys.readouterr().err
        assert "Failed to aggregate hydro simulation data: /x/hydros" in err


# ---------------------------------------------------------------------------
# End-to-end-ish: a corrupt existing parquet drives the reader through the
# real (unpatched) code path to NovomodeloReadError, proving the antipattern fix
# rather than only the CLI wiring.
# ---------------------------------------------------------------------------


class TestMissingColumnsWarn:
    """A file present without a column a reader needs is a warning naming the
    column, not a silent empty frame."""

    def test_lp_max_generation_without_its_columns_warns(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        bounds = out / "training" / "dictionaries" / "bounds.parquet"
        bounds.parent.mkdir(parents=True)
        pl.DataFrame(
            {"entity_type_code": [0], "entity_id": [1], "stage_id": [0]}
        ).write_parquet(bounds)

        with dx.collect() as found:
            result = read_novomodelo_lp_max_generation(out)

        assert result.is_empty()
        [warning] = found
        assert warning.code == "report-file-unreadable"
        assert warning.title == (
            "output/training/dictionaries/bounds.parquet could not be read"
        )
        assert warning.notes[-1] == (
            "Reason: no bound_type_code or bound_value columns"
        )

    def test_total_flows_without_the_outflow_warns(self, tmp_path: Path) -> None:
        means = pl.DataFrame(
            {"entity_id": [1], "stage_id": [0], "incremental_inflow_m3s": [10.0]}
        )

        with dx.collect() as found:
            result = read_novomodelo_hydro_total_flows(tmp_path / "output", means)

        assert result.is_empty()
        [warning] = found
        assert warning.title == "output/simulation/hydros could not be read"
        assert "total inflow" in warning.summary
        assert warning.notes[-1] == "Reason: no outflow_m3s column"

    def test_withdrawal_without_its_column_is_a_case_without_withdrawal(
        self, tmp_path: Path
    ) -> None:
        bounds = tmp_path / "constraints" / "hydro_bounds.parquet"
        bounds.parent.mkdir(parents=True)
        pl.DataFrame({"hydro_id": [1], "stage_id": [0]}).write_parquet(bounds)

        with dx.collect() as found:
            result = read_novomodelo_hydro_withdrawal(tmp_path / "output")

        assert result.is_empty()
        assert found == []

    def test_withdrawal_reads_the_stage_rows_of_a_case_with_block_rows(
        self, tmp_path: Path
    ) -> None:
        bounds = tmp_path / "constraints" / "hydro_bounds.parquet"
        bounds.parent.mkdir(parents=True)
        pl.DataFrame(
            {
                "hydro_id": [1, 1, 1, 1],
                "stage_id": [0, 0, 0, 1],
                "block_id": [None, 0, 1, None],
                "water_withdrawal_m3s": [0.5, None, None, 0.7],
            },
            schema_overrides={"block_id": pl.Int32},
        ).write_parquet(bounds)

        result = read_novomodelo_hydro_withdrawal(tmp_path / "output")

        assert result.rows() == [(1, 0, 0.5), (1, 1, 0.7)]


class TestCorruptDrivesRealReaderPath:
    def test_real_reader_raises_on_corrupt_then_empty_on_absent(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        bounds = out / "training" / "dictionaries" / "bounds.parquet"

        # Corrupt-but-existing -> raises.
        write_corrupt_parquet(bounds)
        with pytest.raises(NovomodeloReadError):
            read_novomodelo_lp_max_generation(out)

        # Same reader, file removed -> empty, no raise.
        bounds.unlink()
        result = read_novomodelo_lp_max_generation(out)
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()


# ---------------------------------------------------------------------------
# hydro_bus_generation reader: loud absence, never empty.
# ---------------------------------------------------------------------------

_HYDRO_BUS_GENERATION_COLUMNS = {
    "stage_id",
    "block_id",
    "hydro_id",
    "bus_id",
    "turbined_m3s",
    "generation_mw",
    "generation_mwh",
    "scenario_id",
}


class TestHydroBusGenerationValidPartition:
    def test_returns_seven_columns_plus_scenario_id(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        write_hydro_bus_generation_partition(
            out,
            scenario_id=1,
            rows=[
                {
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 1,
                    "bus_id": 5,
                    "turbined_m3s": 100.0,
                    "generation_mw": 10.0,
                    "generation_mwh": 730.0,
                },
                {
                    "stage_id": 0,
                    "block_id": 1,
                    "hydro_id": 1,
                    "bus_id": 5,
                    "turbined_m3s": 120.0,
                    "generation_mw": 12.0,
                    "generation_mwh": 876.0,
                },
            ],
        )

        df = read_novomodelo_hydro_bus_generation(out).collect()

        assert _HYDRO_BUS_GENERATION_COLUMNS.issubset(set(df.columns))
        assert df.height == 2
        assert df["scenario_id"].to_list() == [1, 1]
        assert df["generation_mwh"].to_list() == [730.0, 876.0]

    def test_dtypes_match_writer_contract(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        write_hydro_bus_generation_partition(
            out,
            scenario_id=1,
            rows=[
                {
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 1,
                    "bus_id": 5,
                    "turbined_m3s": 100.0,
                    "generation_mw": 10.0,
                    "generation_mwh": 730.0,
                },
            ],
        )

        schema = read_novomodelo_hydro_bus_generation(out).collect_schema()
        assert schema["stage_id"] == pl.Int32
        assert schema["block_id"] == pl.Int32
        assert schema["hydro_id"] == pl.Int32
        assert schema["bus_id"] == pl.Int32
        assert schema["turbined_m3s"] == pl.Float64
        assert schema["generation_mw"] == pl.Float64
        assert schema["generation_mwh"] == pl.Float64
        assert schema["scenario_id"] == pl.Int64


class TestHydroBusGenerationAbsentRaisesNamed:
    def test_missing_directory_raises_typed_error_naming_the_dir(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        out.mkdir()

        with pytest.raises(NovomodeloPartitionMissingError) as excinfo:
            read_novomodelo_hydro_bus_generation(out)

        message = str(excinfo.value)
        assert str(out / "simulation" / "hydro_bus_generation") in message

    def test_missing_directory_error_is_typed_not_bare(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()

        # A narrower type than plain Exception -- callers can distinguish
        # this from every other failure mode this reader can raise, and it
        # is never caught-and-discarded inside the reader itself.
        with pytest.raises(NovomodeloPartitionMissingError):
            read_novomodelo_hydro_bus_generation(out)
        assert issubclass(NovomodeloPartitionMissingError, Exception)
        assert NovomodeloPartitionMissingError is not Exception

    def test_missing_output_dir_itself_also_raises_named_error(
        self, tmp_path: Path
    ) -> None:
        # The output dir doesn't exist at all -- still the named error, never a
        # bare FileNotFoundError from deeper I/O and never an empty frame.
        out = tmp_path / "does-not-exist"

        with pytest.raises(NovomodeloPartitionMissingError):
            read_novomodelo_hydro_bus_generation(out)


class TestHydroBusGenerationPresentButEmpty:
    def test_present_empty_directory_returns_empty_not_raise(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        sim_dir = out / "simulation" / "hydro_bus_generation"
        sim_dir.mkdir(parents=True)  # present, but no scenario partitions written.

        df = read_novomodelo_hydro_bus_generation(out).collect()

        assert df.is_empty()
        assert set(df.columns) == _HYDRO_BUS_GENERATION_COLUMNS

    def test_present_empty_directory_emits_diagnostic_not_silence(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        sim_dir = out / "simulation" / "hydro_bus_generation"
        sim_dir.mkdir(parents=True)

        with dx.collect() as collected:
            read_novomodelo_hydro_bus_generation(out)

        empties = [
            d for d in collected if d.code == "hydro-bus-generation-partition-empty"
        ]
        assert len(empties) == 1
        assert empties[0].severity is Severity.WARNING

    def test_absent_and_present_empty_are_not_conflated(self, tmp_path: Path) -> None:
        out = tmp_path / "output"

        # Absent -> named error.
        with pytest.raises(NovomodeloPartitionMissingError):
            read_novomodelo_hydro_bus_generation(out)

        # Same output dir, partition directory now created (still empty) ->
        # no raise, distinct outcome from the absent case above.
        (out / "simulation" / "hydro_bus_generation").mkdir(parents=True)
        df = read_novomodelo_hydro_bus_generation(out).collect()
        assert df.is_empty()


class TestHydroBusGenerationNullableBlockId:
    def test_stage_level_null_block_id_row_does_not_crash_aggregation(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        write_hydro_bus_generation_partition(
            out,
            scenario_id=1,
            rows=[
                {
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 1,
                    "bus_id": 5,
                    "turbined_m3s": 100.0,
                    "generation_mw": 10.0,
                    "generation_mwh": 730.0,
                },
                {
                    "stage_id": 0,
                    "block_id": 1,
                    "hydro_id": 1,
                    "bus_id": 5,
                    "turbined_m3s": 120.0,
                    "generation_mw": 12.0,
                    "generation_mwh": 876.0,
                },
                {
                    # Stage-level row: no single representative block.
                    "stage_id": 0,
                    "block_id": None,
                    "hydro_id": 1,
                    "bus_id": 5,
                    "turbined_m3s": 0.0,
                    "generation_mw": 0.0,
                    "generation_mwh": 0.0,
                },
            ],
        )

        df = read_novomodelo_hydro_bus_generation(out).collect()
        assert df.height == 3
        assert df.filter(pl.col("block_id").is_null()).height == 1

        # A downstream energy aggregation across the block dimension must
        # not choke on the null block_id.
        totals = df.group_by(["hydro_id", "stage_id"]).agg(
            pl.col("generation_mwh").sum()
        )
        assert totals["generation_mwh"].to_list() == [730.0 + 876.0 + 0.0]


class TestHydroBusGenerationCorruptRaisesNovomodeloReadError:
    def test_corrupt_parquet_raises_novomodeloreaderror_not_named_absence(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        write_corrupt_sim_parquet(out, "hydro_bus_generation")

        # Present-but-corrupt is a different failure than absence: the
        # existing NovomodeloReadError contract applies, not the new named
        # partition-missing error.
        with pytest.raises(NovomodeloReadError):
            read_novomodelo_hydro_bus_generation(out)


# ---------------------------------------------------------------------------
# read_novomodelo_hydro_metadata carries no "bus_id"; the plant->bus label comes
# from the hydro_bus_generation partition via read_novomodelo_hydro_bus_labels.
# ---------------------------------------------------------------------------


class TestHydroMetadataHasNoBusId:
    def test_bus_id_key_is_absent(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir(parents=True)
        write_minimal_hydros_json(tmp_path, [hydro_with_group(0, 100, name="H0")])

        result = read_novomodelo_hydro_metadata(out)

        assert "bus_id" not in result[0]

    def test_absent_hydros_json_returns_empty_dict(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir(parents=True)
        assert read_novomodelo_hydro_metadata(out) == {}


def _write_hydros_text(case_dir: Path, text: str) -> None:
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "system" / "hydros.json").write_text(text, encoding="utf-8")


_HYDRO_MEANS = pl.DataFrame(
    {
        "entity_id": [0, 1],
        "stage_id": [0, 0],
        "incremental_inflow_m3s": [10.0, 4.0],
        "outflow_m3s": [30.0, 50.0],
    }
)


#: Each ``system/`` entity loader, its file, and whether the conversion always
#: writes that file (so its absence is noted).
_SYSTEM_ENTITY_LOADERS = [
    (read_novomodelo_lines, "lines.json", False),
    (read_novomodelo_hydros, "hydros.json", True),
    (read_novomodelo_thermals, "thermals.json", False),
    (read_novomodelo_non_controllable_sources, "non_controllable_sources.json", False),
    (read_novomodelo_buses, "buses.json", True),
]


@pytest.mark.parametrize(("read", "filename", "always_written"), _SYSTEM_ENTITY_LOADERS)
class TestSystemEntityLoaders:
    def test_absent_file_is_noted_only_when_the_conversion_always_writes_it(
        self,
        tmp_path: Path,
        read: Callable[[Path], list[dict]],
        filename: str,
        always_written: bool,
    ) -> None:
        with dx.collect() as found:
            assert read(tmp_path / "output") == []

        expected = [
            ("report-file-missing", Severity.INFO, f"system/{filename} not found")
        ]
        assert [(d.code, d.severity, d.title) for d in found] == (
            expected if always_written else []
        )

    def test_unparseable_file_is_one_warning(
        self,
        tmp_path: Path,
        read: Callable[[Path], list[dict]],
        filename: str,
        always_written: bool,
    ) -> None:
        (tmp_path / "system").mkdir()
        (tmp_path / "system" / filename).write_text("{not json", encoding="utf-8")

        with dx.collect() as found:
            assert read(tmp_path / "output") == []

        assert [(d.code, d.severity, d.title) for d in found] == [
            (
                "report-file-unreadable",
                Severity.WARNING,
                f"system/{filename} could not be read",
            )
        ]

    def test_present_file_yields_its_entries(
        self,
        tmp_path: Path,
        read: Callable[[Path], list[dict]],
        filename: str,
        always_written: bool,
    ) -> None:
        entries = [{"id": 0, "name": "A"}, {"id": 1, "name": "B"}]
        (tmp_path / "system").mkdir()
        (tmp_path / "system" / filename).write_text(
            json.dumps({filename.removesuffix(".json"): entries}), encoding="utf-8"
        )

        with dx.collect() as found:
            assert read(tmp_path / "output") == entries

        assert found == []


def test_scan_simulation_entity_leaves_out_coupling_auxiliary_rows(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    write_sim_entity_partition(
        output,
        "hydros",
        0,
        [
            {"stage_id": 0, "hydro_id": 0, "spillage_m3s": 1.0},
            {"stage_id": 0, "hydro_id": 1288, "spillage_m3s": 9.0},
        ],
    )
    write_sim_entity_partition(output, "costs", 0, [{"stage_id": 0, "total": 5.0}])

    hydros = scan_simulation_entity(output, "hydros")
    costs = scan_simulation_entity(output, "costs")

    assert hydros is not None
    assert costs is not None
    assert hydros.collect()["hydro_id"].to_list() == [0]
    assert costs.collect().height == 1


class TestReadNovomodeloHydros:
    def test_coupling_auxiliary_plants_are_left_out(self, tmp_path: Path) -> None:
        plant = {"id": 0, "name": "A"}
        auxiliary = {"id": 1288, "name": "ACOPL-288 BELO MONTE"}
        _write_hydros_text(tmp_path, json.dumps({"hydros": [plant, auxiliary]}))

        assert read_novomodelo_hydros(tmp_path / "output") == [plant]

    def test_coupling_auxiliary_plants_are_counted(self, tmp_path: Path) -> None:
        hydros = [
            {"id": 0, "name": "A"},
            {"id": 1262, "name": "ACOPL-262 SALTO"},
            {"id": 1288, "name": "ACOPL-288 BELO MONTE"},
        ]
        _write_hydros_text(tmp_path, json.dumps({"hydros": hydros}))

        assert read_novomodelo_coupling_auxiliary_count(tmp_path / "output") == 2

    def test_file_without_the_plant_list_is_a_case_without_plants(
        self, tmp_path: Path
    ) -> None:
        _write_hydros_text(tmp_path, "{}")

        with dx.collect() as found:
            assert read_novomodelo_hydros(tmp_path / "output") == []

        assert found == []

    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            ("{not json", "Reason: JSONDecodeError: "),
            ("[]", "Reason: no list of hydro plants"),
            ('{"hydros": 3}', "Reason: no list of hydro plants"),
            ('{"hydros": [{"name": "H0"}]}', "Reason: a hydro plant without an id"),
            ('{"hydros": [{"id": "0"}]}', "Reason: a hydro plant without an id"),
        ],
    )
    def test_unusable_file_is_one_warning(
        self, tmp_path: Path, text: str, reason: str
    ) -> None:
        _write_hydros_text(tmp_path, text)

        with dx.collect() as found:
            assert read_novomodelo_hydros(tmp_path / "output") == []

        [warning] = found
        assert (warning.code, warning.severity, warning.title) == (
            "report-file-unreadable",
            Severity.WARNING,
            "system/hydros.json could not be read",
        )
        assert warning.notes[-1].startswith(reason)

    def test_every_reader_of_an_unreadable_file_degrades_to_one_finding(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        write_sim_entity_partition(
            out,
            "hydros",
            0,
            [
                {
                    "hydro_id": 0,
                    "stage_id": 0,
                    "block_id": 0,
                    "spillage_m3s": 5.0,
                    "equivalent_productivity_mw_per_m3s": 1.0,
                }
            ],
        )
        _write_hydros_text(tmp_path, "{not json")

        with dx.collect() as found:
            assert read_novomodelo_hydro_metadata(out) == {}
            assert read_novomodelo_productivity_detail(out) == {}
            assert read_novomodelo_hydro_total_flows(out, _HYDRO_MEANS).is_empty()
            assert read_novomodelo_spillage_energy(out).is_empty()

        assert len(found) == 4
        assert [(d.code, d.title) for d in dx.finalize_diagnostics(found)] == [
            ("report-file-unreadable", "system/hydros.json could not be read")
        ]
        assert "minimum storage as zero" in found[0].summary

    def test_total_inflow_adds_the_upstream_plants_outflow(
        self, tmp_path: Path
    ) -> None:
        write_minimal_hydros_json(
            tmp_path, [{"id": 0, "downstream_id": 1}, {"id": 1, "downstream_id": None}]
        )

        result = read_novomodelo_hydro_total_flows(tmp_path / "output", _HYDRO_MEANS)

        assert result.sort("entity_id")["total_inflow_m3s"].to_list() == [10.0, 34.0]


class TestHydroBusLabels:
    """The plant->bus label map is derived from the hydro_bus_generation

    partition's distinct (hydro_id, bus_id) pairs; a plant genuinely present
    at more than one bus keeps every one of its buses here rather than
    collapsing onto one -- exclusion/diagnosis of that ambiguity is the
    analyze layer's job (see ``analyze._bus_name_lookups``), not this reader's.
    """

    def test_single_bus_plant_maps_to_a_one_element_frozenset(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        write_hydro_bus_generation_partition(
            out,
            scenario_id=0,
            rows=[
                {
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 1,
                    "bus_id": 5,
                    "turbined_m3s": 0.0,
                    "generation_mw": 10.0,
                    "generation_mwh": 730.0,
                },
                {
                    "stage_id": 1,
                    "block_id": 0,
                    "hydro_id": 1,
                    "bus_id": 5,
                    "turbined_m3s": 0.0,
                    "generation_mw": 12.0,
                    "generation_mwh": 730.0,
                },
            ],
        )

        result = read_novomodelo_hydro_bus_labels(out)

        assert result == {1: frozenset({5})}

    def test_synthetic_two_bus_plant_keeps_every_bus(self, tmp_path: Path) -> None:
        """A synthetic two-bus plant, at the reader level:

        no real deck produces this today, but the
        partition itself imposes no one-bus-per-plant constraint, so a plant
        recorded at two buses across scenarios/stages must not be collapsed
        down to a single one here.
        """
        out = tmp_path / "output"
        write_hydro_bus_generation_partition(
            out,
            scenario_id=0,
            rows=[
                {
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 9,
                    "bus_id": 100,
                    "turbined_m3s": 0.0,
                    "generation_mw": 10.0,
                    "generation_mwh": 730.0,
                },
                {
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 9,
                    "bus_id": 101,
                    "turbined_m3s": 0.0,
                    "generation_mw": 5.0,
                    "generation_mwh": 365.0,
                },
                {
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 1,
                    "bus_id": 100,
                    "turbined_m3s": 0.0,
                    "generation_mw": 20.0,
                    "generation_mwh": 1460.0,
                },
            ],
        )

        result = read_novomodelo_hydro_bus_labels(out)

        assert result == {9: frozenset({100, 101}), 1: frozenset({100})}

    def test_missing_partition_raises_named_error(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir(parents=True)

        with pytest.raises(NovomodeloPartitionMissingError):
            read_novomodelo_hydro_bus_labels(out)

    def test_present_but_empty_partition_returns_empty_dict(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        (out / "simulation" / "hydro_bus_generation").mkdir(parents=True)

        assert read_novomodelo_hydro_bus_labels(out) == {}


# ---------------------------------------------------------------------------
# Loaded buses and the home bus of a plant split across buses.
# ---------------------------------------------------------------------------


def _write_load_stats(case_dir: Path, columns: dict[str, list]) -> None:
    path = case_dir / "scenarios" / "load_seasonal_stats.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(columns).write_parquet(path)


class TestReadNovomodeloLoadBusIds:
    def test_bus_without_load_at_any_stage_is_left_out(self, tmp_path: Path) -> None:
        _write_load_stats(
            tmp_path,
            {
                "bus_id": [0, 0, 1, 1],
                "stage_id": [0, 1, 0, 1],
                "mean_mw": [100.0, 80.0, 0.0, 0.0],
            },
        )

        assert read_novomodelo_load_bus_ids(tmp_path / "output") == {0}

    def test_bus_with_load_at_one_stage_counts(self, tmp_path: Path) -> None:
        _write_load_stats(
            tmp_path,
            {
                "bus_id": [0, 1, 0, 1],
                "stage_id": [0, 0, 1, 1],
                "mean_mw": [50.0, 0.0, 48.0, 9.0],
            },
        )

        assert read_novomodelo_load_bus_ids(tmp_path / "output") == {0, 1}

    def test_absent_file_yields_empty_set(self, tmp_path: Path) -> None:
        assert read_novomodelo_load_bus_ids(tmp_path / "output") == frozenset()

    def test_file_without_mean_mw_raises(self, tmp_path: Path) -> None:
        _write_load_stats(tmp_path, {"bus_id": [0, 1], "stage_id": [0, 0]})

        with pytest.raises(NovomodeloReadError, match="load_seasonal_stats"):
            read_novomodelo_load_bus_ids(tmp_path / "output")


class TestBusMetadataHasLoad:
    def test_flags_each_bus_by_its_load(self, tmp_path: Path) -> None:
        (tmp_path / "system").mkdir()
        (tmp_path / "system" / "buses.json").write_text(
            json.dumps({"buses": [{"id": 0, "name": "SE"}, {"id": 5, "name": "IV"}]}),
            encoding="utf-8",
        )
        _write_load_stats(
            tmp_path, {"bus_id": [0, 5], "stage_id": [0, 0], "mean_mw": [9e4, 0.0]}
        )

        assert read_novomodelo_bus_metadata(tmp_path / "output") == {
            0: {"name": "SE", "has_load": True},
            5: {"name": "IV", "has_load": False},
        }


class TestHydroHomeBus:
    def test_single_bus_plant_is_home_there_whatever_the_load(self) -> None:
        assert hydro_home_bus({5}, frozenset()) == 5

    def test_split_plant_is_home_at_its_one_loaded_bus(self) -> None:
        """Itaipu: 50 Hz units on SE, 60 Hz units on the transshipment bus."""
        assert hydro_home_bus([0, 5, 5], frozenset({0, 1, 2, 3})) == 0

    @pytest.mark.parametrize(
        "load_bus_ids",
        [frozenset(), frozenset({0, 5})],
        ids=["none-loaded", "both-loaded"],
    )
    def test_split_plant_without_one_loaded_bus_has_no_home(
        self, load_bus_ids: frozenset[int]
    ) -> None:
        assert hydro_home_bus({0, 5}, load_bus_ids) is None

    def test_row_names_the_plant_its_buses_and_their_load(self) -> None:
        row = split_plant_row(47, "ITAIPU", [5, 0, 5], frozenset({0, 5}))

        assert row == [47, "ITAIPU", "0, 5", 2]


# ---------------------------------------------------------------------------
# read_novomodelo_bus_aggregates rewired onto hydro_bus_generation;
# the _load_entity_bus_map silent-failure clause is gone.
# ---------------------------------------------------------------------------

# One stage (id 0), two blocks: hours 100.0 and 200.0 -> 300.0 total. Shared
# by every fixture below so the hand-computed weighted means in the test
# bodies use the same denominator.
_STAGE_BLOCKS = {0: [(0, 100.0), (1, 200.0)]}


def _build_full_bus_aggregates_case(tmp_path: Path) -> Path:
    """Build a case with hydro + thermal + NCS all on bus_id=5, stage_id=0.

    Two scenarios, so every per-bus variable has exactly two per-scenario
    values and the p10/p50/p90 the production code computes can be
    hand-derived with the plain two-point linear-interpolation formula
    ``lo + q * (hi - lo)``.
    """
    out = tmp_path / "output"
    out.mkdir(parents=True)
    write_stages_json(tmp_path, _STAGE_BLOCKS)

    # Hydro: one plant on bus 5. generation_mwh is block_mw * block_hours,
    # exactly as novomodelo's writer produces it.
    write_hydro_bus_generation_partition(
        out,
        scenario_id=0,
        rows=[
            {
                "stage_id": 0,
                "block_id": 0,
                "hydro_id": 1,
                "bus_id": 5,
                "turbined_m3s": 0.0,
                "generation_mw": 10.0,
                "generation_mwh": 10.0 * 100.0,
            },
            {
                "stage_id": 0,
                "block_id": 1,
                "hydro_id": 1,
                "bus_id": 5,
                "turbined_m3s": 0.0,
                "generation_mw": 20.0,
                "generation_mwh": 20.0 * 200.0,
            },
        ],
    )
    write_hydro_bus_generation_partition(
        out,
        scenario_id=1,
        rows=[
            {
                "stage_id": 0,
                "block_id": 0,
                "hydro_id": 1,
                "bus_id": 5,
                "turbined_m3s": 0.0,
                "generation_mw": 12.0,
                "generation_mwh": 12.0 * 100.0,
            },
            {
                "stage_id": 0,
                "block_id": 1,
                "hydro_id": 1,
                "bus_id": 5,
                "turbined_m3s": 0.0,
                "generation_mw": 22.0,
                "generation_mwh": 22.0 * 200.0,
            },
        ],
    )

    # Thermal: one plant on bus 5, mapped via system/thermals.json.
    write_entity_bus_json(tmp_path, "thermals", [{"id": 1, "bus_id": 5}])
    write_sim_entity_partition(
        out,
        "thermals",
        scenario_id=0,
        rows=[
            {
                "scenario_id": 0,
                "stage_id": 0,
                "block_id": 0,
                "thermal_id": 1,
                "generation_mw": 50.0,
            },
            {
                "scenario_id": 0,
                "stage_id": 0,
                "block_id": 1,
                "thermal_id": 1,
                "generation_mw": 70.0,
            },
        ],
    )
    write_sim_entity_partition(
        out,
        "thermals",
        scenario_id=1,
        rows=[
            {
                "scenario_id": 1,
                "stage_id": 0,
                "block_id": 0,
                "thermal_id": 1,
                "generation_mw": 60.0,
            },
            {
                "scenario_id": 1,
                "stage_id": 0,
                "block_id": 1,
                "thermal_id": 1,
                "generation_mw": 80.0,
            },
        ],
    )

    # NCS: one source on bus 5, mapped via system/non_controllable_sources.json.
    write_entity_bus_json(
        tmp_path, "non_controllable_sources", [{"id": 1, "bus_id": 5}]
    )
    write_sim_entity_partition(
        out,
        "non_controllables",
        scenario_id=0,
        rows=[
            {
                "scenario_id": 0,
                "stage_id": 0,
                "block_id": 0,
                "non_controllable_id": 1,
                "generation_mw": 5.0,
            },
            {
                "scenario_id": 0,
                "stage_id": 0,
                "block_id": 1,
                "non_controllable_id": 1,
                "generation_mw": 15.0,
            },
        ],
    )
    write_sim_entity_partition(
        out,
        "non_controllables",
        scenario_id=1,
        rows=[
            {
                "scenario_id": 1,
                "stage_id": 0,
                "block_id": 0,
                "non_controllable_id": 1,
                "generation_mw": 8.0,
            },
            {
                "scenario_id": 1,
                "stage_id": 0,
                "block_id": 1,
                "non_controllable_id": 1,
                "generation_mw": 18.0,
            },
        ],
    )
    return out


class TestBusAggregatesHydroReportsNonZero:
    """Hydro generation per bus is non-zero on a 0.13 output dir --
    the regression this guards against."""

    def test_hydro_gen_mw_is_nonzero_and_matches_hand_computed_value(
        self, tmp_path: Path
    ) -> None:
        out = _build_full_bus_aggregates_case(tmp_path)

        result = read_novomodelo_bus_aggregates(out)

        row = result.filter((pl.col("bus_id") == 5) & (pl.col("stage_id") == 0))
        assert row.height == 1
        p10 = row["hydro_gen_mw_p10"][0]
        p50 = row["hydro_gen_mw_p50"][0]
        p90 = row["hydro_gen_mw_p90"][0]
        assert p10 is not None
        assert p10 > 0.0
        assert p50 > 0.0
        assert p90 > 0.0

        # Hand computation: sum(generation_mwh) / total_stage_hours per
        # scenario, then the plain two-point linear-interpolation formula.
        scenario0 = (10.0 * 100.0 + 20.0 * 200.0) / 300.0  # 16.666...
        scenario1 = (12.0 * 100.0 + 22.0 * 200.0) / 300.0  # 18.666...
        lo, hi = sorted([scenario0, scenario1])
        assert p10 == pytest.approx(lo + 0.1 * (hi - lo))
        assert p50 == pytest.approx(lo + 0.5 * (hi - lo))
        assert p90 == pytest.approx(lo + 0.9 * (hi - lo))

    def test_hydros_json_absence_does_not_affect_hydro_generation(
        self, tmp_path: Path
    ) -> None:
        """Hydro generation per bus is reported with no hydros.json in the
        case: the hydro branch takes no plant->bus map from it."""
        out = _build_full_bus_aggregates_case(tmp_path)
        # No system/hydros.json is written by the builder above; assert
        # that explicitly, then confirm hydro generation is still reported.
        assert not (tmp_path / "system" / "hydros.json").exists()

        result = read_novomodelo_bus_aggregates(out)

        row = result.filter((pl.col("bus_id") == 5) & (pl.col("stage_id") == 0))
        assert row["hydro_gen_mw_p50"][0] > 0.0


class TestBusAggregatesThermalNcsWeightedMean:
    """Thermal and NCS per-bus aggregates are the block-hours-weighted mean of
    ``generation_mw`` per scenario, as ``_agg_entity_by_bus`` computes it."""

    def test_thermal_and_ncs_gen_mw_are_block_hours_weighted_means(
        self, tmp_path: Path
    ) -> None:
        out = _build_full_bus_aggregates_case(tmp_path)

        result = read_novomodelo_bus_aggregates(out)
        row = result.filter((pl.col("bus_id") == 5) & (pl.col("stage_id") == 0))
        assert row.height == 1

        # Thermal: weighted mean per scenario = sum(mw*hours)/sum(hours).
        thermal_s0 = (50.0 * 100.0 + 70.0 * 200.0) / 300.0  # 63.333...
        thermal_s1 = (60.0 * 100.0 + 80.0 * 200.0) / 300.0  # 73.333...
        lo, hi = sorted([thermal_s0, thermal_s1])
        assert row["thermal_gen_mw_p10"][0] == pytest.approx(lo + 0.1 * (hi - lo))
        assert row["thermal_gen_mw_p50"][0] == pytest.approx(lo + 0.5 * (hi - lo))
        assert row["thermal_gen_mw_p90"][0] == pytest.approx(lo + 0.9 * (hi - lo))

        # NCS: same formula.
        ncs_s0 = (5.0 * 100.0 + 15.0 * 200.0) / 300.0  # 11.666...
        ncs_s1 = (8.0 * 100.0 + 18.0 * 200.0) / 300.0  # 14.666...
        lo, hi = sorted([ncs_s0, ncs_s1])
        assert row["ncs_gen_mw_p10"][0] == pytest.approx(lo + 0.1 * (hi - lo))
        assert row["ncs_gen_mw_p50"][0] == pytest.approx(lo + 0.5 * (hi - lo))
        assert row["ncs_gen_mw_p90"][0] == pytest.approx(lo + 0.9 * (hi - lo))


class TestBusAggregatesMissingHydroPartitionRaisesNamed:
    """A missing hydro_bus_generation partition surfaces the
    named error through read_novomodelo_bus_aggregates -- it is not caught and
    converted to empty here, even though thermal/NCS data is present."""

    def test_missing_partition_raises_through_bus_aggregates(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        out.mkdir(parents=True)
        write_stages_json(tmp_path, _STAGE_BLOCKS)

        # Thermal data present and valid -- proves the hydro absence isn't
        # incidentally masked by every other branch also being empty.
        write_entity_bus_json(tmp_path, "thermals", [{"id": 1, "bus_id": 5}])
        write_sim_entity_partition(
            out,
            "thermals",
            scenario_id=0,
            rows=[
                {
                    "scenario_id": 0,
                    "stage_id": 0,
                    "block_id": 0,
                    "thermal_id": 1,
                    "generation_mw": 50.0,
                }
            ],
        )

        with pytest.raises(NovomodeloPartitionMissingError):
            read_novomodelo_bus_aggregates(out)


class TestBusAggregatesEnergyNotReweighted:
    """Energy aggregates use generation_mwh un-re-weighted -- pins one
    (bus, stage) energy figure against a hand-computed expectation. Single
    scenario so every percentile column collapses to the same exact value,
    isolating the sum(generation_mwh)/stage_hours computation from
    percentile-interpolation noise."""

    def test_single_scenario_hydro_energy_figure_matches_hand_computation(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        out.mkdir(parents=True)
        write_stages_json(tmp_path, _STAGE_BLOCKS)

        # Two hydro plants on the same bus, one scenario. A null-block_id
        # stage-level row is included to prove it is summed in
        # without corrupting the result -- its
        # generation_mwh is 0.0, a real value novomodelo could legitimately
        # write for a slack/violation-only stage-level row.
        write_hydro_bus_generation_partition(
            out,
            scenario_id=0,
            rows=[
                {
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 1,
                    "bus_id": 5,
                    "turbined_m3s": 0.0,
                    "generation_mw": 10.0,
                    "generation_mwh": 1000.0,  # 10.0 * 100.0 hours
                },
                {
                    "stage_id": 0,
                    "block_id": 1,
                    "hydro_id": 1,
                    "bus_id": 5,
                    "turbined_m3s": 0.0,
                    "generation_mw": 20.0,
                    "generation_mwh": 4000.0,  # 20.0 * 200.0 hours
                },
                {
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 2,
                    "bus_id": 5,
                    "turbined_m3s": 0.0,
                    "generation_mw": 5.0,
                    "generation_mwh": 500.0,  # 5.0 * 100.0 hours
                },
                {
                    "stage_id": 0,
                    "block_id": 1,
                    "hydro_id": 2,
                    "bus_id": 5,
                    "turbined_m3s": 0.0,
                    "generation_mw": 8.0,
                    "generation_mwh": 1600.0,  # 8.0 * 200.0 hours
                },
                {
                    "stage_id": 0,
                    "block_id": None,
                    "hydro_id": 1,
                    "bus_id": 5,
                    "turbined_m3s": 0.0,
                    "generation_mw": 0.0,
                    "generation_mwh": 0.0,
                },
            ],
        )

        result = read_novomodelo_bus_aggregates(out)
        row = result.filter((pl.col("bus_id") == 5) & (pl.col("stage_id") == 0))
        assert row.height == 1

        # Hand computation, un-re-weighted: raw sum of generation_mwh across
        # both hydros and every block (including the null-block row), then
        # a SINGLE division by the stage's total hours -- never multiplied
        # by hours a second time.
        total_energy_mwh = 1000.0 + 4000.0 + 500.0 + 1600.0 + 0.0
        total_stage_hours = 100.0 + 200.0
        expected_hydro_gen_mw = total_energy_mwh / total_stage_hours

        assert row["hydro_gen_mw_p10"][0] == pytest.approx(expected_hydro_gen_mw)
        assert row["hydro_gen_mw_p50"][0] == pytest.approx(expected_hydro_gen_mw)
        assert row["hydro_gen_mw_p90"][0] == pytest.approx(expected_hydro_gen_mw)


class TestBusAggregatesHydroFallsBackToBlockZeroMwWithoutStagesJson:
    """Without ``stages.json`` (``block_hours`` is ``None``), the hydro
    branch has no stage-hours denominator to convert an energy total back into
    a power figure, so it mirrors every sibling aggregator in this module
    (``_agg_entity_by_bus``, :func:`read_novomodelo_hydro_means`,
    :func:`read_novomodelo_thermal_means`, :func:`read_novomodelo_bus_means`)
    and reports the ``block_id == 0`` representative block's ``generation_mw``
    -- already a genuine MW figure -- summed across hydros per bus, never the
    raw ``generation_mwh`` sum. This is the only fixture in the module that
    omits ``stages.json``."""

    def test_no_stages_json_uses_block_zero_generation_mw_not_energy_sum(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        out.mkdir(parents=True)
        # Deliberately no stages.json anywhere -- _load_block_hours -> None.

        # Two hydro plants on bus 5, one scenario. block_id=0's generation_mw
        # is the value the fixed branch must report (summed across the two
        # plants); block_id=1 and the null-block-id stage-level row carry
        # deliberately huge generation_mw/generation_mwh values so the
        # assertion below fails loudly if either leaks into the sum instead
        # of being excluded, exactly like the sibling aggregators' block_id
        # == 0 fallback excludes every non-representative block.
        write_hydro_bus_generation_partition(
            out,
            scenario_id=0,
            rows=[
                {
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 1,
                    "bus_id": 5,
                    "turbined_m3s": 0.0,
                    "generation_mw": 10.0,
                    "generation_mwh": 1000.0,
                },
                {
                    "stage_id": 0,
                    "block_id": 1,
                    "hydro_id": 1,
                    "bus_id": 5,
                    "turbined_m3s": 0.0,
                    "generation_mw": 999.0,
                    "generation_mwh": 999000.0,
                },
                {
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 2,
                    "bus_id": 5,
                    "turbined_m3s": 0.0,
                    "generation_mw": 5.0,
                    "generation_mwh": 500.0,
                },
                {
                    "stage_id": 0,
                    "block_id": None,
                    "hydro_id": 1,
                    "bus_id": 5,
                    "turbined_m3s": 0.0,
                    "generation_mw": 777.0,
                    "generation_mwh": 777000.0,
                },
            ],
        )

        result = read_novomodelo_bus_aggregates(out)
        row = result.filter((pl.col("bus_id") == 5) & (pl.col("stage_id") == 0))
        assert row.height == 1

        expected_mw = 10.0 + 5.0  # block_id==0 generation_mw, summed across plants.
        assert row["hydro_gen_mw_p50"][0] == pytest.approx(expected_mw)

        # Not the raw MWh sum (across ALL blocks, including the null-block row)
        # relabeled as MW.
        wrong_energy_sum = 1000.0 + 999000.0 + 500.0 + 777000.0
        assert row["hydro_gen_mw_p50"][0] != pytest.approx(wrong_energy_sum)


# ---------------------------------------------------------------------------
# _load_entity_bus_map: bare except is gone, remaining callers
# (thermal, NCS) still have a real bus_id to read.
# ---------------------------------------------------------------------------


class TestLoadEntityBusMapNoBareExcept:
    """No bare except remains; a corrupt JSON raises a typed error
    rather than yielding {}."""

    def test_corrupt_json_raises_novomodeloreaderror_not_empty_dict(
        self, tmp_path: Path
    ) -> None:
        sysdir = tmp_path / "system"
        sysdir.mkdir(parents=True)
        (sysdir / "thermals.json").write_text(
            "{ this is not valid json ", encoding="utf-8"
        )

        with pytest.raises(NovomodeloReadError):
            _load_entity_bus_map(tmp_path / "output", "thermals")

    def test_missing_bus_id_field_raises_instead_of_being_filtered(
        self, tmp_path: Path
    ) -> None:
        # The old code silently dropped an entry lacking "bus_id" via
        # ``if "bus_id" in e``. That filter is gone: a thermal/NCS entry
        # missing its bus_id is a real problem now, not a skippable row.
        write_entity_bus_json(tmp_path, "thermals", [{"id": 1}])

        with pytest.raises(NovomodeloReadError):
            _load_entity_bus_map(tmp_path / "output", "thermals")

    def test_missing_file_still_returns_empty_dict(self, tmp_path: Path) -> None:
        # The deliberate "optional file" branch (path is None) is untouched.
        result = _load_entity_bus_map(tmp_path / "output", "thermals")
        assert result == {}

    def test_valid_json_still_returns_map(self, tmp_path: Path) -> None:
        write_entity_bus_json(
            tmp_path, "thermals", [{"id": 1, "bus_id": 5}, {"id": 2, "bus_id": 6}]
        )

        result = _load_entity_bus_map(tmp_path / "output", "thermals")
        assert result == {1: 5, 2: 6}


# ---------------------------------------------------------------------------
# Golden regeneration: the reader-level proof underneath the
# per-bus chart goldens. The chart goldens themselves are guarded by
# hand-built fixtures in test_chart_helpers.py that never touch these
# readers; the two classes below are the honest substitute --
# proving, on a synthetic 0.13 output dir, that
# read_novomodelo_bus_aggregates' hydro branch is both correctly weighted
# and reconciles exactly with the unchanged plant-keyed path.
# ---------------------------------------------------------------------------


class TestEnergyNotReweightedHandComputedProof:
    """Hand-computes one (bus, stage) energy figure from
    ``stages.json`` ``blocks[].hours`` and a synthetic ``hydro_bus_generation``
    partition, and proves -- rather than merely asserts -- that
    ``read_novomodelo_bus_aggregates`` uses ``generation_mwh`` un-re-weighted.

    Both the correct and the incorrect (double-hours-weighted) hand
    computations are derived programmatically from the same underlying
    ``rows``/``stage_blocks`` inputs, and the reader's actual output is
    checked against both -- matching the correct one, diverging sharply
    from the wrong one -- so the test demonstrates *which* value is right
    instead of pinning a number that could coincidentally satisfy either
    formula.
    """

    def test_reader_matches_un_reweighted_energy_not_the_double_weighted_one(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        out.mkdir(parents=True)
        stage_blocks = {0: [(0, 100.0), (1, 200.0)]}
        write_stages_json(tmp_path, stage_blocks)

        # One plant, one bus, one scenario: generation_mwh is exactly
        # generation_mw * block_hours, as novomodelo's writer produces it.
        rows: list[dict[str, object]] = [
            {
                "stage_id": 0,
                "block_id": 0,
                "hydro_id": 1,
                "bus_id": 5,
                "turbined_m3s": 0.0,
                "generation_mw": 10.0,
                "generation_mwh": 10.0 * 100.0,  # 1000.0
            },
            {
                "stage_id": 0,
                "block_id": 1,
                "hydro_id": 1,
                "bus_id": 5,
                "turbined_m3s": 0.0,
                "generation_mw": 20.0,
                "generation_mwh": 20.0 * 200.0,  # 4000.0
            },
        ]
        write_hydro_bus_generation_partition(out, scenario_id=0, rows=rows)

        result = read_novomodelo_bus_aggregates(out)
        row = result.filter((pl.col("bus_id") == 5) & (pl.col("stage_id") == 0))
        assert row.height == 1
        actual = row["hydro_gen_mw_p50"][0]

        # Hand computation #1 (CORRECT, per stages.json): stage 0's total
        # hours is 100.0 + 200.0 = 300.0. generation_mwh is ALREADY
        # hours-weighted energy per block, so the bus/stage MW figure is
        # the raw sum of generation_mwh across blocks divided by that
        # total -- one division, no second multiplication by hours.
        total_hours = sum(hours for _, hours in stage_blocks[0])  # 300.0
        total_energy_mwh = sum(as_float(r["generation_mwh"]) for r in rows)  # 5000.0
        correct = total_energy_mwh / total_hours  # 16.666...

        # Hand computation #2 (WRONG): the mistake the standing project
        # rule warns against -- treating generation_mwh as if it still
        # needed hours weighting, i.e. multiplying it by hours a second
        # time before dividing by total hours.
        hours_by_block = dict(stage_blocks[0])
        wrong_double_weighted = (
            sum(
                as_float(r["generation_mwh"]) * hours_by_block[as_int(r["block_id"])]
                for r in rows
            )
            / total_hours
        )  # 3000.0 -- a nonsensical order-of-magnitude jump

        # Sanity: the two hand-computed formulas actually diverge, so a
        # match against "correct" is a real discriminator, not a tautology.
        assert correct != pytest.approx(wrong_double_weighted)
        assert actual == pytest.approx(correct)
        assert actual != pytest.approx(wrong_double_weighted)


class TestBusSumMatchesPlantTotalCrossCheck:
    """The strongest guard here: total hydro
    generation summed over buses (the new ``read_novomodelo_bus_aggregates``
    source) must equal the plant-keyed total from the unchanged
    ``simulation/hydros/`` (``read_novomodelo_hydro_means``), and no bus may gain
    hydro generation it has no plants for.

    Three plants across two hydro buses (bus 5 hosts two plants, bus 6 hosts
    one), plus a third bus (7) carrying a thermal plant and NO hydro plant
    at all, in a single stage/scenario so every percentile column collapses
    to the corresponding scenario value exactly (no interpolation noise).
    """

    _STAGE_BLOCKS = {0: [(0, 100.0), (1, 200.0)]}

    def _build_case(self, tmp_path: Path) -> Path:
        out = tmp_path / "output"
        out.mkdir(parents=True)
        write_stages_json(tmp_path, self._STAGE_BLOCKS)

        # Bus-keyed source (new): hydro_bus_generation partition. Plants 1
        # and 2 share bus 5; plant 3 is alone on bus 6.
        write_hydro_bus_generation_partition(
            out,
            scenario_id=0,
            rows=[
                {
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 1,
                    "bus_id": 5,
                    "turbined_m3s": 0.0,
                    "generation_mw": 10.0,
                    "generation_mwh": 1000.0,
                },
                {
                    "stage_id": 0,
                    "block_id": 1,
                    "hydro_id": 1,
                    "bus_id": 5,
                    "turbined_m3s": 0.0,
                    "generation_mw": 20.0,
                    "generation_mwh": 4000.0,
                },
                {
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 2,
                    "bus_id": 5,
                    "turbined_m3s": 0.0,
                    "generation_mw": 3.0,
                    "generation_mwh": 300.0,
                },
                {
                    "stage_id": 0,
                    "block_id": 1,
                    "hydro_id": 2,
                    "bus_id": 5,
                    "turbined_m3s": 0.0,
                    "generation_mw": 7.0,
                    "generation_mwh": 1400.0,
                },
                {
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 3,
                    "bus_id": 6,
                    "turbined_m3s": 0.0,
                    "generation_mw": 15.0,
                    "generation_mwh": 1500.0,
                },
                {
                    "stage_id": 0,
                    "block_id": 1,
                    "hydro_id": 3,
                    "bus_id": 6,
                    "turbined_m3s": 0.0,
                    "generation_mw": 25.0,
                    "generation_mwh": 5000.0,
                },
            ],
        )

        # Plant-keyed source (unchanged): simulation/hydros/, carrying the
        # same raw per-block MW values as the partition above, so both
        # sources describe the same underlying generation.
        write_sim_entity_partition(
            out,
            "hydros",
            scenario_id=0,
            rows=[
                {
                    "scenario_id": 0,
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 1,
                    "generation_mw": 10.0,
                },
                {
                    "scenario_id": 0,
                    "stage_id": 0,
                    "block_id": 1,
                    "hydro_id": 1,
                    "generation_mw": 20.0,
                },
                {
                    "scenario_id": 0,
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 2,
                    "generation_mw": 3.0,
                },
                {
                    "scenario_id": 0,
                    "stage_id": 0,
                    "block_id": 1,
                    "hydro_id": 2,
                    "generation_mw": 7.0,
                },
                {
                    "scenario_id": 0,
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": 3,
                    "generation_mw": 15.0,
                },
                {
                    "scenario_id": 0,
                    "stage_id": 0,
                    "block_id": 1,
                    "hydro_id": 3,
                    "generation_mw": 25.0,
                },
            ],
        )

        # A third bus (7) with a thermal plant and NO hydro plant at all --
        # the negative-space check: this bus must not surface a nonzero (or
        # any other bus's) hydro figure.
        write_entity_bus_json(tmp_path, "thermals", [{"id": 1, "bus_id": 7}])
        write_sim_entity_partition(
            out,
            "thermals",
            scenario_id=0,
            rows=[
                {
                    "scenario_id": 0,
                    "stage_id": 0,
                    "block_id": 0,
                    "thermal_id": 1,
                    "generation_mw": 50.0,
                },
                {
                    "scenario_id": 0,
                    "stage_id": 0,
                    "block_id": 1,
                    "thermal_id": 1,
                    "generation_mw": 60.0,
                },
            ],
        )
        return out

    def test_bus_sum_equals_plant_total_and_bus_with_no_plants_stays_null(
        self, tmp_path: Path
    ) -> None:
        out = self._build_case(tmp_path)

        bus_result = read_novomodelo_bus_aggregates(out)
        plant_result = read_novomodelo_hydro_means(out)

        # Plant-keyed total (unchanged simulation/hydros/ path): sum of
        # each plant's block-hours-weighted mean MW at stage 0.
        # (5000/300 + 1700/300 + 6500/300) == 13200/300 == 44.0.
        plant_total = plant_result.filter(pl.col("stage_id") == 0)[
            "generation_mw"
        ].sum()
        assert plant_total == pytest.approx(44.0)

        # Bus-summed total (the new read_novomodelo_bus_aggregates source): sum
        # of each bus's hydro_gen_mw_p50 (== the single-scenario exact
        # value) at stage 0. A null (bus 7 -- no hydro plant) must not
        # silently contribute a stray nonzero via an unguarded sum.
        bus_hydro = bus_result.filter(pl.col("stage_id") == 0)
        bus_total = bus_hydro["hydro_gen_mw_p50"].fill_null(0.0).sum()

        assert bus_total == pytest.approx(plant_total)
        assert bus_total == pytest.approx(44.0)

        # Sign/magnitude sanity: bus 5 (two plants) and bus 6 (one plant)
        # carry distinct, non-overlapping figures; bus 7 (no hydro plant)
        # is null, not a stray copy of another bus's figure.
        bus5 = bus_hydro.filter(pl.col("bus_id") == 5)["hydro_gen_mw_p50"][0]
        bus6 = bus_hydro.filter(pl.col("bus_id") == 6)["hydro_gen_mw_p50"][0]
        bus7_rows = bus_hydro.filter(pl.col("bus_id") == 7)
        assert bus5 == pytest.approx((1000.0 + 4000.0 + 300.0 + 1400.0) / 300.0)
        assert bus6 == pytest.approx((1500.0 + 5000.0) / 300.0)
        assert bus7_rows.height == 1
        assert bus7_rows["hydro_gen_mw_p50"][0] is None


# ---------------------------------------------------------------------------
# read_novomodelo_lines / read_novomodelo_line_bounds /
# read_novomodelo_training_metadata
# ---------------------------------------------------------------------------


class TestReadNovomodeloLines:
    def test_present_returns_lines_list(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        system_dir = tmp_path / "system"
        system_dir.mkdir(parents=True)
        lines = [{"id": 0, "source_bus_id": 0, "target_bus_id": 1}]
        (system_dir / "lines.json").write_text(
            json.dumps({"lines": lines}), encoding="utf-8"
        )

        assert read_novomodelo_lines(out) == lines

    def test_absent_returns_empty_list(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()

        assert read_novomodelo_lines(out) == []

    @pytest.mark.parametrize(
        ("content", "reason"),
        [
            ("not valid json{", "Reason: JSONDecodeError: "),
            ('{"lines": {"id": 0}}', "Reason: no list of lines"),
            ("[]", "Reason: no list of lines"),
        ],
    )
    def test_a_file_without_a_list_of_lines_is_no_lines_and_one_warning(
        self, tmp_path: Path, content: str, reason: str
    ) -> None:
        out = tmp_path / "output"
        system_dir = tmp_path / "system"
        system_dir.mkdir(parents=True)
        (system_dir / "lines.json").write_text(content, encoding="utf-8")

        with dx.collect() as collected:
            assert read_novomodelo_lines(out) == []
            assert read_novomodelo_lines(out) == []

        [warning] = dx.finalize_diagnostics(collected)
        assert warning.code == "report-file-unreadable"
        assert warning.severity is Severity.WARNING
        assert "treats the case as having no lines" in warning.summary
        assert warning.notes[-1].startswith(reason)


class TestReadNovomodeloFphaFitConfig:
    def test_reads_each_fit_range_per_stage(self, tmp_path: Path) -> None:
        write_stages_json(tmp_path, {0: [(0, 1.0)], 1: [(0, 1.0)]})
        system_dir = tmp_path / "system"
        system_dir.mkdir()
        config = {
            "source": "computed",
            "fitting_window": {"volume_min_hm3": 50.0, "volume_max_hm3": 50.0},
            "spillage_fit_max_m3s": 900.0,
            "turbined_fit_max_m3s": 410.0,
        }
        models = [
            {
                "hydro_id": 3,
                "stage_ranges": [
                    {"start_stage_id": 0, "end_stage_id": None, "model": "fpha"}
                    | {"fpha_config": config}
                ],
            }
        ]
        (system_dir / "hydro_production_models.json").write_text(
            json.dumps({"production_models": models}), encoding="utf-8"
        )

        result = read_novomodelo_fpha_fit_config(tmp_path / "output")

        assert result["stage_id"].to_list() == [0, 1]
        assert result["spillage_fit_max_m3s"].to_list() == [900.0, 900.0]
        assert result["turbined_fit_max_m3s"].to_list() == [410.0, 410.0]

    def test_reads_each_season_at_its_stages(self, tmp_path: Path) -> None:
        stages = [
            {"id": stage_id, "season_id": season, "blocks": [{"id": 0, "hours": 1.0}]}
            for stage_id, season in enumerate([8, 9, 8])
        ]
        (tmp_path / "stages.json").write_text(
            json.dumps({"stages": stages}), encoding="utf-8"
        )
        system_dir = tmp_path / "system"
        system_dir.mkdir()
        models = [
            {
                "hydro_id": 3,
                "selection_mode": "seasonal",
                "default_model": "constant_productivity",
                "seasons": [
                    {
                        "season_id": season,
                        "model": "fpha",
                        "fpha_config": {
                            "source": "computed",
                            "turbined_fit_max_m3s": q_max,
                        },
                    }
                    for season, q_max in ((8, 400.0), (9, 420.0))
                ],
            }
        ]
        (system_dir / "hydro_production_models.json").write_text(
            json.dumps({"production_models": models}), encoding="utf-8"
        )

        result = read_novomodelo_fpha_fit_config(tmp_path / "output").sort("stage_id")

        assert result["stage_id"].to_list() == [0, 1, 2]
        assert result["turbined_fit_max_m3s"].to_list() == [400.0, 420.0, 400.0]


class TestReadNovomodeloHydroBlockOperation:
    def test_carries_the_inflow_non_negativity_slack(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        write_stages_json(tmp_path, {0: [(0, 2.0)]})
        write_sim_entity_partition(
            out,
            "hydros",
            0,
            [
                {
                    "stage_id": 0,
                    "block_id": 0,
                    "hydro_id": hydro_id,
                    "turbined_m3s": 10.0,
                    "spillage_m3s": 0.0,
                    "outflow_m3s": 10.0,
                    "incremental_inflow_m3s": incremental,
                    "inflow_nonnegativity_slack_m3s": slack,
                    "storage_initial_hm3": 100.0,
                    "storage_final_hm3": 90.0,
                    "generation_mw": 5.0,
                    "water_value_per_hm3": 1.0,
                    "accumulated_productivity_mw_per_m3s": 0.5,
                }
                for hydro_id, incremental, slack in ((1, -5.0, 5.0), (2, 7.0, 0.0))
            ],
        )

        frame = read_novomodelo_hydro_block_operation(out).sort("hydro_id")

        assert frame.schema["inflow_nonnegativity_slack_m3s"] == pl.Float64
        assert frame["inflow_nonnegativity_slack_m3s"].to_list() == [5.0, 0.0]
        assert frame["incremental_inflow_m3s"].to_list() == [-5.0, 7.0]


class TestReadNovomodeloInflowNonNegativityMethod:
    @staticmethod
    def _write_config(directory: Path, text: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(text, encoding="utf-8")

    def test_absent_config_has_no_method(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()

        assert read_novomodelo_inflow_non_negativity_method(out) is None

    def test_config_without_the_key_has_the_default_method(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        self._write_config(out, json.dumps({"modeling": {}}))

        assert (
            read_novomodelo_inflow_non_negativity_method(out)
            == "truncation_with_penalty"
        )

    def test_returns_the_configured_method(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        config = {"modeling": {"inflow_non_negativity": {"method": "penalty"}}}
        self._write_config(out, json.dumps(config))

        assert read_novomodelo_inflow_non_negativity_method(out) == "penalty"

    def test_reads_the_case_directory_config(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        config = {"modeling": {"inflow_non_negativity": {"method": "none"}}}
        self._write_config(tmp_path, json.dumps(config))

        assert read_novomodelo_inflow_non_negativity_method(out) == "none"

    def test_case_directory_config_wins_over_the_output_directory(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        for directory, method in ((tmp_path, "none"), (out, "penalty")):
            config = {"modeling": {"inflow_non_negativity": {"method": method}}}
            self._write_config(directory, json.dumps(config))

        assert read_novomodelo_inflow_non_negativity_method(out) == "none"

    @pytest.mark.parametrize(
        "text",
        [
            "{",
            "[]",
            '{"modeling": []}',
            '{"modeling": {"inflow_non_negativity": {"method": 3}}}',
        ],
    )
    def test_unusable_config_raises(self, tmp_path: Path, text: str) -> None:
        out = tmp_path / "output"
        out.mkdir()
        self._write_config(tmp_path, text)

        with pytest.raises(NovomodeloReadError, match="config.json"):
            read_novomodelo_inflow_non_negativity_method(out)


class TestReadNovomodeloLineBounds:
    def test_present_returns_raw_frame(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        constraints_dir = tmp_path / "constraints"
        constraints_dir.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "line_id": pa.array([0], type=pa.int32()),
                    "stage_id": pa.array([0], type=pa.int32()),
                    "block_id": pa.array([None], type=pa.int32()),
                    "direct_mw": pa.array([1200.0], type=pa.float64()),
                    "reverse_mw": pa.array([800.0], type=pa.float64()),
                }
            ),
            constraints_dir / "line_bounds.parquet",
        )

        df = read_novomodelo_line_bounds(out)

        assert not df.is_empty()
        assert df["direct_mw"][0] == 1200.0
        assert df["reverse_mw"][0] == 800.0

    def test_absent_returns_empty_frame(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()

        df = read_novomodelo_line_bounds(out)

        assert df.is_empty()
        assert {"line_id", "stage_id", "block_id", "direct_mw", "reverse_mw"}.issubset(
            set(df.columns)
        )

    def test_corrupt_parquet_raises_novomodeloreaderror(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        write_corrupt_parquet(tmp_path / "constraints" / "line_bounds.parquet")

        with pytest.raises(NovomodeloReadError, match="line_bounds.parquet"):
            read_novomodelo_line_bounds(out)


def _write_line_case(
    case_dir: Path,
    rows: list[tuple[int, int, int | None, float, float]],
    capacities: dict[int, tuple[float, float]],
) -> list[dict]:
    """Write ``line_bounds.parquet`` rows ``(line, stage, block, direct, reverse)``
    and return the ``lines.json`` lines for capacities ``{line: (direct, reverse)}``."""
    constraints_dir = case_dir / "constraints"
    constraints_dir.mkdir(parents=True)
    line, stage, block, direct, reverse = zip(*rows, strict=True)
    pq.write_table(
        pa.table(
            {
                "line_id": pa.array(line, type=pa.int32()),
                "stage_id": pa.array(stage, type=pa.int32()),
                "block_id": pa.array(block, type=pa.int32()),
                "direct_mw": pa.array(direct, type=pa.float64()),
                "reverse_mw": pa.array(reverse, type=pa.float64()),
            }
        ),
        constraints_dir / "line_bounds.parquet",
    )
    return [
        {"id": lid, "capacity": {"direct_mw": d, "reverse_mw": r}}
        for lid, (d, r) in capacities.items()
    ]


class TestReadNovomodeloStageLineBounds:
    def test_weights_block_overrides_by_hours_not_the_base_row(
        self, tmp_path: Path
    ) -> None:
        write_stages_json(tmp_path, {0: [(0, 30.0), (1, 69.0), (2, 69.0)]})
        lines = _write_line_case(
            tmp_path,
            [
                (4, 0, None, 6000.0, 4200.0),
                (4, 0, 0, 6000.0, 4200.0),
                (4, 0, 1, 4000.0, 1000.0),
                (4, 0, 2, 4000.0, 1000.0),
            ],
            {4: (6000.0, 4200.0)},
        )

        row = read_novomodelo_stage_line_bounds(tmp_path / "output", lines).row(
            0, named=True
        )

        assert row["direct_mw"] == pytest.approx((30 * 6000 + 138 * 4000) / 168)
        assert row["reverse_mw"] == pytest.approx((30 * 4200 + 138 * 1000) / 168)

    def test_block_without_override_takes_the_base_row(self, tmp_path: Path) -> None:
        write_stages_json(tmp_path, {0: [(0, 1.0), (1, 3.0)]})
        lines = _write_line_case(
            tmp_path,
            [(0, 0, None, 800.0, 600.0), (0, 0, 1, 400.0, 200.0)],
            {0: (5000.0, 5000.0)},
        )

        row = read_novomodelo_stage_line_bounds(tmp_path / "output", lines).row(
            0, named=True
        )

        assert row["direct_mw"] == pytest.approx((800.0 + 3 * 400.0) / 4)
        assert row["reverse_mw"] == pytest.approx((600.0 + 3 * 200.0) / 4)

    def test_block_without_any_row_takes_the_line_capacity(
        self, tmp_path: Path
    ) -> None:
        write_stages_json(tmp_path, {0: [(0, 1.0), (1, 1.0)]})
        lines = _write_line_case(
            tmp_path, [(0, 0, 0, 400.0, 200.0)], {0: (1000.0, 800.0)}
        )

        row = read_novomodelo_stage_line_bounds(tmp_path / "output", lines).row(
            0, named=True
        )

        assert row["direct_mw"] == pytest.approx(700.0)
        assert row["reverse_mw"] == pytest.approx(500.0)

    def test_direction_unbounded_in_one_block_stays_unbounded(
        self, tmp_path: Path
    ) -> None:
        write_stages_json(tmp_path, {0: [(0, 1.0), (1, 1.0)]})
        lines = _write_line_case(
            tmp_path,
            [(0, 0, 0, 99999.0, 300.0), (0, 0, 1, 500.0, 300.0)],
            {0: (99999.0, 300.0)},
        )

        row = read_novomodelo_stage_line_bounds(tmp_path / "output", lines).row(
            0, named=True
        )

        assert row["direct_mw"] == 99999.0
        assert row["reverse_mw"] == pytest.approx(300.0)

    def test_without_block_hours_block_zero_stands_for_the_stage(
        self, tmp_path: Path
    ) -> None:
        lines = _write_line_case(
            tmp_path,
            [(0, 0, 0, 400.0, 200.0), (0, 0, 1, 900.0, 700.0)],
            {0: (1000.0, 1000.0)},
        )

        row = read_novomodelo_stage_line_bounds(tmp_path / "output", lines).row(
            0, named=True
        )

        assert (row["direct_mw"], row["reverse_mw"]) == (400.0, 200.0)

    def test_absent_line_bounds_returns_typed_empty_frame(self, tmp_path: Path) -> None:
        df = read_novomodelo_stage_line_bounds(tmp_path / "output", [])

        assert df.is_empty()
        assert df.columns == ["line_id", "stage_id", "direct_mw", "reverse_mw"]


class TestReadNovomodeloNetworkLines:
    def _write_case(self, case_dir: Path, lines_json: str) -> Path:
        (case_dir / "system").mkdir(parents=True)
        (case_dir / "system" / "lines.json").write_text(lines_json, encoding="utf-8")
        _write_line_case(case_dir, [(0, 0, None, 1200.0, 800.0)], {})
        return case_dir / "output"

    def test_reads_the_stage_bounds_and_the_lines_verbatim(
        self, tmp_path: Path
    ) -> None:
        lines = [{"id": 0, "name": "SE-S", "capacity": {"direct_mw": 1000.0}}]
        output_dir = self._write_case(tmp_path, json.dumps({"lines": lines}))

        with dx.collect() as collected:
            line_bounds, line_meta = read_novomodelo_network_lines(output_dir)

        assert line_meta == lines
        row = line_bounds.row(0, named=True)
        assert (row["direct_mw"], row["reverse_mw"]) == (1200.0, 800.0)
        assert collected == []

    def test_a_case_without_the_files_is_empty_and_notes_the_bounds(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with dx.collect() as collected:
            line_bounds, line_meta = read_novomodelo_network_lines(output_dir)

        assert line_bounds.is_empty()
        assert line_meta == []
        assert [(d.code, d.title) for d in collected] == [
            ("report-file-missing", "constraints/line_bounds.parquet not found")
        ]

    def test_unparseable_lines_json_keeps_the_bounds_and_warns_once(
        self, tmp_path: Path
    ) -> None:
        output_dir = self._write_case(tmp_path, "{not json")

        with dx.collect() as collected:
            line_bounds, line_meta = read_novomodelo_network_lines(output_dir)

        assert line_meta == []
        assert line_bounds.row(0, named=True)["direct_mw"] == 1200.0
        assert [(d.code, d.severity, d.title) for d in collected] == [
            (
                "report-file-unreadable",
                Severity.WARNING,
                "system/lines.json could not be read",
            )
        ]
        assert "no line on the Network tab" in collected[0].summary


class TestReadNovomodeloTrainingMetadata:
    def test_present_returns_dict(self, tmp_path: Path) -> None:
        out = tmp_path / "case" / "output"
        training_dir = out / "training"
        training_dir.mkdir(parents=True)
        (training_dir / "metadata.json").write_text(
            json.dumps({"version": "0.14.3", "duration_seconds": 12.0}),
            encoding="utf-8",
        )

        assert read_novomodelo_training_metadata(out) == {
            "version": "0.14.3",
            "duration_seconds": 12.0,
        }

    def test_absent_returns_empty_dict(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()

        assert read_novomodelo_training_metadata(out) == {}

    def test_unified_path_rule_resolves_the_case_dir_parent_fallback(
        self, tmp_path: Path
    ) -> None:
        """A case layout with no doubled ``output/`` segment resolves through
        the one unified candidate search."""
        case_dir = tmp_path / "case"
        training_dir = case_dir / "training"
        training_dir.mkdir(parents=True)
        (training_dir / "metadata.json").write_text(
            json.dumps({"version": "0.14.3"}), encoding="utf-8"
        )

        assert read_novomodelo_training_metadata(case_dir / "output") == {
            "version": "0.14.3"
        }

    def test_corrupt_json_degrades_to_empty_dict(self, tmp_path: Path) -> None:
        out = tmp_path / "case" / "output"
        training_dir = out / "training"
        training_dir.mkdir(parents=True)
        (training_dir / "metadata.json").write_text("not valid json{", encoding="utf-8")

        with dx.collect() as collected:
            assert read_novomodelo_training_metadata(out) == {}
        assert [(d.code, d.title) for d in collected] == [
            (
                "report-file-unreadable",
                "output/training/metadata.json could not be read",
            )
        ]


class TestNovomodeloSoftwareVersion:
    def test_reads_software_version(self) -> None:
        assert (
            novomodelo_software_version(
                {"software": "novomodelo", "software_version": "0.18.0"}
            )
            == "0.18.0"
        )

    def test_absent_or_non_string_returns_none(self) -> None:
        assert novomodelo_software_version({}) is None
        assert novomodelo_software_version({"software_version": 18}) is None


# ---------------------------------------------------------------------------
# read_novomodelo_scenario_weights: the probability of each simulation scenario
# ---------------------------------------------------------------------------


def _write_scenario_summary(
    output_dir: Path, probabilities: list[float | None]
) -> Path:
    path = output_dir / "simulation" / "scenario_summary.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "scenario_id": pa.array(range(len(probabilities)), type=pa.int32()),
                "probability": pa.array(probabilities, type=pa.float64()),
                "discounted_immediate_cost": pa.array(
                    [1.0] * len(probabilities), type=pa.float64()
                ),
            }
        ),
        path,
    )
    return path


class TestReadNovomodeloScenarioWeights:
    def test_enumerated_run_reads_each_probability(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_scenario_summary(out, [0.25, 0.75])

        weights = read_novomodelo_scenario_weights(out)

        assert weights.probabilities is not None
        assert dict(weights.probabilities.iter_rows()) == {0: 0.25, 1: 0.75}

    def test_sampled_run_weighs_scenarios_the_same(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_scenario_summary(out, [None, None, None])

        assert read_novomodelo_scenario_weights(out).uniform

    def test_training_only_run_weighs_scenarios_the_same(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()

        with dx.collect() as collected:
            assert read_novomodelo_scenario_weights(out).uniform
        assert collected == []

    def test_simulation_without_a_summary_warns(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        (out / "simulation").mkdir(parents=True)

        with dx.collect() as collected:
            assert read_novomodelo_scenario_weights(out).uniform
        [diag] = collected
        assert diag.code == "scenario-probabilities-missing"
        assert diag.severity is Severity.WARNING
        assert "equal weights" in diag.summary

    def test_some_scenarios_without_a_probability_raise(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_scenario_summary(out, [0.5, None])

        with pytest.raises(NovomodeloReadError, match="1 of 2 scenarios"):
            read_novomodelo_scenario_weights(out)

    def test_corrupt_summary_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        write_corrupt_parquet(out / "simulation" / "scenario_summary.parquet")

        with pytest.raises(NovomodeloReadError):
            read_novomodelo_scenario_weights(out)


class TestMeansReadersUseTheScenarioProbabilities:
    """A means reader returns novomodelo's expectation, not the sample mean."""

    @staticmethod
    def _two_scenario_thermals(output_dir: Path) -> None:
        for scenario_id, generation in ((0, 100.0), (1, 200.0)):
            write_sim_entity_partition(
                output_dir,
                "thermals",
                scenario_id,
                [
                    {
                        "stage_id": 0,
                        "block_id": 0,
                        "thermal_id": 0,
                        "generation_mw": generation,
                    }
                ],
            )

    def test_enumerated_run_takes_the_expectation(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        self._two_scenario_thermals(out)
        _write_scenario_summary(out, [0.25, 0.75])

        means = read_novomodelo_thermal_means(out)

        assert means["generation_mw"].to_list() == pytest.approx([175.0])

    def test_sampled_run_takes_the_sample_mean(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        self._two_scenario_thermals(out)
        _write_scenario_summary(out, [None, None])

        means = read_novomodelo_thermal_means(out)

        assert means["generation_mw"].to_list() == pytest.approx([150.0])


# ---------------------------------------------------------------------------
# The load buses' net import over their lines, and the default exchange cost.
# ---------------------------------------------------------------------------


def _write_exchange_case(
    case_dir: Path, flows: dict[int, list[tuple[int, int, float]]]
) -> Path:
    """SE (bus 0) -> NE (bus 1) on line 0 and NE -> FC (bus 2) on line 1, FC
    without load; *flows* maps a scenario to its ``(line, block, MW)`` rows at
    stage 0, whose blocks last 10 h and 30 h."""
    write_stages_json(case_dir, {0: [(0, 10.0), (1, 30.0)]})
    _write_load_stats(
        case_dir,
        {"bus_id": [0, 1, 2], "stage_id": [0, 0, 0], "mean_mw": [9.0, 9.0, 0.0]},
    )
    system = case_dir / "system"
    system.mkdir(parents=True, exist_ok=True)
    (system / "lines.json").write_text(
        json.dumps(
            {
                "lines": [
                    {"id": 0, "source_bus_id": 0, "target_bus_id": 1},
                    {"id": 1, "source_bus_id": 1, "target_bus_id": 2},
                ]
            }
        ),
        encoding="utf-8",
    )
    out = case_dir / "output"
    for scenario_id, rows in flows.items():
        write_sim_entity_partition(
            out,
            "exchanges",
            scenario_id,
            [
                {"stage_id": 0, "block_id": block, "line_id": line, "net_flow_mw": mw}
                for line, block, mw in rows
            ],
        )
    return out


class TestReadNovomodeloBusNetImport:
    def test_a_bus_imports_what_flows_in_less_what_flows_out(
        self, tmp_path: Path
    ) -> None:
        out = _write_exchange_case(
            tmp_path, {0: [(0, 0, 100.0), (0, 1, 200.0), (1, 0, -40.0), (1, 1, -40.0)]}
        )

        net_import = read_novomodelo_bus_net_import(out)

        # Line 0 carries (100 * 10 + 200 * 30) / 40 = 175 MW from SE to NE,
        # and line 1 40 MW from FC to NE; FC has no load and no row.
        assert net_import.to_dicts() == [
            {"entity_id": 0, "stage_id": 0, "net_import_mw": pytest.approx(-175.0)},
            {"entity_id": 1, "stage_id": 0, "net_import_mw": pytest.approx(215.0)},
        ]

    def test_scenarios_weigh_by_their_probability(self, tmp_path: Path) -> None:
        out = _write_exchange_case(
            tmp_path,
            {
                0: [(0, 0, 100.0), (0, 1, 100.0), (1, 0, 0.0), (1, 1, 0.0)],
                1: [(0, 0, 200.0), (0, 1, 200.0), (1, 0, 0.0), (1, 1, 0.0)],
            },
        )
        _write_scenario_summary(out, [0.25, 0.75])

        net_import = read_novomodelo_bus_net_import(out)

        assert net_import["net_import_mw"].to_list() == pytest.approx([-175.0, 175.0])

    def test_a_case_without_lines_has_no_net_import(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        write_sim_entity_partition(
            out,
            "exchanges",
            0,
            [{"stage_id": 0, "block_id": 0, "line_id": 0, "net_flow_mw": 1.0}],
        )

        net_import = read_novomodelo_bus_net_import(out)

        assert net_import.is_empty()
        assert net_import.columns == ["entity_id", "stage_id", "net_import_mw"]

    def _with_bus_prices(self, out: Path) -> None:
        for scenario_id in (0, 1):
            write_sim_entity_partition(
                out,
                "buses",
                scenario_id,
                [
                    {"stage_id": 0, "block_id": block, "bus_id": bus, "spot_price": 1.0}
                    for bus in (0, 1, 2)
                    for block in (0, 1)
                ],
            )

    def test_bus_means_carry_it_and_leave_a_bus_without_load_null(
        self, tmp_path: Path
    ) -> None:
        out = _write_exchange_case(
            tmp_path, {s: [(0, b, 100.0) for b in (0, 1)] for s in (0, 1)}
        )
        self._with_bus_prices(out)

        means = read_novomodelo_bus_means(out)

        assert means.sort("entity_id")["net_import_mw"].to_list() == [
            pytest.approx(-100.0),
            pytest.approx(100.0),
            None,
        ]

    def test_bus_percentiles_carry_its_bands(self, tmp_path: Path) -> None:
        out = _write_exchange_case(
            tmp_path,
            {
                0: [(0, b, 100.0) for b in (0, 1)],
                1: [(0, b, 300.0) for b in (0, 1)],
            },
        )
        self._with_bus_prices(out)

        bands = read_novomodelo_bus_percentiles(out).sort("entity_id")

        assert bands.select(
            "net_import_mw_p10", "net_import_mw_p50", "net_import_mw_p90"
        ).rows() == [
            (-280.0, -200.0, -120.0),
            (120.0, 200.0, 280.0),
            (None, None, None),
        ]

    def _priced_case(self, tmp_path: Path) -> Path:
        out = _write_exchange_case(
            tmp_path, {s: [(0, b, 100.0) for b in (0, 1)] for s in (0, 1)}
        )
        self._with_bus_prices(out)
        _write_scenario_summary(out, [0.5, 0.5])
        return out

    def test_asking_for_the_net_import_is_the_default(self, tmp_path: Path) -> None:
        out = self._priced_case(tmp_path)

        means = read_novomodelo_bus_means(out)
        bands = read_novomodelo_bus_percentiles(out)

        assert means.equals(read_novomodelo_bus_means(out, include_net_import=True))
        assert bands.equals(
            read_novomodelo_bus_percentiles(out, include_net_import=True)
        )
        assert means.sort("entity_id")["net_import_mw"].to_list() == [
            pytest.approx(-100.0),
            pytest.approx(100.0),
            None,
        ]
        assert "net_import_mw_p50" in bands.columns

    def test_by_default_a_malformed_line_list_is_reported_by_each_reader(
        self, tmp_path: Path
    ) -> None:
        out = self._priced_case(tmp_path)
        (out.parent / "system" / "lines.json").write_text("{not json", encoding="utf-8")

        with dx.collect() as found:
            means = read_novomodelo_bus_means(out)
            bands = read_novomodelo_bus_percentiles(out)

        assert [d.code for d in found] == ["report-file-unreadable"] * 2
        assert means["net_import_mw"].null_count() == means.height
        assert "net_import_mw_p50" not in bands.columns

    def test_without_the_net_import_a_malformed_line_list_is_not_read(
        self, tmp_path: Path
    ) -> None:
        out = self._priced_case(tmp_path)
        with_net_import = read_novomodelo_bus_means(out)
        bands_with_net_import = read_novomodelo_bus_percentiles(out)
        (out.parent / "system" / "lines.json").write_text("{not json", encoding="utf-8")

        with dx.collect() as found:
            means = read_novomodelo_bus_means(out, include_net_import=False)
            bands = read_novomodelo_bus_percentiles(out, include_net_import=False)

        assert found == []
        assert means.schema == with_net_import.schema
        assert means["net_import_mw"].null_count() == means.height
        assert means.drop("net_import_mw").equals(with_net_import.drop("net_import_mw"))
        assert bands.equals(
            bands_with_net_import.drop(
                [c for c in bands_with_net_import.columns if "net_import" in c]
            )
        )

    def test_without_the_net_import_a_case_reads_as_one_without_lines(
        self, tmp_path: Path
    ) -> None:
        out = self._priced_case(tmp_path)
        (out.parent / "system" / "lines.json").unlink()

        means = read_novomodelo_bus_means(out)
        bands = read_novomodelo_bus_percentiles(out)

        assert means.equals(read_novomodelo_bus_means(out, include_net_import=False))
        assert bands.equals(
            read_novomodelo_bus_percentiles(out, include_net_import=False)
        )
        assert means.schema["net_import_mw"] == pl.Float64
        assert not any("net_import" in column for column in bands.columns)


class TestReadNovomodeloDefaultExchangeCost:
    def test_reads_the_line_exchange_cost(self, tmp_path: Path) -> None:
        (tmp_path / "penalties.json").write_text(
            json.dumps({"line": {"exchange_cost": 0.0002}}), encoding="utf-8"
        )

        with dx.collect() as found:
            cost = read_novomodelo_default_exchange_cost(tmp_path / "output")

        assert cost == 0.0002
        assert found == []

    def test_a_missing_file_is_noted(self, tmp_path: Path) -> None:
        with dx.collect() as found:
            cost = read_novomodelo_default_exchange_cost(tmp_path / "output")

        assert cost is None
        assert [d.code for d in found] == ["report-file-missing"]

    @pytest.mark.parametrize(
        "text", ["{not json", json.dumps({"line": {}}), json.dumps([1])]
    )
    def test_a_file_without_a_usable_cost_is_reported(
        self, tmp_path: Path, text: str
    ) -> None:
        (tmp_path / "penalties.json").write_text(text, encoding="utf-8")

        with dx.collect() as found:
            cost = read_novomodelo_default_exchange_cost(tmp_path / "output")

        assert cost is None
        assert [(d.code, d.severity) for d in found] == [
            ("report-file-unreadable", Severity.WARNING)
        ]


class TestReadNovomodeloGnlDeliveredCost:
    """The anticipated generation delivered over the horizon, at each plant's
    price, discounted and weighted by the scenario probabilities."""

    @staticmethod
    def _case(tmp_path: Path, *, anticipated: bool = True) -> Path:
        case_dir = tmp_path / "case"
        output_dir = case_dir / "output"
        (case_dir / "system").mkdir(parents=True)
        plant: dict[str, object] = {"id": 7, "name": "GNL", "cost_per_mwh": 100.0}
        if anticipated:
            plant["anticipated_config"] = {"lead_time_hours": 1440.0}
        (case_dir / "system" / "thermals.json").write_text(
            json.dumps({"thermals": [plant, {"id": 8, "cost_per_mwh": 50.0}]}),
            encoding="utf-8",
        )
        (case_dir / "constraints").mkdir()
        pl.DataFrame(
            {
                "thermal_id": [7],
                "stage_id": [1],
                "block_id": [None],
                "min_generation_mw": [0.0],
                "max_generation_mw": [500.0],
                "cost_per_mwh": [200.0],
            },
            schema_overrides={"block_id": pl.Int32},
        ).write_parquet(case_dir / "constraints" / "thermal_bounds.parquet")
        for scenario, mwh in ((0, 10.0), (1, 30.0)):
            rows: list[dict[str, object]] = [
                {
                    "stage_id": stage,
                    "block_id": block,
                    "thermal_id": thermal,
                    "generation_mwh": mwh if thermal == 7 else 99.0,
                    "is_anticipated": thermal == 7 and anticipated,
                }
                for stage in (0, 1)
                for block in (0, 1)
                for thermal in (7, 8)
            ]
            write_sim_entity_partition(output_dir, "thermals", scenario, rows)
            write_sim_entity_partition(
                output_dir,
                "costs",
                scenario,
                [
                    {"stage_id": stage, "block_id": block, "discount_factor": factor}
                    for stage, factor in ((0, 1.0), (1, 0.5))
                    for block in (0, 1)
                ],
            )
        pl.DataFrame(
            {"scenario_id": [0, 1], "probability": [0.75, 0.25]}
        ).write_parquet(output_dir / "simulation" / "scenario_summary.parquet")
        return output_dir

    def test_prices_each_stage_and_weighs_the_scenarios(self, tmp_path: Path) -> None:
        output_dir = self._case(tmp_path)

        # Per scenario: stage 0 at the base 100 $/MWh, stage 1 at its 200 $/MWh
        # stage row discounted by 0.5, two blocks each: 2 x mwh x (100 + 100).
        assert read_novomodelo_gnl_delivered_cost(output_dir) == pytest.approx(
            0.75 * 4_000.0 + 0.25 * 12_000.0
        )

    def test_none_without_an_anticipated_plant(self, tmp_path: Path) -> None:
        assert (
            read_novomodelo_gnl_delivered_cost(self._case(tmp_path, anticipated=False))
            is None
        )

    def test_none_without_thermal_output(self, tmp_path: Path) -> None:
        assert read_novomodelo_gnl_delivered_cost(tmp_path / "output") is None


class TestReadNovomodeloConvergenceBounds:
    COLUMNS = [
        "iteration",
        "lower_bound",
        "upper_bound",
        "upper_bound_std",
        "upper_bound_kind",
    ]

    @staticmethod
    def _write(tmp_path: Path, frame: pl.DataFrame) -> Path:
        output_dir = tmp_path / "output"
        (output_dir / "training").mkdir(parents=True)
        frame.write_parquet(output_dir / "training" / "convergence.parquet")
        return output_dir

    @staticmethod
    def _current(**overrides: pl.Series) -> pl.DataFrame:
        columns = {
            "iteration": pl.Series("iteration", [2, 1], dtype=pl.Int32),
            "lower_bound": pl.Series("lower_bound", [200.0, 100.0]),
            "upper_bound": pl.Series("upper_bound", [260.0, 150.0]),
            "upper_bound_std": pl.Series("upper_bound_std", [None, None], pl.Float64),
            "upper_bound_kind": pl.Series("upper_bound_kind", ["exact", "exact"]),
            "forward_passes": pl.Series("forward_passes", [5, 5]),
        }
        return pl.DataFrame(list({**columns, **overrides}.values()))

    def test_returns_the_five_columns_sorted_by_iteration(self, tmp_path: Path) -> None:
        result = read_novomodelo_convergence_bounds(
            self._write(tmp_path, self._current())
        )

        assert result.columns == self.COLUMNS
        assert dict(result.schema) == {
            "iteration": pl.Int64,
            "lower_bound": pl.Float64,
            "upper_bound": pl.Float64,
            "upper_bound_std": pl.Float64,
            "upper_bound_kind": pl.Utf8,
        }
        assert result["iteration"].to_list() == [1, 2]
        assert result["lower_bound"].to_list() == [100.0, 200.0]
        assert result["upper_bound"].to_list() == [150.0, 260.0]
        assert result["upper_bound_std"].to_list() == [None, None]
        assert result["upper_bound_kind"].to_list() == ["exact", "exact"]

    def test_a_statistical_bound_keeps_its_std_and_kind(self, tmp_path: Path) -> None:
        frame = self._current(
            upper_bound_std=pl.Series("upper_bound_std", [7.0, 9.0]),
            upper_bound_kind=pl.Series(
                "upper_bound_kind", ["statistical", "statistical"]
            ),
        )

        result = read_novomodelo_convergence_bounds(self._write(tmp_path, frame))

        assert result["upper_bound_std"].to_list() == [9.0, 7.0]
        assert result["upper_bound_kind"].to_list() == ["statistical", "statistical"]

    def test_the_schema_before_0_14_reads_the_legacy_upper_bound(
        self, tmp_path: Path
    ) -> None:
        frame = pl.DataFrame(
            {
                "iteration": [1, 2],
                "lower_bound": [100.0, 200.0],
                "upper_bound_mean": [150.0, 260.0],
            }
        )

        result = read_novomodelo_convergence_bounds(self._write(tmp_path, frame))

        assert result.columns == self.COLUMNS
        assert result["upper_bound"].to_list() == [150.0, 260.0]
        assert result["upper_bound_kind"].to_list() == [None, None]
        assert result.schema["upper_bound_kind"] == pl.Utf8
        assert result["upper_bound_std"].to_list() == [None, None]
        assert result.schema["upper_bound_std"] == pl.Float64

    def test_upper_bound_wins_over_the_legacy_column(self, tmp_path: Path) -> None:
        frame = self._current().with_columns(upper_bound_mean=pl.Series([999.0, 999.0]))

        result = read_novomodelo_convergence_bounds(self._write(tmp_path, frame))

        assert result["upper_bound"].to_list() == [150.0, 260.0]

    def test_without_a_training_directory_the_empty_frame_is_silent(
        self, tmp_path: Path
    ) -> None:
        with dx.collect() as found:
            result = read_novomodelo_convergence_bounds(tmp_path / "output")

        assert result.is_empty()
        assert result.columns == self.COLUMNS
        assert found == []

    def test_an_absent_file_in_the_training_directory_is_noted(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "output" / "training").mkdir(parents=True)

        with dx.collect() as found:
            result = read_novomodelo_convergence_bounds(tmp_path / "output")

        assert result.is_empty()
        assert result.columns == self.COLUMNS
        assert result.schema["upper_bound_kind"] == pl.Utf8
        assert [d.code for d in found] == ["report-file-missing"]

    def test_a_file_of_garbage_bytes_raises(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        write_corrupt_parquet(output_dir / "training" / "convergence.parquet")

        with pytest.raises(NovomodeloReadError, match="convergence.parquet"):
            read_novomodelo_convergence_bounds(output_dir)

    @pytest.mark.parametrize(
        ("drop", "rename", "reason"),
        [
            ("lower_bound", None, "no lower_bound column"),
            ("iteration", None, "no iteration column"),
            ("upper_bound", None, "no upper_bound column"),
            ("upper_bound", "zsup", "no upper_bound column"),
        ],
    )
    def test_a_file_lacking_a_required_column_is_reported(
        self, tmp_path: Path, drop: str, rename: str | None, reason: str
    ) -> None:
        frame = self._current().drop(drop)
        if rename:
            frame = frame.with_columns(**{rename: pl.Series([1.0, 2.0])})

        with dx.collect() as found:
            result = read_novomodelo_convergence_bounds(self._write(tmp_path, frame))

        assert result.is_empty()
        assert result.columns == self.COLUMNS
        assert [(d.code, d.severity) for d in found] == [
            ("report-file-unreadable", Severity.WARNING)
        ]
        assert f"Reason: {reason}" in found[0].notes

    @pytest.mark.parametrize("lacking", [None, "lower_bound"])
    def test_its_findings_collapse_with_those_of_the_convergence_reader(
        self, tmp_path: Path, lacking: str | None
    ) -> None:
        if lacking:
            output_dir = self._write(tmp_path, self._current().drop(lacking))
        else:
            output_dir = tmp_path / "output"
            (output_dir / "training").mkdir(parents=True)

        with dx.collect() as found:
            read_novomodelo_convergence(output_dir)
            read_novomodelo_convergence_bounds(output_dir)

        assert len(found) == 2
        assert len(dx.finalize_diagnostics(found)) == 1


class TestReadNovomodeloRequiredEntities:
    ENTRIES = [{"id": 1, "name": "SUL", "bus_id": 0}, {"id": 0, "name": "SUDESTE"}]

    @staticmethod
    def _write(tmp_path: Path, filename: str, content: str | bytes) -> Path:
        path = tmp_path / "system" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)
        return path

    @staticmethod
    def _fails[E: Exception](output_dir: Path, filename: str, error: type[E]) -> E:
        with dx.collect() as found, pytest.raises(error) as raised:
            read_novomodelo_required_entities(output_dir, filename)
        assert found == []
        return raised.value

    @pytest.mark.parametrize("entity", ["hydros", "buses", "thermals", "lines"])
    def test_a_readable_file_is_returned_in_file_order_and_silent(
        self, tmp_path: Path, entity: str
    ) -> None:
        self._write(tmp_path, f"{entity}.json", json.dumps({entity: self.ENTRIES}))

        with dx.collect() as found:
            result = read_novomodelo_required_entities(
                tmp_path / "output", f"{entity}.json"
            )

        assert result == self.ENTRIES
        assert found == []

    def test_an_empty_list_is_returned_as_empty(self, tmp_path: Path) -> None:
        self._write(tmp_path, "lines.json", json.dumps({"lines": []}))

        with dx.collect() as found:
            result = read_novomodelo_required_entities(
                tmp_path / "output", "lines.json"
            )

        assert result == []
        assert found == []

    def test_a_file_in_the_output_directory_is_found_and_named_in_the_error(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        self._write(output_dir, "buses.json", json.dumps({"buses": []}))
        assert read_novomodelo_required_entities(output_dir, "buses.json") == []

        bad = self._write(output_dir, "hydros.json", "not json")
        exc = self._fails(output_dir, "hydros.json", FieldParseError)

        assert exc.path == str(bad)

    def test_an_absent_file_raises_source_file_error(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"

        exc = self._fails(output_dir, "hydros.json", SourceFileError)

        assert exc.path == str(tmp_path / "system" / "hydros.json")
        assert exc.path.endswith("system/hydros.json")
        assert exc.field is None
        assert str(exc) == (
            f"system/hydros.json was not found for the Novomodelo output {output_dir}."
        )
        assert diagnostic_from_exception(exc, context="Comparison").code == (
            "source-file-missing"
        )

    def test_text_that_is_not_json_raises_field_parse_error(
        self, tmp_path: Path
    ) -> None:
        path = self._write(tmp_path, "hydros.json", "not json")

        exc = self._fails(tmp_path / "output", "hydros.json", FieldParseError)

        assert exc.path == str(path)
        assert exc.field == "hydros"
        assert str(exc).startswith(
            "system/hydros.json could not be parsed: JSONDecodeError: "
        )
        assert str(exc).endswith(".")
        assert isinstance(exc.__cause__, ValueError)
        assert diagnostic_from_exception(exc, context="Comparison").code == (
            "source-field-parse"
        )

    def test_a_file_that_cannot_be_decoded_raises_field_parse_error(
        self, tmp_path: Path
    ) -> None:
        path = self._write(tmp_path, "lines.json", b"\xff\xfe{")

        exc = self._fails(tmp_path / "output", "lines.json", FieldParseError)

        assert exc.path == str(path)
        assert exc.field == "lines"
        assert "UnicodeDecodeError" in str(exc)

    @pytest.mark.parametrize(
        "text",
        [
            '{"hydros": {"id": 0, "name": "FURNAS"}}',
            '{"hydros": null}',
            '{"buses": [{"id": 0, "name": "SUDESTE"}]}',
            "{}",
            "[]",
            '"hydros"',
        ],
    )
    def test_no_list_under_the_key_raises_field_parse_error(
        self, tmp_path: Path, text: str
    ) -> None:
        path = self._write(tmp_path, "hydros.json", text)

        exc = self._fails(tmp_path / "output", "hydros.json", FieldParseError)

        assert exc.path == str(path)
        assert exc.field == "hydros"
        assert "holds no list of hydros" in str(exc)
        assert diagnostic_from_exception(exc, context="Comparison").code == (
            "source-field-parse"
        )

    @pytest.mark.parametrize(
        ("filename", "payload", "field", "position"),
        [
            ("hydros.json", [{"id": 0}], "name", 1),
            ("hydros.json", [{"name": "FURNAS"}], "id", 1),
            ("hydros.json", [{"id": True, "name": "FURNAS"}], "id", 1),
            ("hydros.json", [{"id": "0", "name": "FURNAS"}], "id", 1),
            ("hydros.json", [{"id": 0.0, "name": "FURNAS"}], "id", 1),
            ("hydros.json", ["FURNAS"], "id", 1),
            ("hydros.json", [{"id": 0, "name": ""}], "name", 1),
            ("hydros.json", [{"id": 0, "name": 7}], "name", 1),
            (
                "buses.json",
                [{"id": 0, "name": "SE"}, {"id": 1, "name": None}],
                "name",
                2,
            ),
            (
                "thermals.json",
                [{"id": 4, "name": "ANGRA 1"}, {"id": 5, "name": " "}],
                "name",
                2,
            ),
            (
                "lines.json",
                [{"id": 0, "source_bus_id": 0, "target_bus_id": 1}],
                "name",
                1,
            ),
            ("lines.json", [{"id": 0, "name": "SE-S"}, {"name": "S-NE"}], "id", 2),
        ],
    )
    def test_a_bad_entry_raises_field_parse_error_at_its_position(
        self,
        tmp_path: Path,
        filename: str,
        payload: list[object],
        field: str,
        position: int,
    ) -> None:
        key = filename.removesuffix(".json")
        path = self._write(tmp_path, filename, json.dumps({key: payload}))

        exc = self._fails(tmp_path / "output", filename, FieldParseError)

        assert exc.path == str(path)
        assert exc.field == field
        assert str(exc).startswith(f"Entry {position} of system/{filename} ")
        assert diagnostic_from_exception(exc, context="Comparison").code == (
            "source-field-parse"
        )

    def test_the_first_bad_entry_decides_the_field(self, tmp_path: Path) -> None:
        payload = {"hydros": [{"id": 0, "name": "A"}, {"id": 1}, {"name": "C"}]}
        self._write(tmp_path, "hydros.json", json.dumps(payload))

        exc = self._fails(tmp_path / "output", "hydros.json", FieldParseError)

        assert exc.field == "name"
        assert str(exc).startswith("Entry 2 ")

    def test_nothing_above_debug_is_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)
        output_dir = tmp_path / "output"
        self._write(tmp_path, "buses.json", json.dumps({"buses": self.ENTRIES}))
        self._write(tmp_path, "hydros.json", "not json")

        read_novomodelo_required_entities(output_dir, "buses.json")
        self._fails(output_dir, "hydros.json", FieldParseError)
        self._fails(output_dir, "thermals.json", SourceFileError)

        assert [r for r in caplog.records if r.levelno > logging.DEBUG] == []

    def test_messages_name_no_repository_internals(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        self._write(tmp_path, "hydros.json", "not json")
        self._write(tmp_path, "buses.json", json.dumps({"buses": [{"id": 0}]}))
        self._write(tmp_path, "lines.json", "{}")
        messages = [
            str(self._fails(output_dir, name, error))
            for name, error in (
                ("hydros.json", FieldParseError),
                ("buses.json", FieldParseError),
                ("lines.json", FieldParseError),
                ("thermals.json", SourceFileError),
            )
        ]

        for leak in (
            "read_novomodelo",
            "find_system_json",
            "docs/",
            "plans/",
            "ticket-",
        ):
            assert not [m for m in messages if leak in m]


class TestReadNovomodeloRequiredGenericConstraints:
    ENTRIES = [
        {"id": 1, "name": "VMINOP_1", "slack": True},
        {"id": 0, "name": "RE_1", "expression": "hydro_generation(id=0)"},
    ]

    @staticmethod
    def _write(case_dir: Path, content: str | bytes) -> Path:
        path = case_dir / "constraints" / "generic_constraints.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)
        return path

    @staticmethod
    def _fails[E: Exception](case_dir: Path, error: type[E]) -> E:
        with dx.collect() as found, pytest.raises(error) as raised:
            read_novomodelo_required_generic_constraints(case_dir)
        assert found == []
        return raised.value

    def test_a_readable_file_is_returned_in_file_order_and_silent(
        self, tmp_path: Path
    ) -> None:
        self._write(tmp_path, json.dumps({"constraints": self.ENTRIES}))

        with dx.collect() as found:
            result = read_novomodelo_required_generic_constraints(tmp_path)

        assert result == self.ENTRIES
        assert found == []

    def test_an_empty_list_is_returned_as_empty(self, tmp_path: Path) -> None:
        self._write(tmp_path, json.dumps({"constraints": []}))

        with dx.collect() as found:
            result = read_novomodelo_required_generic_constraints(tmp_path)

        assert result == []
        assert found == []

    def test_an_absent_file_raises_source_file_error_naming_the_expected_path(
        self, tmp_path: Path
    ) -> None:
        exc = self._fails(tmp_path, SourceFileError)

        assert exc.path == str(tmp_path / "constraints" / "generic_constraints.json")
        assert exc.field is None
        assert str(exc) == (
            "constraints/generic_constraints.json was not found for the "
            f"Novomodelo case {tmp_path}."
        )
        assert diagnostic_from_exception(exc, context="Comparison").code == (
            "source-file-missing"
        )

    def test_a_case_without_the_constraints_directory_is_an_absent_file(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "system").mkdir()

        self._fails(tmp_path, SourceFileError)

    def test_text_that_is_not_json_raises_field_parse_error(
        self, tmp_path: Path
    ) -> None:
        path = self._write(tmp_path, "not json")

        exc = self._fails(tmp_path, FieldParseError)

        assert exc.path == str(path)
        assert exc.field == "constraints"
        assert str(exc).startswith(
            "constraints/generic_constraints.json could not be parsed: "
            "JSONDecodeError: "
        )
        assert str(exc).endswith(".")
        assert isinstance(exc.__cause__, ValueError)
        assert diagnostic_from_exception(exc, context="Comparison").code == (
            "source-field-parse"
        )

    def test_a_file_that_cannot_be_decoded_raises_field_parse_error(
        self, tmp_path: Path
    ) -> None:
        path = self._write(tmp_path, b"\xff\xfe{")

        exc = self._fails(tmp_path, FieldParseError)

        assert exc.path == str(path)
        assert exc.field == "constraints"
        assert "UnicodeDecodeError" in str(exc)

    def test_a_path_that_is_a_directory_raises_field_parse_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "constraints" / "generic_constraints.json"
        path.mkdir(parents=True)

        exc = self._fails(tmp_path, FieldParseError)

        assert exc.path == str(path)
        assert exc.field == "constraints"
        assert isinstance(exc.__cause__, OSError)

    @pytest.mark.parametrize(
        "text",
        [
            '{"constraints": {}}',
            '{"constraints": {"id": 0, "name": "RE_1"}}',
            '{"constraints": null}',
            '{"hydros": [{"id": 0, "name": "RE_1"}]}',
            "{}",
            "[]",
            '[{"id": 0, "name": "RE_1"}]',
            '"constraints"',
        ],
    )
    def test_no_list_under_the_key_raises_field_parse_error(
        self, tmp_path: Path, text: str
    ) -> None:
        path = self._write(tmp_path, text)

        exc = self._fails(tmp_path, FieldParseError)

        assert exc.path == str(path)
        assert exc.field == "constraints"
        assert "holds no list of constraints" in str(exc)
        assert diagnostic_from_exception(exc, context="Comparison").code == (
            "source-field-parse"
        )

    @pytest.mark.parametrize(
        ("payload", "field", "position"),
        [
            ([{"id": 0}], "name", 1),
            ([{"name": "RE_1"}], "id", 1),
            ([{"id": True, "name": "A"}], "id", 1),
            ([{"id": False, "name": "A"}], "id", 1),
            ([{"id": "0", "name": "RE_1"}], "id", 1),
            ([{"id": 0.0, "name": "RE_1"}], "id", 1),
            ([{"id": None, "name": "RE_1"}], "id", 1),
            (["RE_1"], "id", 1),
            ([[0, "RE_1"]], "id", 1),
            ([None], "id", 1),
            ([{"id": 0, "name": ""}], "name", 1),
            ([{"id": 0, "name": " "}], "name", 1),
            ([{"id": 0, "name": "\t\n"}], "name", 1),
            ([{"id": 0, "name": 7}], "name", 1),
            ([{"id": 0, "name": None}], "name", 1),
            ([{"id": 0, "name": ["RE_1"]}], "name", 1),
            ([{"id": 0, "name": "RE_1"}, {"id": 1, "name": " "}], "name", 2),
            ([{"id": 0, "name": "RE_1"}, {"name": "VMINOP_1"}], "id", 2),
        ],
    )
    def test_a_bad_entry_raises_field_parse_error_at_its_position(
        self, tmp_path: Path, payload: list[object], field: str, position: int
    ) -> None:
        path = self._write(tmp_path, json.dumps({"constraints": payload}))

        exc = self._fails(tmp_path, FieldParseError)

        assert exc.path == str(path)
        assert exc.field == field
        assert str(exc).startswith(
            f"Entry {position} of constraints/generic_constraints.json "
        )
        assert str(exc).endswith(
            "has no integer id." if field == "id" else "has no name."
        )
        assert diagnostic_from_exception(exc, context="Comparison").code == (
            "source-field-parse"
        )

    def test_the_first_bad_entry_decides_the_field(self, tmp_path: Path) -> None:
        payload = [{"id": 0, "name": "A"}, {"id": 1}, {"name": "C"}]
        self._write(tmp_path, json.dumps({"constraints": payload}))

        exc = self._fails(tmp_path, FieldParseError)

        assert exc.field == "name"
        assert str(exc).startswith("Entry 2 ")

    def test_other_keys_of_an_entry_and_a_padded_name_are_not_checked(
        self, tmp_path: Path
    ) -> None:
        entries = [{"id": 3, "name": " re_1 ", "description": 5, "slack": "yes"}]
        self._write(tmp_path, json.dumps({"constraints": entries}))

        assert read_novomodelo_required_generic_constraints(tmp_path) == entries

    def test_the_lenient_reader_stays_lenient_on_the_same_file(
        self, tmp_path: Path
    ) -> None:
        self._write(tmp_path, json.dumps({"constraints": [{"id": 0, "name": " "}]}))

        with dx.collect() as found:
            assert read_novomodelo_generic_constraints(tmp_path) == [
                {"id": 0, "name": " "}
            ]
        assert found == []

        self._write(tmp_path, "not json")

        with dx.collect() as found:
            assert read_novomodelo_generic_constraints(tmp_path) == []
        assert [d.code for d in found] == ["report-file-unreadable"]

    def test_nothing_above_debug_is_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)
        good, bad, absent = tmp_path / "good", tmp_path / "bad", tmp_path / "absent"
        self._write(good, json.dumps({"constraints": self.ENTRIES}))
        self._write(bad, "not json")

        read_novomodelo_required_generic_constraints(good)
        self._fails(bad, FieldParseError)
        self._fails(absent, SourceFileError)

        assert [r for r in caplog.records if r.levelno > logging.DEBUG] == []

    def test_messages_name_no_repository_internals(self, tmp_path: Path) -> None:
        cases = {
            "not_json": ("not json", FieldParseError),
            "no_id": (json.dumps({"constraints": [{"name": "A"}]}), FieldParseError),
            "no_list": ("{}", FieldParseError),
        }
        messages = []
        for name, (text, error) in cases.items():
            self._write(tmp_path / name, text)
            messages.append(str(self._fails(tmp_path / name, error)))
        messages.append(str(self._fails(tmp_path / "absent", SourceFileError)))

        for leak in (
            "read_novomodelo",
            "find_system_json",
            "docs/",
            "plans/",
            "ticket-",
        ):
            assert not [m for m in messages if leak in m]


class TestReadNovomodeloStageRiskAversion:
    """``None`` is an unknown measure, ``{}`` no ``stages.json``: never truthiness."""

    CVAR = {"cvar": {"alpha": 0.25, "lambda": 0.35}}
    EFFECT = (
        "the report cannot tell whether the run is risk-averse and does not "
        "judge its upper bound."
    )

    @staticmethod
    def _write(directory: Path, stages: object) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "stages.json"
        path.write_text(json.dumps({"stages": stages}), encoding="utf-8")
        return path

    @staticmethod
    def _read(
        output_dir: Path,
    ) -> tuple[dict[int, float] | None, list[dx.Diagnostic]]:
        with dx.collect() as found:
            result = read_novomodelo_stage_risk_aversion(output_dir)
        return result, found

    def _assert_unknown(self, output_dir: Path) -> dx.Diagnostic:
        result, found = self._read(output_dir)
        assert result is None
        assert [(d.code, d.severity) for d in found] == [
            ("report-file-unreadable", Severity.WARNING)
        ]
        return found[0]

    def test_expectation_and_cvar_stages_map_to_their_weights(
        self, tmp_path: Path
    ) -> None:
        self._write(
            tmp_path,
            [
                {"id": 0, "risk_measure": "expectation"},
                {"id": 1, "risk_measure": self.CVAR},
            ],
        )

        result, found = self._read(tmp_path / "output")

        assert result == {0: 0.0, 1: 0.35}
        assert found == []

    def test_stages_without_a_risk_measure_weigh_zero(self, tmp_path: Path) -> None:
        self._write(tmp_path, [{"id": 0}, {"id": 1}, {"id": 5, "blocks": []}])

        result, found = self._read(tmp_path / "output")

        assert result == {0: 0.0, 1: 0.0, 5: 0.0}
        assert found == []

    @pytest.mark.parametrize(("lam", "weight"), [(0, 0.0), (0.0, 0.0), (1, 1.0)])
    def test_a_cvar_lambda_is_the_weight_as_a_float(
        self, tmp_path: Path, lam: float, weight: float
    ) -> None:
        measure = {"cvar": {"alpha": 0.5, "lambda": lam}}
        self._write(tmp_path, [{"id": 3, "risk_measure": measure}])

        result, found = self._read(tmp_path / "output")

        assert result == {3: weight}
        assert result is not None
        assert isinstance(result[3], float)
        assert found == []

    @pytest.mark.parametrize("spelling", ["expectation", "Expectation", "EXPECTATION"])
    def test_expectation_is_read_without_regard_to_ascii_case(
        self, tmp_path: Path, spelling: str
    ) -> None:
        self._write(tmp_path, [{"id": 0, "risk_measure": spelling}])

        result, found = self._read(tmp_path / "output")

        assert result == {0: 0.0}
        assert found == []

    @pytest.mark.parametrize(
        "text",
        [
            "cvar",
            "CVaR",
            "",
            " expectation",
            "expectation ",
            "expectation\n",
            "risk-neutral",
            "EXPECTATİON",
        ],
    )
    def test_any_other_string_is_unknown(self, tmp_path: Path, text: str) -> None:
        self._write(tmp_path, [{"id": 0, "risk_measure": text}])

        diagnostic = self._assert_unknown(tmp_path / "output")

        assert "a stage without a risk measure the report can read" in str(
            diagnostic.notes
        )

    @pytest.mark.parametrize(
        "measure",
        [
            {"cvar": {"alpha": 0.25}},
            {"cvar": {"alpha": 0.25, "lambda": "0.35"}},
            {"cvar": {"alpha": 0.25, "lambda": True}},
            {"cvar": {"alpha": 0.25, "lambda": None}},
            {"cvar": 0.35},
            {"cvar": None},
            {"cvar": []},
            {"lambda": 0.35},
            {},
            None,
            [],
            7,
            True,
        ],
    )
    def test_a_malformed_measure_is_unknown(
        self, tmp_path: Path, measure: object
    ) -> None:
        self._write(tmp_path, [{"id": 0, "risk_measure": measure}])

        self._assert_unknown(tmp_path / "output")

    @pytest.mark.parametrize(
        "stage",
        [
            {"risk_measure": "expectation"},
            {"id": "0", "risk_measure": "expectation"},
            {"id": 0.0, "risk_measure": "expectation"},
            {"id": True, "risk_measure": "expectation"},
            {"id": None},
            "stage",
            7,
            None,
        ],
    )
    def test_a_stage_without_an_integer_id_is_unknown(
        self, tmp_path: Path, stage: object
    ) -> None:
        self._write(tmp_path, [stage])

        self._assert_unknown(tmp_path / "output")

    @pytest.mark.parametrize("bad_position", [0, 1, 2])
    def test_one_unusable_stage_makes_the_whole_case_unknown(
        self, tmp_path: Path, bad_position: int
    ) -> None:
        stages: list[dict] = [{"id": i, "risk_measure": self.CVAR} for i in range(3)]
        stages[bad_position]["risk_measure"] = "cvar"
        self._write(tmp_path, stages)

        diagnostic = self._assert_unknown(tmp_path / "output")

        assert diagnostic.notes[-1].endswith(
            "a stage without a risk measure the report can read"
        )

    @pytest.mark.parametrize(
        "content",
        [
            b"not json",
            b"",
            b"\xff\xfe{",
            b"[1]",
            b'"stages"',
            b"{}",
            b'{"stages": []}',
            b'{"stages": null}',
            b'{"stages": "0"}',
            b'{"stages": {"id": 0}}',
        ],
    )
    def test_a_file_without_a_stage_list_is_unknown(
        self, tmp_path: Path, content: bytes
    ) -> None:
        (tmp_path / "stages.json").write_bytes(content)

        diagnostic = self._assert_unknown(tmp_path / "output")

        assert diagnostic.notes[-1] == "Reason: no list of stages"

    def test_no_file_is_empty_and_silent(self, tmp_path: Path) -> None:
        (tmp_path / "output").mkdir()

        result, found = self._read(tmp_path / "output")

        assert result == {}
        assert result is not None
        assert found == []

    def test_the_diagnostic_names_the_file_and_its_effect(self, tmp_path: Path) -> None:
        path = tmp_path / "stages.json"
        path.write_text("not json", encoding="utf-8")

        diagnostic = self._assert_unknown(tmp_path / "output")

        assert diagnostic.title == "stages.json could not be read"
        assert diagnostic.summary == f"stages.json could not be read, so {self.EFFECT}"
        assert diagnostic.notes == [f"Path: {path}", "Reason: no list of stages"]

    def test_the_diagnostic_path_is_the_first_searched_file_that_exists(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "stages.json").write_text("not json", encoding="utf-8")
        assert f"Path: {output_dir / 'stages.json'}" in (
            self._assert_unknown(output_dir).notes
        )

        (tmp_path / "stages.json").write_text("not json", encoding="utf-8")
        assert f"Path: {tmp_path / 'stages.json'}" in (
            self._assert_unknown(output_dir).notes
        )

    def test_the_output_directory_is_read_when_the_case_directory_has_none(
        self, tmp_path: Path
    ) -> None:
        self._write(tmp_path / "output", [{"id": 0, "risk_measure": self.CVAR}])

        result, found = self._read(tmp_path / "output")

        assert result == {0: 0.35}
        assert found == []

    def test_the_case_directory_is_read_before_the_output_directory(
        self, tmp_path: Path
    ) -> None:
        self._write(tmp_path, [{"id": 0, "risk_measure": self.CVAR}])
        self._write(tmp_path / "output", [{"id": 0, "risk_measure": "expectation"}])

        result, _ = self._read(tmp_path / "output")

        assert result == {0: 0.35}

    @pytest.mark.parametrize("case_dir_file", ["not json", '{"stages": []}'])
    def test_a_file_without_stages_falls_through_to_the_next_location(
        self, tmp_path: Path, case_dir_file: str
    ) -> None:
        (tmp_path / "stages.json").write_text(case_dir_file, encoding="utf-8")
        self._write(tmp_path / "output", [{"id": 0, "risk_measure": self.CVAR}])

        result, found = self._read(tmp_path / "output")

        assert result == {0: 0.35}
        assert found == []

    def test_the_first_non_empty_list_decides_even_when_unusable(
        self, tmp_path: Path
    ) -> None:
        self._write(tmp_path, [{"id": 0, "risk_measure": "cvar"}])
        self._write(tmp_path / "output", [{"id": 0, "risk_measure": "expectation"}])

        self._assert_unknown(tmp_path / "output")

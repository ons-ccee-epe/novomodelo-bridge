"""Tests for the fail-loud contract of the Cobre output readers.

ROBUST-01: a read/parse/aggregation failure on an **already-existing**
Cobre output file must raise :class:`CobreReadError` (so the comparison
engine never reports a false "no divergence" on unreadable data), while a
genuinely **absent** optional output must still yield an empty frame.  The
CLI maps :class:`CobreReadError` to exit code 2 (distinct from exit 1 =
"mismatches found").
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import typer

from cobre_bridge.cli.args import CompareArgs
from cobre_bridge.cobre.readers import (
    CobreReadError,
    _load_entity_bus_map,
    cobre_software_version,
    read_cobre_bus_aggregates,
    read_cobre_bus_means,
    read_cobre_hydro_bus_generation,
    read_cobre_hydro_bus_labels,
    read_cobre_hydro_means,
    read_cobre_hydro_metadata,
    read_cobre_hydro_per_stage_bounds,
    read_cobre_hydro_withdrawal,
    read_cobre_line_bounds,
    read_cobre_line_means,
    read_cobre_lines,
    read_cobre_lp_max_generation,
    read_cobre_thermal_means,
    read_cobre_thermal_metadata,
    read_cobre_training_metadata,
)
from cobre_bridge.core import diagnostics as dx
from cobre_bridge.core.diagnostics import Severity
from cobre_bridge.core.errors import CobrePartitionMissingError

# ---------------------------------------------------------------------------
# Fixtures: build corrupt vs. valid-but-empty Cobre output layouts on disk.
# ---------------------------------------------------------------------------

_GARBAGE = b"this is definitely not a valid parquet file -- no PAR1 footer"


def _write_corrupt_sim_parquet(output_dir: Path, entity: str) -> Path:
    """Create ``output/simulation/<entity>/scenario_id=0000/data.parquet``
    containing non-parquet garbage and return its path.
    """
    p = output_dir / "simulation" / entity / "scenario_id=0000" / "data.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_GARBAGE)
    return p


def _write_hydro_bus_generation_partition(
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


def _write_sim_entity_partition(
    output_dir: Path,
    entity: str,
    scenario_id: int,
    rows: list[dict[str, object]],
) -> Path:
    """Write a synthetic hive-partitioned ``simulation/<entity>`` parquet.

    Column dtypes are inferred by pyarrow from the Python values in *rows*
    (unlike :func:`_write_hydro_bus_generation_partition`, which pins the
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


def _write_stages_json(
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


def _write_entity_bus_json(
    case_dir: Path, entity: str, entries: list[dict[str, int]]
) -> Path:
    """Write a minimal ``system/<entity>.json`` with ``id``/``bus_id`` entries."""
    sysdir = case_dir / "system"
    sysdir.mkdir(parents=True, exist_ok=True)
    path = sysdir / f"{entity}.json"
    path.write_text(json.dumps({entity: entries}), encoding="utf-8")
    return path


def _write_minimal_hydros_json(case_dir: Path, hydros: list[dict[str, object]]) -> Path:
    """Write a minimal ``system/hydros.json`` (physics fields only, no bus)."""
    sysdir = case_dir / "system"
    sysdir.mkdir(parents=True, exist_ok=True)
    path = sysdir / "hydros.json"
    path.write_text(json.dumps({"hydros": hydros}), encoding="utf-8")
    return path


def _write_corrupt_parquet(path: Path) -> Path:
    """Create a corrupt parquet at *path* (parents made) and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_GARBAGE)
    return path


# ---------------------------------------------------------------------------
# Corrupt *existing* file -> CobreReadError.
# ---------------------------------------------------------------------------


class TestCorruptExistingRaises:
    def test_hydro_means_corrupt_parquet_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_corrupt_sim_parquet(out, "hydros")
        with pytest.raises(CobreReadError, match="scan parquets"):
            read_cobre_hydro_means(out)

    def test_thermal_means_corrupt_parquet_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_corrupt_sim_parquet(out, "thermals")
        with pytest.raises(CobreReadError):
            read_cobre_thermal_means(out)

    def test_bus_means_corrupt_parquet_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_corrupt_sim_parquet(out, "buses")
        with pytest.raises(CobreReadError):
            read_cobre_bus_means(out)

    def test_line_means_corrupt_parquet_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_corrupt_sim_parquet(out, "exchanges")
        with pytest.raises(CobreReadError):
            read_cobre_line_means(out)

    def test_lp_max_generation_corrupt_bounds_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_corrupt_parquet(out / "training" / "dictionaries" / "bounds.parquet")
        with pytest.raises(CobreReadError, match="bounds.parquet"):
            read_cobre_lp_max_generation(out)

    def test_hydro_withdrawal_corrupt_constraints_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_corrupt_parquet(tmp_path / "constraints" / "hydro_bounds.parquet")
        with pytest.raises(CobreReadError, match="hydro_bounds.parquet"):
            read_cobre_hydro_withdrawal(out)

    def test_hydro_per_stage_bounds_corrupt_constraints_raises(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        _write_corrupt_parquet(tmp_path / "constraints" / "hydro_bounds.parquet")
        with pytest.raises(CobreReadError, match="hydro_bounds.parquet"):
            read_cobre_hydro_per_stage_bounds(out)


# ---------------------------------------------------------------------------
# Absent file -> empty DataFrame (no raise).  PRESERVED behavior.
# ---------------------------------------------------------------------------


class TestAbsentReturnsEmpty:
    def test_hydro_means_absent_dir_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_cobre_hydro_means(out)
        assert df.is_empty()

    def test_thermal_means_absent_dir_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_cobre_thermal_means(out)
        assert df.is_empty()

    def test_bus_means_absent_dir_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_cobre_bus_means(out)
        assert df.is_empty()

    def test_line_means_absent_dir_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_cobre_line_means(out)
        assert df.is_empty()

    def test_lp_max_generation_absent_file_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_cobre_lp_max_generation(out)
        assert df.is_empty()

    def test_hydro_withdrawal_absent_file_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_cobre_hydro_withdrawal(out)
        assert df.is_empty()

    def test_hydro_per_stage_bounds_absent_file_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_cobre_hydro_per_stage_bounds(out)
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
        (sysdir / "thermals.json").write_text("{ this is not valid json ")
        out.mkdir()
        assert read_cobre_thermal_metadata(out) == {}


# ---------------------------------------------------------------------------
# CLI maps CobreReadError -> exit code 2 (distinct from 1 = mismatches).
# ---------------------------------------------------------------------------


class TestCliExitCodeTwoOnCobreReadError:
    def test_results_cli_exits_2_when_reader_raises(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from cobre_bridge.cli.compare import _run_newave_comparison

        args = CompareArgs(
            source_dir=tmp_path / "newave",
            cobre_output_dir=tmp_path / "output",
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
            raise CobreReadError("Failed to aggregate hydro simulation data: /x/hydros")

        with (
            patch(
                "cobre_bridge.cobre.readers.read_cobre_lines",
                return_value=[],
            ),
            patch(
                "cobre_bridge.newave.case.NewaveCase.from_directory",
                return_value=MagicMock(),
            ),
            patch(
                "cobre_bridge.comparators.newave.alignment.build_entity_alignment",
                return_value=object(),
            ),
            patch(
                "cobre_bridge.comparators.newave.results.compare_results",
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
# real (unpatched) code path to CobreReadError, proving the antipattern fix
# rather than only the CLI wiring.
# ---------------------------------------------------------------------------


class TestCorruptDrivesRealReaderPath:
    def test_real_reader_raises_on_corrupt_then_empty_on_absent(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        bounds = out / "training" / "dictionaries" / "bounds.parquet"

        # Corrupt-but-existing -> raises.
        _write_corrupt_parquet(bounds)
        with pytest.raises(CobreReadError):
            read_cobre_lp_max_generation(out)

        # Same reader, file removed -> empty, no raise.
        bounds.unlink()
        result = read_cobre_lp_max_generation(out)
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
        _write_hydro_bus_generation_partition(
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

        df = read_cobre_hydro_bus_generation(out).collect()

        assert _HYDRO_BUS_GENERATION_COLUMNS.issubset(set(df.columns))
        assert df.height == 2
        assert df["scenario_id"].to_list() == [1, 1]
        assert df["generation_mwh"].to_list() == [730.0, 876.0]

    def test_dtypes_match_writer_contract(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_hydro_bus_generation_partition(
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

        schema = read_cobre_hydro_bus_generation(out).collect_schema()
        assert schema["stage_id"] == pl.Int32
        assert schema["block_id"] == pl.Int32
        assert schema["hydro_id"] == pl.Int32
        assert schema["bus_id"] == pl.Int32
        assert schema["turbined_m3s"] == pl.Float64
        assert schema["generation_mw"] == pl.Float64
        assert schema["generation_mwh"] == pl.Float64
        assert schema["scenario_id"] == pl.Int64


class TestHydroBusGenerationAbsentRaisesNamed:
    def test_missing_directory_raises_typed_error_naming_dir_and_version(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        out.mkdir()

        with pytest.raises(CobrePartitionMissingError) as excinfo:
            read_cobre_hydro_bus_generation(out)

        message = str(excinfo.value)
        assert str(out / "simulation" / "hydro_bus_generation") in message
        assert "0.13.0" in message

    def test_missing_directory_error_is_typed_not_bare(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()

        # A narrower type than plain Exception -- callers can distinguish
        # this from every other failure mode this reader can raise, and it
        # is never caught-and-discarded inside the reader itself.
        with pytest.raises(CobrePartitionMissingError):
            read_cobre_hydro_bus_generation(out)
        assert issubclass(CobrePartitionMissingError, Exception)
        assert CobrePartitionMissingError is not Exception

    def test_missing_output_dir_itself_also_raises_named_error(
        self, tmp_path: Path
    ) -> None:
        # The output dir doesn't exist at all (e.g. `compare` pointed at a
        # stale/pre-0.13 case dir) -- still the named error, never a bare
        # FileNotFoundError from deeper I/O and never an empty frame.
        out = tmp_path / "does-not-exist"

        with pytest.raises(CobrePartitionMissingError):
            read_cobre_hydro_bus_generation(out)


class TestHydroBusGenerationPresentButEmpty:
    def test_present_empty_directory_returns_empty_not_raise(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        sim_dir = out / "simulation" / "hydro_bus_generation"
        sim_dir.mkdir(parents=True)  # present, but no scenario partitions written.

        df = read_cobre_hydro_bus_generation(out).collect()

        assert df.is_empty()
        assert set(df.columns) == _HYDRO_BUS_GENERATION_COLUMNS

    def test_present_empty_directory_emits_diagnostic_not_silence(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        sim_dir = out / "simulation" / "hydro_bus_generation"
        sim_dir.mkdir(parents=True)

        with dx.collect() as collected:
            read_cobre_hydro_bus_generation(out)

        empties = [
            d for d in collected if d.code == "hydro-bus-generation-partition-empty"
        ]
        assert len(empties) == 1
        assert empties[0].severity is Severity.WARNING

    def test_absent_and_present_empty_are_not_conflated(self, tmp_path: Path) -> None:
        out = tmp_path / "output"

        # Absent -> named error.
        with pytest.raises(CobrePartitionMissingError):
            read_cobre_hydro_bus_generation(out)

        # Same output dir, partition directory now created (still empty) ->
        # no raise, distinct outcome from the absent case above.
        (out / "simulation" / "hydro_bus_generation").mkdir(parents=True)
        df = read_cobre_hydro_bus_generation(out).collect()
        assert df.is_empty()


class TestHydroBusGenerationNullableBlockId:
    def test_stage_level_null_block_id_row_does_not_crash_aggregation(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        _write_hydro_bus_generation_partition(
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

        df = read_cobre_hydro_bus_generation(out).collect()
        assert df.height == 3
        assert df.filter(pl.col("block_id").is_null()).height == 1

        # A downstream energy aggregation across the block dimension must
        # not choke on the null block_id.
        totals = df.group_by(["hydro_id", "stage_id"]).agg(
            pl.col("generation_mwh").sum()
        )
        assert totals["generation_mwh"].to_list() == [730.0 + 876.0 + 0.0]


class TestHydroBusGenerationCorruptRaisesCobreReadError:
    def test_corrupt_parquet_raises_cobrereaderror_not_named_absence(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        _write_corrupt_sim_parquet(out, "hydro_bus_generation")

        # Present-but-corrupt is a different failure than absence: the
        # existing CobreReadError contract applies, not the new named
        # partition-missing error.
        with pytest.raises(CobreReadError):
            read_cobre_hydro_bus_generation(out)


# ---------------------------------------------------------------------------
# read_cobre_hydro_metadata drops "bus_id"; the
# plant->bus label is re-sourced from the hydro_bus_generation partition via
# read_cobre_hydro_bus_labels.
# ---------------------------------------------------------------------------


class TestHydroMetadataHasNoBusId:
    def test_bus_id_key_is_absent_even_when_present_in_source_json(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        out.mkdir(parents=True)
        # A stray "bus_id" in hydros.json (e.g. a stale pre-0.13 case) must be
        # ignored -- the metadata reader's job is plant physics only, not a
        # bus source, so the key must never resurface in its return value.
        _write_minimal_hydros_json(
            tmp_path,
            [{"id": 0, "name": "H0", "bus_id": 100}],
        )

        result = read_cobre_hydro_metadata(out)

        assert "bus_id" not in result[0]

    def test_absent_hydros_json_returns_empty_dict(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir(parents=True)
        assert read_cobre_hydro_metadata(out) == {}


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
        _write_hydro_bus_generation_partition(
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

        result = read_cobre_hydro_bus_labels(out)

        assert result == {1: frozenset({5})}

    def test_synthetic_two_bus_plant_keeps_every_bus(self, tmp_path: Path) -> None:
        """A synthetic two-bus plant, at the reader level:

        no real deck produces this today, but the
        partition itself imposes no one-bus-per-plant constraint, so a plant
        recorded at two buses across scenarios/stages must not be collapsed
        down to a single one here.
        """
        out = tmp_path / "output"
        _write_hydro_bus_generation_partition(
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

        result = read_cobre_hydro_bus_labels(out)

        assert result == {9: frozenset({100, 101}), 1: frozenset({100})}

    def test_missing_partition_raises_named_error(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir(parents=True)

        with pytest.raises(CobrePartitionMissingError):
            read_cobre_hydro_bus_labels(out)

    def test_present_but_empty_partition_returns_empty_dict(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        (out / "simulation" / "hydro_bus_generation").mkdir(parents=True)

        assert read_cobre_hydro_bus_labels(out) == {}


# ---------------------------------------------------------------------------
# read_cobre_bus_aggregates rewired onto hydro_bus_generation;
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
    _write_stages_json(tmp_path, _STAGE_BLOCKS)

    # Hydro: one plant on bus 5. generation_mwh is block_mw * block_hours,
    # exactly as cobre's writer produces it.
    _write_hydro_bus_generation_partition(
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
    _write_hydro_bus_generation_partition(
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
    _write_entity_bus_json(tmp_path, "thermals", [{"id": 1, "bus_id": 5}])
    _write_sim_entity_partition(
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
    _write_sim_entity_partition(
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
    _write_entity_bus_json(
        tmp_path, "non_controllable_sources", [{"id": 1, "bus_id": 5}]
    )
    _write_sim_entity_partition(
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
    _write_sim_entity_partition(
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

        result = read_cobre_bus_aggregates(out)

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
        """The hydro branch no longer depends on a
        hydro plant->bus map at all, so it is unaffected by hydros.json
        being entirely absent -- proof that no caller of
        ``_load_entity_bus_map`` passes ``"hydros"`` any more."""
        out = _build_full_bus_aggregates_case(tmp_path)
        # No system/hydros.json is written by the builder above; assert
        # that explicitly, then confirm hydro generation is still reported.
        assert not (tmp_path / "system" / "hydros.json").exists()

        result = read_cobre_bus_aggregates(out)

        row = result.filter((pl.col("bus_id") == 5) & (pl.col("stage_id") == 0))
        assert row["hydro_gen_mw_p50"][0] > 0.0


class TestBusAggregatesThermalNcsUnchanged:
    """Thermal and NCS per-bus aggregates are unchanged --
    pinned numerically against the pre-change formula (block-
    hours-weighted mean of generation_mw, exactly as ``_agg_entity_by_bus``
    computed before and still computes now, untouched)."""

    def test_thermal_and_ncs_gen_mw_match_pre_change_weighted_mean(
        self, tmp_path: Path
    ) -> None:
        out = _build_full_bus_aggregates_case(tmp_path)

        result = read_cobre_bus_aggregates(out)
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
    named error through read_cobre_bus_aggregates -- it is not caught and
    converted to empty here, even though thermal/NCS data is present."""

    def test_missing_partition_raises_through_bus_aggregates(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        out.mkdir(parents=True)
        _write_stages_json(tmp_path, _STAGE_BLOCKS)

        # Thermal data present and valid -- proves the hydro absence isn't
        # incidentally masked by every other branch also being empty.
        _write_entity_bus_json(tmp_path, "thermals", [{"id": 1, "bus_id": 5}])
        _write_sim_entity_partition(
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

        with pytest.raises(CobrePartitionMissingError):
            read_cobre_bus_aggregates(out)


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
        _write_stages_json(tmp_path, _STAGE_BLOCKS)

        # Two hydro plants on the same bus, one scenario. A null-block_id
        # stage-level row is included to prove it is summed in
        # without corrupting the result -- its
        # generation_mwh is 0.0, a real value cobre could legitimately
        # write for a slack/violation-only stage-level row.
        _write_hydro_bus_generation_partition(
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

        result = read_cobre_bus_aggregates(out)
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
    """Regression: without ``stages.json`` (``block_hours`` is
    ``None``), the hydro branch has no stage-hours denominator to convert an
    energy total back into a power figure, so it must mirror every sibling
    aggregator in this module (``_agg_entity_by_bus``,
    :func:`read_cobre_hydro_means`, :func:`read_cobre_thermal_means`,
    :func:`read_cobre_bus_means`) and report the ``block_id == 0``
    representative block's ``generation_mw`` -- already a genuine MW figure
    -- summed across hydros per bus. The pre-fix code instead relabeled the
    raw ``generation_mwh`` sum as ``hydro_gen_mw``, off by the block-hours
    factor; this is the only fixture in the module that omits
    ``stages.json``, so this path was previously untested."""

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
        _write_hydro_bus_generation_partition(
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

        result = read_cobre_bus_aggregates(out)
        row = result.filter((pl.col("bus_id") == 5) & (pl.col("stage_id") == 0))
        assert row.height == 1

        expected_mw = 10.0 + 5.0  # block_id==0 generation_mw, summed across plants.
        assert row["hydro_gen_mw_p50"][0] == pytest.approx(expected_mw)

        # The pre-fix behaviour relabeled the raw MWh sum (across ALL blocks,
        # including the null-block row) as MW -- assert the fix does not
        # reproduce that order-of-magnitude-inflated figure.
        wrong_energy_sum = 1000.0 + 999000.0 + 500.0 + 777000.0
        assert row["hydro_gen_mw_p50"][0] != pytest.approx(wrong_energy_sum)


# ---------------------------------------------------------------------------
# _load_entity_bus_map: bare except is gone, remaining callers
# (thermal, NCS) still have a real bus_id to read.
# ---------------------------------------------------------------------------


class TestLoadEntityBusMapNoBareExcept:
    """No bare except remains; a corrupt JSON raises a typed error
    rather than yielding {}."""

    def test_corrupt_json_raises_cobrereaderror_not_empty_dict(
        self, tmp_path: Path
    ) -> None:
        sysdir = tmp_path / "system"
        sysdir.mkdir(parents=True)
        (sysdir / "thermals.json").write_text("{ this is not valid json ")

        with pytest.raises(CobreReadError):
            _load_entity_bus_map(tmp_path / "output", "thermals", "thermal_id")

    def test_missing_bus_id_field_raises_instead_of_being_filtered(
        self, tmp_path: Path
    ) -> None:
        # The old code silently dropped an entry lacking "bus_id" via
        # ``if "bus_id" in e``. That filter is gone: a thermal/NCS entry
        # missing its bus_id is a real problem now, not a skippable row.
        _write_entity_bus_json(tmp_path, "thermals", [{"id": 1}])

        with pytest.raises(CobreReadError):
            _load_entity_bus_map(tmp_path / "output", "thermals", "thermal_id")

    def test_missing_file_still_returns_empty_dict(self, tmp_path: Path) -> None:
        # The deliberate "optional file" branch (path is None) is untouched.
        result = _load_entity_bus_map(tmp_path / "output", "thermals", "thermal_id")
        assert result == {}

    def test_valid_json_still_returns_map(self, tmp_path: Path) -> None:
        _write_entity_bus_json(
            tmp_path, "thermals", [{"id": 1, "bus_id": 5}, {"id": 2, "bus_id": 6}]
        )

        result = _load_entity_bus_map(tmp_path / "output", "thermals", "thermal_id")
        assert result == {1: 5, 2: 6}


# ---------------------------------------------------------------------------
# Golden regeneration: the reader-level proof underneath the
# per-bus chart goldens. The chart goldens themselves are guarded by
# hand-built fixtures in test_chart_helpers.py that never touch these
# readers; the two classes below are the honest substitute --
# proving, on a synthetic 0.13 output dir, that
# read_cobre_bus_aggregates' hydro branch is both correctly weighted
# and reconciles exactly with the unchanged plant-keyed path.
# ---------------------------------------------------------------------------


class TestEnergyNotReweightedHandComputedProof:
    """Hand-computes one (bus, stage) energy figure from
    ``stages.json`` ``blocks[].hours`` and a synthetic ``hydro_bus_generation``
    partition, and proves -- rather than merely asserts -- that
    ``read_cobre_bus_aggregates`` uses ``generation_mwh`` un-re-weighted.

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
        _write_stages_json(tmp_path, stage_blocks)

        # One plant, one bus, one scenario: generation_mwh is exactly
        # generation_mw * block_hours, as cobre's writer produces it.
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
        _write_hydro_bus_generation_partition(out, scenario_id=0, rows=rows)

        result = read_cobre_bus_aggregates(out)
        row = result.filter((pl.col("bus_id") == 5) & (pl.col("stage_id") == 0))
        assert row.height == 1
        actual = row["hydro_gen_mw_p50"][0]

        # Hand computation #1 (CORRECT, per stages.json): stage 0's total
        # hours is 100.0 + 200.0 = 300.0. generation_mwh is ALREADY
        # hours-weighted energy per block, so the bus/stage MW figure is
        # the raw sum of generation_mwh across blocks divided by that
        # total -- one division, no second multiplication by hours.
        total_hours = sum(hours for _, hours in stage_blocks[0])  # 300.0
        total_energy_mwh = sum(float(r["generation_mwh"]) for r in rows)  # 5000.0
        correct = total_energy_mwh / total_hours  # 16.666...

        # Hand computation #2 (WRONG): the mistake the standing project
        # rule warns against -- treating generation_mwh as if it still
        # needed hours weighting, i.e. multiplying it by hours a second
        # time before dividing by total hours.
        hours_by_block = dict(stage_blocks[0])
        wrong_double_weighted = (
            sum(
                float(r["generation_mwh"]) * hours_by_block[int(r["block_id"])]
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
    generation summed over buses (the new ``read_cobre_bus_aggregates``
    source) must equal the plant-keyed total from the unchanged
    ``simulation/hydros/`` (``read_cobre_hydro_means``), and no bus may gain
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
        _write_stages_json(tmp_path, self._STAGE_BLOCKS)

        # Bus-keyed source (new): hydro_bus_generation partition. Plants 1
        # and 2 share bus 5; plant 3 is alone on bus 6.
        _write_hydro_bus_generation_partition(
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
        _write_sim_entity_partition(
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
        _write_entity_bus_json(tmp_path, "thermals", [{"id": 1, "bus_id": 7}])
        _write_sim_entity_partition(
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

        bus_result = read_cobre_bus_aggregates(out)
        plant_result = read_cobre_hydro_means(out)

        # Plant-keyed total (unchanged simulation/hydros/ path): sum of
        # each plant's block-hours-weighted mean MW at stage 0.
        # (5000/300 + 1700/300 + 6500/300) == 13200/300 == 44.0.
        plant_total = plant_result.filter(pl.col("stage_id") == 0)[
            "generation_mw"
        ].sum()
        assert plant_total == pytest.approx(44.0)

        # Bus-summed total (the new read_cobre_bus_aggregates source): sum
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
# read_cobre_lines / read_cobre_line_bounds /
# read_cobre_training_metadata -- the three previously-missing readers that
# every ad-hoc lines.json / line_bounds.parquet / training/metadata.json
# site now routes through.
# ---------------------------------------------------------------------------


class TestReadCobreLines:
    def test_present_returns_lines_list(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        system_dir = tmp_path / "system"
        system_dir.mkdir(parents=True)
        lines = [{"id": 0, "source_bus_id": 0, "target_bus_id": 1}]
        (system_dir / "lines.json").write_text(json.dumps({"lines": lines}))

        assert read_cobre_lines(out) == lines

    def test_absent_returns_empty_list(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()

        assert read_cobre_lines(out) == []

    def test_corrupt_json_raises_cobrereaderror(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        system_dir = tmp_path / "system"
        system_dir.mkdir(parents=True)
        (system_dir / "lines.json").write_text("not valid json{")

        with pytest.raises(CobreReadError, match="lines.json"):
            read_cobre_lines(out)


class TestReadCobreLineBounds:
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

        df = read_cobre_line_bounds(out)

        assert not df.is_empty()
        assert df["direct_mw"][0] == 1200.0
        assert df["reverse_mw"][0] == 800.0

    def test_absent_returns_empty_frame(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()

        df = read_cobre_line_bounds(out)

        assert df.is_empty()
        assert {"line_id", "stage_id", "block_id", "direct_mw", "reverse_mw"}.issubset(
            set(df.columns)
        )

    def test_corrupt_parquet_raises_cobrereaderror(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_corrupt_parquet(tmp_path / "constraints" / "line_bounds.parquet")

        with pytest.raises(CobreReadError, match="line_bounds.parquet"):
            read_cobre_line_bounds(out)


class TestReadCobreTrainingMetadata:
    def test_present_returns_dict(self, tmp_path: Path) -> None:
        out = tmp_path / "case" / "output"
        training_dir = out / "training"
        training_dir.mkdir(parents=True)
        (training_dir / "metadata.json").write_text(
            json.dumps({"version": "0.14.3", "duration_seconds": 12.0})
        )

        assert read_cobre_training_metadata(out) == {
            "version": "0.14.3",
            "duration_seconds": 12.0,
        }

    def test_absent_returns_empty_dict(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()

        assert read_cobre_training_metadata(out) == {}

    def test_unified_path_rule_resolves_the_case_dir_parent_fallback(
        self, tmp_path: Path
    ) -> None:
        """A case layout with no doubled ``output/`` segment (the shape
        ``export._read_cobre_version`` used to resolve via its own fallback
        but ``dashboard.load_output_metadata`` could not) still
        resolves through the one unified candidate search."""
        case_dir = tmp_path / "case"
        training_dir = case_dir / "training"
        training_dir.mkdir(parents=True)
        (training_dir / "metadata.json").write_text(json.dumps({"version": "0.14.3"}))

        assert read_cobre_training_metadata(case_dir / "output") == {
            "version": "0.14.3"
        }

    def test_corrupt_json_degrades_to_empty_dict(self, tmp_path: Path) -> None:
        out = tmp_path / "case" / "output"
        training_dir = out / "training"
        training_dir.mkdir(parents=True)
        (training_dir / "metadata.json").write_text("not valid json{")

        assert read_cobre_training_metadata(out) == {}


class TestCobreSoftwareVersion:
    def test_reads_software_version(self) -> None:
        assert (
            cobre_software_version({"software": "cobre", "software_version": "0.18.0"})
            == "0.18.0"
        )

    def test_falls_back_to_the_pre_software_key(self) -> None:
        assert cobre_software_version({"cobre_version": "0.17.0"}) == "0.17.0"

    def test_prefers_software_version_when_both_are_present(self) -> None:
        assert (
            cobre_software_version(
                {"software_version": "0.18.0", "cobre_version": "0.17.0"}
            )
            == "0.18.0"
        )

    def test_absent_or_non_string_returns_none(self) -> None:
        assert cobre_software_version({}) is None
        assert cobre_software_version({"software_version": 18}) is None

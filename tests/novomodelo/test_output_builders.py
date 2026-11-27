"""Self-tests of ``build_solved_case``, read back through the public readers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.novomodelo.readers import (
    NovomodeloReadError,
    novomodelo_software_version,
    read_novomodelo_block_hours,
    read_novomodelo_bus_aggregates,
    read_novomodelo_bus_means,
    read_novomodelo_buses,
    read_novomodelo_convergence,
    read_novomodelo_fpha_planes,
    read_novomodelo_hydro_block_operation,
    read_novomodelo_hydro_means,
    read_novomodelo_hydros,
    read_novomodelo_line_means,
    read_novomodelo_lines,
    read_novomodelo_non_controllable_sources,
    read_novomodelo_scenario_weights,
    read_novomodelo_stage_costs,
    read_novomodelo_thermal_means,
    read_novomodelo_thermals,
    read_novomodelo_training_duration,
    read_novomodelo_training_metadata,
)
from tests.novomodelo_output_builders import (
    GARBAGE_BYTES,
    SolvedCaseSpec,
    build_solved_case,
    corrupt_file,
    drop_column,
    drop_partition,
)


def test_default_spec_names_report_buses() -> None:
    submarkets = {"SUDESTE", "SUL", "NORDESTE", "NORTE", "SE", "NE", "S", "N"}

    assert set(SolvedCaseSpec().buses) <= submarkets


def test_default_spec_reads_back_without_diagnostics(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")

    with dx.collect() as found:
        counts = [
            len(read_novomodelo_hydros(output)),
            len(read_novomodelo_buses(output)),
            len(read_novomodelo_thermals(output)),
            len(read_novomodelo_lines(output)),
        ]

    assert output == tmp_path / "case" / "output"
    assert counts == [2, 2, 1, 1]
    assert found == []


def test_builder_writes_only_under_case_dir(tmp_path: Path) -> None:
    build_solved_case(tmp_path / "nested" / "case")

    assert [p.name for p in tmp_path.iterdir()] == ["nested"]


def test_two_builds_of_one_spec_write_identical_files(tmp_path: Path) -> None:
    build_solved_case(tmp_path / "a")
    build_solved_case(tmp_path / "b")

    written = sorted(
        p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*.*")
    )

    assert len(written) == 21
    assert [
        (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()
        for rel in written
    ] == [True] * 21


def test_stages_hold_one_block_each_and_the_risk_measure(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case", SolvedCaseSpec(stages=4))

    hours = read_novomodelo_block_hours(output)

    assert hours["stage_id"].to_list() == [0, 1, 2, 3]
    assert hours["hours"].to_list() == [744.0] * 4


@pytest.mark.parametrize(
    "risk_measure",
    ["expectation", {"cvar": {"alpha": 0.5, "lambda": 0.25}}],
)
def test_risk_measure_is_written_on_every_stage(
    tmp_path: Path, risk_measure: str | dict[str, Any]
) -> None:
    case = tmp_path / "case"
    build_solved_case(case, SolvedCaseSpec(stages=3, risk_measure=risk_measure))

    stages = json.loads((case / "stages.json").read_text(encoding="utf-8"))["stages"]

    assert [s["risk_measure"] for s in stages] == [risk_measure] * 3


def test_current_convergence_reads_as_canonical_columns(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")

    frame = read_novomodelo_convergence(output)

    assert frame.columns == ["iteration", "lower_bound", "upper_bound"]
    assert frame.height == 3
    assert frame["iteration"].to_list() == [1, 2, 3]
    assert frame["lower_bound"].to_list() == pytest.approx([5.0e8, 7.5e8, 8.75e8])
    assert frame["upper_bound"].to_list() == pytest.approx([1.5e9, 1.25e9, 1.125e9])


def test_scale_multiplies_the_bounds(tmp_path: Path) -> None:
    base = read_novomodelo_convergence(build_solved_case(tmp_path / "a"))
    scaled = read_novomodelo_convergence(
        build_solved_case(tmp_path / "b", SolvedCaseSpec(scale=3.0))
    )

    assert scaled["lower_bound"].to_list() == pytest.approx(
        [3 * v for v in base["lower_bound"]]
    )
    assert scaled["upper_bound"].to_list() == pytest.approx(
        [3 * v for v in base["upper_bound"]]
    )


def test_current_schema_columns(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")

    columns = pl.read_parquet(output / "training" / "convergence.parquet").columns

    assert {"upper_bound", "upper_bound_std", "upper_bound_kind"} <= set(columns)
    assert "upper_bound_mean" not in columns


def test_pre_0_14_schema_columns(tmp_path: Path) -> None:
    output = build_solved_case(
        tmp_path / "case", SolvedCaseSpec(convergence_schema="pre_0_14")
    )

    columns = pl.read_parquet(output / "training" / "convergence.parquet").columns

    assert {"upper_bound_mean", "upper_bound_std"} <= set(columns)
    assert "upper_bound" not in columns
    assert "upper_bound_kind" not in columns


def test_exact_upper_bound_has_a_null_standard_deviation(tmp_path: Path) -> None:
    output = build_solved_case(
        tmp_path / "case", SolvedCaseSpec(upper_bound_kind="exact")
    )

    table = pl.read_parquet(output / "training" / "convergence.parquet")

    assert table["upper_bound_kind"].to_list() == ["exact"] * 3
    assert table["upper_bound_std"].null_count() == 3


@pytest.mark.parametrize(
    ("software_version", "version_key", "expected"),
    [
        ("0.21.0", "software_version", "0.21.0"),
        ("0.20.0", "cobre_version", None),
        (None, "software_version", None),
        (None, "cobre_version", None),
    ],
)
def test_version_key_variants(
    tmp_path: Path,
    software_version: str | None,
    version_key: str,
    expected: str | None,
) -> None:
    output = build_solved_case(
        tmp_path / "case",
        SolvedCaseSpec(software_version=software_version, version_key=version_key),
    )

    metadata = read_novomodelo_training_metadata(output)

    assert novomodelo_software_version(metadata) == expected
    assert (version_key in metadata) is (software_version is not None)


def test_training_duration_follows_the_iteration_count(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case", SolvedCaseSpec(iterations=10))

    assert read_novomodelo_training_duration(output) == pytest.approx(3.0)


def test_id_offset_shifts_every_entity_id(tmp_path: Path) -> None:
    low = build_solved_case(tmp_path / "low", SolvedCaseSpec(id_offset=0))
    high = build_solved_case(tmp_path / "high", SolvedCaseSpec(id_offset=10))

    def furnas_id(output: Path) -> int:
        return next(
            h["id"] for h in read_novomodelo_hydros(output) if h["name"] == "FURNAS"
        )

    assert furnas_id(low) == 0
    assert furnas_id(high) == 10
    assert [b["id"] for b in read_novomodelo_buses(high)] == [10, 11]
    assert [t["id"] for t in read_novomodelo_thermals(high)] == [10]
    assert [ln["id"] for ln in read_novomodelo_lines(high)] == [10]


def test_entities_reference_buses_by_the_offset_ids(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case", SolvedCaseSpec(id_offset=10))

    (line,) = read_novomodelo_lines(output)
    (thermal,) = read_novomodelo_thermals(output)
    hydros = read_novomodelo_hydros(output)

    assert (line["source_bus_id"], line["target_bus_id"]) == (10, 11)
    assert thermal["bus_id"] == 11
    assert {h["unit_groups"][0]["bus_id"] for h in hydros} == {10}


def test_line_with_an_unknown_bus_raises_naming_the_line(tmp_path: Path) -> None:
    spec = SolvedCaseSpec(lines=(("SE-NE", "SUDESTE", "NORDESTE"),))

    with pytest.raises(ValueError, match="SE-NE"):
        build_solved_case(tmp_path / "case", spec)


@pytest.mark.parametrize(
    "override",
    [{"convergence_schema": "pre_0_15"}, {"version_key": "novomodelo_version"}],
)
def test_unknown_choice_raises_naming_the_field(
    tmp_path: Path, override: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match=next(iter(override))):
        build_solved_case(tmp_path / "case", SolvedCaseSpec(**override))


_MEANS_READERS: list[Callable[[Path], pl.DataFrame]] = [
    read_novomodelo_hydro_means,
    read_novomodelo_thermal_means,
    read_novomodelo_bus_means,
    read_novomodelo_line_means,
    read_novomodelo_stage_costs,
    read_novomodelo_bus_aggregates,
]


@pytest.mark.parametrize("reader", _MEANS_READERS, ids=lambda r: r.__name__)
def test_means_reader_returns_rows_without_warnings(
    tmp_path: Path, reader: Callable[[Path], pl.DataFrame]
) -> None:
    output = build_solved_case(tmp_path / "case")

    with dx.collect() as found:
        frame = reader(output)

    assert frame.height >= 1
    assert [
        d for d in found if d.severity in (dx.Severity.WARNING, dx.Severity.ERROR)
    ] == []


def test_hydro_means_carry_a_value_in_every_column(tmp_path: Path) -> None:
    frame = read_novomodelo_hydro_means(build_solved_case(tmp_path / "case"))

    assert frame.height == 4
    assert sum(frame.null_count().row(0)) == 0


def test_every_entity_is_simulated_in_every_stage(tmp_path: Path) -> None:
    spec = SolvedCaseSpec(stages=3)
    output = build_solved_case(tmp_path / "case", spec)

    assert read_novomodelo_hydro_means(output).height == 2 * 3
    assert read_novomodelo_thermal_means(output).height == 1 * 3
    assert read_novomodelo_bus_means(output).height == 2 * 3
    assert read_novomodelo_line_means(output).height == 1 * 3
    assert read_novomodelo_stage_costs(output).height == 3


def test_scenarios_sets_the_partitions_and_the_summary_rows(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case", SolvedCaseSpec(scenarios=3))

    partitions = sorted(
        p.parent.name for p in (output / "simulation" / "hydros").rglob("*.parquet")
    )
    summary = pl.read_parquet(output / "simulation" / "scenario_summary.parquet")

    assert partitions == ["scenario_id=0000", "scenario_id=0001", "scenario_id=0002"]
    assert summary.schema == {"scenario_id": pl.Int32, "probability": pl.Float64}
    assert summary["scenario_id"].to_list() == [0, 1, 2]
    assert summary["probability"].null_count() == 3


def test_scenarios_weigh_the_same_without_a_warning(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")

    with dx.collect() as found:
        weights = read_novomodelo_scenario_weights(output)

    assert weights.uniform
    assert found == []


def test_scenarios_differ_so_that_an_average_is_not_a_single_scenario(
    tmp_path: Path,
) -> None:
    output = build_solved_case(tmp_path / "case")

    scenario_means = [
        pl.read_parquet(
            output / "simulation" / "thermals" / f"scenario_id={s:04d}" / "data.parquet"
        )["generation_mw"].mean()
        for s in (0, 1)
    ]

    assert scenario_means[0] != scenario_means[1]


def test_fpha_planes_hold_two_per_hydro_and_stage(tmp_path: Path) -> None:
    default = read_novomodelo_fpha_planes(build_solved_case(tmp_path / "a"))
    longer = read_novomodelo_fpha_planes(
        build_solved_case(tmp_path / "b", SolvedCaseSpec(stages=3))
    )

    assert default is not None
    assert longer is not None
    assert default.height == 2 * 2 * 2
    assert longer.height == 2 * 2 * 3
    assert default.columns == [
        "hydro_id",
        "stage_id",
        "plane_id",
        "gamma_0",
        "gamma_v",
        "gamma_q",
        "gamma_s",
        "kappa",
    ]
    assert default["plane_id"].unique().sort().to_list() == [0, 1]
    assert default.select(pl.struct("hydro_id", "stage_id", "plane_id")).n_unique() == 8


def test_without_fpha_planes_the_output_has_no_hydro_models(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case", SolvedCaseSpec(fpha_planes=False))

    assert read_novomodelo_fpha_planes(output) is None
    assert not (output / "hydro_models").exists()


@pytest.mark.parametrize(
    ("reader", "column"),
    [
        (read_novomodelo_hydro_means, "generation_mw"),
        (read_novomodelo_hydro_means, "water_value_per_hm3"),
        (read_novomodelo_thermal_means, "generation_mw"),
        (read_novomodelo_bus_means, "spot_price"),
        (read_novomodelo_line_means, "net_flow_mw"),
        (read_novomodelo_stage_costs, "immediate_cost"),
        (read_novomodelo_stage_costs, "future_cost"),
        (read_novomodelo_stage_costs, "thermal_cost"),
    ],
)
def test_scale_multiplies_every_simulated_flow_price_and_cost(
    tmp_path: Path, reader: Callable[[Path], pl.DataFrame], column: str
) -> None:
    base = reader(build_solved_case(tmp_path / "a"))
    doubled = reader(build_solved_case(tmp_path / "b", SolvedCaseSpec(scale=2.0)))

    assert base.height >= 1
    assert (base[column] > 0).all()
    assert doubled[column].to_list() == pytest.approx([2 * v for v in base[column]])


def test_scale_does_not_change_which_rows_exist(tmp_path: Path) -> None:
    base = read_novomodelo_thermal_means(build_solved_case(tmp_path / "a"))
    doubled = read_novomodelo_thermal_means(
        build_solved_case(tmp_path / "b", SolvedCaseSpec(scale=2.0))
    )

    assert doubled.select("entity_id", "stage_id").equals(
        base.select("entity_id", "stage_id")
    )


def test_id_offset_moves_the_simulated_ids_but_not_the_values(tmp_path: Path) -> None:
    low = read_novomodelo_hydro_means(build_solved_case(tmp_path / "low"))
    high = read_novomodelo_hydro_means(
        build_solved_case(tmp_path / "high", SolvedCaseSpec(id_offset=10))
    )

    assert low["entity_id"].unique().sort().to_list() == [0, 1]
    assert high["entity_id"].unique().sort().to_list() == [10, 11]
    assert high.drop("entity_id").equals(low.drop("entity_id"))


def test_hydro_generation_by_bus_sits_on_the_first_bus(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case", SolvedCaseSpec(id_offset=10))

    by_bus = (
        read_novomodelo_bus_aggregates(output)
        .group_by("bus_id")
        .agg(pl.col("hydro_gen_mw_p50").max())
        .sort("bus_id")
    )
    first, second = by_bus["hydro_gen_mw_p50"].to_list()

    assert by_bus["bus_id"].to_list() == [10, 11]
    assert first > 0
    assert second is None


def test_an_entity_list_left_empty_writes_no_partition(tmp_path: Path) -> None:
    spec = SolvedCaseSpec(lines=())
    output = build_solved_case(tmp_path / "case", spec)

    assert not (output / "simulation" / "exchanges").exists()
    assert (output / "simulation" / "thermals").is_dir()
    assert read_novomodelo_line_means(output).is_empty()


def test_an_empty_hydro_list_writes_neither_hydro_partition(tmp_path: Path) -> None:
    spec = SolvedCaseSpec(hydros=())
    output = build_solved_case(tmp_path / "case", spec)

    assert not (output / "simulation" / "hydros").exists()
    assert not (output / "simulation" / "hydro_bus_generation").exists()
    assert (output / "simulation" / "thermals").is_dir()


def _bus_load_at_scenario_zero(output: Path) -> list[float]:
    path = output / "simulation" / "buses" / "scenario_id=0000" / "data.parquet"
    return pl.read_parquet(path).sort("stage_id", "bus_id")["load_mw"].to_list()


def test_every_hydro_row_carries_the_accumulated_productivity_unscaled(
    tmp_path: Path,
) -> None:
    for scale in (1.0, 2.0):
        output = build_solved_case(
            tmp_path / f"scale{scale}", SolvedCaseSpec(scale=scale)
        )
        rows = pl.read_parquet(output / "simulation" / "hydros")

        assert rows.height == 2 * 2 * 2
        assert rows["accumulated_productivity_mw_per_m3s"].to_list() == [0.08] * 8
        assert rows["equivalent_productivity_mw_per_m3s"].to_list() == [0.08] * 8


def test_a_default_build_reads_back_as_hydro_block_operation(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")

    with dx.collect() as found:
        operation = read_novomodelo_hydro_block_operation(output)

    assert operation.height == 2 * 2 * 2
    assert sum(operation.null_count().row(0)) == 0
    assert found == []


def test_load_stats_write_one_row_per_bus_and_stage(tmp_path: Path) -> None:
    spec = SolvedCaseSpec(stages=3, load_stats=True, id_offset=10)
    output = build_solved_case(tmp_path / "case", spec)

    stats = pl.read_parquet(
        tmp_path / "case" / "scenarios" / "load_seasonal_stats.parquet"
    )

    assert stats.schema == {
        "bus_id": pl.Int32,
        "stage_id": pl.Int32,
        "mean_mw": pl.Float64,
        "std_mw": pl.Float64,
    }
    assert stats.height == 2 * 3
    assert stats.select("bus_id", "stage_id").unique().height == 2 * 3
    assert stats["bus_id"].unique().sort().to_list() == [10, 11]
    assert stats["std_mw"].to_list() == [0.0] * 6
    assert stats.sort("stage_id", "bus_id")["mean_mw"].to_list() == pytest.approx(
        _bus_load_at_scenario_zero(output)
    )


def test_load_stats_follow_the_scale_of_the_simulated_load(tmp_path: Path) -> None:
    spec = SolvedCaseSpec(scale=2.0, load_stats=True)
    build_solved_case(tmp_path / "scaled", spec)
    build_solved_case(tmp_path / "base", SolvedCaseSpec(load_stats=True))

    def means(case: str) -> list[float]:
        path = tmp_path / case / "scenarios" / "load_seasonal_stats.parquet"
        return pl.read_parquet(path)["mean_mw"].to_list()

    assert means("scaled") == pytest.approx([2 * v for v in means("base")])


def test_non_controllables_write_the_sources_and_the_partition(
    tmp_path: Path,
) -> None:
    spec = SolvedCaseSpec(non_controllables=("WIND_A", "SOLAR_B"), id_offset=10)
    output = build_solved_case(tmp_path / "case", spec)

    with dx.collect() as found:
        sources = read_novomodelo_non_controllable_sources(output)
    partition = pl.read_parquet(
        output
        / "simulation"
        / "non_controllables"
        / "scenario_id=0000"
        / "data.parquet"
    )

    assert found == []
    assert sources == [
        {"id": 10, "name": "WIND_A", "bus_id": 10},
        {"id": 11, "name": "SOLAR_B", "bus_id": 10},
    ]
    assert partition.columns == [
        "stage_id",
        "block_id",
        "non_controllable_id",
        "generation_mw",
    ]
    assert partition.height == 2 * 2
    assert partition["non_controllable_id"].unique().sort().to_list() == [10, 11]
    assert (partition["generation_mw"] > 0).all()
    assert len(list((output / "simulation" / "non_controllables").glob("*"))) == 2


@pytest.mark.parametrize(
    "spec",
    [SolvedCaseSpec(), SolvedCaseSpec(load_stats=False, non_controllables=())],
    ids=["default", "explicitly-off"],
)
def test_the_options_default_to_writing_nothing(
    tmp_path: Path, spec: SolvedCaseSpec
) -> None:
    output = build_solved_case(tmp_path / "case", spec)

    assert not (tmp_path / "case" / "scenarios").exists()
    assert not (tmp_path / "case" / "system" / "non_controllable_sources.json").exists()
    assert not (output / "simulation" / "non_controllables").exists()


def test_the_options_complete_the_bus_readers(tmp_path: Path) -> None:
    spec = SolvedCaseSpec(load_stats=True, non_controllables=("WIND_A",))
    output = build_solved_case(tmp_path / "case", spec)

    with dx.collect() as found:
        means = read_novomodelo_bus_means(output)
        aggregates = read_novomodelo_bus_aggregates(output)

    assert means["net_import_mw"].null_count() == 0
    assert {"ncs_gen_mw_p50", "net_load_mw_p50"} <= set(aggregates.columns)
    assert found == []


def test_two_builds_with_the_options_on_write_identical_files(tmp_path: Path) -> None:
    spec = SolvedCaseSpec(load_stats=True, non_controllables=("WIND_A",))
    build_solved_case(tmp_path / "a", spec)
    build_solved_case(tmp_path / "b", spec)

    written = sorted(
        p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*.*")
    )

    assert len(written) == 21 + 4
    assert {
        Path("scenarios/load_seasonal_stats.parquet"),
        Path("system/non_controllable_sources.json"),
    } <= set(written)
    assert all(
        (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()
        for rel in written
    )


def test_drop_partition_removes_a_directory(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")

    drop_partition(output, "simulation/thermals")

    assert not (output / "simulation" / "thermals").exists()
    assert (output / "simulation" / "hydros").is_dir()
    assert read_novomodelo_thermal_means(output).is_empty()


def test_drop_partition_deletes_a_file(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")

    drop_partition(output, "training/convergence.parquet")

    assert not (output / "training" / "convergence.parquet").exists()
    assert (output / "training" / "metadata.json").is_file()


def test_drop_column_rewrites_every_file_below_a_directory(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")
    files = sorted((output / "simulation" / "buses").rglob("data.parquet"))
    before = list(pl.read_parquet_schema(files[0]))

    drop_column(output, "simulation/buses", "spot_price")

    assert len(files) == 2
    assert [list(pl.read_parquet_schema(f)) for f in files] == [
        [c for c in before if c != "spot_price"]
    ] * 2
    assert "spot_price" in before


def test_drop_column_rewrites_one_named_file(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")
    path = output / "training" / "convergence.parquet"
    rows = pl.read_parquet(path).height

    drop_column(output, "training/convergence.parquet", "gap_percent")

    remaining = pl.read_parquet(path)
    assert "gap_percent" not in remaining.columns
    assert "lower_bound" in remaining.columns
    assert remaining.height == rows


def test_a_reader_sees_the_column_a_corruptor_dropped(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")

    drop_column(output, "simulation/buses", "spot_price")

    frame = read_novomodelo_bus_means(output)

    assert frame.height >= 1
    assert frame["spot_price"].is_null().all()


def test_corrupt_file_makes_the_first_partition_unreadable(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")
    first, second = sorted((output / "simulation" / "hydros").rglob("data.parquet"))

    corrupt_file(output, "simulation/hydros")

    assert first.read_bytes() == GARBAGE_BYTES
    assert pl.read_parquet(second).height > 0
    with pytest.raises(NovomodeloReadError):
        read_novomodelo_hydro_means(output)


def test_corrupt_file_overwrites_a_named_file(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")

    corrupt_file(output, "training/convergence.parquet")

    assert (output / "training" / "convergence.parquet").read_bytes() == GARBAGE_BYTES


@pytest.mark.parametrize(
    "corruptor",
    [
        drop_partition,
        lambda output, relative: drop_column(output, relative, "spot_price"),
        corrupt_file,
    ],
    ids=["drop_partition", "drop_column", "corrupt_file"],
)
def test_a_corruptor_names_the_path_it_does_not_find(
    tmp_path: Path, corruptor: Callable[[Path, str], None]
) -> None:
    output = build_solved_case(tmp_path / "case")

    with pytest.raises(FileNotFoundError, match="simulation/missing"):
        corruptor(output, "simulation/missing")


@pytest.mark.parametrize("relative", ["..", "../..", ".", "../outside.txt"])
def test_a_corruptor_leaves_files_outside_the_output_alone(
    tmp_path: Path, relative: str
) -> None:
    output = build_solved_case(tmp_path / "case")
    outside = output.parent / "outside.txt"
    outside.write_text("keep", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        drop_partition(output, relative)

    assert outside.read_text(encoding="utf-8") == "keep"
    assert (output / "simulation").is_dir()


def test_drop_column_names_a_column_the_file_lacks(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")
    files = sorted((output / "simulation" / "buses").rglob("data.parquet"))
    before = [f.read_bytes() for f in files]

    with pytest.raises(KeyError, match="no_such_column"):
        drop_column(output, "simulation/buses", "no_such_column")

    assert [f.read_bytes() for f in files] == before


def test_drop_column_rewrites_nothing_when_a_later_file_lacks_the_column(
    tmp_path: Path,
) -> None:
    output = build_solved_case(tmp_path / "case")
    first, second = sorted((output / "simulation" / "buses").rglob("data.parquet"))
    drop_column(output, str(second.relative_to(output)), "spot_price")
    before = first.read_bytes()

    with pytest.raises(KeyError, match="spot_price"):
        drop_column(output, "simulation/buses", "spot_price")

    assert first.read_bytes() == before


def test_a_directory_without_a_data_file_is_not_found(tmp_path: Path) -> None:
    output = build_solved_case(tmp_path / "case")

    with pytest.raises(FileNotFoundError, match="data.parquet"):
        corrupt_file(output, "hydro_models")

    assert (output / "hydro_models" / "fpha_hyperplanes.parquet").is_file()

"""Unit tests for the ANALYZE-layer adapters."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl
import pytest

from novomodelo_bridge.comparators.analyze import (
    aggregate_percentile_band,
    build_results_dataset,
    bus_groups_and_pct,
    cost_percent_deltas,
    fpha_metric_summary,
    novomodelo_sum_and_newave_sin,
    per_bus_band_from_pct,
    per_bus_sums_from_frame,
    per_bus_sums_from_results,
    per_stage_sum_from_frame,
    per_stage_sum_from_results,
    plant_max_reldiff_ranking,
    plant_percentile_arrays,
    productivity_scatter_errors,
    results_footer_counts,
    spillage_lookups,
    summary_frame_from_results,
    tidy_from_results,
    tidy_percentiles_from_percentile_data,
    tidy_results_dataset,
    top_divergences_from_results,
)
from novomodelo_bridge.comparators.dataset import (
    SUMMARY_SCHEMA,
    TIDY_SCHEMA,
    ComparisonDataset,
)
from novomodelo_bridge.comparators.model import (
    PercentileData,
    ResultComparison,
    build_results_summary,
)
from novomodelo_bridge.comparators.verdict import (
    CompareVerdict,
    build_compare_verdict,
)
from novomodelo_bridge.core import diagnostics as dx


def _make_results() -> list[ResultComparison]:
    return [
        ResultComparison(
            entity_type="hydro",
            entity_name="ITAIPU",
            newave_code=10,
            novomodelo_id=0,
            stage=0,
            variable="generation_mw",
            newave_value=100.0,
            novomodelo_value=110.0,
            abs_diff=10.0,
            rel_diff=0.1,
        ),
        ResultComparison(
            entity_type="hydro",
            entity_name="TUCURUI",
            newave_code=20,
            novomodelo_id=1,
            stage=1,
            variable="generation_mw",
            newave_value=50.0,
            novomodelo_value=40.0,
            abs_diff=10.0,
            rel_diff=0.2,
        ),
        ResultComparison(
            entity_type="thermal",
            entity_name="ANGRA",
            newave_code=30,
            novomodelo_id=2,
            stage=0,
            variable="generation_mw",
            newave_value=0.0,
            novomodelo_value=5.0,
            abs_diff=5.0,
            rel_diff=None,
        ),
    ]


def _empty_summary() -> pl.DataFrame:
    return ComparisonDataset.empty().summary


def test_tidy_from_results_row_count_and_sources() -> None:
    results = _make_results()

    out = tidy_from_results(results)

    assert out.height == 2 * len(results)
    assert set(out["source"].unique().to_list()) == {"newave", "novomodelo"}


def test_tidy_from_results_schema_conforms() -> None:
    out = tidy_from_results(_make_results())

    assert list(out.columns) == list(TIDY_SCHEMA)
    expected = {name: dtype() for name, dtype in TIDY_SCHEMA.items()}
    assert dict(out.schema) == expected


def test_tidy_from_results_values_and_sentinels() -> None:
    out = tidy_from_results(_make_results())

    newave_row = out.filter((pl.col("entity_id") == 0) & (pl.col("source") == "newave"))
    assert newave_row["value"].to_list() == [100.0]
    novomodelo_row = out.filter(
        (pl.col("entity_id") == 0) & (pl.col("source") == "novomodelo")
    )
    assert novomodelo_row["value"].to_list() == [110.0]
    assert out["bus"].unique().to_list() == [-1]
    assert out["block"].unique().to_list() == [-1]


def test_tidy_from_results_empty_returns_schema_frame() -> None:
    out = tidy_from_results([])

    assert out.height == 0
    assert list(out.columns) == list(TIDY_SCHEMA)


def test_tidy_percentiles_unpivots_triplets() -> None:
    pct = PercentileData(
        hydro=pl.DataFrame(
            {
                "entity_id": [0, 1],
                "stage_id": [0, 0],
                "generation_mw_p10": [90.0, 45.0],
                "generation_mw_p50": [100.0, 50.0],
                "generation_mw_p90": [110.0, 55.0],
            }
        )
    )

    out = tidy_percentiles_from_percentile_data(pct)

    assert out.height == 6
    assert set(out["source"].unique().to_list()) == {"p10", "p50", "p90"}
    assert out["variable"].unique().to_list() == ["generation_mw"]
    assert out["entity_type"].unique().to_list() == ["hydro"]
    assert out["entity_name"].unique().to_list() == [""]
    assert out["bus"].unique().to_list() == [-1]
    assert out["block"].unique().to_list() == [-1]
    p50 = out.filter((pl.col("entity_id") == 0) & (pl.col("source") == "p50"))
    assert p50["value"].to_list() == [100.0]


def test_tidy_percentiles_multiple_entity_types() -> None:
    pct = PercentileData(
        hydro=pl.DataFrame(
            {
                "entity_id": [0],
                "stage_id": [0],
                "generation_mw_p10": [1.0],
                "generation_mw_p50": [2.0],
                "generation_mw_p90": [3.0],
            }
        ),
        thermal=pl.DataFrame(
            {
                "entity_id": [5],
                "stage_id": [2],
                "generation_mw_p10": [10.0],
                "generation_mw_p50": [20.0],
                "generation_mw_p90": [30.0],
            }
        ),
    )

    out = tidy_percentiles_from_percentile_data(pct)

    assert set(out["entity_type"].unique().to_list()) == {"hydro", "thermal"}
    assert out.height == 6


def test_tidy_percentiles_empty_returns_schema_frame() -> None:
    out = tidy_percentiles_from_percentile_data(PercentileData())

    assert out.height == 0
    assert list(out.columns) == list(TIDY_SCHEMA)
    expected = {name: dtype() for name, dtype in TIDY_SCHEMA.items()}
    assert dict(out.schema) == expected


def test_tidy_percentiles_missing_identifier_skipped() -> None:
    pct = PercentileData(
        hydro=pl.DataFrame(
            {
                "entity_id": [0],
                "generation_mw_p10": [1.0],
                "generation_mw_p50": [2.0],
                "generation_mw_p90": [3.0],
            }
        )
    )

    out = tidy_percentiles_from_percentile_data(pct)

    assert out.height == 0
    assert list(out.columns) == list(TIDY_SCHEMA)


def test_tidy_results_dataset_validates() -> None:
    pct = PercentileData(
        hydro=pl.DataFrame(
            {
                "entity_id": [0, 1],
                "stage_id": [0, 0],
                "generation_mw_p10": [90.0, 45.0],
                "generation_mw_p50": [100.0, 50.0],
                "generation_mw_p90": [110.0, 55.0],
            }
        )
    )

    tidy = tidy_results_dataset(_make_results(), pct)
    dataset = ComparisonDataset(tidy=tidy, summary=_empty_summary(), metadata={})

    dataset.validate()
    assert tidy.height == 2 * len(_make_results()) + 6


def test_tidy_results_dataset_empty_inputs_validate() -> None:
    tidy = tidy_results_dataset([], PercentileData())
    dataset = ComparisonDataset(tidy=tidy, summary=_empty_summary(), metadata={})

    dataset.validate()
    assert tidy.height == 0


def _make_many_results(n: int) -> list[ResultComparison]:
    return [
        ResultComparison(
            entity_type="hydro",
            entity_name=f"PLANT_{i:03d}",
            newave_code=i,
            novomodelo_id=i,
            stage=i % 4,
            variable="generation_mw",
            newave_value=float(i),
            novomodelo_value=float(i) + float(i),
            abs_diff=float(i),
            rel_diff=None if i == 0 else 1.0,
        )
        for i in range(n)
    ]


def test_summary_frame_matches_build_results_summary() -> None:
    results = _make_results()

    frame = summary_frame_from_results(results, 1e-2)
    summary = build_results_summary(results, 1e-2)

    assert list(frame.columns) == list(SUMMARY_SCHEMA)
    expected = {name: dtype() for name, dtype in SUMMARY_SCHEMA.items()}
    assert dict(frame.schema) == expected
    assert frame.height == len(summary.by_variable)

    for row in frame.iter_rows(named=True):
        stats = summary.by_variable[row["variable"]]
        assert row["count"] == stats.count
        assert row["mean_abs_diff"] == stats.mean_abs_diff
        assert row["max_abs_diff"] == stats.max_abs_diff
        assert row["mean_smape"] == stats.mean_smape
        assert row["max_smape"] == stats.max_smape
        assert row["within_tol_rate"] == stats.within_tol_rate
        assert row["correlation"] == stats.correlation


def test_top_divergences_sorted_and_truncated() -> None:
    results = _make_many_results(50)

    top = top_divergences_from_results(results, 5)

    assert len(top) == 5
    abs_diffs = [d["abs_diff"] for d in top]
    assert abs_diffs == sorted(abs_diffs, reverse=True)
    assert top[0]["abs_diff"] == max(r.abs_diff for r in results)
    assert set(top[0].keys()) == {
        "entity_type",
        "entity_name",
        "novomodelo_id",
        "stage",
        "variable",
        "newave_value",
        "novomodelo_value",
        "abs_diff",
        "rel_diff",
    }


def test_top_divergences_tie_break_deterministic() -> None:
    results = _make_results()  # two rows share abs_diff == 10.0

    top = top_divergences_from_results(results, 3)

    # abs_diff 10.0 ties resolved by entity_name (ITAIPU before TUCURUI),
    # the 5.0 row (ANGRA) comes last despite the earlier name.
    assert [d["entity_name"] for d in top] == ["ITAIPU", "TUCURUI", "ANGRA"]


def test_summary_and_top_divergences_empty_inputs() -> None:
    frame = summary_frame_from_results([], 1e-2)
    top = top_divergences_from_results([])

    assert frame.height == 0
    assert list(frame.columns) == list(SUMMARY_SCHEMA)
    expected = {name: dtype() for name, dtype in SUMMARY_SCHEMA.items()}
    assert dict(frame.schema) == expected
    assert top == []


def test_build_results_dataset_validates_and_carries_metadata() -> None:
    results = _make_results()
    pct = PercentileData(
        hydro=pl.DataFrame(
            {
                "entity_id": [0, 1],
                "stage_id": [0, 0],
                "generation_mw_p10": [90.0, 45.0],
                "generation_mw_p50": [100.0, 50.0],
                "generation_mw_p90": [110.0, 55.0],
            }
        ),
        nw_costs={"deficit": 1.0},
        novomodelo_costs={"deficit": 2.0},
        nw_bus_names={0: "SUDESTE"},
        nw_hydro_names={0: "ITAIPU", 1: "TUCURUI"},
    )

    dataset = build_results_dataset(results, pct, 1e-2)

    dataset.validate()
    assert len(dataset.metadata["top_divergences"]) == len(results)
    assert dataset.render.nw_costs == {"deficit": 1.0}
    assert dataset.render.novomodelo_costs == {"deficit": 2.0}
    assert dataset.render.nw_bus_names == {0: "SUDESTE"}
    assert dataset.metadata["nw_hydro_names"] == {0: "ITAIPU", 1: "TUCURUI"}
    assert dataset.tidy.height == 2 * len(results) + 6


def test_build_results_dataset_empty_inputs_validate() -> None:
    dataset = build_results_dataset([], PercentileData(), 1e-2)

    dataset.validate()
    assert dataset.tidy.height == 0
    assert dataset.summary.height == 0


def test_build_results_dataset_populates_render_inputs() -> None:
    """``results`` + every drained ``pct`` frame land on ``dataset.render``."""
    results = _make_results()
    pct = PercentileData(
        thermal=pl.DataFrame({"entity_id": [0], "stage_id": [0]}),
        nw_max_stage=12,
        novomodelo_training_seconds=3.5,
    )

    dataset = build_results_dataset(results, pct, 1e-2)

    assert dataset.render.results == list(results)
    assert dataset.render.nw_max_stage == 12
    assert dataset.render.novomodelo_training_seconds == 3.5
    assert isinstance(dataset.render.thermal, pl.DataFrame)
    assert isinstance(dataset.render.gc_constraints, list)


def test_metadata_json_holds_only_provenance_at_top_level(tmp_path: Path) -> None:
    """``to_dir``'s top-level ``metadata.json`` holds only provenance keys.

    Render inputs (``results`` plus the drained ``PercentileData`` frames)
    serialize into the nested render payload, never as a top-level
    ``metadata.json`` key — while the genuine provenance key
    ``top_divergences`` is kept at the top level.
    """
    import dataclasses
    import json

    from novomodelo_bridge.comparators.dataset import RenderInputs

    results = _make_results()
    pct = PercentileData(
        hydro=pl.DataFrame({"entity_id": [0], "stage_id": [0]}),
        thermal=pl.DataFrame({"entity_id": [0], "stage_id": [0]}),
        nw_costs={"deficit": 1.0},
        nw_max_stage=5,
    )
    dataset = build_results_dataset(results, pct, 1e-2)

    paths = dataset.to_dir(tmp_path)
    metadata_path = paths[2]
    written = json.loads(metadata_path.read_text(encoding="utf-8"))

    # No RenderInputs field name leaks in as a top-level metadata.json key.
    render_field_names = {f.name for f in dataclasses.fields(RenderInputs)}
    leaked = render_field_names & written.keys()
    assert not leaked, f"render fields leaked into top-level metadata.json: {leaked}"
    # Genuine provenance keys survive at the top level.
    assert "top_divergences" in written
    assert "footer_counts" in written
    assert "nw_hydro_names" in written


def test_render_inputs_fields_match_report_builder_consumption() -> None:
    """Every ``RenderInputs`` field is read by ``report_builder``, and vice versa.

    Guards the sync invariant between :class:`RenderInputs` and
    ``report_builder.build_comparison_report``: a contributor who adds a new
    typed render field but forgets to wire it into the report (or removes a
    consumer without dropping the field) is caught here, rather than
    surfacing as a silently-unused field or an ``AttributeError`` at render
    time.
    """
    import dataclasses
    import inspect
    import re

    from novomodelo_bridge.comparators import report_builder
    from novomodelo_bridge.comparators.dataset import RenderInputs

    field_names = {f.name for f in dataclasses.fields(RenderInputs)}
    source = inspect.getsource(report_builder)
    consumed = set(re.findall(r"dataset\.render\.(\w+)", source))

    assert consumed == field_names


# -------------------------------------------------------------------
# footer/summary count metadata
# -------------------------------------------------------------------


def test_results_footer_counts_matches_legacy_summary() -> None:
    results = _make_results()

    footer = results_footer_counts(results)
    legacy = build_results_summary(results, 1e-2)

    assert footer["total"] == legacy.total
    assert footer["by_entity_type"] == legacy.by_entity_type


def test_build_results_dataset_carries_footer_counts() -> None:
    results = _make_results()

    dataset = build_results_dataset(results, PercentileData(), 1e-2)

    legacy = build_results_summary(results, 1e-2)
    footer = dataset.metadata["footer_counts"]
    assert isinstance(footer, dict)
    assert footer["total"] == legacy.total
    assert footer["by_entity_type"] == legacy.by_entity_type


# ---------------------------------------------------------------------------
# chart-aggregation primitives
# ---------------------------------------------------------------------------


def _band_result(stage: int, novomodelo_id: int, var: str) -> ResultComparison:
    return ResultComparison(
        entity_type="hydro",
        entity_name=f"H{novomodelo_id}",
        newave_code=novomodelo_id + 10,
        novomodelo_id=novomodelo_id,
        stage=stage,
        variable=var,
        newave_value=float(stage * 100 + novomodelo_id),
        novomodelo_value=float(stage * 100 + novomodelo_id) - 5.0,
        abs_diff=5.0,
        rel_diff=None,
    )


def test_aggregate_percentile_band_sums_across_entities() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 1],
            "stage_id": [1, 1, 2, 2],
            "storage_final_hm3_p10": [80.0, 160.0, 85.0, 165.0],
            "storage_final_hm3_p90": [100.0, 200.0, 105.0, 205.0],
        }
    )
    p10, p90 = aggregate_percentile_band(pct, "storage_final_hm3", [1, 2], None)
    assert p10 == [240.0, 250.0]
    assert p90 == [300.0, 310.0]


def test_aggregate_percentile_band_missing_column_returns_empty() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [1, 1],
            # only the _p90 column present -> _p10 missing
            "storage_final_hm3_p90": [100.0, 200.0],
        }
    )
    assert aggregate_percentile_band(pct, "storage_final_hm3", [1], None) == ([], [])


def test_aggregate_percentile_band_none_returns_empty() -> None:
    assert aggregate_percentile_band(None, "storage_final_hm3", [1, 2], None) == (
        [],
        [],
    )
    empty = pl.DataFrame()
    assert aggregate_percentile_band(empty, "storage_final_hm3", [1], None) == ([], [])


def test_aggregate_percentile_band_missing_stage_defaults_to_zero() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0],
            "stage_id": [1],
            "v_p10": [10.0],
            "v_p90": [20.0],
        }
    )
    # stage 3 is absent from the frame -> 0.0, not a KeyError.
    p10, p90 = aggregate_percentile_band(pct, "v", [1, 3], None)
    assert p10 == [10.0, 0.0]
    assert p90 == [20.0, 0.0]


def test_aggregate_percentile_band_filters_entity_ids() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [1, 1],
            "v_p10": [80.0, 160.0],
            "v_p90": [100.0, 200.0],
        }
    )
    # keep only entity 1.
    p10, p90 = aggregate_percentile_band(pct, "v", [1], {1})
    assert p10 == [160.0]
    assert p90 == [200.0]


def test_per_stage_sum_from_results_filters_entity_type_and_variable() -> None:
    results = [
        _band_result(stage=1, novomodelo_id=0, var="storage_final_hm3"),
        _band_result(stage=1, novomodelo_id=1, var="storage_final_hm3"),
        _band_result(stage=2, novomodelo_id=0, var="storage_final_hm3"),
        # different variable / type -> excluded.
        _band_result(stage=1, novomodelo_id=9, var="other"),
        ResultComparison(
            entity_type="thermal",
            entity_name="T",
            newave_code=99,
            novomodelo_id=99,
            stage=1,
            variable="storage_final_hm3",
            newave_value=1000.0,
            novomodelo_value=1.0,
            abs_diff=999.0,
            rel_diff=None,
        ),
    ]
    nw, cb, matched = per_stage_sum_from_results(results, "hydro", "storage_final_hm3")
    # newave_value = stage*100 + novomodelo_id; novomodelo_value = that - 5.
    # stage 1: entities 0,1 -> 100 + 101 = 201; stage 2: entity 0 -> 200.
    assert nw == {1: 100.0 + 101.0, 2: 200.0}
    assert cb == {1: 95.0 + 96.0, 2: 195.0}
    assert matched == {0, 1}


def test_per_stage_sum_from_results_empty_variable_matches_all() -> None:
    results = [
        ResultComparison(
            entity_type="thermal",
            entity_name="T1",
            newave_code=21,
            novomodelo_id=5,
            stage=1,
            variable="generation_mw",
            newave_value=50.0,
            novomodelo_value=45.0,
            abs_diff=5.0,
            rel_diff=None,
        ),
        ResultComparison(
            entity_type="thermal",
            entity_name="T2",
            newave_code=22,
            novomodelo_id=6,
            stage=1,
            variable="commitment",  # different variable, same type -> still kept
            newave_value=1.0,
            novomodelo_value=1.0,
            abs_diff=0.0,
            rel_diff=None,
        ),
    ]
    nw, cb, matched = per_stage_sum_from_results(results, "thermal", "")
    assert nw == {1: 51.0}
    assert cb == {1: 46.0}
    assert matched == {5, 6}


def test_per_stage_sum_from_results_no_match_returns_empty() -> None:
    results = [_band_result(stage=1, novomodelo_id=0, var="v")]
    nw, cb, matched = per_stage_sum_from_results(results, "bus", "v")
    assert nw == {}
    assert cb == {}
    assert matched == set()


def test_per_stage_sum_from_frame_matches_legacy() -> None:
    df = pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 1],
            "stage_id": [2, 2, 1, 1],
            "turbined_m3s": [10.0, 20.0, 30.0, 40.0],
        }
    )
    out = per_stage_sum_from_frame(df, "turbined_m3s", None)
    # sorted by stage_id.
    assert list(out.keys()) == [1, 2]
    assert out == {1: 70.0, 2: 30.0}


def test_per_stage_sum_from_frame_filters_matched_ids() -> None:
    df = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [1, 1],
            "turbined_m3s": [30.0, 40.0],
        }
    )
    assert per_stage_sum_from_frame(df, "turbined_m3s", {1}) == {1: 40.0}


def test_per_stage_sum_from_frame_empty_returns_empty() -> None:
    assert per_stage_sum_from_frame(None, "turbined_m3s", None) == {}
    assert per_stage_sum_from_frame(pl.DataFrame(), "turbined_m3s", None) == {}
    # frame present but column absent.
    df = pl.DataFrame({"entity_id": [0], "stage_id": [1], "other": [1.0]})
    assert per_stage_sum_from_frame(df, "turbined_m3s", None) == {}


# ---------------------------------------------------------------------------
# per-bus roll-up + per-plant percentile extraction
# ---------------------------------------------------------------------------


def _hydro_row(
    novomodelo_id: int,
    stage: int,
    variable: str,
    nw: float,
    cb: float,
    *,
    entity_type: str = "hydro",
) -> ResultComparison:
    return ResultComparison(
        entity_type=entity_type,
        entity_name=f"H{novomodelo_id}",
        newave_code=novomodelo_id + 10,
        novomodelo_id=novomodelo_id,
        stage=stage,
        variable=variable,
        newave_value=nw,
        novomodelo_value=cb,
        abs_diff=abs(nw - cb),
        rel_diff=None,
    )


def test_per_bus_sums_from_results_buckets_and_skips_fictitious() -> None:
    # Four hydro rows: ids 0,1 -> SUDESTE; id 2 -> SUL; id 3 -> NOFICT1 (dropped).
    #
    # The bus label is re-sourced from the hydro_bus_generation
    # partition's distinct (hydro_id, bus_id) pairs (merged onto hydro_meta as
    # "bus_ids" by the results-comparison orchestrator) -- read_novomodelo_hydro_metadata
    # itself no longer carries a "bus_id" key.
    results = [
        _hydro_row(0, 1, "storage_final_hm3", 100.0, 90.0),
        _hydro_row(1, 1, "storage_final_hm3", 200.0, 180.0),
        _hydro_row(2, 1, "storage_final_hm3", 50.0, 48.0),
        _hydro_row(3, 1, "storage_final_hm3", 999.0, 999.0),
        # a non-matching variable / type must be ignored.
        _hydro_row(0, 1, "generation_mw", 1.0, 1.0),
        _hydro_row(5, 1, "storage_final_hm3", 7.0, 7.0, entity_type="thermal"),
    ]
    hydro_meta: dict[int, dict[str, object]] = {
        0: {"bus_ids": {100}},
        1: {"bus_ids": {100}},
        2: {"bus_ids": {101}},
        3: {"bus_ids": {199}},
    }
    bus_meta: dict[int, dict[str, object]] = {
        100: {"name": "SUDESTE"},
        101: {"name": "SUL"},
        199: {"name": "NOFICT1"},
    }

    out = per_bus_sums_from_results(results, "storage_final_hm3", hydro_meta, bus_meta)

    assert set(out) == {"SUDESTE", "SUL"}  # NOFICT1 excluded.
    assert out["SUDESTE"]["nw"] == {1: 300.0}
    assert out["SUDESTE"]["cb"] == {1: 270.0}
    assert out["SUDESTE"]["ids"] == {0, 1}
    assert out["SUL"]["nw"] == {1: 50.0}
    assert out["SUL"]["cb"] == {1: 48.0}
    assert out["SUL"]["ids"] == {2}


def test_per_bus_sums_from_results_skips_plants_without_bus_ids() -> None:
    results = [
        _hydro_row(0, 1, "storage_final_hm3", 100.0, 90.0),
        _hydro_row(1, 1, "storage_final_hm3", 200.0, 180.0),
    ]
    # id 1 carries no "bus_ids" label at all (e.g. it never appears in the
    # hydro_bus_generation partition) -> skipped entirely, same as the legacy
    # bus_id-is-None skip.
    hydro_meta: dict[int, dict[str, object]] = {
        0: {"bus_ids": {100}},
        1: {},
    }
    bus_meta: dict[int, dict[str, object]] = {100: {"name": "SUDESTE"}}

    out = per_bus_sums_from_results(results, "storage_final_hm3", hydro_meta, bus_meta)

    assert set(out) == {"SUDESTE"}
    assert out["SUDESTE"]["ids"] == {0}
    assert out["SUDESTE"]["nw"] == {1: 100.0}


def test_per_bus_sums_from_results_excludes_multi_bus_plant() -> None:
    """A plant recorded at two buses is neither collapsed onto

    one of them nor added to both (which would double-count its aggregate
    value); it is dropped from every bus's roll-up and a Diagnostic is raised.
    """
    results = [
        _hydro_row(0, 1, "storage_final_hm3", 100.0, 90.0),  # single-bus, kept.
        _hydro_row(9, 1, "storage_final_hm3", 500.0, 480.0),  # two-bus, excluded.
    ]
    hydro_meta: dict[int, dict[str, object]] = {
        0: {"bus_ids": {100}},
        # synthetic two-bus plant; named so the fix (include the plant
        # name, not just its numeric id) is checked.
        9: {"bus_ids": {100, 101}, "name": "AMBIGUOUS_PLANT"},
    }
    bus_meta: dict[int, dict[str, object]] = {
        100: {"name": "SUDESTE"},
        101: {"name": "SUL"},
    }

    with dx.collect() as collected:
        out = per_bus_sums_from_results(
            results, "storage_final_hm3", hydro_meta, bus_meta
        )

    # Plant 9's value appears in NEITHER bus's bucket -- not collapsed onto
    # SUDESTE or SUL, and not double-counted into both.
    assert out["SUDESTE"]["ids"] == {0}
    assert out["SUDESTE"]["nw"] == {1: 100.0}
    assert "SUL" not in out

    ambiguous = [d for d in collected if d.code == "hydro-multi-bus-ambiguous"]
    assert len(ambiguous) == 1
    assert "9" in ambiguous[0].summary
    assert "AMBIGUOUS_PLANT" in ambiguous[0].summary


def test_per_bus_sums_from_frame_matches_legacy_closure() -> None:
    df = pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 2],
            "stage_id": [1, 1, 2, 1],
            "water_withdrawal_violation_pos_m3s": [5.0, 3.0, 6.0, 9.0],
        }
    )
    hydro_meta: dict[int, dict[str, object]] = {
        0: {"bus_ids": {100}},
        1: {"bus_ids": {100}},
        2: {"bus_ids": {199}},  # NOFICT -> dropped.
    }
    bus_meta: dict[int, dict[str, object]] = {
        100: {"name": "SUDESTE"},
        199: {"name": "NOFICT1"},
    }

    out = per_bus_sums_from_frame(
        df, "water_withdrawal_violation_pos_m3s", {0, 1, 2}, hydro_meta, bus_meta
    )

    assert set(out) == {"SUDESTE"}
    assert out["SUDESTE"]["sum"] == {1: 8.0, 2: 6.0}
    assert out["SUDESTE"]["ids"] == {0, 1}


def test_per_bus_sums_from_frame_excludes_multi_bus_plant() -> None:
    """Frame-sourced variant of the two-bus exclusion."""
    df = pl.DataFrame(
        {
            "entity_id": [0, 9],
            "stage_id": [1, 1],
            "water_withdrawal_violation_pos_m3s": [5.0, 50.0],
        }
    )
    hydro_meta: dict[int, dict[str, object]] = {
        0: {"bus_ids": {100}},
        9: {"bus_ids": {100, 101}},
    }
    bus_meta: dict[int, dict[str, object]] = {
        100: {"name": "SUDESTE"},
        101: {"name": "SUL"},
    }

    with dx.collect() as collected:
        out = per_bus_sums_from_frame(
            df, "water_withdrawal_violation_pos_m3s", {0, 9}, hydro_meta, bus_meta
        )

    assert out["SUDESTE"]["sum"] == {1: 5.0}
    assert out["SUDESTE"]["ids"] == {0}
    assert "SUL" not in out
    assert any(d.code == "hydro-multi-bus-ambiguous" for d in collected)


def test_per_bus_sums_from_frame_empty_or_missing_column_returns_empty() -> None:
    hydro_meta: dict[int, dict[str, object]] = {0: {"bus_ids": {100}}}
    bus_meta: dict[int, dict[str, object]] = {100: {"name": "SUDESTE"}}
    assert per_bus_sums_from_frame(None, "v", None, hydro_meta, bus_meta) == {}
    df = pl.DataFrame({"entity_id": [0], "stage_id": [1], "other": [1.0]})
    assert per_bus_sums_from_frame(df, "v", None, hydro_meta, bus_meta) == {}


def test_per_bus_sums_from_results_empty_hydro_meta_diagnoses_map_empty() -> None:
    """A non-empty hydro_meta that resolves to zero usable

    bus labels must not silently produce an empty map -- a Diagnostic names
    the failure instead.
    """
    results = [_hydro_row(0, 1, "storage_final_hm3", 100.0, 90.0)]
    # Every plant lacks a "bus_ids" label (e.g. the merge step never ran).
    hydro_meta: dict[int, dict[str, object]] = {0: {}}
    bus_meta: dict[int, dict[str, object]] = {}

    with dx.collect() as collected:
        out = per_bus_sums_from_results(
            results, "storage_final_hm3", hydro_meta, bus_meta
        )

    assert out == {}
    assert any(d.code == "hydro-to-bus-map-empty" for d in collected)


def test_per_bus_sums_from_results_dedupes_repeated_ambiguity_diagnostic() -> None:
    """Regression: report_builder.py threads the SAME hydro_meta/
    bus_meta object pair into ~11 per-bus chart calls per comparison run
    (one per chart variable); the ambiguous-plant diagnostic must fire once
    per run, not once per call -- compare has no diagnostics de-dup sink, so
    the un-memoized helper would log the identical warning ~11x."""
    results = [_hydro_row(0, 1, "storage_final_hm3", 100.0, 90.0)]
    hydro_meta: dict[int, dict[str, object]] = {
        0: {"bus_ids": {100}},
        9: {"bus_ids": {100, 101}},
    }
    bus_meta: dict[int, dict[str, object]] = {100: {"name": "SUDESTE"}}

    with dx.collect() as collected:
        for _ in range(11):
            per_bus_sums_from_results(
                results, "storage_final_hm3", hydro_meta, bus_meta
            )

    ambiguous = [d for d in collected if d.code == "hydro-multi-bus-ambiguous"]
    assert len(ambiguous) == 1

    # A genuinely different hydro_meta/bus_meta pair (a later, distinct
    # comparison run) is unaffected by the memo -- the diagnostic is
    # per-pair, not globally suppressed after its first ever emission.
    other_hydro_meta: dict[int, dict[str, object]] = {9: {"bus_ids": {100, 101}}}
    with dx.collect() as collected_other:
        per_bus_sums_from_results(
            results, "storage_final_hm3", other_hydro_meta, bus_meta
        )
    assert any(d.code == "hydro-multi-bus-ambiguous" for d in collected_other)


def test_per_bus_band_from_pct_sums_across_bus_ids() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 1],
            "stage_id": [1, 1, 2, 2],
            "generation_mw_p10": [80.0, 160.0, 85.0, 165.0],
            "generation_mw_p90": [100.0, 200.0, 105.0, 205.0],
        }
    )
    per_bus_ids = {"SUDESTE": {0, 1}}

    out = per_bus_band_from_pct(pct, "generation_mw", per_bus_ids)

    assert out["SUDESTE"][1] == (240.0, 300.0)
    assert out["SUDESTE"][2] == (250.0, 310.0)


def test_per_bus_band_from_pct_missing_column_returns_empty() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [1, 1],
            # only _p90 present -> _p10 missing.
            "generation_mw_p90": [100.0, 200.0],
        }
    )
    assert per_bus_band_from_pct(pct, "generation_mw", {"SUDESTE": {0, 1}}) == {}
    assert per_bus_band_from_pct(None, "generation_mw", {"SUDESTE": {0}}) == {}


def test_per_bus_band_from_pct_skips_empty_id_set() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0],
            "stage_id": [1],
            "v_p10": [10.0],
            "v_p90": [20.0],
        }
    )
    out = per_bus_band_from_pct(pct, "v", {"SUDESTE": set(), "SUL": {0}})
    assert "SUDESTE" not in out
    assert out["SUL"][1] == (10.0, 20.0)


def test_plant_percentile_arrays_rounds_and_defaults() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0, 0],
            "stage_id": [1, 2],
            "storage_final_hm3_p10": [80.123, 85.0],
            "storage_final_hm3_p90": [100.456, 105.0],
        }
    )
    # stage 3 is absent for this plant -> defaults to 0.0.
    out = plant_percentile_arrays(pct, [("storage_final_hm3", "Storage", [1, 2, 3])], 0)
    assert out["storage_final_hm3_p10"] == [80.12, 85.0, 0.0]
    assert out["storage_final_hm3_p90"] == [100.46, 105.0, 0.0]


def test_plant_percentile_arrays_missing_column_returns_empty() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0],
            "stage_id": [1],
            # neither storage_final_hm3 _p10 nor _p90 present.
            "other_p10": [1.0],
            "other_p90": [2.0],
        }
    )
    assert plant_percentile_arrays(pct, [("storage_final_hm3", "", [1, 2])], 0) == {}


def test_plant_percentile_arrays_unknown_plant_returns_empty() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0],
            "stage_id": [1],
            "v_p10": [1.0],
            "v_p90": [2.0],
        }
    )
    # plant 99 has no rows.
    assert plant_percentile_arrays(pct, [("v", "", [1])], 99) == {}
    assert plant_percentile_arrays(None, [("v", "", [1])], 0) == {}


def test_plant_percentile_arrays_per_variable_stage_axes() -> None:
    """Each variable aligns to its OWN stage axis from the single filtered sub."""
    pct = pl.DataFrame(
        {
            "entity_id": [0, 0],
            "stage_id": [1, 2],
            "a_p10": [10.0, 11.0],
            "a_p90": [20.0, 21.0],
            "b_p10": [30.0, 31.0],
            "b_p90": [40.0, 41.0],
        }
    )
    # Variable "a" spans stages [1, 2]; "b" spans only [2] (its own axis).
    out = plant_percentile_arrays(pct, [("a", "A", [1, 2]), ("b", "B", [2])], 0)
    assert out["a_p10"] == [10.0, 11.0]
    assert out["a_p90"] == [20.0, 21.0]
    assert out["b_p10"] == [31.0]
    assert out["b_p90"] == [41.0]


def test_plant_percentile_arrays_filters_once_per_plant() -> None:
    """The per-plant ``entity_id`` filter runs exactly once regardless of var count."""

    class _CountingFrame:
        """Minimal pl.DataFrame proxy counting ``filter`` invocations."""

        def __init__(self, inner: pl.DataFrame) -> None:
            self._inner = inner
            self.filter_calls = 0

        def is_empty(self) -> bool:
            return self._inner.is_empty()

        def filter(self, *args: object, **kwargs: object) -> pl.DataFrame:
            self.filter_calls += 1
            return self._inner.filter(*args, **kwargs)

    pct = pl.DataFrame(
        {
            "entity_id": [0, 0],
            "stage_id": [1, 2],
            "a_p10": [10.0, 11.0],
            "a_p90": [20.0, 21.0],
            "b_p10": [30.0, 31.0],
            "b_p90": [40.0, 41.0],
        }
    )
    counting = _CountingFrame(pct)

    # Three variables, yet only ONE filter call for the single plant.
    out = plant_percentile_arrays(
        cast("pl.DataFrame", counting),
        [("a", "A", [1, 2]), ("b", "B", [1, 2]), ("missing", "M", [1, 2])],
        0,
    )
    assert counting.filter_calls == 1
    assert set(out) == {"a_p10", "a_p90", "b_p10", "b_p90"}


# ---------------------------------------------------------------------------
# system/network draw-only primitives
# ---------------------------------------------------------------------------


def _bus_rc(
    name: str, novomodelo_id: int, stage: int, variable: str
) -> ResultComparison:
    return ResultComparison(
        entity_type="bus",
        entity_name=name,
        newave_code=novomodelo_id + 10,
        novomodelo_id=novomodelo_id,
        stage=stage,
        variable=variable,
        newave_value=float(stage * 10 + novomodelo_id),
        novomodelo_value=float(stage * 10 + novomodelo_id) - 1.0,
        abs_diff=1.0,
        rel_diff=None,
    )


def _spill_rc(stage: int, variable: str, nw: float) -> ResultComparison:
    return ResultComparison(
        entity_type="system_spillage",
        entity_name="SIN",
        newave_code=0,
        novomodelo_id=0,
        stage=stage,
        variable=variable,
        newave_value=nw,
        novomodelo_value=nw - 1.0,
        abs_diff=1.0,
        rel_diff=None,
    )


def test_novomodelo_sum_and_newave_sin_sums_both_sides() -> None:
    novomodelo_hydro = pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 1],
            "stage_id": [1, 1, 2, 2],
            "stored_energy_final_mwh": [900.0, 1900.0, 1000.0, 2000.0],
        }
    )
    nw_sin = pl.DataFrame(
        {
            "newave_code": [0, 0],
            "stage": [2, 3],
            "variable": ["EARMF", "EARMF"],
            "value": [3.0, 4.0],
        }
    )

    novomodelo_by_stage, nw_by_stage = novomodelo_sum_and_newave_sin(
        novomodelo_hydro, "stored_energy_final_mwh", nw_sin, "EARMF", 730.0, 1, None
    )

    assert novomodelo_by_stage == {1: 2800.0, 2: 3000.0}
    # stage 2 -> 2-1=1, stage 3 -> 3-1=2; value * 730.
    assert nw_by_stage == {1: 3.0 * 730.0, 2: 4.0 * 730.0}


def test_novomodelo_sum_and_newave_sin_applies_factor_and_offset() -> None:
    novomodelo_hydro = pl.DataFrame({"entity_id": [0], "stage_id": [5], "v": [10.0]})
    nw_sin = pl.DataFrame(
        {
            "stage": [10, 11, None, 12],
            "variable": [" earmf ", "EARMF", "EARMF", "OTHER"],
            "value": [1.0, None, 5.0, 9.0],
        }
    )

    novomodelo_by_stage, nw_by_stage = novomodelo_sum_and_newave_sin(
        novomodelo_hydro, "v", nw_sin, "EARMF", 2.0, 3, None
    )

    assert novomodelo_by_stage == {5: 10.0}
    # Row 1: " earmf " strips/uppercases to EARMF -> stage 10-3=7, 1.0*2=2.0.
    # Row 2: value None -> skipped. Row 3: stage None -> skipped.
    # Row 4: variable OTHER -> filtered out.
    assert nw_by_stage == {7: 2.0}


def test_novomodelo_sum_and_newave_sin_empty_returns_empty() -> None:
    empty = pl.DataFrame(schema={"entity_id": pl.Int64, "stage_id": pl.Int64})
    assert novomodelo_sum_and_newave_sin(empty, "v", None, None, 1.0, 0, None) == (
        {},
        {},
    )

    novomodelo_hydro = pl.DataFrame({"entity_id": [0], "stage_id": [1], "v": [5.0]})
    # Missing variable column -> ({}, {}).
    assert novomodelo_sum_and_newave_sin(
        novomodelo_hydro, "absent", None, None, 1.0, 0, None
    ) == ({}, {})
    # No nw_sin -> only Novomodelo side populated.
    cb, nw = novomodelo_sum_and_newave_sin(
        novomodelo_hydro, "v", None, "EARMF", 1.0, 0, None
    )
    assert cb == {1: 5.0}
    assert nw == {}


def test_bus_groups_and_pct_groups_by_name() -> None:
    results = [
        _bus_rc("Sudeste", 0, 1, "deficit_mw"),
        _bus_rc("sul", 1, 1, "deficit_mw"),
        _bus_rc("NORTE", 2, 1, "deficit_mw"),
        _bus_rc("Sudeste", 0, 2, "deficit_mw"),
        _bus_rc("Sudeste", 0, 1, "marginal_cost"),  # other variable, excluded
    ]
    pct = pl.DataFrame(
        {
            "entity_id": [0, 1, 2],
            "stage_id": [1, 1, 1],
            "deficit_mw_p10": [1.0, 2.0, 3.0],
            "deficit_mw_p90": [4.0, 5.0, 6.0],
        }
    )

    buses, pct_by_eid = bus_groups_and_pct(results, "deficit_mw", pct)

    assert set(buses) == {"SUDESTE", "SUL", "NORTE"}
    assert len(buses["SUDESTE"]) == 2
    assert pct_by_eid[0][1]["deficit_mw_p10"] == 1.0
    assert pct_by_eid[2][1]["deficit_mw_p90"] == 6.0


def test_bus_groups_and_pct_missing_column_empty_lookup() -> None:
    results = [_bus_rc("SUDESTE", 0, 1, "deficit_mw")]
    pct = pl.DataFrame(
        {
            "entity_id": [0],
            "stage_id": [1],
            # only _p10 present, _p90 absent -> empty lookup.
            "deficit_mw_p10": [1.0],
        }
    )

    buses, pct_by_eid = bus_groups_and_pct(results, "deficit_mw", pct)

    assert set(buses) == {"SUDESTE"}
    assert pct_by_eid == {}
    # None frame also yields empty lookup.
    _, pct_none = bus_groups_and_pct(results, "deficit_mw", None)
    assert pct_none == {}


def test_spillage_lookups_builds_nw_and_cb() -> None:
    results = [
        _spill_rc(1, "VERTOT", 100.0),
        _spill_rc(2, "VERTOT", 110.0),
        _spill_rc(1, "VERTcont", 60.0),
    ]
    novomodelo_spill_energy = pl.DataFrame(
        {
            "stage_id": [1, 2],
            "total_mw": [96.0, 106.0],
            "reservoir_mw": [58.0, 63.0],
            "rorov_mw": [38.0, 43.0],
        }
    )

    nw_lookup, cb_lookup = spillage_lookups(results, novomodelo_spill_energy)

    assert nw_lookup["VERTOT"] == {1: 100.0, 2: 110.0}
    assert nw_lookup["VERTcont"] == {1: 60.0}
    assert cb_lookup["spill_energy_total_mw"] == {1: 96.0, 2: 106.0}
    assert cb_lookup["spill_energy_reservoir_mw"] == {1: 58.0, 2: 63.0}
    assert cb_lookup["spill_energy_rorov_mw"] == {1: 38.0, 2: 43.0}


def test_spillage_lookups_empty_returns_empty() -> None:
    empty = pl.DataFrame(
        schema={
            "stage_id": pl.Int64,
            "total_mw": pl.Float64,
            "reservoir_mw": pl.Float64,
            "rorov_mw": pl.Float64,
        }
    )
    nw_lookup, cb_lookup = spillage_lookups([], empty)
    assert nw_lookup == {}
    assert cb_lookup == {}


# -------------------------------------------------------------------
# compare verdict builder
# -------------------------------------------------------------------


def _summary_dataset(rows: list[dict[str, object]]) -> ComparisonDataset:
    """A dataset with an empty tidy frame and a hand-built summary frame.

    The summary frame conforms to ``SUMMARY_SCHEMA``; ``tidy`` is empty so the
    dataset validates while exercising only the summary-driven verdict path.
    """
    summary = pl.DataFrame(rows, schema=SUMMARY_SCHEMA)
    dataset = ComparisonDataset(
        tidy=pl.DataFrame(schema=TIDY_SCHEMA),
        summary=summary,
    )
    dataset.validate()
    return dataset


class TestCompareVerdict:
    """Tests for ``build_compare_verdict`` reading only ``dataset.summary``."""

    def test_verdict_empty_dataset_returns_zeroed_verdict(self) -> None:
        dataset = build_results_dataset([], PercentileData(), 1e-2)

        verdict = build_compare_verdict(dataset)

        assert verdict == CompareVerdict(
            within_tol=0,
            total=0,
            worst_variable=None,
            worst_smape=0.0,
            all_within_tol=False,
        )

    def test_verdict_all_within_tol_true_when_every_variable_perfect(self) -> None:
        dataset = _summary_dataset(
            [
                {
                    "variable": "storage_max",
                    "count": 3,
                    "mean_abs_diff": 0.0,
                    "max_abs_diff": 0.0,
                    "mean_smape": 0.0,
                    "max_smape": 0.0,
                    "within_tol_rate": 1.0,
                    "correlation": None,
                },
                {
                    "variable": "generation_max",
                    "count": 2,
                    "mean_abs_diff": 0.0,
                    "max_abs_diff": 0.0,
                    "mean_smape": 0.01,
                    "max_smape": 0.02,
                    "within_tol_rate": 1.0,
                    "correlation": None,
                },
            ]
        )

        verdict = build_compare_verdict(dataset)

        assert verdict.within_tol == 2
        assert verdict.total == 2
        assert verdict.all_within_tol is True

    def test_verdict_tie_break_picks_lexicographically_first_variable(self) -> None:
        dataset = _summary_dataset(
            [
                {
                    "variable": "b_var",
                    "count": 1,
                    "mean_abs_diff": 1.0,
                    "max_abs_diff": 1.0,
                    "mean_smape": 0.1,
                    "max_smape": 0.5,
                    "within_tol_rate": 0.0,
                    "correlation": None,
                },
                {
                    "variable": "a_var",
                    "count": 1,
                    "mean_abs_diff": 1.0,
                    "max_abs_diff": 1.0,
                    "mean_smape": 0.1,
                    "max_smape": 0.5,
                    "within_tol_rate": 0.0,
                    "correlation": None,
                },
            ]
        )

        verdict = build_compare_verdict(dataset)

        assert verdict.worst_variable == "a_var"
        assert verdict.worst_smape == 0.5


# ---------------------------------------------------------------------------
# plant_max_reldiff_ranking (charts._shared._plant_max_reldiff_table)
# ---------------------------------------------------------------------------


def _reldiff_result(
    name: str,
    code: int,
    novomodelo_id: int,
    stage: int,
    var: str,
    rel_diff: float | None,
    entity_type: str = "hydro",
) -> ResultComparison:
    return ResultComparison(
        entity_type=entity_type,
        entity_name=name,
        newave_code=code,
        novomodelo_id=novomodelo_id,
        stage=stage,
        variable=var,
        newave_value=100.0,
        novomodelo_value=100.0 * (1.0 + (rel_diff or 0.0)),
        abs_diff=100.0 * (rel_diff or 0.0),
        rel_diff=rel_diff,
    )


def test_plant_max_reldiff_ranking_orders_worst_first_and_computes_median() -> None:
    results = [
        _reldiff_result("ITAIPU", 10, 0, 0, "gen", 0.03),
        # Same (plant, variable), later stage: the larger rel_diff must win.
        _reldiff_result("ITAIPU", 10, 0, 1, "gen", 0.05),
        _reldiff_result("ITAIPU", 10, 0, 0, "vol", 0.20),
        _reldiff_result("TUCURUI", 20, 1, 0, "gen", 0.10),
        # Unmatched comparison (rel_diff is None) -> excluded from the ranking
        # and from the median, mirroring the "-" cell it renders as.
        _reldiff_result("TUCURUI", 20, 1, 0, "vol", None),
        # Different entity_type -> filtered out entirely.
        ResultComparison(
            entity_type="thermal",
            entity_name="ANGRA",
            newave_code=30,
            novomodelo_id=2,
            stage=0,
            variable="gen",
            newave_value=10.0,
            novomodelo_value=99.0,
            abs_diff=89.0,
            rel_diff=8.9,
        ),
    ]
    variables = [("gen", "Generation"), ("vol", "Volume")]

    max_rd, plant_keys, medians = plant_max_reldiff_ranking(results, "hydro", variables)

    assert max_rd == {
        ("ITAIPU", 10, "gen"): 0.05,
        ("ITAIPU", 10, "vol"): 0.20,
        ("TUCURUI", 20, "gen"): 0.10,
    }
    # ITAIPU's worst cell (0.20) beats TUCURUI's (0.10) -> ITAIPU sorts first.
    assert plant_keys == [("ITAIPU", 10), ("TUCURUI", 20)]
    assert medians["gen"] == pytest.approx(0.075)
    assert medians["vol"] == pytest.approx(0.20)


def test_plant_max_reldiff_ranking_no_matching_rows_returns_none_medians() -> None:
    results = [_reldiff_result("ANGRA", 30, 2, 0, "gen", 0.1, entity_type="thermal")]

    max_rd, plant_keys, medians = plant_max_reldiff_ranking(
        results, "hydro", [("gen", "Generation")]
    )

    assert max_rd == {}
    assert plant_keys == []
    # statistics.median on the empty column -> None (the render layer's "-" cell).
    assert medians == {"gen": None}


# ---------------------------------------------------------------------------
# productivity_scatter_errors (charts.productivity)
# ---------------------------------------------------------------------------


def test_productivity_scatter_errors_computes_mean_and_max_relative_error() -> None:
    nw_vals = [100.0, 50.0, 200.0]
    cb_vals = [110.0, 45.0, 200.0]  # rel errs: 0.1, 0.1, 0.0

    mean_rel, max_rel = productivity_scatter_errors(nw_vals, cb_vals)

    assert mean_rel == pytest.approx(0.2 / 3)
    assert max_rel == pytest.approx(0.1)


def test_productivity_scatter_errors_excludes_reference_below_guard() -> None:
    # The first pair's reference is below the 1e-12 guard and is dropped;
    # only the second pair (rel err 0.1) contributes.
    nw_vals = [1e-13, 100.0]
    cb_vals = [5.0, 90.0]

    mean_rel, max_rel = productivity_scatter_errors(nw_vals, cb_vals)

    assert mean_rel == pytest.approx(0.1)
    assert max_rel == pytest.approx(0.1)


def test_productivity_scatter_errors_empty_input_returns_zero_zero() -> None:
    assert productivity_scatter_errors([], []) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# fpha_metric_summary (charts.fpha.fpha_metrics_table)
# ---------------------------------------------------------------------------


def test_fpha_metric_summary_aggregates_per_plant_and_sorts_worst_first() -> None:
    metrics = pl.DataFrame(
        {
            "novomodelo_id": [0, 0, 1],
            "plant_name": ["ITAIPU", "ITAIPU", "TUCURUI"],
            "stage": [0, 1, 0],
            "n_planes_newave": [3, 5, 2],
            "n_planes_novomodelo": [4, 6, 2],
            "n_v": [1, 1, 3],
            "nmae": [0.02, 0.04, 0.10],
            "bias": [0.01, -0.01, 0.05],
            "max_abs_dev": [1.0, 2.0, 3.0],
            "gh_max_ratio": [1.01, 0.98, 1.10],
        }
    )

    agg = fpha_metric_summary(metrics)
    rows = {int(r["novomodelo_id"]): r for r in agg.iter_rows(named=True)}

    itaipu = rows[0]
    assert itaipu["plant_name"] == "ITAIPU"
    assert itaipu["planes_nw"] == 5
    assert itaipu["planes_cb"] == 6
    assert itaipu["mean_nmae"] == pytest.approx(3.0)
    assert itaipu["worst_nmae"] == pytest.approx(4.0)
    assert itaipu["mean_bias"] == pytest.approx(0.0)
    assert itaipu["ghr_min"] == pytest.approx(0.98)
    assert itaipu["ghr_max"] == pytest.approx(1.01)

    tucurui = rows[1]
    assert tucurui["mean_nmae"] == pytest.approx(10.0)
    assert tucurui["worst_nmae"] == pytest.approx(10.0)

    # Sorted worst-NMAE first: TUCURUI (10%) before ITAIPU (4%).
    assert list(agg["novomodelo_id"]) == [1, 0]


# ---------------------------------------------------------------------------
# cost_percent_deltas (charts.costs.cost_breakdown_table)
# ---------------------------------------------------------------------------


def test_cost_percent_deltas_computes_rows_and_totals_sorted_by_abs_diff() -> None:
    categories = [
        ("Thermal", 100.0, 110.0, "#111"),
        ("Deficit", 0.0, 5.0, "#222"),
        ("Exchange", 50.0, 45.0, "#333"),
    ]

    rows, total_nw, total_cb, total_diff, total_pct = cost_percent_deltas(categories)

    assert rows == [
        ("Thermal", 100.0, 110.0, 10.0, pytest.approx(10.0), "#111"),
        # nw_v == 0.0 is below the 0.01 guard -> pct is None, not a ZeroDivisionError.
        ("Deficit", 0.0, 5.0, 5.0, None, "#222"),
        ("Exchange", 50.0, 45.0, -5.0, pytest.approx(-10.0), "#333"),
    ]
    assert total_nw == pytest.approx(150.0)
    assert total_cb == pytest.approx(160.0)
    assert total_diff == pytest.approx(10.0)
    assert total_pct == pytest.approx(10.0 / 150.0 * 100.0)


def test_cost_percent_deltas_zero_total_denominator_yields_none_pct() -> None:
    categories = [("Only", 0.0, 20.0, "#000")]

    rows, total_nw, total_cb, total_diff, total_pct = cost_percent_deltas(categories)

    assert rows == [("Only", 0.0, 20.0, 20.0, None, "#000")]
    assert total_nw == 0.0
    assert total_cb == pytest.approx(20.0)
    assert total_diff == pytest.approx(20.0)
    assert total_pct is None

"""Tests for the canonical :class:`ComparisonDataset` tidy-frame schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from novomodelo_bridge.comparators.dataset import (
    SUMMARY_SCHEMA,
    TIDY_SCHEMA,
    VALID_SOURCES,
    ComparisonDataset,
    RenderInputs,
    SchemaError,
    _metadata_from_json,
    _metadata_to_json,
)
from novomodelo_bridge.comparators.model import ResultComparison
from tests.golden_utils import _strip_chart_id

if TYPE_CHECKING:
    from pathlib import Path


def _one_row_tidy(source: str) -> pl.DataFrame:
    """Build a single-row tidy frame matching ``TIDY_SCHEMA`` with given source."""
    return pl.DataFrame(
        {
            "entity_type": ["hydro"],
            "entity_id": [0],
            "entity_name": ["ITAIPU"],
            "bus": [0],
            "stage": [0],
            "block": [0],
            "variable": ["generation_mw"],
            "source": [source],
            "value": [123.0],
        },
        schema=TIDY_SCHEMA,
    )


def test_empty_dataset_validates() -> None:
    """A fresh empty dataset passes validation without raising."""
    ComparisonDataset.empty().validate()


def test_validate_rejects_unknown_source() -> None:
    """A tidy row with a source outside VALID_SOURCES is rejected by name."""
    dataset = ComparisonDataset(
        tidy=_one_row_tidy("x"),
        summary=pl.DataFrame(schema=SUMMARY_SCHEMA),
    )
    with pytest.raises(SchemaError, match="x"):
        dataset.validate()


def test_validate_rejects_missing_column() -> None:
    """A tidy frame missing the ``block`` column is rejected, naming ``block``."""
    tidy = _one_row_tidy("newave").drop("block")
    dataset = ComparisonDataset(
        tidy=tidy,
        summary=pl.DataFrame(schema=SUMMARY_SCHEMA),
    )
    with pytest.raises(SchemaError, match="block"):
        dataset.validate()


def test_tidy_schema_columns_order() -> None:
    """The tidy schema declares the documented 9-column order exactly."""
    assert list(TIDY_SCHEMA) == [
        "entity_type",
        "entity_id",
        "entity_name",
        "bus",
        "stage",
        "block",
        "variable",
        "source",
        "value",
    ]


def test_valid_sources_membership() -> None:
    """VALID_SOURCES holds exactly the documented literal source labels."""
    assert VALID_SOURCES == frozenset({"newave", "novomodelo", "p10", "p50", "p90"})


def test_empty_dataset_has_typed_frames() -> None:
    """``empty()`` builds frames typed per the declared schemas."""
    dataset = ComparisonDataset.empty()
    assert dataset.tidy.height == 0
    assert dataset.summary.height == 0
    assert list(dataset.tidy.columns) == list(TIDY_SCHEMA)
    assert list(dataset.summary.columns) == list(SUMMARY_SCHEMA)
    assert dataset.metadata == {}


def test_validate_rejects_bad_summary_columns() -> None:
    """A summary frame with the wrong columns is rejected by validation."""
    dataset = ComparisonDataset(
        tidy=pl.DataFrame(schema=TIDY_SCHEMA),
        summary=pl.DataFrame({"variable": ["generation_mw"]}),
    )
    with pytest.raises(SchemaError, match="summary frame columns"):
        dataset.validate()


def _small_summary() -> pl.DataFrame:
    """Build a single-row summary frame matching ``SUMMARY_SCHEMA``."""
    return pl.DataFrame(
        {
            "variable": ["generation_mw"],
            "count": [1],
            "mean_abs_diff": [0.5],
            "max_abs_diff": [0.5],
            "mean_smape": [0.1],
            "max_smape": [0.1],
            "within_tol_rate": [1.0],
            "correlation": [0.99],
        },
        schema=SUMMARY_SCHEMA,
    )


def test_roundtrip_tidy_and_summary_equal(tmp_path: Path) -> None:
    """A round-tripped dataset preserves the tidy and summary frames exactly."""
    dataset = ComparisonDataset(
        tidy=_one_row_tidy("newave"),
        summary=_small_summary(),
    )
    paths = dataset.to_dir(tmp_path)
    reloaded = ComparisonDataset.from_dir(paths[0].parent)
    assert_frame_equal(reloaded.tidy, dataset.tidy)
    assert_frame_equal(reloaded.summary, dataset.summary)


def test_roundtrip_polars_metadata_frame(tmp_path: Path) -> None:
    """A ``pl.DataFrame`` stored in metadata survives the round-trip."""
    frame = pl.DataFrame({"line_id": [0, 1], "limit": [100.0, 200.0]})
    dataset = ComparisonDataset(
        tidy=_one_row_tidy("novomodelo"),
        summary=_small_summary(),
        metadata={"summary_counts_frame": frame},
    )
    reloaded = ComparisonDataset.from_dir(dataset.to_dir(tmp_path)[0].parent)
    value = reloaded.metadata["summary_counts_frame"]
    assert isinstance(value, pl.DataFrame)
    assert_frame_equal(value, frame)


def test_roundtrip_int_keyed_metadata_coerces_to_string_keys(tmp_path: Path) -> None:
    """An int-keyed metadata dict round-trips with string keys per JSON semantics."""
    dataset = ComparisonDataset(
        tidy=_one_row_tidy("newave"),
        summary=_small_summary(),
        metadata={"names": {0: "ITAIPU"}},
    )
    reloaded = ComparisonDataset.from_dir(dataset.to_dir(tmp_path)[0].parent)
    assert reloaded.metadata["names"] == {"0": "ITAIPU"}


def test_to_dir_writes_three_files(tmp_path: Path) -> None:
    """``to_dir`` writes all three artifact files to disk."""
    dataset = ComparisonDataset(
        tidy=_one_row_tidy("newave"),
        summary=_small_summary(),
    )
    dataset.to_dir(tmp_path)
    assert (tmp_path / "comparison.parquet").exists()
    assert (tmp_path / "summary.parquet").exists()
    assert (tmp_path / "metadata.json").exists()


def test_from_dir_missing_file_raises(tmp_path: Path) -> None:
    """``from_dir`` raises ``FileNotFoundError`` naming the missing summary file."""
    dataset = ComparisonDataset(
        tidy=_one_row_tidy("newave"),
        summary=_small_summary(),
    )
    dataset.to_dir(tmp_path)
    (tmp_path / "summary.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="summary.parquet"):
        ComparisonDataset.from_dir(tmp_path)


def test_non_serializable_metadata_raises(tmp_path: Path) -> None:
    """A non-JSON-native, non-frame metadata value raises ``TypeError`` by key."""
    dataset = ComparisonDataset(
        tidy=_one_row_tidy("newave"),
        summary=_small_summary(),
        metadata={"f": object()},
    )
    with pytest.raises(TypeError, match="f"):
        dataset.to_dir(tmp_path)


def test_frozenset_nested_in_metadata_value_raises(tmp_path: Path) -> None:
    """A frozenset nested inside a metadata dict value raises ``TypeError``.

    Guards why ``compare_results``/``_merge_hydro_bus_ids`` store the hydro
    ``bus_ids`` label as a sorted list, not the reader's frozenset — the latter
    crashed compare-artifact export.
    """
    dataset = ComparisonDataset(
        tidy=_one_row_tidy("newave"),
        summary=_small_summary(),
        metadata={"novomodelo_hydro_meta": {0: {"bus_ids": frozenset({7})}}},
    )
    with pytest.raises(TypeError, match="novomodelo_hydro_meta"):
        dataset.to_dir(tmp_path)


def test_roundtrip_empty_polars_metadata_frame_preserves_schema() -> None:
    """An empty polars metadata frame round-trips with its columns and dtypes.

    Regression for FINDING 1: an empty frame's ``records`` is ``[]``, which on
    naive reconstruction yields a zero-column frame; the ``columns`` schema map
    must restore the original columns and dtypes.
    """
    empty = pl.DataFrame(schema={"line_id": pl.Int64, "limit": pl.Float64})
    view = _metadata_to_json({"bound_table": empty})
    reloaded = _metadata_from_json(view)
    value = reloaded["bound_table"]
    assert isinstance(value, pl.DataFrame)
    assert_frame_equal(value, empty)


def test_metadata_from_json_frame_missing_records_raises() -> None:
    """A frame wrapper without a ``records`` key raises ``TypeError`` by key.

    Regression for FINDING 2: a malformed/truncated metadata.json must raise the
    documented ``TypeError`` instead of a bare ``KeyError``.
    """
    view: dict[str, object] = {
        "line_bounds": {"__frame__": "polars", "columns": {"id": "Int64"}}
    }
    with pytest.raises(TypeError, match="records"):
        _metadata_from_json(view)


def test_metadata_from_json_unknown_frame_kind_raises() -> None:
    """A frame wrapper with an unknown ``kind`` raises ``TypeError`` by key."""
    view: dict[str, object] = {"line_bounds": {"__frame__": "duckdb", "records": []}}
    with pytest.raises(TypeError, match="line_bounds"):
        _metadata_from_json(view)


def test_validate_rejects_wrong_tidy_dtype() -> None:
    """A tidy frame with ``value`` typed Int64 is rejected, naming the column.

    Regression for FINDING 4: column names/order match but the dtype differs.
    """
    tidy = _one_row_tidy("newave").with_columns(pl.col("value").cast(pl.Int64))
    dataset = ComparisonDataset(tidy=tidy, summary=_small_summary())
    with pytest.raises(SchemaError, match="value"):
        dataset.validate()


def test_validate_rejects_wrong_summary_dtype() -> None:
    """A summary frame with ``count`` typed Float64 is rejected by validation."""
    summary = _small_summary().with_columns(pl.col("count").cast(pl.Float64))
    dataset = ComparisonDataset(tidy=_one_row_tidy("newave"), summary=summary)
    with pytest.raises(SchemaError, match="count"):
        dataset.validate()


# -------------------------------------------------------------------
# RenderInputs round-trip
# -------------------------------------------------------------------


def _make_result_comparison() -> ResultComparison:
    return ResultComparison(
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
    )


def _populated_render() -> RenderInputs:
    """A ``RenderInputs`` exercising every field kind: polars frames,
    int-keyed and str-keyed dicts, a list of dicts, scalars, and the
    ``results`` object-graph list."""
    return RenderInputs(
        results=[_make_result_comparison()],
        nw_costs={"deficit": 1.0},
        novomodelo_costs={"deficit": 2.0},
        nw_bus_names={0: "SUDESTE"},
        novomodelo_bus_meta={0: {"name": "SUDESTE"}},
        novomodelo_hydro_meta={0: {"name": "ITAIPU", "bus_ids": [0]}},
        bus=pl.DataFrame({"entity_id": [0], "stage_id": [0]}),
        line_bounds=pl.DataFrame({"line_id": [0], "max_flow_mw": [100.0]}),
        gc_constraints=[{"id": 1, "name": "c1"}],
        nw_offset=1,
        nw_max_stage=5,
        novomodelo_training_seconds=2.5,
    )


def test_roundtrip_render_frame_fields_preserve_type_and_values(
    tmp_path: Path,
) -> None:
    """Polars render fields (including ``line_bounds``) keep their type/values."""
    dataset = ComparisonDataset(
        tidy=_one_row_tidy("newave"),
        summary=_small_summary(),
        render=_populated_render(),
    )
    reloaded = ComparisonDataset.from_dir(dataset.to_dir(tmp_path)[0].parent)

    assert isinstance(reloaded.render.bus, pl.DataFrame)
    assert_frame_equal(reloaded.render.bus, dataset.render.bus)
    assert isinstance(reloaded.render.line_bounds, pl.DataFrame)
    assert_frame_equal(reloaded.render.line_bounds, dataset.render.line_bounds)


def test_roundtrip_render_int_keyed_dicts_restore_int_keys(tmp_path: Path) -> None:
    """``nw_bus_names``/``novomodelo_bus_meta``/``novomodelo_hydro_meta`` keep ``int`` keys.

    Unlike a generic ``metadata`` dict (whose non-string-key coercion on
    round-trip is accepted, see
    ``test_roundtrip_int_keyed_metadata_coerces_to_string_keys``), these three
    render fields are looked up by ``int`` id downstream
    (``report_builder``/``analyze._bus_name_lookups``) — a stringified key
    there is a silent lookup miss, so :func:`_render_from_json` restores them.
    """
    dataset = ComparisonDataset(
        tidy=_one_row_tidy("newave"),
        summary=_small_summary(),
        render=_populated_render(),
    )
    reloaded = ComparisonDataset.from_dir(dataset.to_dir(tmp_path)[0].parent)

    assert reloaded.render.nw_bus_names == {0: "SUDESTE"}
    assert reloaded.render.novomodelo_bus_meta == {0: {"name": "SUDESTE"}}
    assert reloaded.render.novomodelo_hydro_meta == {
        0: {"name": "ITAIPU", "bus_ids": [0]}
    }


def test_roundtrip_render_results_list_reconstructs_result_comparisons(
    tmp_path: Path,
) -> None:
    """``results`` round-trips as a ``list[ResultComparison]``, not raw dicts."""
    dataset = ComparisonDataset(
        tidy=_one_row_tidy("newave"),
        summary=_small_summary(),
        render=_populated_render(),
    )
    reloaded = ComparisonDataset.from_dir(dataset.to_dir(tmp_path)[0].parent)

    assert reloaded.render.results == [_make_result_comparison()]
    assert all(isinstance(r, ResultComparison) for r in reloaded.render.results)


def test_roundtrip_render_scalars_and_list_fields(tmp_path: Path) -> None:
    """The remaining scalar / list-of-dict render fields survive verbatim."""
    dataset = ComparisonDataset(
        tidy=_one_row_tidy("newave"),
        summary=_small_summary(),
        render=_populated_render(),
    )
    reloaded = ComparisonDataset.from_dir(dataset.to_dir(tmp_path)[0].parent)

    assert reloaded.render.nw_costs == {"deficit": 1.0}
    assert reloaded.render.novomodelo_costs == {"deficit": 2.0}
    assert reloaded.render.gc_constraints == [{"id": 1, "name": "c1"}]
    assert reloaded.render.nw_offset == 1
    assert reloaded.render.nw_max_stage == 5
    assert reloaded.render.novomodelo_training_seconds == 2.5


def test_render_from_json_malformed_result_record_raises() -> None:
    """A ``results`` record missing a ``ResultComparison`` field raises by key."""
    from novomodelo_bridge.comparators.dataset import _render_from_json

    view = {"results": [{"entity_type": "hydro"}]}
    with pytest.raises(TypeError, match="results"):
        _render_from_json(view)


def test_empty_render_reproduces_report_byte_identically(tmp_path: Path) -> None:
    """``to_dir`` -> ``from_dir`` reproduces an empty-render report exactly.

    Compared via :func:`_strip_chart_id` (the repo-wide golden-comparison
    normalizer) because each chart div gets a fresh random id per
    ``build_comparison_report`` call, regardless of round-tripping — see
    ``test_chart_helpers``'s own golden tests for the same normalization.
    """
    from novomodelo_bridge.comparators.report_builder import build_comparison_report

    dataset = ComparisonDataset(tidy=_one_row_tidy("newave"), summary=_small_summary())
    html_before = build_comparison_report(dataset)

    reloaded = ComparisonDataset.from_dir(dataset.to_dir(tmp_path)[0].parent)
    html_after = build_comparison_report(reloaded)

    assert _strip_chart_id(html_after) == _strip_chart_id(html_before)


def test_render_round_trip_reproduces_report_byte_identically(tmp_path: Path) -> None:
    """``to_dir`` -> ``from_dir`` -> ``build_comparison_report`` reproduces the
    in-memory report byte-identically: a reloaded
    dataset's render inputs are not degraded relative to the in-memory ones.
    Compared via :func:`_strip_chart_id` for the same reason as
    :func:`test_empty_render_reproduces_report_byte_identically`.
    """
    from novomodelo_bridge.comparators.report_builder import build_comparison_report

    dataset = ComparisonDataset(
        tidy=_one_row_tidy("newave"),
        summary=_small_summary(),
        metadata={"footer_counts": {"total": 1, "by_entity_type": {"hydro": 1}}},
        render=_populated_render(),
    )
    html_before = build_comparison_report(dataset)

    reloaded = ComparisonDataset.from_dir(dataset.to_dir(tmp_path)[0].parent)
    html_after = build_comparison_report(reloaded)

    assert _strip_chart_id(html_after) == _strip_chart_id(html_before)

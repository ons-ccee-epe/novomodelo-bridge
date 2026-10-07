"""Golden-file regression tests pinning the canonical dataset numbers.

These tests snapshot the canonical ``ComparisonDataset`` numbers for the
``compare results`` subcommand, built from synthetic in-memory fixtures (NO
real the source model case, NO ``inewave`` I/O). They pin:

- the canonical tidy frame (``dataset.tidy``),
- the per-variable summary frame (``dataset.summary``),
- the on-disk ``summary.json`` written by :func:`export.write_artifacts`.

Comparison is EXACT: frames via ``polars.testing.assert_frame_equal(...,
check_exact=True, check_dtypes=True)`` and the JSON records via plain ``==``.
This is correct because the math is deterministic and the JSON float round-trip
was verified to preserve the float ``repr`` byte-for-byte.

Regeneration recipe (only when an intentional, reviewed output change is made)::

    NOVOMODELO_BRIDGE_UPDATE_GOLDENS=1 .venv/bin/pytest tests/comparators/test_golden_dataset.py

When ``NOVOMODELO_BRIDGE_UPDATE_GOLDENS=1`` the tests WRITE the golden files and pass;
otherwise they READ the goldens and assert equality. Goldens are NEVER silently
overwritten on mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from novomodelo_bridge.comparators import export
from novomodelo_bridge.comparators.analyze import build_results_dataset
from novomodelo_bridge.comparators.model import PercentileData, ResultComparison
from tests.golden_utils import assert_frame_golden, assert_json_golden


def _make_results() -> list[ResultComparison]:
    """Three result comparisons spanning two entity types."""
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


def _make_percentiles() -> PercentileData:
    """One hydro percentile frame over two entities at stage 0."""
    return PercentileData(
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


_TIDY_SORT = ["entity_type", "entity_id", "stage", "variable", "source"]


def test_golden_results_tidy() -> None:
    dataset = build_results_dataset(_make_results(), _make_percentiles(), 1e-2)

    actual = dataset.tidy.sort(_TIDY_SORT)

    assert_frame_golden(actual, "dataset_results_tidy.json")


def test_golden_results_summary() -> None:
    dataset = build_results_dataset(_make_results(), _make_percentiles(), 1e-2)

    actual = dataset.summary.sort(["variable"])

    assert_frame_golden(actual, "dataset_results_summary.json")


def test_golden_results_summary_json_on_disk(tmp_path: Path) -> None:
    dataset = build_results_dataset(_make_results(), _make_percentiles(), 1e-2)

    export.write_artifacts(
        dataset,
        command="compare newave",
        source_dir=Path("/fake/nw"),
        novomodelo_output_dir=Path("/fake/novomodelo"),
        tolerance=1e-2,
        out_dir=tmp_path,
        formats=["json"],
    )
    records = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert_json_golden(records, "dataset_results_summary_json.json")

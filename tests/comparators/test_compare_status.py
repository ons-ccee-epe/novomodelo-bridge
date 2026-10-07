"""Unit tests for the shared ``compare_status`` --json status vocabulary."""

from __future__ import annotations

import polars as pl

from novomodelo_bridge.comparators.analyze import build_results_dataset
from novomodelo_bridge.comparators.model import PercentileData, ResultComparison
from novomodelo_bridge.comparators.verdict import compare_status


def test_compare_status_empty_dataset_returns_no_comparable_rows() -> None:
    dataset = build_results_dataset([], PercentileData(), 1e-2)

    assert dataset.tidy.is_empty()
    assert compare_status(dataset) == "no-comparable-rows"


def test_compare_status_empty_paired_results_with_percentiles_returns_no_comparable_rows() -> (
    None
):
    """Zero paired comparisons but non-empty percentiles: ``tidy`` is non-empty
    (percentile rows) while ``summary`` is empty (no variable had a paired
    comparison) — the status must still be ``"no-comparable-rows"``, not
    ``"mismatch"`` from a stray ``total == 0`` reading of an empty summary."""
    pct = PercentileData(
        hydro=pl.DataFrame(
            {
                "entity_id": [0],
                "stage_id": [0],
                "generation_mw_p10": [90.0],
                "generation_mw_p50": [100.0],
                "generation_mw_p90": [110.0],
            }
        )
    )
    dataset = build_results_dataset([], pct, 1e-2)

    assert not dataset.tidy.is_empty()
    assert dataset.summary.is_empty()
    assert compare_status(dataset) == "no-comparable-rows"


def test_compare_status_all_within_tol_returns_ok() -> None:
    results = [
        ResultComparison(
            entity_type="hydro",
            entity_name="ITAIPU",
            newave_code=10,
            novomodelo_id=0,
            stage=0,
            variable="generation_mw",
            newave_value=100.0,
            novomodelo_value=100.0,
            abs_diff=0.0,
            rel_diff=0.0,
        ),
    ]
    dataset = build_results_dataset(results, PercentileData(), 1e-2)

    assert compare_status(dataset) == "ok"


def test_compare_status_any_mismatch_returns_mismatch() -> None:
    results = [
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
    ]
    dataset = build_results_dataset(results, PercentileData(), 1e-2)

    assert compare_status(dataset) == "mismatch"

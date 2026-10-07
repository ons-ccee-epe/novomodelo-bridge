"""Console==dataset parity regression tests.

The compare-data-layer migration's central invariant is single-source-of-truth:
every number on the console must trace back to ONE ``ComparisonDataset`` analysis
(``dataset.summary`` rows / ``dataset.metadata`` counts), never to a second,
independently computed path. The historical bug this guards against was a
two-path divergence where console and HTML aggregated separately and drifted.

This module renders the dataset-driven console printers, parses the numbers back
out of the rendered text, and asserts those parsed cells equal the cells produced
by formatting the matching ``dataset.summary`` / ``dataset.metadata`` values with
the printer's OWN helpers (``report._fmt_metric`` and the percent/correlation
format strings). The ONLY computation the test performs is parsing printed text:
every expected cell is derived FROM the dataset, never recomputed from raw
``ResultComparison`` rows. If a future edit makes a printer recompute a
statistic instead of reading the dataset, the parsed cells diverge from the
dataset-derived expected cells and the test fails.

The summary printer now renders a Rich table, so its single-source invariant is
enforced solely by the parse-vs-dataset tests below.

Fixtures are hermetic synthetic in-memory data (NO real the source model case, NO
``inewave`` I/O), copied from ``tests/comparators/test_analyze.py`` and
``tests/comparators/test_golden_dataset.py``.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import polars as pl

from novomodelo_bridge.comparators.analyze import build_results_dataset
from novomodelo_bridge.comparators.model import (
    PercentileData,
    ResultComparison,
)
from novomodelo_bridge.comparators.verdict import build_compare_verdict
from novomodelo_bridge.ui.compare_summary import (
    _fmt_metric,
    print_results_summary_from_dataset,
)

_NW = Path("/fake/nw")
_NOVOMODELO = Path("/fake/novomodelo")
_RESULTS_TOL = 1e-2

# Number of value columns in the per-variable results table, after the variable
# name: Count, Mean|D|, Max|D|, WithinTol, sMAPE, r.
_RESULTS_VALUE_COLS = 6


# ---------------------------------------------------------------------------
# Synthetic fixtures (copied verbatim from tests/comparators/test_analyze.py)
# ---------------------------------------------------------------------------


def _make_single_point_results() -> list[ResultComparison]:
    """One comparison for a variable that appears exactly once.

    A single-data-point variable group has ``len(nw_vals) <= 1``, so
    ``build_results_summary`` leaves ``correlation`` at its ``None`` default
    (Pearson is undefined for fewer than two points). This drives the dataset
    summary's ``correlation`` cell to ``None`` and exercises the printer's
    ``"N/A"`` rendering branch.
    """
    return [
        ResultComparison(
            entity_type="hydro",
            entity_name="ITAIPU",
            newave_code=10,
            novomodelo_id=0,
            stage=0,
            variable="lonely_var",
            newave_value=100.0,
            novomodelo_value=110.0,
            abs_diff=10.0,
            rel_diff=0.1,
        ),
    ]


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


def _one_hydro_pct() -> PercentileData:
    """The one-hydro-frame percentile data."""
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


def _capture(func: object, *args: object, **kwargs: object) -> str:
    """Run a stdout-writing printer and return the captured text."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args, **kwargs)  # type: ignore[operator]
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Parsers — the ONLY computation the test does is parsing printed text.
# ---------------------------------------------------------------------------


def _parse_results_table(text: str) -> dict[str, list[str]]:
    """Parse the per-variable results table into ``{variable: [cells]}``.

    Skips the title block, the ``=`` / ``-`` rules, the column header, the blank
    lines, and the ``Summary:`` footer. Each surviving data row is split on runs
    of whitespace (the fixtures' variable names are space-free); the first token
    is the variable and the remaining tokens are the value cells. A clear
    ``AssertionError`` is raised if a data row's token count is unexpected.
    """
    table: dict[str, list[str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("✓ ", "⚠ ")):  # leading compare verdict line
            continue
        if line.startswith(("Novomodelo vs", "NEWAVE case:", "Novomodelo output:")):
            continue
        if line.startswith("Summary:"):
            continue
        if set(line) <= {"=", "-", "━", "─"}:  # text underline or Rich table rule
            continue
        tokens = line.split()
        if tokens[0] == "Variable":
            continue
        assert len(tokens) == 1 + _RESULTS_VALUE_COLS, (
            f"unexpected results data row token count {len(tokens)} "
            f"(expected {1 + _RESULTS_VALUE_COLS}) in line: {raw!r}"
        )
        table[tokens[0]] = tokens[1:]
    return table


def _parse_results_footer(text: str) -> tuple[int, dict[str, int]]:
    """Parse the ``Summary: N comparisons across K entity types (...)`` footer.

    Returns the total comparison count and the per-entity-type count map parsed
    from the parenthetical ``(c0 t0, c1 t1, ...)`` list.
    """
    footer = next(
        line for line in text.splitlines() if line.strip().startswith("Summary:")
    ).strip()
    after = footer[len("Summary:") :].strip()
    total = int(after.split(maxsplit=1)[0])

    open_paren = footer.index("(")
    close_paren = footer.rindex(")")
    inner = footer[open_paren + 1 : close_paren]

    by_entity_type: dict[str, int] = {}
    for part in inner.split(","):
        count_str, etype = part.split()
        by_entity_type[etype] = int(count_str)
    return total, by_entity_type


# ---------------------------------------------------------------------------
# Group 1: parsed-console-equals-dataset (results)
# ---------------------------------------------------------------------------


def _expected_results_cells(row: dict[str, object]) -> list[str]:
    """Format a ``dataset.summary`` row with the printer's OWN helpers.

    The cell order matches the printed table:
    ``[Count, Mean|D|, Max|D|, WithinTol, sMAPE, r]``. Every cell is derived from
    the dataset row — nothing is recomputed from raw comparison rows.
    """
    correlation = row["correlation"]
    corr = f"{float(correlation):.4f}" if correlation is not None else "N/A"
    return [
        str(int(row["count"])),  # type: ignore[arg-type]
        _fmt_metric(float(row["mean_abs_diff"])),  # type: ignore[arg-type]
        _fmt_metric(float(row["max_abs_diff"])),  # type: ignore[arg-type]
        f"{float(row['within_tol_rate']) * 100:.1f}%",  # type: ignore[arg-type]
        f"{float(row['mean_smape']) * 100:.1f}%",  # type: ignore[arg-type]
        corr,
    ]


def test_results_table_cells_equal_dataset_summary() -> None:
    dataset = build_results_dataset(_make_results(), _one_hydro_pct(), _RESULTS_TOL)
    text = _capture(
        print_results_summary_from_dataset,
        dataset,
        _NW,
        _NOVOMODELO,
        verdict=build_compare_verdict(dataset),
    )

    parsed = _parse_results_table(text)
    summary_rows = {row["variable"]: row for row in dataset.summary.to_dicts()}

    # Every dataset variable is printed and every printed variable is in the dataset.
    assert set(parsed) == set(summary_rows)

    for variable, row in summary_rows.items():
        expected_cells = _expected_results_cells(row)
        assert parsed[variable] == expected_cells, (
            f"console cells for {variable!r} diverged from dataset.summary"
        )


def test_results_footer_equals_dataset_footer_counts() -> None:
    dataset = build_results_dataset(_make_results(), _one_hydro_pct(), _RESULTS_TOL)
    text = _capture(
        print_results_summary_from_dataset,
        dataset,
        _NW,
        _NOVOMODELO,
        verdict=build_compare_verdict(dataset),
    )

    total, by_entity_type = _parse_results_footer(text)
    footer_counts = dataset.metadata["footer_counts"]
    assert isinstance(footer_counts, dict)

    assert total == footer_counts["total"]
    assert by_entity_type == footer_counts["by_entity_type"]


def test_results_correlation_none_renders_na_from_dataset() -> None:
    # Arrange: a single-data-point variable -> dataset.summary.correlation is None.
    dataset = build_results_dataset(
        _make_single_point_results(), PercentileData(), _RESULTS_TOL
    )
    summary_rows = {row["variable"]: row for row in dataset.summary.to_dicts()}
    row = summary_rows["lonely_var"]
    # The N/A branch must be genuinely reached, not asserted blindly.
    assert row["correlation"] is None, (
        "fixture must yield correlation=None to exercise the N/A branch; "
        f"got {row['correlation']!r}"
    )

    # Act: render the dataset-driven printer and parse the variable's row.
    text = _capture(
        print_results_summary_from_dataset,
        dataset,
        _NW,
        _NOVOMODELO,
        verdict=build_compare_verdict(dataset),
    )
    parsed = _parse_results_table(text)

    # Assert: the printed ``r`` cell is "N/A", and "N/A" is exactly what the
    # dataset-derived expected cells produce from correlation=None — proving the
    # printed N/A traces to dataset.summary.correlation, not a recompute.
    expected_cells = _expected_results_cells(row)
    assert (
        expected_cells[-1] == "N/A"
    )  # derived FROM dataset.summary (correlation None)
    assert parsed["lonely_var"][-1] == "N/A"  # the ``r`` column is the last cell
    assert parsed["lonely_var"] == expected_cells

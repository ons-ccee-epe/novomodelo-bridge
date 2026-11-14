"""Console==dataset parity regression tests.

The compare-data-layer migration's central invariant is single-source-of-truth:
every number on the console must trace back to ONE ``ComparisonDataset`` analysis
(``dataset.summary`` rows / ``dataset.metadata`` counts), never to a second,
independently computed path. The historical bug this guards against was a
two-path divergence where console and HTML aggregated separately and drifted.

This module renders the dataset-driven console printers, parses the numbers back
out of the rendered text, and asserts those parsed cells equal the cells produced
by formatting the matching ``dataset.summary`` / ``dataset.metadata`` values with
the printer's OWN helpers (``core.agreement.format_gap``, ``variable_label`` and
the count/correlation format strings). The ONLY computation the test performs is parsing printed text:
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
from collections.abc import Callable
from pathlib import Path

import polars as pl

from novomodelo_bridge.comparators.analyze import build_results_dataset
from novomodelo_bridge.comparators.model import (
    PercentileData,
    ResultComparison,
)
from novomodelo_bridge.comparators.verdict import build_compare_verdict
from novomodelo_bridge.core.agreement import (
    ENTITY_GROUPS,
    Band,
    format_gap,
    is_policy_row,
    variable_label,
)
from novomodelo_bridge.core.pandas_utils import as_float, as_int
from novomodelo_bridge.ui.compare_summary import print_results_summary_from_dataset

_NW = Path("/fake/nw")
_NOVOMODELO = Path("/fake/novomodelo")
_RESULTS_TOL = 0.05

# Value columns of the agreement table after the variable label: Points, Gap,
# r, Agreement.
_RESULTS_VALUE_COLS = 4


# ---------------------------------------------------------------------------
# Synthetic fixtures (copied verbatim from tests/comparators/test_analyze.py)
# ---------------------------------------------------------------------------


def _make_single_point_results() -> list[ResultComparison]:
    """One comparison for a variable that appears exactly once.

    A single-data-point variable group has ``len(nw_vals) <= 1``, so
    ``build_results_summary`` leaves ``correlation`` at its ``None`` default
    (Pearson is undefined for fewer than two points). This drives the dataset
    summary's ``correlation`` cell to ``None`` and exercises the printer's
    ``"—"`` rendering branch.
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
    """Result comparisons spanning two entity types, plus a policy row the
    table leaves out."""
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
        ResultComparison(
            entity_type="hydro",
            entity_name="ITAIPU",
            newave_code=10,
            novomodelo_id=0,
            stage=0,
            variable="water_value_per_hm3",
            newave_value=100.0,
            novomodelo_value=900.0,
            abs_diff=800.0,
            rel_diff=8.0,
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


def _capture(func: Callable[..., object], *args: object, **kwargs: object) -> str:
    """Run a stdout-writing printer and return the captured text."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Parsers — the ONLY computation the test does is parsing printed text.
# ---------------------------------------------------------------------------


def _parse_results_table(text: str) -> dict[tuple[str, str], list[str]]:
    """Parse the agreement table into ``{(group heading, label): [cells]}``.

    A line that is exactly a group heading opens that group; a line ending in a
    band name is a variable row, whose last four whitespace-separated tokens are
    the value cells and whose leading tokens are the label. Everything else (the
    verdict, the header block, rules, the footnote and the footer) is skipped.
    """
    bands = {band.value for band in Band}
    headings = set(ENTITY_GROUPS.values())
    table: dict[tuple[str, str], list[str]] = {}
    group: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line in headings:
            group = line
            continue
        tokens = line.split()
        if not tokens or tokens[-1] not in bands or line.startswith(("✓ ", "⚠ ")):
            continue
        assert group is not None, f"variable row before any group heading: {raw!r}"
        assert len(tokens) > _RESULTS_VALUE_COLS, f"short variable row: {raw!r}"
        label = " ".join(tokens[:-_RESULTS_VALUE_COLS])
        table[(group, label)] = tokens[-_RESULTS_VALUE_COLS:]
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


def _expected_key(row: dict[str, object]) -> tuple[str, str]:
    return (
        ENTITY_GROUPS[str(row["entity_type"])],
        variable_label(str(row["variable"])),
    )


def _expected_results_cells(row: dict[str, object]) -> list[str]:
    """Format a ``dataset.summary`` row with the printer's OWN helpers.

    The cell order matches the printed table: ``[Points, Gap, r, Agreement]``.
    Every cell is derived from the dataset row — nothing is recomputed from raw
    comparison rows.
    """
    correlation = row["correlation"]
    gap = row["gap"]
    return [
        f"{as_int(row['count']):,}",
        format_gap(None if gap is None else as_float(gap)),
        "—" if correlation is None else f"{as_float(correlation):.3f}",
        str(row["band"]),
    ]


def _operation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if not is_policy_row(str(row["entity_type"]), str(row["variable"]))
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
    all_rows = dataset.summary.to_dicts()
    summary_rows = {_expected_key(row): row for row in _operation_rows(all_rows)}

    # The fixture's policy row is in the dataset but not on the console.
    assert len(summary_rows) == len(all_rows) - 1
    # Every operation row is printed and every printed row is in the dataset.
    assert set(parsed) == set(summary_rows)

    for key, row in summary_rows.items():
        expected_cells = _expected_results_cells(row)
        assert parsed[key] == expected_cells, (
            f"console cells for {key!r} diverged from dataset.summary"
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


def test_results_correlation_none_renders_dash_from_dataset() -> None:
    # Arrange: a single-data-point variable -> dataset.summary.correlation is None.
    dataset = build_results_dataset(
        _make_single_point_results(), PercentileData(), _RESULTS_TOL
    )
    summary_rows = {row["variable"]: row for row in dataset.summary.to_dicts()}
    row = summary_rows["lonely_var"]
    # The dash branch must be genuinely reached, not asserted blindly.
    assert row["correlation"] is None, (
        "fixture must yield correlation=None to exercise the dash branch; "
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

    # Assert: the printed ``r`` cell is "—", and "—" is exactly what the
    # dataset-derived expected cells produce from correlation=None — proving the
    # printed dash traces to dataset.summary.correlation, not a recompute.
    expected_cells = _expected_results_cells(row)
    assert expected_cells[2] == "—"
    assert parsed[_expected_key(row)] == expected_cells

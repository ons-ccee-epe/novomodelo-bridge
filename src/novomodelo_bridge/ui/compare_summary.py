"""Output formatting for results comparison."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from cobre_bridge.core.summary_counts import footer_counts
from cobre_bridge.ui.console import (
    compare_row_style,
    get_console,
    make_table,
    render_compare_verdict,
)

if TYPE_CHECKING:
    from rich.console import Console

    from cobre_bridge.comparators.dataset import ComparisonDataset
    from cobre_bridge.comparators.verdict import CompareVerdict


def _fmt_metric(x: float) -> str:
    """Format a metric value for the summary table.

    Scientific notation for very large or very small magnitudes (so columns
    like ``lower_bound`` stay readable instead of printing a 10-digit float),
    thousands-grouped fixed-point otherwise.
    """
    ax = abs(x)
    if ax >= 1e6 or (0 < ax < 1e-3):
        return f"{x:.3e}"
    return f"{x:,.3f}"


def print_results_summary_from_dataset(
    dataset: ComparisonDataset,
    newave_dir: Path,
    cobre_output_dir: Path,
    *,
    verdict: CompareVerdict,
    reference_label: str = "NEWAVE",
    console: Console | None = None,
) -> None:
    """Print the results comparison summary (Rich table) from the canonical dataset.

    The per-variable table is read from ``dataset.summary`` rows and the footer
    from ``dataset.metadata["footer_counts"]`` (``total`` / ``by_entity_type``,
    populated by ``analyze.build_results_dataset``), so every number traces back to
    a single analysis.

    Parameters
    ----------
    dataset:
        The canonical comparison dataset for the results subcommand.
    newave_dir:
        Path to the source model case directory.
    cobre_output_dir:
        Path to the Cobre output directory.
    verdict:
        The pre-built compare verdict to render at the top of the summary.
        Built by the caller (``cli/compare.py``) via ``build_compare_verdict``
        so this ``ui`` module never calls into ``comparators``.
    reference_label:
        Display name for the reference model in the printed header/labels.
        Defaults to ``"NEWAVE"`` (``compare newave``); ``compare decomp``
        overrides it with ``"DECOMP"``.
    console:
        The stdout console to render through. Defaults to :func:`get_console`
        so direct callers (tests) keep working unchanged; the CLI passes its
        ``--no-color``-aware console so this summary honours the flag.
    """
    target = console or get_console()

    render_compare_verdict(verdict, console=target)

    target.print()
    target.print(
        f"Cobre vs {reference_label} Results Comparison",
        soft_wrap=True,
        markup=False,
    )
    target.print("=" * 88, soft_wrap=True)
    target.print(f"{reference_label} case:  {newave_dir}", soft_wrap=True, markup=False)
    target.print(f"Cobre output: {cobre_output_dir}", soft_wrap=True, markup=False)

    # Per-variable table. WithinTol = share within the (relative) tolerance; sMAPE =
    # mean symmetric error (robust to near-zero source-model references).
    summary_rows = {row["variable"]: row for row in dataset.summary.to_dicts()}
    rows: list[list[str]] = []
    row_styles: list[str | None] = []
    for var in sorted(summary_rows):
        stats = summary_rows[var]
        correlation = stats["correlation"]
        corr = f"{float(correlation):.4f}" if correlation is not None else "N/A"
        rows.append(
            [
                var,
                str(int(stats["count"])),
                _fmt_metric(float(stats["mean_abs_diff"])),
                _fmt_metric(float(stats["max_abs_diff"])),
                f"{float(stats['within_tol_rate']) * 100:.1f}%",
                f"{float(stats['mean_smape']) * 100:.1f}%",
                corr,
            ]
        )
        # Style-only colour (never alters cell text): green iff every comparison
        # for this variable is within tolerance.
        row_styles.append(
            compare_row_style(within_tol=float(stats["within_tol_rate"]) == 1.0)
        )

    target.print(
        make_table(
            ["Variable", "Count", "Mean|D|", "Max|D|", "WithinTol", "sMAPE", "r"],
            rows,
            justify=["left", "right", "right", "right", "right", "right", "right"],
            row_styles=row_styles,
        )
    )

    total, by_entity_type = footer_counts(dataset.metadata)

    entity_parts = [
        f"{count} {etype}" for etype, count in sorted(by_entity_type.items())
    ]
    entity_str = ", ".join(entity_parts) if entity_parts else "none"

    target.print()
    target.print(
        f"Summary: {total} comparisons across "
        f"{len(by_entity_type)} entity types ({entity_str})",
        soft_wrap=True,
        markup=False,
    )
    target.print()

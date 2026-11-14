"""Output formatting for results comparison."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.text import Text

from novomodelo_bridge.core.agreement import (
    ENTITY_GROUPS,
    FAR_FACTOR,
    VARIABLE_LABELS,
    Band,
    format_gap,
    is_policy_row,
    variable_label,
)
from novomodelo_bridge.core.summary_counts import footer_counts
from novomodelo_bridge.ui.console import (
    BAND_STYLE,
    get_console,
    make_table,
    render_compare_verdict,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import Console

    from novomodelo_bridge.comparators.charts.costs import PairedCost
    from novomodelo_bridge.comparators.dataset import ComparisonDataset
    from novomodelo_bridge.comparators.verdict import CompareVerdict

_VARIABLE_ORDER: dict[str, int] = {name: i for i, name in enumerate(VARIABLE_LABELS)}


def _group_order(entity_type: str) -> tuple[int, str]:
    groups = list(ENTITY_GROUPS)
    return (
        groups.index(entity_type) if entity_type in groups else len(groups),
        entity_type,
    )


def _agreement_rows(
    dataset: ComparisonDataset,
) -> tuple[list[list[object]], list[str | None]]:
    """The table rows, an unstyled heading row before each entity group, and
    the style of each row."""
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in dataset.summary.to_dicts():
        entity_type, variable = str(row["entity_type"]), str(row["variable"])
        if not is_policy_row(entity_type, variable):
            by_group.setdefault(entity_type, []).append(row)

    rows: list[list[object]] = []
    styles: list[str | None] = []
    for entity_type in sorted(by_group, key=_group_order):
        rows.append([ENTITY_GROUPS.get(entity_type, entity_type), "", "", "", ""])
        styles.append("bold")
        for stats in sorted(
            by_group[entity_type],
            key=lambda r: (_VARIABLE_ORDER.get(str(r["variable"]), 99), r["variable"]),
        ):
            row_band = Band(stats["band"])
            gap = stats["gap"]
            correlation = stats["correlation"]
            rows.append(
                [
                    f"  {variable_label(str(stats['variable']))}",
                    f"{int(stats['count']):,}",
                    format_gap(None if gap is None else float(gap)),
                    "—" if correlation is None else f"{float(correlation):.3f}",
                    Text(row_band.value, style=BAND_STYLE[row_band]),
                ]
            )
            styles.append(None)
    return rows, styles


def _money(value: float | None) -> str:
    return "—" if value is None else f"{value / 1e9:,.3f}"


def print_results_summary_from_dataset(
    dataset: ComparisonDataset,
    newave_dir: Path,
    novomodelo_output_dir: Path,
    *,
    verdict: CompareVerdict,
    training: Sequence[PairedCost] = (),
    reference_label: str = "NEWAVE",
    console: Console | None = None,
) -> None:
    """Print the results comparison summary (Rich table) from the canonical dataset.

    The agreement table is read from ``dataset.summary`` rows, grouped by
    entity type, and leaves out the policy rows; the footer comes from
    ``dataset.metadata["footer_counts"]`` (``total`` / ``by_entity_type``,
    populated by ``analyze.build_results_dataset``), so every number traces
    back to a single analysis.

    Parameters
    ----------
    dataset:
        The canonical comparison dataset for the results subcommand.
    newave_dir:
        Path to the source model case directory.
    novomodelo_output_dir:
        Path to the Novomodelo output directory.
    verdict:
        The pre-built compare verdict to render at the top of the summary.
        Built by the caller (``cli/compare.py``) via ``build_compare_verdict``
        so this ``ui`` module never calls into ``comparators``.
    training:
        The final lower bound and simulated mean cost of each model, printed
        below the table; built by the caller for the same reason.
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
        f"Novomodelo vs {reference_label} Results Comparison",
        soft_wrap=True,
        markup=False,
    )
    target.print("=" * 88, soft_wrap=True)
    target.print(f"{reference_label} case:  {newave_dir}", soft_wrap=True, markup=False)
    target.print(
        f"Novomodelo output: {novomodelo_output_dir}", soft_wrap=True, markup=False
    )

    rows, row_styles = _agreement_rows(dataset)
    if rows:
        target.print(
            make_table(
                ["Variable", "Points", "Gap", "r", "Agreement"],
                rows,
                justify=["left", "right", "right", "right", "left"],
                row_styles=row_styles,
            )
        )
        target.print(
            f"Gap = Σ|Novomodelo − {reference_label}| / Σ|{reference_label}| over "
            f"every stage and entity: close up to {format_gap(verdict.tolerance)}, "
            f"far beyond {format_gap(FAR_FACTOR * verdict.tolerance)}.",
            style="dim",
            soft_wrap=True,
            markup=False,
        )

    if training:
        target.print()
        target.print(
            make_table(
                ["Bounds and cost (10⁹ R$)", reference_label, "Novomodelo", "Δ%"],
                [
                    [
                        row.label,
                        _money(row.reference),
                        _money(row.novomodelo),
                        "—" if row.pct is None else f"{row.pct:+.1f}%",
                    ]
                    for row in training
                ],
                justify=["left", "right", "right", "right"],
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

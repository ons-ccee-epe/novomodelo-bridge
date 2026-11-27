"""Handler of ``compare novomodelo``: two Novomodelo outputs, baseline against candidate."""

from __future__ import annotations

from typing import TYPE_CHECKING

from novomodelo_bridge.cli.args import CompareArgs, ResolvedCompareArgs
from novomodelo_bridge.cli.compare import (
    _resolve_compare_formats,
    _resolve_compare_settings,
    _training_costs,
    _write_html_compare_report,
)
from novomodelo_bridge.cli.failure import _emit_convert_json, _fail
from novomodelo_bridge.cli.verdict import build_verdict, version_dataset_summary
from novomodelo_bridge.ui.console import print_status, render_diagnostics, spinner

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from novomodelo_bridge.comparators.dataset import ComparisonDataset
    from novomodelo_bridge.core.diagnostics import Diagnostic

#: The directory, under the candidate output, that holds the artifacts by default.
ARTIFACTS_DIRNAME: str = "version_comparison_artifacts"


def _export_version_artifacts(
    dataset: ComparisonDataset,
    formats: set[str],
    args: ResolvedCompareArgs,
    diagnostics: Sequence[Diagnostic],
) -> Path:
    """Write the artifacts *formats* asks for and return the directory they go to.

    A write failure must not change the comparison exit code, so an ``OSError``
    is warned on stderr and swallowed. The ``Artifacts written to`` status line
    is the only thing ``--json`` suppresses: the files are still written.
    """
    from novomodelo_bridge.comparators.version.manifest import write_version_artifacts

    out_dir = args.out_dir or args.novomodelo_output_dir / ARTIFACTS_DIRNAME
    try:
        write_version_artifacts(
            dataset,
            command="compare novomodelo",
            tolerance=args.tolerance,
            out_dir=out_dir,
            formats=sorted(formats & {"csv", "parquet", "json"}),
            diagnostics=diagnostics,
        )
    except OSError as exc:
        print_status(
            f"Warning: failed to write artifacts: {exc}",
            console=args.err_console(),
            style="#F5A623",
        )
    else:
        if not args.json_output:
            print_status(f"Artifacts written to {out_dir}")
    return out_dir


def _run_novomodelo_comparison(
    args: CompareArgs, *, labels: tuple[str | None, str | None] = (None, None)
) -> None:
    """Execute the compare novomodelo subcommand.

    ``source_dir`` is the baseline output and ``novomodelo_output_dir`` the
    candidate output; *labels* are the names the user chose for them, ``None``
    for the default. Informational like the other compare commands: a
    divergence never changes the exit code (0), and the ``--json`` status is
    decoupled from it. Only a failure to read the outputs does: a missing
    directory, an absent required file or a malformed entity list (1), and a
    directory that is not an output or an unreadable file (2).
    """
    from novomodelo_bridge.comparators.export import MANIFEST_FILE
    from novomodelo_bridge.comparators.verdict import (
        build_compare_verdict,
        compare_status,
    )
    from novomodelo_bridge.comparators.version.facts import facts_of
    from novomodelo_bridge.comparators.version.report import build_version_report
    from novomodelo_bridge.comparators.version.results import build_version_dataset
    from novomodelo_bridge.core import diagnostics as dx
    from novomodelo_bridge.core.errors import (
        FieldParseError,
        NovomodeloOutputError,
        NovomodeloPartitionMissingError,
        SourceFileError,
    )
    from novomodelo_bridge.ui.compare_summary import print_results_summary_from_dataset

    args = _resolve_compare_settings(args)
    formats = _resolve_compare_formats("compare novomodelo", args)

    with dx.collect() as sink:
        try:
            with spinner(
                "Comparing results…",
                verbose=args.verbose > 0,
                quiet=args.quiet,
                no_color=args.no_color,
            ):
                dataset = build_version_dataset(
                    args.source_dir,
                    args.novomodelo_output_dir,
                    tolerance=args.tolerance,
                    labels=labels,
                )
        # Not a NovomodeloOutputError: unlisted, a missing partition ends in a traceback.
        except (NovomodeloOutputError, NovomodeloPartitionMissingError) as exc:
            _fail("compare novomodelo", args, exc, 2, found_before=sink)
        except (SourceFileError, FieldParseError) as exc:
            _fail("compare novomodelo", args, exc, 1, found_before=sink)
        report_html = build_version_report(dataset) if "html" in formats else None
    diagnostics = dx.finalize_diagnostics(sink)

    facts = facts_of(dataset)
    verdict = build_compare_verdict(dataset)

    if not args.json_output:
        if not args.quiet:
            print_results_summary_from_dataset(
                dataset,
                args.source_dir,
                args.novomodelo_output_dir,
                verdict=verdict,
                training=_training_costs(dataset),
                reference_label=facts.baseline_label,
                candidate_label=facts.candidate_label,
                verbose=args.verbose > 0,
                console=args.out_console(),
            )
        render_diagnostics(
            diagnostics,
            heading="Comparison notes",
            console=args.err_console(),
            quiet=args.quiet,
            verbose=args.verbose > 0,
            details_file=MANIFEST_FILE,
        )

    out_dir = _export_version_artifacts(dataset, formats, args, diagnostics)

    if report_html is not None:
        _write_html_compare_report(report_html, out_dir, args)

    if args.json_output:
        _emit_convert_json(
            build_verdict(
                "compare novomodelo",
                compare_status(dataset),
                version_dataset_summary(dataset),
                diagnostics,
            )
        )

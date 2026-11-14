"""Compare-command handlers for ``compare newave`` / ``compare decomp``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from novomodelo_bridge.cli.args import CompareArgs, ResolvedCompareArgs, _parse_formats
from novomodelo_bridge.cli.config import load_config
from novomodelo_bridge.cli.failure import _emit_convert_json, _fail
from novomodelo_bridge.cli.verdict import (
    build_verdict,
    compare_summary,
    decomp_dataset_summary,
)
from novomodelo_bridge.core.agreement import DEFAULT_TOLERANCE
from novomodelo_bridge.core.provenance import hash_input_files
from novomodelo_bridge.ui.console import print_status, render_diagnostics, spinner

if TYPE_CHECKING:
    from pathlib import Path

    from novomodelo_bridge.comparators.charts.costs import PairedCost
    from novomodelo_bridge.comparators.dataset import ComparisonDataset
    from novomodelo_bridge.comparators.newave.alignment import EntityAlignment
    from novomodelo_bridge.newave.case import NewaveCase
    from novomodelo_bridge.newave.id_map import NewaveIdMap


def _load_compare_context(
    newave_dir: Path,
    novomodelo_output_dir: Path,
    *,
    command: str,
    args: CompareArgs,
) -> tuple[NewaveCase, NewaveIdMap, EntityAlignment, list[dict[str, object]]]:
    """Load the source model case, id-map, entity alignment, and lines.json.

    Setup for `compare newave`. Builds the parsed
    case once (so the id-map reuses its cached readers), loads lines.json, and
    builds the entity alignment.

    Exits the process with code 1 via :func:`_fail` (a rendered diagnostic, or
    under ``--json`` a verdict envelope) if the source model case directory is
    missing (``FileNotFoundError`` from ``NewaveCase.from_directory``).
    """
    from novomodelo_bridge.comparators.newave.alignment import build_entity_alignment
    from novomodelo_bridge.newave.case import NewaveCase
    from novomodelo_bridge.novomodelo.readers import read_novomodelo_lines

    try:
        case = NewaveCase.from_directory(newave_dir)
    except FileNotFoundError as exc:
        _fail(command, args, exc, 1)

    id_map = case.id_map
    lines_json = read_novomodelo_lines(novomodelo_output_dir)
    alignment = build_entity_alignment(id_map, case, lines_json)
    return case, id_map, alignment, lines_json


def _resolve_compare_formats(command: str, args: CompareArgs) -> set[str]:
    """Parse ``--format`` before the comparison runs; a bad token exits 2."""
    try:
        return _parse_formats(args.format)
    except ValueError as exc:
        _fail(command, args, exc, 2)


def _export_compare_artifacts(
    dataset: ComparisonDataset,
    *,
    formats: set[str],
    args: CompareArgs,
    command: str,
    source_dir: Path,
    novomodelo_output_dir: Path,
    tolerance: float,
    out_dir_arg: Path | None,
    input_files: list[dict[str, object]] | None = None,
    diagnostics: list[dict[str, object]] | None = None,
    quiet_status: bool = False,
) -> Path:
    """Write the machine-readable comparison artifacts *formats* asks for.

    Shared by `compare newave` and `compare decomp` (``source_dir`` names the
    source-case directory generically — the manifest field itself is
    reused across both callers). Returns the resolved out_dir for the
    handler's HTML report.

    A write failure must NOT change the comparison exit code, so an
    ``OSError`` is warned and swallowed.

    *quiet_status* (set by ``--json``) gates ONLY the ``Artifacts written to …``
    stdout status line so stdout stays pure JSON; the file export still runs and
    the ``OSError`` write-failure warning still reaches stderr.
    """
    from novomodelo_bridge.comparators.export import write_artifacts

    out_dir: Path = out_dir_arg or (novomodelo_output_dir / "comparison_artifacts")
    export_formats = formats & {"csv", "parquet", "json"}

    try:
        write_artifacts(
            dataset,
            command=command,
            source_dir=source_dir,
            novomodelo_output_dir=novomodelo_output_dir,
            tolerance=tolerance,
            out_dir=out_dir,
            formats=sorted(export_formats),
            input_files=input_files,
            diagnostics=diagnostics,
        )
        if not quiet_status:
            print_status(f"Artifacts written to {out_dir}")
    except OSError as exc:
        print_status(
            f"Warning: failed to write artifacts: {exc}",
            console=args.err_console(),
            style="#F5A623",
        )

    return out_dir


def _build_html_compare_report(
    dataset: ComparisonDataset, formats: set[str], *, reference_label: str
) -> str | None:
    """Build the HTML report when *formats* asks for it (``html``/``all``).

    Call it inside the run's ``diagnostics.collect()`` block: the report's
    charts emit diagnostics, which outside a sink are only logged and never
    reach the comparison notes, the ``comparison.json`` manifest or the
    ``--json`` envelope.
    """
    if "html" not in formats:
        return None
    from novomodelo_bridge.comparators.report_builder import build_comparison_report

    return build_comparison_report(dataset, reference_label=reference_label)


def _write_html_compare_report(html: str, out_dir: Path, args: CompareArgs) -> None:
    """Write the HTML report built by :func:`_build_html_compare_report`.

    The file is still written under ``--json`` (it is a ``--format``
    artifact); only its stdout advisory is routed to stderr so stdout stays
    pure JSON.
    """
    report_path = out_dir / "report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    print_status(
        f"HTML report written to {report_path}",
        console=args.err_console() if args.json_output else None,
    )


def _resolve_compare_settings(args: CompareArgs) -> ResolvedCompareArgs:
    """Return *args* with tolerance/format/out_dir filled in from config.

    Implements the bottom two rungs of the precedence chain
    **CLI flag > env var > config file > built-in default**: Typer has already
    resolved flag-or-env into ``args`` (a non-``None`` value means the flag or
    env var supplied it), so this fills in the config-file value, then the
    built-in default, for whatever is still ``None``. ``args`` is frozen, so
    the resolved settings come back as a new instance; the input is untouched.

    Loads the config once via :func:`load_config` (discovery from cwd) and emits
    one stderr WARNING note per load warning (e.g. a malformed config file). It
    never raises and never changes the exit code; config provenance stays off
    stdout so a ``--json``-style payload remains deterministic.
    """
    cfg = load_config()

    if args.tolerance is not None:
        resolved_tolerance = args.tolerance
    elif cfg.results_tolerance is not None:
        resolved_tolerance = cfg.results_tolerance
    else:
        resolved_tolerance = DEFAULT_TOLERANCE

    # Format: only consult config when neither flag nor env supplied it. Leave
    # ``None`` (rather than pre-expanding) so ``_parse_formats`` stays the single
    # validator/expander downstream.
    resolved_format = args.format
    if resolved_format is None and cfg.formats is not None:
        resolved_format = list(cfg.formats)

    # Out-dir: only consult config when neither flag nor env supplied it. The
    # config value may itself be ``None``, which keeps the derived
    # ``<novomodelo_output_dir>/comparison_artifacts`` default downstream.
    resolved_out_dir = args.out_dir if args.out_dir is not None else cfg.out_dir

    # Surface any config-load warnings on stderr only (never stdout), matching
    # the side-file advisory pattern used by ``_export_compare_artifacts``.
    for warning in cfg.warnings:
        print_status(
            warning,
            console=args.err_console(),
            style="#F5A623",
        )

    return ResolvedCompareArgs(
        json_output=args.json_output,
        verbose=args.verbose,
        log_file=args.log_file,
        no_color=args.no_color,
        quiet=args.quiet,
        source_dir=args.source_dir,
        novomodelo_output_dir=args.novomodelo_output_dir,
        tolerance=resolved_tolerance,
        format=resolved_format,
        out_dir=resolved_out_dir,
    )


def _training_costs(dataset: ComparisonDataset) -> tuple[PairedCost, ...]:
    from novomodelo_bridge.comparators.charts.costs import training_costs

    render = dataset.render
    return training_costs(
        render.nw_convergence,
        render.novomodelo_convergence,
        render.nw_costs,
        render.novomodelo_costs,
    )


def _run_newave_comparison(args: CompareArgs) -> None:
    """Execute the compare newave subcommand.

    ``compare newave`` is informational (a descriptive NEWAVE-vs-Novomodelo
    divergence report), so a divergence never changes the exit code (0). Only a
    failure to compare does: a missing source case (1) or an unreadable
    Novomodelo output (2), both routed through :func:`_fail`.
    """
    args = _resolve_compare_settings(args)

    from novomodelo_bridge.comparators.newave.results import compare_results
    from novomodelo_bridge.comparators.verdict import (
        build_compare_verdict,
        compare_status,
    )
    from novomodelo_bridge.core import diagnostics as dx
    from novomodelo_bridge.core.errors import NovomodeloPartitionMissingError
    from novomodelo_bridge.novomodelo.readers import NovomodeloReadError
    from novomodelo_bridge.ui.compare_summary import print_results_summary_from_dataset

    newave_dir: Path = args.source_dir
    novomodelo_output_dir: Path = args.novomodelo_output_dir
    tolerance: float = args.tolerance
    formats = _resolve_compare_formats("compare newave", args)

    case, id_map, alignment, _lines_json = _load_compare_context(
        newave_dir, novomodelo_output_dir, command="compare newave", args=args
    )

    # NovomodeloPartitionMissingError (a BridgeError; output predates a partition
    # this compare needs) and NovomodeloReadError (a RuntimeError; a malformed
    # output file) are disjoint hierarchies — both must stay in the ``except``,
    # or the dropped one crashes with a bare traceback instead of exit 2.
    with dx.collect() as compare_diagnostics:
        try:
            with spinner(
                "Comparing results…",
                verbose=args.verbose > 0,
                quiet=args.quiet,
                no_color=args.no_color,
            ):
                dataset = compare_results(
                    case=case,
                    id_map=id_map,
                    alignment=alignment,
                    novomodelo_output_dir=novomodelo_output_dir,
                    tolerance=tolerance,
                )
        except (NovomodeloReadError, NovomodeloPartitionMissingError) as exc:
            _fail("compare newave", args, exc, 2)
        report_html = _build_html_compare_report(
            dataset, formats, reference_label="NEWAVE"
        )

    verdict = build_compare_verdict(dataset)

    # Print text summary (sourced from the dataset). Under --json the Rich tables
    # are suppressed in favour of a single machine-readable verdict on stdout;
    # under --quiet the summary is suppressed too, but diagnostics still render.
    if not args.json_output:
        if not args.quiet:
            print_results_summary_from_dataset(
                dataset,
                newave_dir,
                novomodelo_output_dir,
                verdict=verdict,
                training=_training_costs(dataset),
                console=args.out_console(),
            )
        render_diagnostics(
            compare_diagnostics,
            heading="Comparison notes",
            console=args.err_console(),
            quiet=args.quiet,
        )

    out_dir = _export_compare_artifacts(
        dataset,
        formats=formats,
        args=args,
        command="compare newave",
        source_dir=newave_dir,
        novomodelo_output_dir=novomodelo_output_dir,
        tolerance=tolerance,
        out_dir_arg=args.out_dir,
        input_files=hash_input_files(case.files),
        diagnostics=[d.to_dict() for d in compare_diagnostics],
        quiet_status=args.json_output,
    )

    if report_html is not None:
        _write_html_compare_report(report_html, out_dir, args)

    if args.json_output:
        # ``status`` is DECOUPLED from the exit code (this command always
        # exits 0) and uses the shared ``compare_status`` vocabulary, so an
        # empty dataset reports "no-comparable-rows", not "mismatch".
        status = compare_status(dataset)
        _emit_convert_json(
            build_verdict(
                "compare newave",
                status,
                compare_summary(verdict),
                compare_diagnostics,
            )
        )


def _run_decomp_comparison(args: CompareArgs) -> None:
    """Execute the compare decomp subcommand.

    Informational like ``compare newave``: divergences never change the exit
    code (0) -- the command describes them rather than judging them. Two
    failures stop it, the same as on the NEWAVE track: a missing source deck
    (exit 1) and an unreadable *existing* output file (exit 2) -- reporting a
    zero-vs-zero match on data we could not read would be worse than stopping.
    """
    from novomodelo_bridge.comparators.decomp.results import build_decomp_dataset
    from novomodelo_bridge.comparators.verdict import (
        build_compare_verdict,
        compare_status,
    )
    from novomodelo_bridge.core import diagnostics as dx
    from novomodelo_bridge.core.errors import (
        FieldParseError,
        NovomodeloPartitionMissingError,
    )
    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.novomodelo.readers import NovomodeloReadError
    from novomodelo_bridge.ui.compare_summary import print_results_summary_from_dataset

    # Resolved before the read (unlike the pre-dataset ordering) so
    # ``build_decomp_dataset`` below gets a concrete tolerance rather than the
    # raw, possibly-``None`` CLI value — mirrors ``_run_newave_comparison``.
    args = _resolve_compare_settings(args)
    formats = _resolve_compare_formats("compare decomp", args)

    try:
        decomp_case = DecompCase.from_directory(args.source_dir)
    except FileNotFoundError as exc:
        _fail("compare decomp", args, exc, 1)

    with dx.collect() as compare_diagnostics:
        try:
            with spinner(
                "Comparing results…",
                verbose=args.verbose > 0,
                quiet=args.quiet,
                no_color=args.no_color,
            ):
                dataset = build_decomp_dataset(
                    args.source_dir,
                    args.novomodelo_output_dir,
                    tolerance=args.tolerance,
                )
        except (
            NovomodeloReadError,
            NovomodeloPartitionMissingError,
            FieldParseError,
            FileNotFoundError,
            ValueError,
        ) as exc:
            _fail("compare decomp", args, exc, 2)
        report_html = _build_html_compare_report(
            dataset, formats, reference_label="DECOMP"
        )

    verdict = build_compare_verdict(dataset)

    if not args.json_output:
        if not args.quiet:
            print_results_summary_from_dataset(
                dataset,
                args.source_dir,
                args.novomodelo_output_dir,
                verdict=verdict,
                training=_training_costs(dataset),
                reference_label="DECOMP",
                console=args.out_console(),
            )
        render_diagnostics(
            compare_diagnostics,
            heading="Comparison notes",
            console=args.err_console(),
            quiet=args.quiet,
        )

    out_dir = _export_compare_artifacts(
        dataset,
        formats=formats,
        args=args,
        command="compare decomp",
        source_dir=args.source_dir,
        novomodelo_output_dir=args.novomodelo_output_dir,
        tolerance=args.tolerance,
        out_dir_arg=args.out_dir,
        input_files=hash_input_files(decomp_case.files),
        diagnostics=[d.to_dict() for d in compare_diagnostics],
        quiet_status=args.json_output,
    )

    if report_html is not None:
        _write_html_compare_report(report_html, out_dir, args)

    if args.json_output:
        # ``status`` is DECOUPLED from the exit code (this command always
        # exits 0, mirroring ``compare newave``) and uses the shared
        # ``compare_status`` vocabulary.
        summary = decomp_dataset_summary(dataset)
        status = compare_status(dataset)
        _emit_convert_json(
            build_verdict("compare decomp", status, summary, compare_diagnostics)
        )

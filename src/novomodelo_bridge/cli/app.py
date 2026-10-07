"""Command-line interface entry point for cobre-bridge."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from cobre_bridge import __version__
from cobre_bridge.cli.args import (
    CheckArgs,
    CompareArgs,
    ConvertArgs,
    DashboardArgs,
)
from cobre_bridge.cli.check import _run_check, _run_decomp_check
from cobre_bridge.cli.compare import _run_decomp_comparison, _run_newave_comparison
from cobre_bridge.cli.convert import _run_decomp_conversion, _run_newave_conversion
from cobre_bridge.cli.dashboard import _run_dashboard

# noqa: F401 below -- re-exported so `cli._NULL_HANDLER` keeps resolving for the
# ``test_configure_logging_levels`` import + the ``cli._configure_logging`` spy sites.
from cobre_bridge.cli.logging_config import NULL_HANDLER as _NULL_HANDLER  # noqa: F401
from cobre_bridge.cli.logging_config import configure_logging as _configure_logging
from cobre_bridge.cli.logging_config import restore_log_file_handler
from cobre_bridge.cli.validate import (
    _partition_validation_warnings as _partition_validation_warnings,
)
from cobre_bridge.cobre.compat import MIN_COBRE_VERSION as MIN_COBRE_VERSION
from cobre_bridge.ui.console import print_status

# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------

# Shared flags reused across the leaf commands.
_VerboseOpt = Annotated[
    int,
    typer.Option(
        "--verbose",
        "-v",
        count=True,
        help="Increase console log verbosity (-v INFO, -vv DEBUG).",
    ),
]
_LogFileOpt = Annotated[
    Path | None,
    typer.Option(
        "--log-file",
        metavar="PATH",
        help="Write the full DEBUG log to PATH (the console verbosity is unaffected).",
    ),
]
_NoColorOpt = Annotated[
    bool,
    typer.Option(
        "--no-color",
        help="Disable coloured output (also honoured via the NO_COLOR env var).",
    ),
]
_QuietOpt = Annotated[
    bool,
    typer.Option(
        "--quiet",
        help="Suppress the summary and info notes; warnings/errors still show.",
    ),
]
_FormatOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--format",
        metavar="FORMAT",
        envvar="COBRE_BRIDGE_FORMAT",
        help=(
            "Output format(s): console,html,csv,parquet,json,all. "
            "Comma-separated and/or repeatable. Overridable via "
            "COBRE_BRIDGE_FORMAT or cobre-bridge.toml. "
            "(default: console,parquet,json)"
        ),
    ),
]
_OutDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--out-dir",
        envvar="COBRE_BRIDGE_OUT_DIR",
        help=(
            "Directory for file artifacts. Overridable via COBRE_BRIDGE_OUT_DIR "
            "or cobre-bridge.toml. "
            "(default: <cobre_output_dir>/comparison_artifacts)."
        ),
    ),
]
_JsonOpt = Annotated[
    bool,
    typer.Option(
        "--json",
        help=(
            "Emit a single machine-readable JSON verdict to stdout and "
            "suppress the human-readable (Rich) output."
        ),
    ),
]
_ToleranceOpt = Annotated[
    float | None,
    typer.Option(
        envvar="COBRE_BRIDGE_RESULTS_TOLERANCE",
        help=(
            "Relative tolerance for results comparison (default 1e-2; "
            "overridable via COBRE_BRIDGE_RESULTS_TOLERANCE or cobre-bridge.toml)."
        ),
    ),
]

app = typer.Typer(
    name="cobre-bridge",
    help="Convert power system data to Cobre input format.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=True,
)
convert_app = typer.Typer(help="Convert data from a source format to Cobre JSON.")
compare_app = typer.Typer(help="Compare source model inputs/results against Cobre.")
check_app = typer.Typer(help="Validate source-model inputs without converting.")
app.add_typer(convert_app, name="convert")
app.add_typer(compare_app, name="compare")
app.add_typer(check_app, name="check")


def _version_callback(value: bool) -> None:
    if value:
        print_status(f"cobre-bridge {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Convert power system data to Cobre input format."""


@convert_app.command("newave")
def _convert_newave(
    src: Annotated[Path, typer.Argument(help="Path to the NEWAVE case directory.")],
    dst: Annotated[
        Path, typer.Argument(help="Path to the output Cobre case directory.")
    ],
    validate: Annotated[
        bool,
        typer.Option(
            "--validate",
            help="After conversion, validate the output with the cobre package.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite destination directory if it already contains files.",
        ),
    ] = False,
    diagnostics_json: Annotated[
        Path | None,
        typer.Option(
            "--diagnostics-json",
            metavar="PATH",
            help="Also write the conversion diagnostics (counts + findings) as JSON.",
        ),
    ] = None,
    json_output: _JsonOpt = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Run the full conversion in memory and report what would be "
                "written, without creating or modifying the destination directory."
            ),
        ),
    ] = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Convert a NEWAVE case directory to a Cobre case directory."""
    _configure_logging(verbose, log_file)
    _run_newave_conversion(
        ConvertArgs(
            src=src,
            dst=dst,
            validate=validate,
            force=force,
            diagnostics_json=diagnostics_json,
            json_output=json_output,
            dry_run=dry_run,
            verbose=verbose,
            log_file=log_file,
            no_color=no_color,
            quiet=quiet,
        )
    )


@convert_app.command("decomp")
def _convert_decomp(
    src: Annotated[Path, typer.Argument(help="Path to the DECOMP deck directory.")],
    dst: Annotated[
        Path, typer.Argument(help="Path to the output Cobre case directory.")
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite destination directory if it already contains files.",
        ),
    ] = False,
    validate: Annotated[
        bool,
        typer.Option(
            "--validate",
            help="After conversion, validate the output with the cobre package.",
        ),
    ] = False,
    diagnostics_json: Annotated[
        Path | None,
        typer.Option(
            "--diagnostics-json",
            metavar="PATH",
            help="Also write the conversion diagnostics (counts + findings) as JSON.",
        ),
    ] = None,
    json_output: _JsonOpt = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Run the full conversion in memory and report what would be "
                "written, without creating or modifying the destination directory."
            ),
        ),
    ] = False,
    no_fcf: Annotated[
        bool,
        typer.Option(
            "--no-fcf",
            help=(
                "Skip importing the deck's boundary FCF. By default, when the "
                "deck declares cortes/cortesh files (its FC records), they are "
                "imported as a terminal-stage cobre policy checkpoint via an "
                "in-process 1-iteration cobre pass (slow; requires cobre-python). "
                "Pass this for a quick conversion without the terminal FCF. "
                "The FCF is always skipped under --dry-run."
            ),
        ),
    ] = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Convert a DECOMP deck revision to a Cobre case directory.

    Deck features the conversion leaves out (``check decomp`` lists them per
    deck) are reported as warnings, never dropped silently. The boundary FCF is
    imported by default whenever the deck declares its cut files; ``--no-fcf``
    skips it.
    """
    _configure_logging(verbose, log_file)
    _run_decomp_conversion(
        ConvertArgs(
            src=src,
            dst=dst,
            force=force,
            validate=validate,
            diagnostics_json=diagnostics_json,
            json_output=json_output,
            dry_run=dry_run,
            no_fcf=no_fcf,
            verbose=verbose,
            log_file=log_file,
            no_color=no_color,
            quiet=quiet,
        )
    )


@compare_app.command("decomp")
def _compare_decomp(
    decomp_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to the DECOMP deck directory (deck + dec_oper_*.csv "
            "result files, all directly in it)."
        ),
    ],
    cobre_output_dir: Annotated[
        Path, typer.Argument(help="Path to the Cobre output directory.")
    ],
    tolerance: _ToleranceOpt = None,
    fmt: _FormatOpt = None,
    out_dir: _OutDirOpt = None,
    json_output: _JsonOpt = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Compare a DECOMP run's published operation against Cobre's simulation.

    Informational: always exits 0, reporting divergences without failing.

    Two caveats apply to the generated report. First, the Overview tab's NPV
    cost cards compare DECOMP's undiscounted-nominal costs against Cobre's
    time-discounted costs: DECOMP's own cost report carries no per-stage
    discount factor, so none is fabricated on that side, and the two totals
    are not on the same time-value footing. Second, percentile bands are
    omitted (or labelled low-N where one would otherwise appear) for a
    deterministic tree with too few scenarios to report a spread without
    synthesizing it.
    """
    _configure_logging(verbose, log_file)
    _run_decomp_comparison(
        CompareArgs(
            source_dir=decomp_dir,
            cobre_output_dir=cobre_output_dir,
            format=fmt,
            out_dir=out_dir,
            tolerance=tolerance,
            json_output=json_output,
            verbose=verbose,
            log_file=log_file,
            no_color=no_color,
            quiet=quiet,
        )
    )


@compare_app.command("newave")
def _compare_newave(
    newave_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to the NEWAVE case directory (case + MEDIAS-*.CSV "
            "result files, all directly in it)."
        ),
    ],
    cobre_output_dir: Annotated[
        Path, typer.Argument(help="Path to the Cobre output directory.")
    ],
    tolerance: _ToleranceOpt = None,
    fmt: _FormatOpt = None,
    out_dir: _OutDirOpt = None,
    json_output: _JsonOpt = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Compare NEWAVE published results against Cobre simulation output.

    Informational: always exits 0, reporting divergences without failing.
    """
    _configure_logging(verbose, log_file)
    _run_newave_comparison(
        CompareArgs(
            source_dir=newave_dir,
            cobre_output_dir=cobre_output_dir,
            tolerance=tolerance,
            format=fmt,
            out_dir=out_dir,
            json_output=json_output,
            verbose=verbose,
            log_file=log_file,
            no_color=no_color,
            quiet=quiet,
        )
    )


@check_app.command("newave")
def _check_newave(
    src: Annotated[Path, typer.Argument(help="Path to the NEWAVE case directory.")],
    json_output: _JsonOpt = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Validate a NEWAVE case directory without converting or writing any files."""
    _configure_logging(verbose, log_file)
    _run_check(
        CheckArgs(
            src=src,
            json_output=json_output,
            verbose=verbose,
            log_file=log_file,
            no_color=no_color,
            quiet=quiet,
        )
    )


@check_app.command("decomp")
def _check_decomp(
    src: Annotated[Path, typer.Argument(help="Path to the DECOMP deck directory.")],
    json_output: _JsonOpt = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Validate a DECOMP deck revision without converting or writing any files.

    Also reports what the conversion will leave behind, so a deferred feature
    is never a silent omission.
    """
    _configure_logging(verbose, log_file)
    _run_decomp_check(
        CheckArgs(
            src=src,
            json_output=json_output,
            verbose=verbose,
            log_file=log_file,
            no_color=no_color,
            quiet=quiet,
        )
    )


@app.command("dashboard")
def _dashboard(
    case_dir: Annotated[Path, typer.Argument(help="Path to the Cobre case directory.")],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output HTML file path (default: <case_dir>/dashboard.html).",
        ),
    ] = None,
    open_browser: Annotated[
        bool,
        typer.Option(
            "--open",
            help=(
                "Open the generated dashboard in the default web browser "
                "after writing it."
            ),
        ),
    ] = False,
    json_output: _JsonOpt = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Generate an interactive HTML dashboard from Cobre simulation results."""
    _configure_logging(verbose, log_file)
    _run_dashboard(
        DashboardArgs(
            case_dir=case_dir,
            output=output,
            open_browser=open_browser,
            json_output=json_output,
            verbose=verbose,
            log_file=log_file,
            no_color=no_color,
            quiet=quiet,
        )
    )


def main() -> None:
    """Console entry point: run the Typer app, restoring the logger afterwards.

    The thin wrapper restores the ``cobre_bridge`` logger ``propagate`` flag that
    ``_configure_logging`` flips, and removes + closes any ``--log-file``
    ``FileHandler`` it attached, so a real CLI run never leaks logging state.
    """
    pkg_logger = logging.getLogger("cobre_bridge")
    prior_propagate = pkg_logger.propagate
    try:
        app()
    finally:
        pkg_logger.propagate = prior_propagate
        restore_log_file_handler(pkg_logger)


if __name__ == "__main__":
    main()

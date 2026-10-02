"""Export commands for novomodelo-bridge: export Novomodelo artifacts to other formats."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Annotated

import typer

from novomodelo_bridge.cli.args import CommonArgs
from novomodelo_bridge.cli.logging_config import configure_logging as _configure_logging
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.dessem.exporters.fcf import EXPORT_PHASE_LABELS
from novomodelo_bridge.dessem.exporters.fcf.pipeline import ExportSettings, run_export
from novomodelo_bridge.ui.console import (
    conversion_progress,
    get_console,
    print_status,
    render_diagnostics,
    render_error,
)

export_app = typer.Typer(
    help="Export Novomodelo artifacts to external formats (DESSEM, etc.)."
)

# Default hydro codes file bundled with the package
_DEFAULT_HYDRO_CODES = resources.files(
    "novomodelo_bridge.dessem.exporters.fcf.mapping"
).joinpath("decomp_hydro_codes.json")


@dataclass(frozen=True)
class ExportDessemFcfArgs(CommonArgs):
    """Options for ``export dessem-fcf``."""

    case_dir: Path
    revision: str
    output: Path | None
    force: bool
    hydro_codes: Path | None
    include_terminal: bool


# Shared flags (re-use from app.py pattern)
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
        help="Write the full DEBUG log to PATH.",
    ),
]
_NoColorOpt = Annotated[
    bool,
    typer.Option("--no-color", help="Disable coloured output."),
]
_QuietOpt = Annotated[
    bool,
    typer.Option("--quiet", help="Suppress info messages; warnings/errors still show."),
]
_JsonOpt = Annotated[
    bool,
    typer.Option(
        "--json",
        help="Emit a single machine-readable JSON verdict to stdout.",
    ),
]


def _outputs_exist(output_dir: Path, revision: str) -> bool:
    """Check if output files already exist."""
    return any(
        [
            (output_dir / f"mapcut.{revision}").exists(),
            (output_dir / f"cortdeco.{revision}").exists(),
            (output_dir / "export_manifest.json").exists(),
        ]
    )


@export_app.command("dessem-fcf")
def _export_dessem_fcf(
    case_dir: Annotated[
        Path, typer.Argument(help="Path to the solved Novomodelo case directory.")
    ],
    revision: Annotated[
        str,
        typer.Option(
            "--revision",
            "-r",
            help="PMO revision (e.g., rv0, rv1). Required.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory. Default: CASE_DIR/dessem_fcf/",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing outputs."),
    ] = False,
    hydro_codes: Annotated[
        Path | None,
        typer.Option(
            "--hydro-codes",
            metavar="PATH",
            help=(
                "Path to hydro codes JSON mapping file. "
                "Default: bundled decomp_hydro_codes.json"
            ),
        ),
    ] = None,
    include_terminal: Annotated[
        bool,
        typer.Option(
            "--include-terminal",
            help="Include terminal pool in ECO CSV output.",
        ),
    ] = False,
    json_output: _JsonOpt = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Export Novomodelo FCF to DESSEM-compatible mapcut/cortdeco format.

    Reads a Novomodelo policy checkpoint from a solved case directory and writes
    DECOMP-format mapcut and cortdeco binary files that DESSEM can consume.

    The --revision flag is required and determines the output file extension
    (mapcut.rv0, cortdeco.rv0, etc.).

    The --hydro-codes flag points to a JSON file mapping Novomodelo hydro positions
    to DECOMP plant codes. If not provided, uses the bundled default mapping.

    \b
    Warnings logged during export:
    - P3: travel-time (tempo de viagem) coefficients not yet supported
    - P14: GNL (anticipated thermal) coefficients emitted as zeros
    """
    _configure_logging(verbose, log_file)

    args = ExportDessemFcfArgs(
        case_dir=case_dir,
        revision=revision,
        output=output,
        force=force,
        hydro_codes=hydro_codes,
        include_terminal=include_terminal,
        json_output=json_output,
        verbose=verbose,
        log_file=log_file,
        no_color=no_color,
        quiet=quiet,
    )

    if not case_dir.is_dir():
        render_error(
            f"Case directory not found: {case_dir}", console=args.err_console()
        )
        raise typer.Exit(1)

    # Resolve hydro codes path: use bundled default if not provided
    if hydro_codes is not None:
        hydro_codes_path = hydro_codes
        if not hydro_codes_path.is_file():
            render_error(
                f"Hydro codes file not found: {hydro_codes_path}",
                console=args.err_console(),
            )
            raise typer.Exit(1)
        hydro_codes_str = str(hydro_codes_path)
    else:
        # Use bundled default
        hydro_codes_str = str(_DEFAULT_HYDRO_CODES)

    output_dir = output or (case_dir / "dessem_fcf")

    if not force and _outputs_exist(output_dir, revision):
        render_error(
            f"Outputs exist at {output_dir}. Use --force to overwrite.",
            console=args.err_console(),
        )
        raise typer.Exit(1)

    settings = ExportSettings(
        eco_subdirectory="eco",
        content_subdirectory="content",
        hydro_codes_path=hydro_codes_str,
        include_terminal_pool=include_terminal,
    )

    # Minimum delay per phase (seconds) to keep progress readable in interactive use.
    # Only applied when progress is visible (TTY, not quiet, not verbose).
    show_progress = sys.stderr.isatty() and not quiet and verbose == 0
    min_phase_delay = 0.12 if show_progress else 0.0

    def _paced_step(step_fn):
        """Wrap step callback with minimum delay for readability."""

        def wrapper(label: str) -> None:
            step_fn(label)
            if min_phase_delay > 0:
                time.sleep(min_phase_delay)

        return wrapper

    err_console = get_console(stderr=True, no_color=no_color)

    try:
        with dx.collect() as export_diags:
            with conversion_progress(
                len(EXPORT_PHASE_LABELS),
                verbose=verbose > 0,
                quiet=quiet,
                no_color=no_color,
            ) as step:
                exit_code = run_export(
                    case_dir=case_dir,
                    revision=revision,
                    output_dir=output_dir,
                    settings=settings,
                    on_phase=_paced_step(step),
                )

        if exit_code == 0:
            if not quiet:
                print_status(
                    f"Export completed: {output_dir}",
                    console=args.out_console(),
                )

        # Render diagnostics (warnings about P3, P14, etc.)
        render_diagnostics(
            export_diags,
            heading="Export notes",
            console=err_console,
            quiet=quiet,
        )

        raise typer.Exit(exit_code)

    except typer.Exit:
        raise
    except Exception as exc:
        render_error(f"Export failed: {exc}", console=args.err_console())
        raise typer.Exit(1) from exc

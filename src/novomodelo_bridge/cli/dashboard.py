"""Dashboard-command handler for ``dashboard``."""

from __future__ import annotations

import webbrowser
from pathlib import Path

from cobre_bridge.cli.args import DashboardArgs
from cobre_bridge.cli.failure import _emit_convert_json, _fail
from cobre_bridge.cli.verdict import build_verdict, dashboard_summary
from cobre_bridge.core.errors import CobreOutputError
from cobre_bridge.ui.console import (
    get_console,
    print_status,
    render_diagnostics,
    spinner,
)


def _run_dashboard(args: DashboardArgs) -> None:
    """Execute the dashboard subcommand."""
    from cobre_bridge.dashboard import build_dashboard

    case_dir: Path = args.case_dir.resolve()
    if not (case_dir / "output" / "simulation").exists():
        _fail(
            "dashboard",
            args,
            CobreOutputError(f"no simulation output found in {case_dir}"),
            1,
        )

    from cobre_bridge.core import diagnostics as dx

    output_path: Path = args.output or (case_dir / "dashboard.html")
    if not args.json_output and not args.quiet:
        print_status(
            f"Building dashboard from {case_dir} ...", console=args.out_console()
        )
    with dx.collect() as dash_diags:
        with spinner(
            "Building dashboard…",
            verbose=args.verbose > 0,
            quiet=args.quiet,
            no_color=args.no_color,
        ):
            build_dashboard(case_dir, output_path)
    size_kb = output_path.stat().st_size / 1024
    if not args.json_output:
        if not args.quiet:
            print_status(
                f"Dashboard written to {output_path} ({size_kb:.0f} KB)",
                console=args.out_console(),
            )
        render_diagnostics(
            dash_diags,
            heading="Dashboard notes",
            console=get_console(stderr=True),
            quiet=args.quiet,
        )

    if args.json_output:
        # --json: one machine-readable verdict to stdout. Emitted BEFORE the
        # --open block so the stdout JSON flushes regardless of browser outcome;
        # a built dashboard is always a success (the only failure path exits 1
        # before the build). ``size_kb`` keeps the precise float — only the
        # human line above rounds it with ``:.0f``.
        _emit_convert_json(
            build_verdict(
                "dashboard",
                "ok",
                dashboard_summary(str(output_path), size_kb),
                dash_diags,
            )
        )

    # The --open advisory stays off the stdout --json payload; it already goes to
    # stderr (err_console below), so the two compose cleanly.
    if args.open_browser:
        err_console = args.err_console()
        try:
            opened = webbrowser.open(output_path.resolve().as_uri())
        except (webbrowser.Error, OSError):
            opened = False
        if not opened:
            print_status(
                f"Note: could not open a browser for {output_path}; open it manually.",
                console=err_console,
                style="#F5A623",
            )

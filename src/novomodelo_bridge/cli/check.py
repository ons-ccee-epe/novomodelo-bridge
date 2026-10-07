"""Check-command handlers for ``check newave`` / ``check decomp``."""

from __future__ import annotations

import typer

from cobre_bridge.cli.args import CheckArgs
from cobre_bridge.cli.failure import _emit_convert_json
from cobre_bridge.cli.verdict import build_verdict, check_summary
from cobre_bridge.core.preflight import PreflightResult, PreflightVerdict
from cobre_bridge.ui.console import render_checklist

#: Preflight verdict → process exit code: ``OK`` is clean (0), ``WARNINGS`` is
#: advisory (1), and ``WILL_NOT_CONVERT`` is the most severe (2). Kept as data
#: so the mapping is directly unit-testable and impossible to drift from in the
#: handler.
_VERDICT_EXIT_CODE: dict[PreflightVerdict, int] = {
    PreflightVerdict.OK: 0,
    PreflightVerdict.WARNINGS: 1,
    PreflightVerdict.WILL_NOT_CONVERT: 2,
}


def _gate_check_exit(verdict: PreflightVerdict) -> None:
    """Preflight exit-code gate: ``OK`` (0) returns; 1/2 raise ``typer.Exit``."""
    exit_code = _VERDICT_EXIT_CODE[verdict]
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def _render_and_exit(args: CheckArgs, result: PreflightResult, command: str) -> None:
    """Render a preflight result and exit per its verdict.

    The one shared body for ``check newave`` and ``check decomp`` (same contract
    on both tracks). With ``--json`` it emits one machine-readable verdict to
    stdout — the checklist rides under ``summary`` and ``status`` passes the
    preflight verdict through verbatim, never recomputed — otherwise it renders
    the ✓/✗ checklist to stdout with diagnostics on stderr. Exits ``OK`` → 0,
    ``WARNINGS`` → 1, ``WILL_NOT_CONVERT`` → 2.
    """
    if args.json_output:
        summary = check_summary(
            [
                {"label": check.label, "passed": check.passed, "detail": check.detail}
                for check in result.checks
            ]
        )
        _emit_convert_json(
            build_verdict(command, result.verdict.value, summary, result.diagnostics)
        )
    else:
        render_checklist(
            result,
            console=args.out_console(),
            diagnostics_console=args.err_console(),
            quiet=args.quiet,
        )

    _gate_check_exit(result.verdict)


def _run_decomp_check(args: CheckArgs) -> None:
    """Execute the check decomp subcommand. Writes no files; runs no pipeline."""
    from cobre_bridge.decomp.preflight import run_decomp_preflight

    _render_and_exit(args, run_decomp_preflight(args.src), "check decomp")


def _run_check(args: CheckArgs) -> None:
    """Execute the check newave subcommand. Writes no files; runs no pipeline."""
    from cobre_bridge.newave.preflight import run_preflight

    _render_and_exit(args, run_preflight(args.src), "check newave")

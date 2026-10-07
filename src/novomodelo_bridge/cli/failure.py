"""Shared CLI failure-verdict helpers used by every command family."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, NoReturn

import typer

from cobre_bridge.cli.verdict import build_verdict
from cobre_bridge.core.errors import diagnostic_from_exception
from cobre_bridge.ui.console import render_diagnostics

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cobre_bridge.cli.args import CommonArgs


def _emit_convert_json(document: dict[str, object]) -> None:
    """Write the ``--json`` verdict *document* to stdout as one JSON object.

    Writes directly to ``sys.stdout`` (NOT through the Rich console, which may
    inject styling/wrapping), with a trailing newline. A fixed insertion order is
    preserved (``sort_keys=False``) so the output is byte-stable.
    """
    json.dump(document, sys.stdout, indent=2, ensure_ascii=False, sort_keys=False)
    sys.stdout.write("\n")


#: Operation named in a failure's heading, category and title, by command family.
_FAILURE_CONTEXT = {
    "convert": "Conversion",
    "compare": "Comparison",
    "dashboard": "Dashboard",
}


def _fail(
    command: str,
    args: CommonArgs,
    exc: Exception,
    code: int,
    *,
    summary: Mapping[str, object] | None = None,
) -> NoReturn:
    """Render + (under --json) emit one failure verdict, then Exit(code)."""
    context = _FAILURE_CONTEXT[command.split()[0]]
    diag = diagnostic_from_exception(exc, context=context)
    if args.json_output:
        _emit_convert_json(build_verdict(command, "error", summary or {}, [diag]))
    else:
        render_diagnostics(
            [diag],
            heading=f"{context} notes",
            console=args.err_console(),
            quiet=args.quiet,
        )
    raise typer.Exit(code=code)

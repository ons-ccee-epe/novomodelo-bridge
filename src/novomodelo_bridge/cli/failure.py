"""Shared CLI failure-verdict helpers used by every command family."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any, NoReturn

import typer
from typer.core import TyperGroup

from novomodelo_bridge.cli.verdict import build_verdict
from novomodelo_bridge.core.diagnostics import Diagnostic, Severity
from novomodelo_bridge.core.errors import diagnostic_from_exception
from novomodelo_bridge.ui.console import render_diagnostics

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typer._click import Context

    from novomodelo_bridge.cli.args import CommonArgs


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
    "export": "Export",
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


#: ``Context.meta`` key under which the root group records whether ``--json`` was passed.
_JSON_REQUESTED = "novomodelo_bridge.json_requested"


def _usage_failure(exc: typer.TyperException, root: str | None) -> NoReturn:
    """Emit the verdict for a usage error *exc*, then Exit with its code.

    The command is the path of subcommands down to the context that rejected the
    command line (``"convert newave"``); an error on the root itself, such as an
    unknown command, names *root*.
    """
    ctx = getattr(exc, "ctx", None)
    path: list[str] = []
    while ctx is not None and ctx.parent is not None:
        path.append(ctx.info_name)
        ctx = ctx.parent
    diag = Diagnostic(
        code="usage-error",
        severity=Severity.ERROR,
        category="Command line failure",
        title="Command line failed",
        summary=exc.format_message(),
        remediation="Run the command with --help to see its arguments and options.",
    )
    command = " ".join(reversed(path)) or root or ""
    _emit_convert_json(build_verdict(command, "error", {}, [diag]))
    raise typer.Exit(code=exc.exit_code)


class JsonUsageErrorGroup(TyperGroup):
    """Root group: under ``--json`` a usage error is a verdict envelope on stdout.

    Click rejects a bad command line (unknown option or command, missing
    argument, invalid value) before any handler runs, so no handler can emit the
    envelope the ``--json`` contract promises for every failure. The root group
    sees both phases: the parse of its own arguments (``make_context``) and the
    parse and run of every subcommand (``invoke``). Without ``--json`` the error
    is left to Click: usage text on stderr, exit 2.
    """

    def make_context(
        self,
        info_name: str | None,
        args: list[str],
        parent: Context | None = None,
        **extra: Any,
    ) -> Context:
        try:
            ctx = super().make_context(info_name, args, parent, **extra)
        except typer.TyperException as exc:
            if "--json" not in args:
                raise
            _usage_failure(exc, self.name)
        ctx.meta[_JSON_REQUESTED] = "--json" in args
        return ctx

    def invoke(self, ctx: Context) -> Any:
        try:
            return super().invoke(ctx)
        except typer.TyperException as exc:
            if not ctx.meta[_JSON_REQUESTED]:
                raise
            _usage_failure(exc, self.name)

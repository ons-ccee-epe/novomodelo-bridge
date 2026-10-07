"""Typed, frozen replacement for the CLI's per-command ``SimpleNamespace``.

Each Typer command builds one of these dataclasses instead of a duck-typed
``SimpleNamespace`` and hands it to its ``_run_*`` handler. Frozen construction
forbids the post-hoc mutation the untyped namespace allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cobre_bridge.ui.console import get_console

if TYPE_CHECKING:
    from rich.console import Console


@dataclass(frozen=True)
class CommonArgs:
    """The five flags every command accepts."""

    json_output: bool
    verbose: int
    log_file: Path | None
    no_color: bool
    quiet: bool

    def out_console(self) -> Console:
        """Build the stdout console for this invocation."""
        return get_console(no_color=self.no_color)

    def err_console(self) -> Console:
        """Build the stderr console for this invocation."""
        return get_console(stderr=True, no_color=self.no_color)


@dataclass(frozen=True)
class ConvertArgs(CommonArgs):
    """Options for ``convert newave`` / ``convert decomp``."""

    src: Path
    dst: Path
    validate: bool
    force: bool
    diagnostics_json: Path | None
    dry_run: bool
    no_fcf: bool = False


@dataclass(frozen=True)
class CompareArgs(CommonArgs):
    """Options for ``compare newave`` / ``compare decomp``.

    ``source_dir`` generalizes the per-command ``newave_dir``/``decomp_dir``
    argument to the name the comparison manifest already treats generically.
    """

    source_dir: Path
    cobre_output_dir: Path
    tolerance: float | None
    format: list[str] | None
    out_dir: Path | None


@dataclass(frozen=True)
class CheckArgs(CommonArgs):
    """Options for ``check newave`` / ``check decomp``."""

    src: Path


@dataclass(frozen=True)
class DashboardArgs(CommonArgs):
    """Options for ``dashboard``."""

    case_dir: Path
    output: Path | None
    open_browser: bool


#: The ``--format`` tokens the compare subcommands accept on the CLI.
_VALID_CLI_FORMATS: frozenset[str] = frozenset(
    {"console", "html", "csv", "parquet", "json", "all"}
)


def _parse_formats(raw: list[str] | None) -> set[str]:
    """Parse ``--format`` tokens (comma-separated and/or repeatable) into a set.

    Defaults to ``{"console", "parquet", "json"}`` when none given, so a plain
    ``compare`` run still writes the queryable data artifacts (the always-on
    behavior agents rely on). Expands ``"all"`` to every concrete format.
    Raises ``ValueError`` naming the offending token on any unknown value.
    """
    if raw is None:
        return {"console", "parquet", "json"}

    formats: set[str] = set()
    for element in raw:
        for token in element.split(","):
            cleaned = token.strip().lower()
            if not cleaned:
                continue
            if cleaned not in _VALID_CLI_FORMATS:
                msg = (
                    f"unknown format '{cleaned}'; allowed formats are "
                    f"{sorted(_VALID_CLI_FORMATS)}"
                )
                raise ValueError(msg)
            if cleaned == "all":
                formats |= {"console", "html", "csv", "parquet", "json"}
            else:
                formats.add(cleaned)
    return formats

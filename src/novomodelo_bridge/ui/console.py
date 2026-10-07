"""Rich terminal rendering for the cobre-bridge CLI.

The single place that turns plain data —
:class:`~cobre_bridge.core.diagnostics.Diagnostic` objects, a
:class:`~cobre_bridge.core.conversion.ConversionReport`, comparator summary tables —
into styled terminal output. Keeping all Rich usage here means the rest of the package
stays presentation-free and the styling (copper-accented palette from
:mod:`cobre_bridge.ui.theme`, severity colours, table boxes) is consistent across
commands.

Stream discipline mirrors the previous CLI contract that tests rely on: *primary
results* (the conversion summary, comparator tables, "written to …" lines) go to
**stdout**; *diagnostics, warnings and errors* go to **stderr**. Build the matching
console with :func:`get_console`.

Colour is opt-out: Rich auto-detects a non-TTY (no ANSI under ``capsys``) and honours
the ``NO_COLOR`` environment variable; ``--no-color`` forces it off explicitly. Plain
status/error lines keep their exact text (styling is a no-op when not a TTY) so they
stay greppable; only the genuinely tabular output changes shape.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

from rich.box import SIMPLE_HEAVY
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table
from rich.text import Text

from cobre_bridge.core.diagnostics import Diagnostic, Severity
from cobre_bridge.core.preflight import PreflightVerdict
from cobre_bridge.ui.theme import COPPER_ACCENT, SEVERITY

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from cobre_bridge.comparators.verdict import CompareVerdict
    from cobre_bridge.core.conversion import ConversionReport
    from cobre_bridge.core.preflight import PreflightResult

#: Width used when the output is not a real terminal (tests, pipes), so rendered
#: tables and panels are deterministic instead of wrapping at Rich's 80-col default.
DEFAULT_WIDTH = 100

#: Cap on detail-table rows actually printed; the overflow is summarised in a caption
#: so a case with hundreds of affected entities does not flood the terminal. The full
#: set is always available via ``--diagnostics-json``.
MAX_TABLE_ROWS = 20

#: Severity → Rich style. Amber/red track the dashboard palette; INFO is muted.
_SEVERITY_STYLE: dict[Severity, str] = {
    Severity.INFO: "dim",
    Severity.WARNING: SEVERITY["warning"],
    Severity.ERROR: f"bold {SEVERITY['error']}",
}

#: Severity → glyph prefix shown in panel titles.
_SEVERITY_GLYPH: dict[Severity, str] = {
    Severity.INFO: "ℹ",
    Severity.WARNING: "⚠",
    Severity.ERROR: "✖",
}

#: Success style reused for the ✓ summary banner and passing checklist lines.
_SUCCESS_STYLE = f"bold {SEVERITY['ok']}"

#: Verdict → (glyph, style, headline text) for the preflight checklist headline.
#: Reuses the existing severity glyphs/colours: ⚠/✖ and the amber/red styles
#: track :data:`_SEVERITY_GLYPH` / :data:`_SEVERITY_STYLE`, and ``✓`` + the green
#: style match :func:`render_conversion_summary`.
_VERDICT_HEADLINE: dict[PreflightVerdict, tuple[str, str, str]] = {
    PreflightVerdict.OK: ("✓", _SUCCESS_STYLE, "Ready to convert"),
    PreflightVerdict.WARNINGS: (
        _SEVERITY_GLYPH[Severity.WARNING],
        _SEVERITY_STYLE[Severity.WARNING],
        "Ready with warnings",
    ),
    PreflightVerdict.WILL_NOT_CONVERT: (
        _SEVERITY_GLYPH[Severity.ERROR],
        _SEVERITY_STYLE[Severity.ERROR],
        "Will not convert",
    ),
}


def get_console(
    *,
    stderr: bool = False,
    no_color: bool = False,
    width: int | None = None,
) -> Console:
    """Build a :class:`rich.console.Console` for stdout (default) or stderr.

    ``no_color=True`` forces colour off while keeping box drawing (the ``NO_COLOR``
    convention). When *width* is not given and the stream is not a TTY, a fixed
    :data:`DEFAULT_WIDTH` is used so non-interactive output (tests, pipes) is
    deterministic. ``highlight=False`` disables Rich's automatic literal
    highlighting so our text renders exactly as written.
    """
    stream = sys.stderr if stderr else sys.stdout
    kwargs: dict[str, object] = {"file": stream, "highlight": False, "emoji": False}
    if no_color:
        kwargs["no_color"] = True
    if width is not None:
        kwargs["width"] = width
    elif not stream.isatty():
        kwargs["width"] = DEFAULT_WIDTH
    return Console(**kwargs)


def print_status(
    message: str, *, console: Console | None = None, style: str | None = None
) -> None:
    """Print a single status line, preserving its exact text (no inserted wrapping)."""
    target = console or get_console()
    target.print(message, style=style, soft_wrap=True)


def render_error(message: str, *, console: Console | None = None) -> None:
    """Print an error line to stderr as ``Error: <message>`` (red on a TTY).

    The literal ``Error: `` prefix and the message text are preserved verbatim so
    existing stderr substring checks keep matching; colour is added only on a TTY.
    """
    target = console or get_console(stderr=True)
    target.print(f"Error: {message}", style=f"bold {SEVERITY['error']}", soft_wrap=True)


def make_table(
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    title: str | None = None,
    justify: Sequence[str] | None = None,
    caption: str | None = None,
    row_styles: Sequence[str | None] | None = None,
) -> Table:
    """Build a styled Rich table from columns + rows (cells coerced to ``str``).

    ``row_styles`` is an optional per-row Rich style aligned to *rows* by index:
    ``row_styles[i]`` styles ``rows[i]`` (a ``None`` entry, or an index past the
    end of a shorter ``row_styles``, leaves that row unstyled). The default of
    ``None`` is byte-identical to passing no row styles — colour is style-only and
    never changes cell text, column count, or justification.
    """
    table = Table(title=title, box=SIMPLE_HEAVY, caption=caption, title_style="bold")
    for index, name in enumerate(columns):
        align = (
            justify[index] if justify is not None and index < len(justify) else "left"
        )
        table.add_column(name, justify=align, no_wrap=False)
    for index, row in enumerate(rows):
        style = (
            row_styles[index]
            if row_styles is not None and index < len(row_styles)
            else None
        )
        table.add_row(*(_cell(value) for value in row), style=style)
    return table


def compare_row_style(*, within_tol: bool) -> str:
    """Return the per-row Rich style for a compare table row.

    Green (:data:`_SUCCESS_STYLE`) when the row is fully within tolerance,
    otherwise the red error style (:data:`_SEVERITY_STYLE`). Reuses the existing
    pass/fail palette so the compare summaries match the checklist colours — no
    new colour value is introduced.
    """
    return _SUCCESS_STYLE if within_tol else _SEVERITY_STYLE[Severity.ERROR]


def _cell(value: object) -> str:
    """Render a table cell value as text (``None`` shows as an em dash)."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def render_conversion_summary(
    report: ConversionReport, *, console: Console | None = None
) -> None:
    """Render a one-line converted-entity summary on stdout (green ✓ on a TTY).

    Kept as a single line — preserving the ``N hydros`` … wording — so it stays
    greppable and reads as a success banner rather than a table.
    """
    target = console or get_console()
    summary = (
        f"Converted {report.hydro_count} hydros · "
        f"{report.thermal_count} thermals · "
        f"{report.bus_count} buses · "
        f"{report.line_count} lines · "
        f"{report.stage_count} stages"
    )
    target.print(f"✓ {summary}", style=_SUCCESS_STYLE, soft_wrap=True)


def render_compare_verdict(
    verdict: CompareVerdict, *, console: Console | None = None
) -> None:
    """Render the one-line compare headline on stdout (green ✓ / amber ⚠ on a TTY).

    Leads ``compare results`` so the user gets the answer
    first: ``✓ N/M variables within tol`` when every variable is within tolerance,
    or ``⚠ N/M variables within tol — worst: <var> sMAPE <pct>%`` otherwise. An
    empty dataset renders ``⚠ no variables compared`` with no worst clause. The
    glyphs/styles reuse :data:`_SUCCESS_STYLE` and the amber
    :data:`_SEVERITY_STYLE` / :data:`_SEVERITY_GLYPH` (no new colour). As a primary
    result it goes to **stdout** (the given *console* or :func:`get_console`), with
    ``print_status``-style soft-wrap so the exact text stays greppable.
    """
    target = console or get_console()
    warn_glyph = _SEVERITY_GLYPH[Severity.WARNING]
    warn_style = _SEVERITY_STYLE[Severity.WARNING]
    if verdict.total == 0:
        target.print(
            f"{warn_glyph} no variables compared", style=warn_style, soft_wrap=True
        )
        return
    if verdict.all_within_tol:
        glyph, style = "✓", _SUCCESS_STYLE
    else:
        glyph, style = warn_glyph, warn_style
    line = f"{glyph} {verdict.within_tol}/{verdict.total} variables within tol"
    # The all_within_tol guard suppresses the meaningless "worst: … sMAPE 0%" on a
    # perfect match, keeping that line exactly "✓ N/N variables within tol".
    if verdict.worst_variable is not None and not verdict.all_within_tol:
        line += (
            f" — worst: {verdict.worst_variable} sMAPE {verdict.worst_smape * 100:.0f}%"
        )
    target.print(line, style=style, soft_wrap=True)


def render_checklist(
    result: PreflightResult,
    *,
    console: Console | None = None,
    diagnostics_console: Console | None = None,
    quiet: bool = False,
) -> None:
    """Render a :class:`~cobre_bridge.core.preflight.PreflightResult` as a ✓/✗
    checklist.

    The headline (``✓ Ready to convert`` / ``⚠ Ready with warnings`` / ``✖ Will not
    convert``) and one ✓/✗ line per :class:`~cobre_bridge.core.preflight.CheckItem`
    are the primary result of ``check`` and go to **stdout** (the given *console* or
    :func:`get_console`). The verdict is taken verbatim from ``result.verdict`` — it is
    never recomputed here.

    Each passing check renders as ``✓ <label>`` (green) and each failed check as
    ``✖ <label>`` (red); a non-empty ``detail`` is appended as ``— <detail>`` (dim).
    With *quiet* set, passing ``✓`` lines are suppressed but the headline and failed
    ``✖`` lines are still shown. The diagnostics are then delegated to
    :func:`render_diagnostics`, on *diagnostics_console* when given, so warnings/errors
    render through the same panels on a separate stream/colour setting than the
    headline; when *diagnostics_console* is ``None`` it falls back to *console* (and
    when both are ``None``, to :func:`render_diagnostics`'s own default stderr
    console) — the two-argument split lets a caller keep the stdout(results)/
    stderr(diagnostics) contract while making both streams ``--no-color``-aware.
    """
    target = console or get_console()

    glyph, style, headline = _VERDICT_HEADLINE[result.verdict]
    target.print(f"{glyph} {headline}", style=style, soft_wrap=True)

    for check in result.checks:
        if quiet and check.passed:
            continue
        item_glyph = "✓" if check.passed else _SEVERITY_GLYPH[Severity.ERROR]
        item_style = _SUCCESS_STYLE if check.passed else _SEVERITY_STYLE[Severity.ERROR]
        line = Text(f"{item_glyph} {check.label}", style=item_style)
        if check.detail:
            line.append(f" — {check.detail}", style="dim")
        target.print(line, soft_wrap=True)

    render_diagnostics(
        result.diagnostics,
        heading="Preflight notes",
        console=diagnostics_console or console,
        quiet=quiet,
    )


def render_diagnostics(
    diagnostics: Sequence[Diagnostic],
    *,
    heading: str,
    console: Console | None = None,
    quiet: bool = False,
) -> None:
    """Render diagnostics grouped by category, ordered INFO < WARNING < ERROR.

    A one-line roll-up under *heading* (``⚠ Conversion notes: 2 warning(s) · 1
    note(s)``) precedes the per-category panels.
    With *quiet* set, INFO-severity diagnostics are suppressed and only the roll-up
    plus warnings/errors are shown. Does nothing when there are no diagnostics.
    """
    if not diagnostics:
        return

    target = console or get_console(stderr=True)
    shown = [d for d in diagnostics if not (quiet and d.severity is Severity.INFO)]

    target.print(_rollup_line(diagnostics, heading))
    if not shown:
        return

    ordered = sorted(shown, key=lambda d: (d.severity.rank, d.category))
    last_category: str | None = None
    for diag in ordered:
        if diag.category != last_category:
            target.print()
            target.print(Text(diag.category, style="bold underline"))
            last_category = diag.category
        target.print(_diagnostic_panel(diag))


def _rollup_line(diagnostics: Sequence[Diagnostic], heading: str) -> Text:
    """Build the ``<heading>: N errors · N warnings · N notes`` header line."""
    counts = {sev: 0 for sev in Severity}
    for diag in diagnostics:
        counts[diag.severity] += 1
    parts: list[str] = []
    if counts[Severity.ERROR]:
        parts.append(f"{counts[Severity.ERROR]} error(s)")
    if counts[Severity.WARNING]:
        parts.append(f"{counts[Severity.WARNING]} warning(s)")
    if counts[Severity.INFO]:
        parts.append(f"{counts[Severity.INFO]} note(s)")
    glyph = _SEVERITY_GLYPH[
        Severity.ERROR
        if counts[Severity.ERROR]
        else Severity.WARNING
        if counts[Severity.WARNING]
        else Severity.INFO
    ]
    summary = " · ".join(parts) if parts else "no findings"
    return Text(f"{glyph} {heading}: {summary}", style="bold")


def _diagnostic_panel(diag: Diagnostic) -> Panel:
    """Render one diagnostic as a titled, severity-coloured panel."""
    style = _SEVERITY_STYLE[diag.severity]
    body = Table.grid(padding=(0, 0))
    body.add_column()
    body.add_row(Text(diag.summary))

    if diag.table is not None:
        rows = diag.table.rows
        capped = rows[:MAX_TABLE_ROWS]
        caption = diag.table.caption
        if len(rows) > MAX_TABLE_ROWS:
            overflow = len(rows) - MAX_TABLE_ROWS
            extra = f"… and {overflow} more (see --diagnostics-json)"
            caption = f"{caption}; {extra}" if caption else extra
        body.add_row("")
        body.add_row(
            make_table(
                diag.table.columns,
                capped,
                justify=diag.table.justify,
                caption=caption,
            )
        )

    for note in diag.notes:
        body.add_row(Text(f"• {note}", style="dim"))

    if diag.remediation:
        body.add_row(Text(f"→ {diag.remediation}", style=COPPER_ACCENT))

    title = f"{_SEVERITY_GLYPH[diag.severity]} {diag.title}"
    return Panel(
        body, title=title, title_align="left", border_style=style, expand=False
    )


# ---------------------------------------------------------------------------
# Live progress (TTY-only)
# ---------------------------------------------------------------------------


def _progress_enabled(*, verbose: bool, quiet: bool) -> bool:
    """Live progress shows only on an interactive stderr, off under --verbose/--quiet.

    Disabling it off-TTY keeps piped/captured output (tests, redirects) clean and
    deterministic, and ``--verbose`` would otherwise interleave DEBUG logs with the
    live region.
    """
    return sys.stderr.isatty() and not verbose and not quiet


@contextmanager
def conversion_progress(
    total_steps: int,
    *,
    verbose: bool = False,
    quiet: bool = False,
    no_color: bool = False,
) -> Iterator[Callable[[str], None]]:
    """Yield a ``step(label)`` callback that advances a transient progress bar.

    The bar (spinner + label + percentage) renders on stderr while a conversion
    runs; each ``step(label)`` updates the description and advances one tick. When
    progress is disabled (non-TTY / ``--verbose`` / ``--quiet``) the yielded
    callback is a no-op, so headless runs and tests are unaffected.
    """
    if not _progress_enabled(verbose=verbose, quiet=quiet):
        yield lambda _label: None
        return

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=get_console(stderr=True, no_color=no_color),
        transient=True,
    )
    with progress:
        task = progress.add_task("Converting", total=total_steps)

        def step(label: str) -> None:
            progress.update(task, description=label, advance=1)

        yield step


@contextmanager
def spinner(
    message: str,
    *,
    verbose: bool = False,
    quiet: bool = False,
    no_color: bool = False,
) -> Iterator[None]:
    """Show an indeterminate stderr spinner for the duration of a whole operation.

    No-op when progress is disabled (non-TTY / ``--verbose`` / ``--quiet``).
    """
    if not _progress_enabled(verbose=verbose, quiet=quiet):
        yield
        return
    with get_console(stderr=True, no_color=no_color).status(message, spinner="dots"):
        yield

"""Unit tests for the Rich rendering layer (``novomodelo_bridge.ui.console``)."""

from __future__ import annotations

import io

from rich.console import Console

from novomodelo_bridge.core.conversion import ConversionReport
from novomodelo_bridge.core.diagnostics import Diagnostic, DiagnosticTable, Severity
from novomodelo_bridge.core.preflight import (
    CheckItem,
    PreflightResult,
    PreflightVerdict,
)
from novomodelo_bridge.ui.console import (
    _SUCCESS_STYLE,
    MAX_TABLE_ROWS,
    _progress_enabled,
    compare_row_style,
    conversion_progress,
    make_table,
    render_checklist,
    render_conversion_summary,
    render_diagnostics,
    render_error,
    spinner,
)


def _console() -> tuple[Console, io.StringIO]:
    """A deterministic, colourless, fixed-width console writing to a buffer."""
    buf = io.StringIO()
    console = Console(file=buf, width=100, no_color=True, highlight=False, emoji=False)
    return console, buf


class TestConversionSummary:
    def test_summary_line_keeps_entity_counts(self) -> None:
        console, buf = _console()
        report = ConversionReport(
            hydro_count=10, thermal_count=5, bus_count=4, line_count=3, stage_count=60
        )
        render_conversion_summary(report, console=console)
        text = buf.getvalue()
        assert "10 hydros" in text
        assert "5 thermals" in text
        assert "60 stages" in text


class TestRenderError:
    def test_error_prefix_and_message_preserved(self) -> None:
        console, buf = _console()
        render_error("source directory 'x' does not exist", console=console)
        assert "Error: source directory 'x' does not exist" in buf.getvalue()


class TestRenderDiagnostics:
    def _gtmin(self) -> Diagnostic:
        return Diagnostic(
            code="thermal-gtmin-above-capacity",
            severity=Severity.WARNING,
            category="Thermal bounds",
            title="GTMIN exceeds capacity (1 plant)",
            summary="one plant affected",
            table=DiagnosticTable(
                columns=["Plant", "Code", "Stages", "GTMIN MW", "Cap MW"],
                rows=[["ANGRA 2", 13, "2-3", 481.3, 423.4]],
                justify=["left", "right", "left", "right", "right"],
            ),
            remediation="Check EXPT FCMAX/GTMIN and MANUTT for these plants.",
        )

    def test_empty_renders_nothing(self) -> None:
        console, buf = _console()
        render_diagnostics([], heading="Conversion notes", console=console)
        assert buf.getvalue() == ""

    def test_renders_title_table_cells_and_remediation(self) -> None:
        console, buf = _console()
        render_diagnostics([self._gtmin()], heading="Conversion notes", console=console)
        text = buf.getvalue()
        assert "GTMIN exceeds capacity" in text
        assert "ANGRA 2" in text  # resolved plant name, not just the code
        assert "2-3" in text  # stage range
        assert "481.3" in text  # GTMIN value
        assert "423.4" in text  # capacity value
        assert "Check EXPT FCMAX/GTMIN" in text  # remediation hint
        assert "→ Check EXPT" in text and "→ →" not in text  # one arrow
        assert "Thermal bounds" in text  # category header

    def test_rollup_counts_warnings_and_notes(self) -> None:
        console, buf = _console()
        note = Diagnostic(
            code="n",
            severity=Severity.INFO,
            category="Entity exclusion",
            title="Note",
            summary="a note",
        )
        render_diagnostics(
            [self._gtmin(), note], heading="Dashboard notes", console=console
        )
        text = buf.getvalue()
        assert "Dashboard notes: 1 warning(s) · 1 note(s)" in text
        assert "1 warning(s)" in text
        assert "1 note(s)" in text

    def test_quiet_suppresses_info_but_keeps_warnings(self) -> None:
        console, buf = _console()
        note = Diagnostic(
            code="n",
            severity=Severity.INFO,
            category="Entity exclusion",
            title="Fictitious note",
            summary="excluded plants",
        )
        render_diagnostics(
            [self._gtmin(), note],
            heading="Conversion notes",
            console=console,
            quiet=True,
        )
        text = buf.getvalue()
        assert "GTMIN exceeds capacity" in text
        assert "Fictitious note" not in text

    def test_long_table_is_capped_with_overflow_caption(self) -> None:
        console, buf = _console()
        rows = [[f"P{i}", i] for i in range(MAX_TABLE_ROWS + 5)]
        diag = Diagnostic(
            code="many",
            severity=Severity.INFO,
            category="Entity exclusion",
            title="Many plants",
            summary="lots",
            table=DiagnosticTable(columns=["Plant", "Code"], rows=rows),
        )
        render_diagnostics([diag], heading="Conversion notes", console=console)
        text = buf.getvalue()
        assert "5 more" in text  # overflow summarised
        assert "P0" in text
        assert f"P{MAX_TABLE_ROWS + 4}" not in text  # last row not printed


class TestRenderChecklist:
    def test_render_checklist_ok_headline_and_items(self) -> None:
        console, buf = _console()
        result = PreflightResult(
            verdict=PreflightVerdict.OK,
            diagnostics=[],
            checks=[
                CheckItem(label="Required files present", passed=True),
                CheckItem(label="Source path is a directory", passed=True),
                CheckItem(label="caso.dat resolved", passed=True),
            ],
        )
        render_checklist(result, console=console)
        text = buf.getvalue()
        assert "✓ Ready to convert" in text
        assert text.count("✓") == 4  # headline + three passing items

    def test_render_checklist_will_not_convert_shows_failed(self) -> None:
        console, buf = _console()
        result = PreflightResult(
            verdict=PreflightVerdict.WILL_NOT_CONVERT,
            diagnostics=[],
            checks=[
                CheckItem(
                    label="File discovery (caso.dat → arquivos.dat)",
                    passed=False,
                    detail="caso.dat not found",
                ),
            ],
        )
        render_checklist(result, console=console)
        text = buf.getvalue()
        assert "✖ Will not convert" in text
        assert "✖ File discovery (caso.dat → arquivos.dat)" in text
        assert "caso.dat not found" in text  # detail appended

    def test_render_checklist_quiet_suppresses_passing(self) -> None:
        console, buf = _console()
        result = PreflightResult(
            verdict=PreflightVerdict.OK,
            diagnostics=[],
            checks=[
                CheckItem(label="Required files present", passed=True),
                CheckItem(label="caso.dat resolved", passed=True),
            ],
        )
        render_checklist(result, console=console, quiet=True)
        text = buf.getvalue()
        assert "✓ Ready to convert" in text  # headline still shown
        assert "Required files present" not in text  # passing item suppressed
        assert "caso.dat resolved" not in text

    def test_render_checklist_delegates_diagnostics(self) -> None:
        console, buf = _console()
        warning = Diagnostic(
            code="optional-file-absent",
            severity=Severity.WARNING,
            category="Preflight",
            title="Optional input absent",
            summary="Optional input 'modif' was not found.",
        )
        result = PreflightResult(
            verdict=PreflightVerdict.WARNINGS,
            diagnostics=[warning],
            checks=[CheckItem(label="Required files present", passed=True)],
        )
        render_checklist(result, console=console)
        text = buf.getvalue()
        assert "⚠ Ready with warnings" in text  # warning headline
        assert "Optional input absent" in text  # delegated diagnostics panel
        assert "modif" in text

    def test_render_checklist_diagnostics_console_routes_separately(self) -> None:
        """``diagnostics_console`` sends diagnostics to a console DIFFERENT from
        the headline/checklist console (the Option C stdout/stderr split)."""
        headline_console, headline_buf = _console()
        diag_console, diag_buf = _console()
        warning = Diagnostic(
            code="optional-file-absent",
            severity=Severity.WARNING,
            category="Preflight",
            title="Optional input absent",
            summary="Optional input 'modif' was not found.",
        )
        result = PreflightResult(
            verdict=PreflightVerdict.WARNINGS,
            diagnostics=[warning],
            checks=[CheckItem(label="Required files present", passed=True)],
        )
        render_checklist(
            result, console=headline_console, diagnostics_console=diag_console
        )
        headline_text = headline_buf.getvalue()
        diag_text = diag_buf.getvalue()

        assert "⚠ Ready with warnings" in headline_text
        assert "Required files present" in headline_text
        assert "Optional input absent" not in headline_text  # not on this console

        assert "Optional input absent" in diag_text
        assert "⚠ Ready with warnings" not in diag_text  # headline stayed put

    def test_render_checklist_omitting_diagnostics_console_matches_console(
        self,
    ) -> None:
        """Omitting ``diagnostics_console`` keeps diagnostics on *console* —
        byte-identical to the pre-Option-C single-console call shape."""
        console, buf = _console()
        warning = Diagnostic(
            code="optional-file-absent",
            severity=Severity.WARNING,
            category="Preflight",
            title="Optional input absent",
            summary="Optional input 'modif' was not found.",
        )
        result = PreflightResult(
            verdict=PreflightVerdict.WARNINGS,
            diagnostics=[warning],
            checks=[CheckItem(label="Required files present", passed=True)],
        )
        render_checklist(result, console=console)
        text = buf.getvalue()
        assert "⚠ Ready with warnings" in text
        assert "Optional input absent" in text  # same console, no split


class TestMakeTableRowStyles:
    """``make_table``'s optional per-row style is style-only and tolerant."""

    @staticmethod
    def _ansi(table_rows: list[list[object]], styles: list[str | None]) -> str:
        """Render a table to a forced-terminal, truecolor buffer and return ANSI."""
        buf = io.StringIO()
        console = Console(
            file=buf,
            force_terminal=True,
            color_system="truecolor",
            width=40,
            highlight=False,
            emoji=False,
        )
        console.print(make_table(["A"], table_rows, row_styles=styles))
        return buf.getvalue()

    def test_make_table_row_styles_applies_per_row_style(self) -> None:
        table = make_table(["A"], [["x"], ["y"]], row_styles=[_SUCCESS_STYLE, None])
        # Direct assertion: the Table carries the style on the styled row only.
        assert table.rows[0].style == _SUCCESS_STYLE
        assert table.rows[1].style is None

        # ANSI assertion: the green truecolor escape (74;139;111 == #4A8B6F) wraps
        # the first row's cell text and is absent from the second.
        ansi = self._ansi([["x"], ["y"]], [_SUCCESS_STYLE, None])
        assert "x" in ansi  # cell text unchanged
        assert "y" in ansi
        # The green #4A8B6F truecolor escape (38;2;74;139;111) wraps the styled
        # row's cell glyph and is wholly absent from the unstyled row.
        assert "38;2;74;139;111mx" in ansi  # green row carries the colour
        assert "38;2;74;139;111my" not in ansi  # unstyled row does not

    def test_make_table_row_styles_none_is_backward_compatible(self) -> None:
        columns = ["A", "B"]
        rows: list[list[object]] = [["x", 1], ["y", 2]]

        def _render(table: object) -> str:
            buf = io.StringIO()
            console = Console(
                file=buf, width=100, no_color=True, highlight=False, emoji=False
            )
            console.print(table)
            return buf.getvalue()

        default = _render(make_table(columns, rows))
        explicit_none = _render(make_table(columns, rows, row_styles=None))
        assert default == explicit_none

    def test_make_table_row_styles_shorter_than_rows_no_error(self) -> None:
        # row_styles shorter than rows: trailing rows are unstyled, no IndexError.
        table = make_table(["A"], [["x"], ["y"], ["z"]], row_styles=[_SUCCESS_STYLE])
        assert table.rows[0].style == _SUCCESS_STYLE
        assert table.rows[1].style is None
        assert table.rows[2].style is None


class TestCompareRowStyle:
    """``compare_row_style`` returns the existing green/red palette, no new colour."""

    def test_within_tol_returns_success_style(self) -> None:
        assert compare_row_style(within_tol=True) == _SUCCESS_STYLE

    def test_not_within_tol_returns_error_style(self) -> None:
        from novomodelo_bridge.core.diagnostics import Severity
        from novomodelo_bridge.ui.console import _SEVERITY_STYLE

        assert compare_row_style(within_tol=False) == _SEVERITY_STYLE[Severity.ERROR]


class TestProgressGating:
    def test_enabled_only_on_interactive_non_verbose_non_quiet(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr("sys.stderr.isatty", lambda: True, raising=False)
        assert _progress_enabled(verbose=False, quiet=False) is True
        assert _progress_enabled(verbose=True, quiet=False) is False
        assert _progress_enabled(verbose=False, quiet=True) is False

    def test_disabled_off_tty(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.stderr.isatty", lambda: False, raising=False)
        assert _progress_enabled(verbose=False, quiet=False) is False

    def test_conversion_progress_noop_off_tty(self, capsys) -> None:
        # Under pytest stderr is captured (not a TTY): the bar is suppressed and the
        # yielded callback is a harmless no-op, so output stays deterministic.
        with conversion_progress(6) as step:
            step("Discovering files")
            step("Writing Parquet")
        assert capsys.readouterr().err == ""

    def test_spinner_noop_off_tty(self, capsys) -> None:
        with spinner("Comparing…"):
            pass
        assert capsys.readouterr().err == ""

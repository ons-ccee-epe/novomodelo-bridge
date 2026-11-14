"""Unit tests for the structured-diagnostics model (``novomodelo_bridge.core.diagnostics``)."""

from __future__ import annotations

import logging
from pathlib import Path

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    collect,
    counted,
    emit,
    exception_reason,
    finalize_diagnostics,
    format_stage_ranges,
    merge_tables,
    report_file_missing_diagnostic,
    report_file_unreadable_diagnostic,
    report_files_unreadable_diagnostic,
    unreadable_file_diagnostic,
)
from novomodelo_bridge.core.errors import BridgeError


class TestSeverity:
    def test_rank_orders_info_warning_error(self) -> None:
        assert Severity.INFO.rank < Severity.WARNING.rank < Severity.ERROR.rank

    def test_log_level_maps_to_logging_levels(self) -> None:
        assert Severity.INFO.log_level == logging.INFO
        assert Severity.WARNING.log_level == logging.WARNING
        assert Severity.ERROR.log_level == logging.ERROR


class TestFormatStageRanges:
    def test_collapses_consecutive_runs(self) -> None:
        assert format_stage_ranges([4, 5, 6, 7, 19]) == "4-7, 19"

    def test_dedupes_and_sorts(self) -> None:
        assert format_stage_ranges([3, 1, 2, 2, 1]) == "1-3"

    def test_singletons_are_not_ranged(self) -> None:
        assert format_stage_ranges([1, 3, 5]) == "1, 3, 5"

    def test_empty_returns_empty_string(self) -> None:
        assert format_stage_ranges([]) == ""


class TestCollectAndEmit:
    def _diag(self, severity: Severity = Severity.WARNING) -> Diagnostic:
        return Diagnostic(
            code="x",
            severity=severity,
            category="Cat",
            title="Title",
            summary="something happened",
        )

    def test_emit_inside_collect_appends_to_sink(self) -> None:
        with collect() as sink:
            emit(self._diag())
            emit(self._diag(Severity.INFO))
        assert len(sink) == 2
        assert [d.severity for d in sink] == [Severity.WARNING, Severity.INFO]

    def test_emit_inside_collect_does_not_log(self, caplog) -> None:
        logger = logging.getLogger("novomodelo_bridge.newave.converters.fake")
        with caplog.at_level(
            logging.INFO, logger="novomodelo_bridge.newave.converters.fake"
        ):
            with collect():
                emit(self._diag(), logger=logger)
        assert caplog.records == []

    def test_emit_without_sink_logs_on_caller_logger(self, caplog) -> None:
        logger = logging.getLogger("novomodelo_bridge.newave.converters.fake")
        with caplog.at_level(
            logging.WARNING, logger="novomodelo_bridge.newave.converters.fake"
        ):
            emit(self._diag(), logger=logger)
        assert "something happened" in caplog.text
        assert caplog.records[0].levelno == logging.WARNING

    def test_collect_restores_previous_sink(self) -> None:
        with collect() as outer:
            emit(self._diag())
            with collect() as inner:
                emit(self._diag(Severity.ERROR))
            emit(self._diag(Severity.INFO))
        assert len(inner) == 1
        assert len(outer) == 2


class TestToDict:
    def test_diagnostic_to_dict_round_trips_fields(self) -> None:
        diag = Diagnostic(
            code="thermal-gtmin-above-capacity",
            severity=Severity.WARNING,
            category="Thermal bounds",
            title="GTMIN exceeds capacity",
            summary="one plant affected",
            table=DiagnosticTable(
                columns=["Plant", "GTMIN MW"],
                rows=[["ANGRA 2", 1350.0]],
                justify=["left", "right"],
                caption="worst case",
            ),
            notes=["a note"],
            remediation="check EXPT",
        )
        payload = diag.to_dict()
        assert payload["code"] == "thermal-gtmin-above-capacity"
        assert payload["severity"] == "warning"
        table = payload["table"]
        assert isinstance(table, dict)
        assert table["columns"] == ["Plant", "GTMIN MW"]
        assert table["rows"] == [["ANGRA 2", 1350.0]]
        assert payload["notes"] == ["a note"]
        assert payload["remediation"] == "check EXPT"

    def test_table_none_serialises_to_none(self) -> None:
        diag = Diagnostic(
            code="c",
            severity=Severity.INFO,
            category="Cat",
            title="t",
            summary="s",
        )
        assert diag.to_dict()["table"] is None


class TestFinalizeDiagnostics:
    def _diag(self, code: str = "x", summary: str = "something happened") -> Diagnostic:
        return Diagnostic(
            code=code,
            severity=Severity.WARNING,
            category="Cat",
            title="Title",
            summary=summary,
        )

    def test_dedups_diagnostics_by_code_and_summary(self) -> None:
        assert len(finalize_diagnostics([self._diag(), self._diag()])) == 1

    def test_keeps_diagnostics_that_differ_in_code_or_summary(self) -> None:
        result = finalize_diagnostics(
            [self._diag(code="a"), self._diag(code="b"), self._diag(summary="other")]
        )
        assert len(result) == 3

    def test_keeps_the_first_occurrence_in_first_met_order(self) -> None:
        first = self._diag(code="a")
        result = finalize_diagnostics(
            [first, self._diag(code="b"), self._diag(code="a"), self._diag(code="c")]
        )
        assert [d.code for d in result] == ["a", "b", "c"]
        assert result[0] is first


class TestCounted:
    def test_singular_for_one(self) -> None:
        assert counted(1, "plant") == "1 plant"

    def test_regular_plural_otherwise(self) -> None:
        assert counted(0, "plant") == "0 plants"
        assert counted(3, "plant") == "3 plants"

    def test_irregular_plural(self) -> None:
        assert counted(2, "series", "series") == "2 series"


class TestMergeTables:
    @staticmethod
    def _row(code: str, row: list[object], title: str = "Finding") -> Diagnostic:
        return Diagnostic(
            code=code,
            severity=Severity.WARNING,
            category="c",
            title=title,
            summary="The listed plants.",
            table=DiagnosticTable(columns=["Code"], rows=[row]),
        )

    def test_one_diagnostic_per_code_with_distinct_rows_and_count(self) -> None:
        merged = merge_tables(
            [self._row("a", [1]), self._row("a", [2]), self._row("a", [1])]
        )
        assert len(merged) == 1
        assert merged[0].title == "Finding (2)"
        assert merged[0].table is not None
        assert merged[0].table.rows == [[1], [2]]

    def test_keeps_first_appearance_order_across_codes(self) -> None:
        plain = Diagnostic(
            code="p", severity=Severity.INFO, category="c", title="T", summary="S."
        )
        merged = merge_tables(
            [self._row("a", [1]), plain, self._row("b", [5]), self._row("a", [2])]
        )
        assert [d.code for d in merged] == ["a", "p", "b"]
        assert merged[1] is plain
        assert merged[0].table is not None
        assert merged[0].table.rows == [[1], [2]]

    def test_a_title_that_already_counts_its_rows_is_recounted(self) -> None:
        merged = merge_tables(
            [self._row("a", [1], "Finding (1)"), self._row("a", [2], "Finding (1)")]
        )
        assert merged[0].title == "Finding (2)"


class TestUnreadableFileDiagnostic:
    def test_names_the_file_and_keeps_the_parser_error_in_the_notes(self) -> None:
        diagnostic = unreadable_file_diagnostic(
            code="x-unreadable",
            file_name="x.dat",
            category="c",
            effect="the case has none of its records.",
            exc=ValueError("bad line 3"),
        )
        assert diagnostic.severity is Severity.WARNING
        assert diagnostic.title == "x.dat could not be read"
        assert diagnostic.summary == (
            "x.dat could not be read, so the case has none of its records."
        )
        assert diagnostic.notes == ["Reason: ValueError: bad line 3"]
        assert diagnostic.remediation == "Check x.dat for a malformed record."

    def test_an_exception_without_text_is_named_by_its_type(self) -> None:
        diagnostic = unreadable_file_diagnostic(
            code="x-unreadable",
            file_name="x.dat",
            category="c",
            effect="nothing.",
            exc=ValueError(),
        )
        assert diagnostic.notes == ["Reason: ValueError"]


class TestReportFileDiagnostics:
    def test_a_missing_file_is_a_note_naming_what_the_report_leaves_out(self) -> None:
        diagnostic = report_file_missing_diagnostic(
            "pmo.dat", Path("/deck/pmo.dat"), "the report has no convergence."
        )
        assert diagnostic.code == "report-file-missing"
        assert diagnostic.severity is Severity.INFO
        assert diagnostic.category == "Case and result files"
        assert diagnostic.title == "pmo.dat not found"
        assert diagnostic.summary == (
            "pmo.dat was not found, so the report has no convergence."
        )
        assert diagnostic.notes == ["Path: /deck/pmo.dat"]

    def test_an_unreadable_file_is_a_warning_with_its_reason(self) -> None:
        diagnostic = report_file_unreadable_diagnostic(
            "system/hydros.json",
            Path("/case/system/hydros.json"),
            "the report names no plant.",
            ValueError("Expecting value"),
        )
        assert diagnostic.code == "report-file-unreadable"
        assert diagnostic.severity is Severity.WARNING
        assert diagnostic.title == "system/hydros.json could not be read"
        assert diagnostic.summary == (
            "system/hydros.json could not be read, so the report names no plant."
        )
        assert diagnostic.notes == [
            "Path: /case/system/hydros.json",
            "Reason: ValueError: Expecting value",
        ]

    def test_an_empty_reason_adds_no_reason_note(self) -> None:
        diagnostic = report_file_unreadable_diagnostic(
            "a.json", Path("/a.json"), "nothing.", ""
        )
        assert diagnostic.notes == ["Path: /a.json"]

    def test_a_stated_reason_is_kept_as_written(self) -> None:
        diagnostic = report_file_unreadable_diagnostic(
            "a.json", Path("/a.json"), "nothing.", "no stage with block hours"
        )
        assert diagnostic.notes == [
            "Path: /a.json",
            "Reason: no stage with block hours",
        ]

    def test_unreadable_files_of_one_kind_are_one_warning_with_a_row_each(
        self,
    ) -> None:
        diagnostic = report_files_unreadable_diagnostic(
            "Interchange tables",
            [("int001.out", ValueError("bad")), ("int002.out", OSError("gone"))],
            "the report has no interchange.",
        )
        assert diagnostic.code == "report-file-unreadable"
        assert diagnostic.severity is Severity.WARNING
        assert diagnostic.title == "Interchange tables that could not be read (2)"
        assert diagnostic.summary == (
            "The listed files could not be read, so the report has no interchange."
        )
        assert diagnostic.table is not None
        assert diagnostic.table.columns == ["File", "Reason"]
        assert diagnostic.table.rows == [
            ["int001.out", "ValueError: bad"],
            ["int002.out", "OSError: gone"],
        ]


class TestExceptionReason:
    def test_a_bridge_message_is_kept_as_written(self) -> None:
        assert exception_reason(BridgeError("pmo.dat lacks a CMO table")) == (
            "pmo.dat lacks a CMO table"
        )

    def test_another_exception_is_led_by_its_type(self) -> None:
        assert exception_reason(KeyError("hydro_id")) == "KeyError: 'hydro_id'"

    def test_an_exception_without_text_is_its_type(self) -> None:
        assert exception_reason(OSError()) == "OSError"

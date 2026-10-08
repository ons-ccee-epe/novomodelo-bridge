"""Unit tests for the structured-diagnostics model (``novomodelo_bridge.core.diagnostics``)."""

from __future__ import annotations

import logging

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    WarningCollector,
    collect,
    emit,
    finalize_diagnostics,
    format_stage_ranges,
)


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
        assert payload["table"]["columns"] == ["Plant", "GTMIN MW"]
        assert payload["table"]["rows"] == [["ANGRA 2", 1350.0]]
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


class TestWarningCollector:
    def test_captures_warning_and_above(self) -> None:
        logger = logging.getLogger("novomodelo_bridge.core.diagnostics.fake")
        collector = WarningCollector()
        logger.addHandler(collector)
        try:
            logger.warning("degraded input A")
            logger.error("degraded input B")
        finally:
            logger.removeHandler(collector)
        assert collector.messages == ["degraded input A", "degraded input B"]

    def test_ignores_info_and_below(self) -> None:
        logger = logging.getLogger("novomodelo_bridge.core.diagnostics.fake2")
        logger.setLevel(logging.DEBUG)
        collector = WarningCollector()
        logger.addHandler(collector)
        try:
            logger.info("not a degradation")
            logger.debug("also not a degradation")
        finally:
            logger.removeHandler(collector)
        assert collector.messages == []


class TestFinalizeDiagnostics:
    def _diag(self, code: str = "x", summary: str = "something happened") -> Diagnostic:
        return Diagnostic(
            code=code,
            severity=Severity.WARNING,
            category="Cat",
            title="Title",
            summary=summary,
        )

    def test_dedups_structured_diagnostics_by_code_and_summary(self) -> None:
        result = finalize_diagnostics([self._diag(), self._diag()], [])
        assert len(result) == 1

    def test_keeps_diagnostics_that_differ_in_code_or_summary(self) -> None:
        result = finalize_diagnostics(
            [self._diag(code="a"), self._diag(code="b"), self._diag(summary="other")],
            [],
        )
        assert len(result) == 3

    def test_wraps_legacy_strings_as_warning_diagnostics(self) -> None:
        result = finalize_diagnostics([], ["legacy warning one"])
        assert len(result) == 1
        assert result[0].code == "legacy-warning"
        assert result[0].severity is Severity.WARNING
        assert result[0].summary == "legacy warning one"

    def test_dedups_legacy_strings_and_preserves_first_occurrence_order(self) -> None:
        result = finalize_diagnostics(
            [], ["first", "second", "first", "third", "second"]
        )
        assert [d.summary for d in result] == ["first", "second", "third"]

    def test_structured_diagnostics_precede_legacy_strings(self) -> None:
        result = finalize_diagnostics([self._diag()], ["legacy"])
        assert result[0].code == "x"
        assert result[1].code == "legacy-warning"

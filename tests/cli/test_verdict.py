"""Unit tests for the unified ``--json`` verdict envelope (``novomodelo_bridge.cli.verdict``)."""

from __future__ import annotations

import pytest

from novomodelo_bridge.cli.verdict import (
    SCHEMA_VERSION,
    _coerce_unmapped_code,
    build_verdict,
    check_summary,
    compare_summary,
    convert_summary,
    dashboard_summary,
    decomp_dataset_summary,
)
from novomodelo_bridge.comparators.verdict import CompareVerdict
from novomodelo_bridge.core.diagnostics import Diagnostic, Severity
from tests.conftest import _empty_fake_dataset, _fake_dataset


def _info_diagnostic() -> Diagnostic:
    """A minimal INFO diagnostic for exercising the verdict builders."""
    return Diagnostic(
        code="some-info",
        severity=Severity.INFO,
        category="Conversion",
        title="An info",
        summary="just so",
    )


class TestBuildVerdict:
    def test_build_verdict_key_order_and_schema_version(self) -> None:
        doc = build_verdict("convert newave", "ok", {"hydros": 1})

        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["schema_version"] == 1
        assert doc["schema_version"] == SCHEMA_VERSION
        assert doc["command"] == "convert newave"
        assert doc["status"] == "ok"
        assert doc["summary"] == {"hydros": 1}
        # Empty diagnostics (the default empty tuple) serializes to ``[]``.
        assert doc["diagnostics"] == []

    def test_build_verdict_serializes_diagnostics(self) -> None:
        diag = _info_diagnostic()

        doc = build_verdict("convert newave", "error", {}, [diag])

        assert doc["diagnostics"] == [diag.to_dict()]
        # The builder copies the passed mapping and injects no keys.
        assert doc["summary"] == {}

    def test_build_verdict_copies_summary(self) -> None:
        summary = convert_summary(1, 1, 1, 1, 1)

        doc = build_verdict("convert newave", "ok", summary)
        returned_summary = doc["summary"]
        assert isinstance(returned_summary, dict)
        returned_summary["hydros"] = 999

        # Mutating the returned summary must not touch the caller's mapping.
        assert summary["hydros"] == 1


class TestSummaryHelpers:
    def test_convert_summary_key_order(self) -> None:
        result = convert_summary(10, 5, 4, 3, 60)

        assert list(result.keys()) == [
            "hydros",
            "thermals",
            "buses",
            "lines",
            "stages",
        ]
        assert result == {
            "hydros": 10,
            "thermals": 5,
            "buses": 4,
            "lines": 3,
            "stages": 60,
        }

    def test_check_summary_wraps_checks(self) -> None:
        rows = [
            {"label": "caso.dat present", "passed": True, "detail": "found"},
            {"label": "hidr.dat present", "passed": False, "detail": "missing"},
        ]

        result = check_summary(rows)

        assert list(result.keys()) == ["checks"]
        assert result["checks"] == rows
        # Each row is copied, not aliased: mutating the source leaves the result.
        rows[0]["passed"] = False
        wrapped = result["checks"]
        assert isinstance(wrapped, list)
        assert wrapped[0]["passed"] is True

    def test_dashboard_summary_shape(self) -> None:
        result = dashboard_summary("dashboard.html", 12.5)

        assert list(result.keys()) == ["output", "size_kb"]
        assert result["output"] == "dashboard.html"
        assert result["size_kb"] == 12.5
        assert isinstance(result["output"], str)
        assert isinstance(result["size_kb"], float)


class TestCompareSummary:
    def test_compare_summary_nulls_worst_when_all_within_tol(self) -> None:
        verdict = CompareVerdict(
            within_tol=2,
            total=2,
            worst_variable="storage",
            worst_smape=0.04,
            all_within_tol=True,
        )

        result = compare_summary(verdict)

        assert list(result.keys()) == [
            "within_tol",
            "total",
            "worst_variable",
            "worst_smape",
            "all_within_tol",
        ]
        assert result["all_within_tol"] is True
        assert result["worst_variable"] is None
        assert result["worst_smape"] == 0.0

    def test_compare_summary_passes_worst_when_divergent(self) -> None:
        verdict = CompareVerdict(
            within_tol=1,
            total=2,
            worst_variable="storage",
            worst_smape=0.04,
            all_within_tol=False,
        )

        result = compare_summary(verdict)

        assert result["all_within_tol"] is False
        assert result["worst_variable"] == "storage"
        assert result["worst_smape"] == pytest.approx(0.04)


class TestCoerceUnmappedCode:
    """`decomp_dataset_summary`'s `unmapped` coercion — regression for the
    network level's `[de, para]` corridor pairs (which crashed `int([...])`)."""

    def test_scalar_code_coerced_to_int(self) -> None:
        assert _coerce_unmapped_code(86) == 86
        assert isinstance(_coerce_unmapped_code(224), int)

    def test_corridor_pair_kept_as_list_of_ints(self) -> None:
        # A `line`-level unmapped entry is a `[submarket_de, submarket_para]`
        # pair, not a scalar code — it must survive as a list, not raise.
        assert _coerce_unmapped_code([-1, 1]) == [-1, 1]
        assert _coerce_unmapped_code((2, -1)) == [2, -1]


class TestDecompDatasetSummary:
    """``decomp_dataset_summary`` -- the ``compare decomp --json`` payload
    builder, sourced entirely from the canonical ``ComparisonDataset``."""

    def test_returns_the_superset_shape_in_the_documented_key_order(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert list(summary.keys()) == [
            "within_tol",
            "total",
            "worst_variable",
            "worst_smape",
            "all_within_tol",
            "stages",
            "variables",
            "unmapped",
        ]

    def test_headline_fields_match_build_compare_verdict(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["within_tol"] == 1
        assert summary["total"] == 2
        assert summary["worst_variable"] == "turbined_m3s"
        assert summary["worst_smape"] == pytest.approx(0.12)
        assert summary["all_within_tol"] is False

    def test_stages_counts_distinct_tidy_stage_values(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["stages"] == 2

    def test_variables_is_the_dataset_summary_verbatim(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["variables"] == dataset.summary.to_dicts()

    def test_unmapped_is_sourced_from_dataset_metadata_with_int_codes(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["unmapped"] == {"hydro": [], "thermal": [86, 224], "bus": []}
        assert all(
            isinstance(code, int)
            for codes in summary["unmapped"].values()
            for code in codes
        )

    def test_all_within_tol_dataset_reports_no_worst_clause(self) -> None:
        dataset = _fake_dataset(all_within_tol=True)

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["within_tol"] == 2
        assert summary["total"] == 2
        assert summary["all_within_tol"] is True
        assert summary["worst_variable"] is None
        assert summary["worst_smape"] == 0.0

    def test_empty_dataset_reports_zero_totals_and_no_stages(self) -> None:
        dataset = _empty_fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["within_tol"] == 0
        assert summary["total"] == 0
        assert summary["all_within_tol"] is False
        assert summary["stages"] == 0
        assert summary["variables"] == []
        assert summary["unmapped"] == {"hydro": [], "thermal": [], "bus": []}

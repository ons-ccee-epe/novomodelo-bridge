"""Unit tests for the unified ``--json`` verdict envelope (``novomodelo_bridge.cli.verdict``)."""

from __future__ import annotations

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
from novomodelo_bridge.comparators.verdict import CompareVerdict, FarVariable
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
        assert doc["schema_version"] == 2
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
    def test_compare_summary_key_order_and_counts(self) -> None:
        verdict = CompareVerdict(
            tolerance=0.05,
            close=3,
            moderate=1,
            far=2,
            far_variables=(
                FarVariable("thermal", "generation_mw", None),
                FarVariable("bus", "spot_price", 0.4),
            ),
        )

        result = compare_summary(verdict)

        assert list(result.keys()) == [
            "tolerance",
            "total",
            "close",
            "moderate",
            "far",
            "far_variables",
        ]
        assert result["tolerance"] == 0.05
        assert result["total"] == 6
        assert (result["close"], result["moderate"], result["far"]) == (3, 1, 2)
        assert result["far_variables"] == [
            {"entity_type": "thermal", "variable": "generation_mw", "gap": None},
            {"entity_type": "bus", "variable": "spot_price", "gap": 0.4},
        ]

    def test_compare_summary_all_close_lists_no_far_variable(self) -> None:
        verdict = CompareVerdict(
            tolerance=0.05, close=2, moderate=0, far=0, far_variables=()
        )

        result = compare_summary(verdict)

        assert result["total"] == result["close"] == 2
        assert result["far_variables"] == []


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

        summary = decomp_dataset_summary(dataset)

        assert list(summary.keys()) == [
            "tolerance",
            "total",
            "close",
            "moderate",
            "far",
            "far_variables",
            "stages",
            "variables",
            "unmapped",
        ]

    def test_headline_fields_match_build_compare_verdict(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset)

        assert summary["tolerance"] == 0.05
        assert summary["total"] == 2
        assert (summary["close"], summary["moderate"], summary["far"]) == (1, 0, 1)
        assert summary["far_variables"] == [
            {"entity_type": "hydro", "variable": "turbined_m3s", "gap": 0.4}
        ]

    def test_stages_counts_distinct_tidy_stage_values(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset)

        assert summary["stages"] == 2

    def test_variables_is_the_dataset_summary_verbatim(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset)

        assert summary["variables"] == dataset.summary.to_dicts()

    def test_unmapped_is_sourced_from_dataset_metadata_with_int_codes(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset)

        unmapped = summary["unmapped"]
        assert unmapped == {"hydro": [], "thermal": [86, 224], "bus": []}
        assert isinstance(unmapped, dict)
        assert all(
            isinstance(code, int) for codes in unmapped.values() for code in codes
        )

    def test_all_close_dataset_lists_no_far_variable(self) -> None:
        dataset = _fake_dataset(all_close=True)

        summary = decomp_dataset_summary(dataset)

        assert summary["close"] == summary["total"] == 2
        assert summary["far_variables"] == []

    def test_empty_dataset_reports_zero_totals_and_no_stages(self) -> None:
        dataset = _empty_fake_dataset()

        summary = decomp_dataset_summary(dataset)

        assert summary["close"] == 0
        assert summary["total"] == 0
        assert summary["stages"] == 0
        assert summary["variables"] == []
        assert summary["unmapped"] == {"hydro": [], "thermal": [], "bus": []}

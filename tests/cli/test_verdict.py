"""Unit tests for the unified ``--json`` verdict envelope (``novomodelo_bridge.cli.verdict``)."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

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
    version_dataset_summary,
)
from novomodelo_bridge.comparators.dataset import ComparisonDataset
from novomodelo_bridge.comparators.verdict import (
    CompareVerdict,
    FarVariable,
    build_compare_verdict,
)
from novomodelo_bridge.comparators.version.results import build_version_dataset
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Diagnostic, Severity
from tests.conftest import _empty_fake_dataset, _fake_dataset
from tests.novomodelo_output_builders import SolvedCaseSpec, build_solved_case


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
                FarVariable("hydro", "spillage_m3s", 0.4, 0.1),
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
            {
                "entity_type": "thermal",
                "variable": "generation_mw",
                "gap": None,
                "gap_excl_ties": None,
            },
            {
                "entity_type": "hydro",
                "variable": "spillage_m3s",
                "gap": 0.4,
                "gap_excl_ties": 0.1,
            },
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
            {
                "entity_type": "hydro",
                "variable": "turbined_m3s",
                "gap": 0.4,
                "gap_excl_ties": 0.2,
            }
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


_FACTS: dict[str, Any] = {
    "baseline_output_dir": "runs/v0.20.0/output",
    "candidate_output_dir": "runs/v0.21.0/output",
    "baseline_version": "0.20.0",
    "candidate_version": "0.21.0",
    "baseline_label": "v0.20.0 (baseline)",
    "candidate_label": "v0.21.0 (candidate)",
    "only_in_baseline": {"hydro": ["TUCURUI"], "thermal": []},
    "only_in_candidate": {"hydro": [], "thermal": ["ANGRA 3"]},
    "unavailable": [{"section": "hydro-bounds", "column": None, "side": "both"}],
    "upper_bound_comparable": True,
    "upper_bound_comparability": "comparable",
    "baseline_upper_bound_kind": "inner_approximation",
    "candidate_upper_bound_kind": "inner_approximation",
    "baseline_gap_percent": 1.25,
    "candidate_gap_percent": 0.75,
    "gap_delta_pp": -0.5,
}

_VERSION_KEYS = [
    "baseline_version",
    "candidate_version",
    "only_in_baseline",
    "only_in_candidate",
    "upper_bound_comparable",
    "baseline_gap_percent",
    "candidate_gap_percent",
    "gap_delta_pp",
]


def _with_facts(dataset: ComparisonDataset, **changes: Any) -> ComparisonDataset:
    dataset.metadata["version_comparison"] = copy.deepcopy({**_FACTS, **changes})
    return dataset


class TestVersionDatasetSummary:
    """``version_dataset_summary`` -- the ``compare novomodelo --json`` payload
    builder: the six headline keys, then eight read from the facts record."""

    def test_keys_are_the_headline_then_the_eight_version_keys_in_order(self) -> None:
        dataset = _with_facts(_fake_dataset())

        result = version_dataset_summary(dataset)

        assert list(result) == [
            "tolerance",
            "total",
            "close",
            "moderate",
            "far",
            "far_variables",
            "baseline_version",
            "candidate_version",
            "only_in_baseline",
            "only_in_candidate",
            "upper_bound_comparable",
            "baseline_gap_percent",
            "candidate_gap_percent",
            "gap_delta_pp",
        ]

    def test_headline_equals_compare_summary_of_the_dataset_verdict(self) -> None:
        dataset = _with_facts(_fake_dataset())

        result = version_dataset_summary(dataset)

        headline = compare_summary(build_compare_verdict(dataset))
        assert {key: result[key] for key in headline} == headline
        assert result["total"] == 2
        assert (result["close"], result["moderate"], result["far"]) == (1, 0, 1)

    def test_version_keys_equal_the_facts_values_unchanged(self) -> None:
        dataset = _with_facts(_fake_dataset())

        result = version_dataset_summary(dataset)

        assert {key: result[key] for key in _VERSION_KEYS} == {
            key: _FACTS[key] for key in _VERSION_KEYS
        }
        assert result["baseline_version"] == "0.20.0"
        assert result["candidate_version"] == "0.21.0"

    def test_missing_gaps_and_versions_stay_none_and_serialize(self) -> None:
        dataset = _with_facts(
            _fake_dataset(),
            baseline_version=None,
            only_in_baseline={"hydro": ["TUCURUI"]},
            upper_bound_comparable=False,
            baseline_gap_percent=None,
            candidate_gap_percent=0.5,
            gap_delta_pp=None,
        )

        result = version_dataset_summary(dataset)

        assert result["baseline_version"] is None
        assert result["only_in_baseline"] == {"hydro": ["TUCURUI"]}
        assert result["upper_bound_comparable"] is False
        assert result["baseline_gap_percent"] is None
        assert result["candidate_gap_percent"] == 0.5
        assert result["gap_delta_pp"] is None
        json.dumps(result, allow_nan=False)

    def test_gap_values_are_numbers_when_both_sides_have_a_gap(self) -> None:
        dataset = _with_facts(_fake_dataset())

        result = version_dataset_summary(dataset)

        assert result["baseline_gap_percent"] == 1.25
        assert result["candidate_gap_percent"] == 0.75
        assert result["gap_delta_pp"] == -0.5
        json.dumps(result, allow_nan=False)

    def test_result_shares_no_container_with_the_dataset_metadata(self) -> None:
        dataset = _with_facts(_fake_dataset(), only_in_baseline={"hydro": ["TUCURUI"]})

        result = version_dataset_summary(dataset)
        only_in_baseline = result["only_in_baseline"]
        assert isinstance(only_in_baseline, dict)
        only_in_baseline["hydro"].append("BELO MONTE")
        only_in_baseline["thermal"] = ["ANGRA 3"]

        stored = dataset.metadata["version_comparison"]
        assert isinstance(stored, dict)
        assert stored["only_in_baseline"] == {"hydro": ["TUCURUI"]}

    @pytest.mark.parametrize(
        "omitted",
        [
            "unavailable",
            "upper_bound_comparability",
            "baseline_label",
            "candidate_label",
            "baseline_output_dir",
            "candidate_output_dir",
            "risk_measures_differ",
        ],
    )
    def test_facts_kept_in_comparison_json_are_not_summary_keys(
        self, omitted: str
    ) -> None:
        dataset = _with_facts(_fake_dataset())

        result = version_dataset_summary(dataset)

        assert omitted not in result

    def test_dataset_without_a_facts_record_raises_type_error(self) -> None:
        dataset = _fake_dataset()

        with pytest.raises(TypeError, match="version_comparison"):
            version_dataset_summary(dataset)

    def test_empty_dataset_reports_zero_total_and_keeps_the_version_keys(self) -> None:
        dataset = _with_facts(_empty_fake_dataset())

        result = version_dataset_summary(dataset)

        assert result["total"] == 0
        assert result["far_variables"] == []
        assert [key for key in result if key in _VERSION_KEYS] == _VERSION_KEYS

    def test_two_identical_builds_are_all_close_and_comparable(
        self, tmp_path: Path
    ) -> None:
        spec = SolvedCaseSpec(load_stats=True)
        baseline = build_solved_case(tmp_path / "baseline", spec)
        candidate = build_solved_case(
            tmp_path / "candidate", replace(spec, id_offset=100)
        )
        with dx.collect():
            dataset = build_version_dataset(baseline, candidate, tolerance=0.05)

        result = version_dataset_summary(dataset)

        assert result["total"] == result["close"]
        assert result["far_variables"] == []
        assert result["upper_bound_comparable"] is True
        assert result["gap_delta_pp"] == 0.0

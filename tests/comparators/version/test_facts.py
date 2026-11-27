"""Tests for ``comparators.version.facts``: the record every surface of a version
comparison reads, built from literal side inputs, alignments, rows and bounds."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from novomodelo_bridge.comparators.analyze import build_results_dataset
from novomodelo_bridge.comparators.dataset import ComparisonDataset
from novomodelo_bridge.comparators.model import PercentileData
from novomodelo_bridge.comparators.version.alignment import NameAlignment
from novomodelo_bridge.comparators.version.availability import Unavailable
from novomodelo_bridge.comparators.version.convergence import (
    UB_COMPARABILITY,
    BoundsComparison,
    FinalBounds,
)
from novomodelo_bridge.comparators.version.facts import (
    METADATA_KEY,
    VersionFacts,
    build_facts,
    facts_of,
)
from novomodelo_bridge.comparators.version.sides import SideInputs

_LABELS = ("baseline label", "candidate label")

_ROWS = [Unavailable("fpha", None, "both"), Unavailable("thermal", None, "baseline")]


def _side(side: str, version: str | None, output_dir: Path) -> SideInputs:
    return SideInputs(
        side=side,
        output_dir=output_dir,
        version=version,
        hydros=[],
        buses=[],
        thermals=None,
        lines=[],
        risk_aversion={},
        partitions={},
    )


def _alignment(
    entity_type: str,
    only_in_baseline: tuple[str, ...] = (),
    only_in_candidate: tuple[str, ...] = (),
) -> NameAlignment:
    return NameAlignment(
        entity_type, (), only_in_baseline, only_in_candidate, duplicates=()
    )


def _comparable() -> BoundsComparison:
    return BoundsComparison(
        baseline=FinalBounds(10, 1000.0, 1005.0, "statistical"),
        candidate=FinalBounds(12, 1000.0, 1008.0, "statistical"),
        comparability="comparable",
        baseline_gap_percent=0.5,
        candidate_gap_percent=0.8,
        gap_delta_pp=0.3,
        risk_measures_differ=False,
    )


def _bounds_missing() -> BoundsComparison:
    return BoundsComparison(
        baseline=None,
        candidate=None,
        comparability="bounds-missing",
        baseline_gap_percent=None,
        candidate_gap_percent=None,
        gap_delta_pp=None,
        risk_measures_differ=False,
    )


def _facts(bounds: BoundsComparison | None = None) -> VersionFacts:
    return build_facts(
        _side("baseline", "0.20.0", Path("base/output")),
        _side("candidate", None, Path("cand/output")),
        [_alignment("hydro", only_in_candidate=("TUCURUI",)), _alignment("bus")],
        _ROWS,
        bounds or _comparable(),
        _LABELS,
    )


def _dataset_with(facts: VersionFacts) -> ComparisonDataset:
    dataset = build_results_dataset([], PercentileData(), 0.05)
    dataset.metadata[METADATA_KEY] = facts.to_dict()
    return dataset


class TestBuildFacts:
    def test_maps_every_input_to_its_field(self) -> None:
        facts = _facts().to_dict()

        assert facts["baseline_version"] == "0.20.0"
        assert facts["candidate_version"] is None
        assert facts["baseline_label"] == "baseline label"
        assert facts["candidate_label"] == "candidate label"
        assert facts["only_in_baseline"] == {"hydro": [], "bus": []}
        assert facts["only_in_candidate"] == {"hydro": ["TUCURUI"], "bus": []}
        assert facts["upper_bound_comparable"] is True
        assert facts["upper_bound_comparability"] == "comparable"
        assert facts["baseline_upper_bound_kind"] == "statistical"
        assert facts["candidate_upper_bound_kind"] == "statistical"
        assert facts["baseline_gap_percent"] == 0.5
        assert facts["candidate_gap_percent"] == 0.8
        assert facts["gap_delta_pp"] == 0.3

    def test_output_dirs_are_stored_as_given_and_never_resolved(self) -> None:
        facts = _facts()

        assert facts.baseline_output_dir == "base/output"
        assert facts.candidate_output_dir == "cand/output"

    def test_one_sided_names_keep_the_alignment_order_in_both_lists(self) -> None:
        facts = build_facts(
            _side("baseline", "0.20.0", Path("b")),
            _side("candidate", "0.21.0", Path("c")),
            [_alignment("hydro", ("B", "A"), ("Z", "Y"))],
            [],
            _comparable(),
            _LABELS,
        )

        assert facts.only_in_baseline == {"hydro": ["B", "A"]}
        assert facts.only_in_candidate == {"hydro": ["Z", "Y"]}
        assert isinstance(facts.only_in_baseline["hydro"], list)

    def test_an_entity_type_without_an_alignment_has_no_key_on_either_side(
        self,
    ) -> None:
        facts = _facts()

        assert "thermal" not in facts.only_in_baseline
        assert "thermal" not in facts.only_in_candidate

    def test_no_alignment_at_all_gives_empty_mappings(self) -> None:
        facts = build_facts(
            _side("baseline", None, Path("b")),
            _side("candidate", None, Path("c")),
            [],
            [],
            _bounds_missing(),
            _LABELS,
        )

        assert facts.only_in_baseline == {}
        assert facts.only_in_candidate == {}
        assert facts.unavailable == []

    def test_unavailable_rows_are_copied_one_for_one_in_the_given_order(self) -> None:
        facts = build_facts(
            _side("baseline", None, Path("b")),
            _side("candidate", None, Path("c")),
            [],
            [
                Unavailable("fpha", None, "both"),
                Unavailable("hydro", "turbined_flow", "candidate"),
                Unavailable("thermal", None, "baseline"),
            ],
            _comparable(),
            _LABELS,
        )

        assert facts.unavailable == [
            {"section": "fpha", "column": None, "side": "both"},
            {"section": "hydro", "column": "turbined_flow", "side": "candidate"},
            {"section": "thermal", "column": None, "side": "baseline"},
        ]

    def test_a_both_row_and_an_entity_list_row_are_stored_unchanged(self) -> None:
        assert _facts().unavailable == [
            {"section": "fpha", "column": None, "side": "both"},
            {"section": "thermal", "column": None, "side": "baseline"},
        ]

    def test_two_alignments_of_one_entity_type_raise_naming_the_type(self) -> None:
        with pytest.raises(ValueError, match="hydro"):
            build_facts(
                _side("baseline", None, Path("b")),
                _side("candidate", None, Path("c")),
                [_alignment("hydro", ("A",)), _alignment("bus"), _alignment("hydro")],
                [],
                _comparable(),
                _LABELS,
            )

    def test_the_inputs_are_not_mutated(self) -> None:
        rows = list(_ROWS)
        alignment = _alignment("hydro", only_in_candidate=("TUCURUI",))

        build_facts(
            _side("baseline", None, Path("b")),
            _side("candidate", None, Path("c")),
            [alignment],
            rows,
            _comparable(),
            _LABELS,
        )

        assert rows == _ROWS
        assert alignment.only_in_candidate == ("TUCURUI",)


class TestBounds:
    def test_a_comparison_without_final_bounds_stores_null_kinds_and_null_gaps(
        self,
    ) -> None:
        facts = _facts(_bounds_missing())

        assert facts.upper_bound_comparable is False
        assert facts.upper_bound_comparability == "bounds-missing"
        assert facts.baseline_upper_bound_kind is None
        assert facts.candidate_upper_bound_kind is None
        assert facts.baseline_gap_percent is None
        assert facts.candidate_gap_percent is None
        assert facts.gap_delta_pp is None

    def test_only_the_side_without_final_bounds_has_a_null_kind(self) -> None:
        bounds = dataclasses.replace(
            _bounds_missing(), candidate=FinalBounds(3, 10.0, 11.0, "statistical")
        )

        facts = _facts(bounds)

        assert facts.baseline_upper_bound_kind is None
        assert facts.candidate_upper_bound_kind == "statistical"

    def test_a_final_bound_whose_output_names_no_kind_stores_a_null_kind(self) -> None:
        bounds = dataclasses.replace(
            _comparable(),
            baseline=FinalBounds(10, 1000.0, 1005.0, None),
            comparability="kind-missing",
        )

        facts = _facts(bounds)

        assert facts.baseline_upper_bound_kind is None
        assert facts.candidate_upper_bound_kind == "statistical"

    @pytest.mark.parametrize("comparability", UB_COMPARABILITY)
    def test_each_comparability_value_is_stored_unchanged(
        self, comparability: str
    ) -> None:
        bounds = dataclasses.replace(_comparable(), comparability=comparability)

        facts = _facts(bounds)

        assert facts.upper_bound_comparability == comparability
        assert facts.upper_bound_comparable is (comparability == "comparable")


class TestSerialization:
    def test_keys_follow_the_field_order_and_omit_risk_measures_differ(self) -> None:
        keys = list(_facts().to_dict())

        assert keys == [field.name for field in dataclasses.fields(VersionFacts)]
        assert "risk_measures_differ" not in keys

    def test_the_dict_round_trips_through_json_with_nulls_not_nan(self) -> None:
        facts = _facts(_bounds_missing())

        text = json.dumps(facts.to_dict(), allow_nan=False)
        restored = VersionFacts.from_dict(json.loads(text))

        assert '"gap_delta_pp": null' in text
        assert restored == facts

    def test_from_dict_raises_key_error_for_a_missing_field(self) -> None:
        data = _facts().to_dict()
        del data["gap_delta_pp"]

        with pytest.raises(KeyError, match="gap_delta_pp"):
            VersionFacts.from_dict(data)

    def test_to_dict_returns_copies_the_record_does_not_share(self) -> None:
        facts = _facts()

        facts.to_dict()["only_in_candidate"]["hydro"].append("OTHER")

        assert facts.only_in_candidate["hydro"] == ["TUCURUI"]

    def test_from_dict_containers_equal_the_mappings_and_are_not_the_same_objects(
        self,
    ) -> None:
        data = _facts().to_dict()

        restored = VersionFacts.from_dict(data)

        for name in ("only_in_baseline", "only_in_candidate", "unavailable"):
            assert getattr(restored, name) == data[name]
            assert getattr(restored, name) is not data[name]
        assert (
            restored.only_in_candidate["hydro"]
            is not data["only_in_candidate"]["hydro"]
        )
        assert restored.unavailable[0] is not data["unavailable"][0]

    def test_mutating_a_restored_record_leaves_the_mapping_unchanged(self) -> None:
        data = _facts().to_dict()
        before = json.dumps(data)

        restored = VersionFacts.from_dict(data)
        restored.unavailable.append({"section": "x", "column": None, "side": "both"})
        restored.unavailable[0]["side"] = "candidate"
        restored.only_in_candidate["hydro"].append("OTHER")
        restored.only_in_baseline["thermal"] = []

        assert json.dumps(data) == before


class TestDatasetMetadata:
    def test_facts_survive_the_dataset_round_trip(self, tmp_path: Path) -> None:
        facts = _facts(_bounds_missing())

        _dataset_with(facts).to_dir(tmp_path)

        assert facts_of(ComparisonDataset.from_dir(tmp_path)) == facts

    def test_facts_of_reads_the_metadata_key(self) -> None:
        facts = _facts()

        assert facts_of(_dataset_with(facts)) == facts

    def test_appending_to_the_record_leaves_the_stored_metadata_unchanged(self) -> None:
        dataset = _dataset_with(_facts())
        stored = copy.deepcopy(dataset.metadata[METADATA_KEY])

        facts_of(dataset).unavailable.append(
            {"section": "x", "column": None, "side": "both"}
        )

        assert dataset.metadata[METADATA_KEY] == stored

    def test_facts_of_raises_type_error_when_the_key_is_absent(self) -> None:
        dataset = build_results_dataset([], PercentileData(), 0.05)

        with pytest.raises(TypeError, match="'version_comparison' must be a dict"):
            facts_of(dataset)

    def test_facts_of_raises_type_error_when_the_value_is_not_a_dict(self) -> None:
        dataset = build_results_dataset([], PercentileData(), 0.05)
        dataset.metadata[METADATA_KEY] = ["not", "a", "dict"]

        with pytest.raises(TypeError, match="must be a dict, got list"):
            facts_of(dataset)

"""Tests for ``comparators.version.alignment``: name and stage alignment of two runs."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from novomodelo_bridge.comparators.version.alignment import (
    AlignedPair,
    DuplicateName,
    NameAlignment,
    StageAlignment,
    align_by_name,
    align_stages,
    name_key,
)

Entry = Mapping[str, object]


def _entries(*named: tuple[int, str]) -> list[Entry]:
    return [{"id": entry_id, "name": name} for entry_id, name in named]


def test_name_key_strips_and_uppercases() -> None:
    assert name_key("  Furnas \t") == "FURNAS"


class TestPairs:
    def test_spelling_and_case_differences_still_pair(self) -> None:
        baseline = _entries((3, " Furnas"), (4, "ITAIPU"))
        candidate = _entries((9, "itaipu "), (7, "FURNAS"), (8, "TUCURUI"))

        result = align_by_name("hydro", baseline, candidate)

        assert result.pairs == (
            AlignedPair("Furnas", 3, 7),
            AlignedPair("ITAIPU", 4, 9),
        )
        assert result.only_in_baseline == ()
        assert result.only_in_candidate == ("TUCURUI",)
        assert result.duplicates == ()

    def test_pairs_follow_baseline_file_order(self) -> None:
        baseline = _entries((1, "Zeta"), (2, "Alpha"), (3, "Mu"))
        candidate = _entries((10, "mu"), (11, "alpha"), (12, "zeta"))

        result = align_by_name("bus", baseline, candidate)

        assert [pair.name for pair in result.pairs] == ["Zeta", "Alpha", "Mu"]

    def test_the_pair_name_is_the_stripped_baseline_spelling(self) -> None:
        result = align_by_name(
            "thermal", _entries((1, "  Gas_A ")), _entries((2, "GAS_A"))
        )

        assert result.pairs == (AlignedPair("Gas_A", 1, 2),)

    def test_entity_type_is_kept_as_passed(self) -> None:
        assert align_by_name("generic_constraint", [], []).entity_type == (
            "generic_constraint"
        )

    def test_names_that_differ_never_pair_by_id(self) -> None:
        baseline = _entries((1, "FURNAS"), (2, "ITAIPU"))
        candidate = _entries((1, "ITUMBIARA"), (2, "TUCURUI"))

        result = align_by_name("hydro", baseline, candidate)

        assert result.pairs == ()
        assert result.only_in_baseline == ("FURNAS", "ITAIPU")
        assert result.only_in_candidate == ("ITUMBIARA", "TUCURUI")

    def test_keys_other_than_id_and_name_are_ignored(self) -> None:
        baseline = [{"id": 1, "name": "A", "bus": 4, "capacity": 10.0}]
        candidate = [{"id": 2, "name": "a", "bus": 9}]

        result = align_by_name("thermal", baseline, candidate)

        assert result.pairs == (AlignedPair("A", 1, 2),)


class TestOneSided:
    def test_entities_only_in_the_baseline(self) -> None:
        result = align_by_name(
            "hydro", _entries((1, "A"), (2, "B")), _entries((5, "A"))
        )

        assert result.pairs == (AlignedPair("A", 1, 5),)
        assert result.only_in_baseline == ("B",)
        assert result.only_in_candidate == ()

    def test_entities_only_in_the_candidate(self) -> None:
        result = align_by_name(
            "hydro", _entries((1, "A")), _entries((5, "A"), (6, "B"))
        )

        assert result.only_in_baseline == ()
        assert result.only_in_candidate == ("B",)

    def test_one_sided_names_are_stripped_and_sorted_by_name_key(self) -> None:
        baseline = _entries((1, " Zeta"), (2, "alpha "), (3, "Beta"))
        candidate = _entries((4, "omega"), (5, "  Delta"), (6, "ZED"))

        result = align_by_name("bus", baseline, candidate)

        assert result.only_in_baseline == ("alpha", "Beta", "Zeta")
        assert result.only_in_candidate == ("Delta", "omega", "ZED")


class TestDuplicates:
    def test_duplicate_on_the_baseline_is_excluded_everywhere_else(self) -> None:
        baseline = _entries((1, "A"), (2, "a "), (3, "B"))
        candidate = _entries((5, "A"), (6, "B"))

        result = align_by_name("hydro", baseline, candidate)

        assert result.pairs == (AlignedPair("B", 3, 6),)
        assert result.only_in_baseline == ()
        assert result.only_in_candidate == ()
        assert result.duplicates == (DuplicateName("A", ("A", "a"), ("A",)),)

    def test_duplicate_on_the_candidate_still_drops_the_baseline_entry(self) -> None:
        baseline = _entries((1, "A"), (2, "B"))
        candidate = _entries((5, "A"), (6, " a"), (7, "B"))

        result = align_by_name("hydro", baseline, candidate)

        assert result.pairs == (AlignedPair("B", 2, 7),)
        assert result.only_in_baseline == ()
        assert result.only_in_candidate == ()
        assert result.duplicates == (DuplicateName("A", ("A",), ("A", "a")),)

    def test_duplicate_on_both_sides_lists_every_spelling_of_both(self) -> None:
        baseline = _entries((1, "Furnas"), (2, "FURNAS"))
        candidate = _entries((5, "furnas"), (6, "Furnas "))

        result = align_by_name("hydro", baseline, candidate)

        assert result.pairs == ()
        assert result.duplicates == (
            DuplicateName("FURNAS", ("Furnas", "FURNAS"), ("furnas", "Furnas")),
        )

    def test_duplicate_absent_from_the_other_side_is_not_one_sided(self) -> None:
        baseline = _entries((1, "A"), (2, "a"))

        result = align_by_name("hydro", baseline, [])

        assert result.only_in_baseline == ()
        assert result.duplicates == (DuplicateName("A", ("A", "a"), ()),)

    def test_duplicates_are_sorted_by_name_key_and_spellings_keep_file_order(
        self,
    ) -> None:
        baseline = _entries((1, "zeta"), (2, "Beta"), (3, "ZETA"), (4, "beta"))
        candidate = _entries((5, "Beta"))

        result = align_by_name("bus", baseline, candidate)

        assert result.duplicates == (
            DuplicateName("BETA", ("Beta", "beta"), ("Beta",)),
            DuplicateName("ZETA", ("zeta", "ZETA"), ()),
        )


class TestEmpty:
    def test_empty_lists_on_both_sides_give_four_empty_tuples(self) -> None:
        assert align_by_name("line", [], []) == NameAlignment("line", (), (), (), ())

    def test_one_empty_side_leaves_the_other_one_sided(self) -> None:
        result = align_by_name("line", _entries((1, "SE-S")), [])

        assert result.pairs == ()
        assert result.only_in_baseline == ("SE-S",)
        assert result.only_in_candidate == ()


class TestInvalidEntries:
    @pytest.mark.parametrize(
        "bad",
        [
            pytest.param({"name": "B"}, id="no-id"),
            pytest.param({"id": 2}, id="no-name"),
            pytest.param({"id": 2, "name": ""}, id="empty-name"),
            pytest.param({"id": 2, "name": "   "}, id="blank-name"),
            pytest.param({"id": 2, "name": None}, id="null-name"),
            pytest.param({"id": 2, "name": 7}, id="non-string-name"),
            pytest.param({"id": True, "name": "B"}, id="bool-id"),
            pytest.param({"id": 2.0, "name": "B"}, id="float-id"),
            pytest.param({"id": "2", "name": "B"}, id="string-id"),
            pytest.param({"id": None, "name": "B"}, id="null-id"),
        ],
    )
    def test_a_bad_baseline_entry_raises_naming_the_entity_type(
        self, bad: Entry
    ) -> None:
        baseline = [{"id": 1, "name": "A"}, bad]
        candidate = _entries((5, "A"))

        with pytest.raises(ValueError, match="thermal"):
            align_by_name("thermal", baseline, candidate)

    def test_a_bad_candidate_entry_raises_naming_the_entity_type_and_side(
        self,
    ) -> None:
        baseline = _entries((1, "A"))
        candidate = [{"id": 5, "name": "A"}, {"id": 6}]

        with pytest.raises(ValueError, match=r"bus: candidate entry 2"):
            align_by_name("bus", baseline, candidate)

    def test_a_bad_entry_is_never_dropped_or_renamed(self) -> None:
        baseline = [{"id": 1, "name": "A"}, {"id": 2}]
        candidate = _entries((5, "A"))

        with pytest.raises(ValueError, match="thermal"):
            align_by_name("thermal", baseline, candidate)


class TestStages:
    def test_common_and_one_sided_stage_ids(self) -> None:
        assert align_stages([0, 1, 2], [1, 2, 3]) == StageAlignment(
            common=(1, 2), only_in_baseline=(0,), only_in_candidate=(3,)
        )

    def test_results_are_sorted_whatever_the_input_order(self) -> None:
        result = align_stages([5, 0, 3, 3], [4, 3, 1, 5])

        assert result.common == (3, 5)
        assert result.only_in_baseline == (0,)
        assert result.only_in_candidate == (1, 4)

    def test_any_iterable_is_accepted(self) -> None:
        result = align_stages(iter(range(3)), (stage for stage in range(2, 5)))

        assert result.common == (2,)
        assert result.only_in_baseline == (0, 1)
        assert result.only_in_candidate == (3, 4)

    def test_empty_inputs_give_empty_tuples(self) -> None:
        assert align_stages([], []) == StageAlignment((), (), ())

    def test_identical_stage_sets_have_nothing_one_sided(self) -> None:
        result = align_stages(range(4), range(4))

        assert result.common == (0, 1, 2, 3)
        assert result.only_in_baseline == ()
        assert result.only_in_candidate == ()

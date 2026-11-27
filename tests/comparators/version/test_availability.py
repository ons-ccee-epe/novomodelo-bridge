"""Tests for ``comparators.version.availability``: the rule that turns two sides'
availability mappings and a declared requirement into a section decision.

Every input is a literal; no fixture tree is read.
"""

from __future__ import annotations

import pytest

from novomodelo_bridge.comparators.version.availability import (
    BOTH_SIDES,
    ENTITY_LIST_KEYS,
    SectionAvailability,
    SectionRequirement,
    Unavailable,
    section_availability,
    unavailable_rows,
    with_entity_lists,
)

_THERMALS = "simulation/thermals"
_BUSES = "simulation/buses"
_THERMAL_LIST = "system/thermals.json"
_LINE_LIST = "system/lines.json"

_FULL = frozenset({"thermal_id", "stage_id", "generation_mw", "scenario_id"})


def _thermal_requirement() -> SectionRequirement:
    return SectionRequirement(
        "thermal",
        {_THERMALS: frozenset({"thermal_id", "stage_id"})},
        {"generation_mw": (_THERMALS, frozenset({"generation_mw"}))},
    )


def _bus_requirement() -> SectionRequirement:
    return SectionRequirement(
        "bus",
        {_BUSES: frozenset({"bus_id"})},
        {
            "spot_price": (_BUSES, frozenset({"spot_price"})),
            "deficit_mw": (_BUSES, frozenset({"deficit_mw"})),
        },
    )


class TestAllPresent:
    def test_both_sides_complete_leaves_the_section_available_with_every_variable(
        self,
    ) -> None:
        side = {_THERMALS: _FULL}

        result = section_availability(_thermal_requirement(), side, side)

        assert result == SectionAvailability("thermal", True, ("generation_mw",), ())

    def test_a_section_with_no_variables_is_available_when_its_partitions_are(
        self,
    ) -> None:
        requirement = SectionRequirement(
            "thermal", {_THERMALS: frozenset({"stage_id"})}
        )
        side = {_THERMALS: _FULL}

        result = section_availability(requirement, side, side)

        assert result.available is True
        assert result.variables == ()
        assert result.unavailable == ()

    def test_an_empty_column_set_satisfies_a_requirement_with_no_columns(self) -> None:
        requirement = SectionRequirement("thermal", {_THERMAL_LIST: frozenset()})
        side = {_THERMAL_LIST: frozenset()}

        assert section_availability(requirement, side, side).available is True

    def test_available_variables_keep_the_requirement_order(self) -> None:
        requirement = SectionRequirement(
            "bus",
            {_BUSES: frozenset()},
            {
                "deficit_mw": (_BUSES, frozenset({"deficit_mw"})),
                "spot_price": (_BUSES, frozenset({"spot_price"})),
                "load_mw": (_BUSES, frozenset({"load_mw"})),
            },
        )
        side = {_BUSES: frozenset({"load_mw", "spot_price", "deficit_mw"})}

        result = section_availability(requirement, side, side)

        assert result.variables == ("deficit_mw", "spot_price", "load_mw")


class TestAbsentPartition:
    def test_absent_on_the_candidate_alone_is_one_candidate_row(self) -> None:
        result = section_availability(
            _thermal_requirement(), {_THERMALS: _FULL}, {_THERMALS: None}
        )

        assert result.available is False
        assert result.variables == ()
        assert result.unavailable == (Unavailable("thermal", None, "candidate"),)

    def test_absent_on_the_baseline_alone_is_one_baseline_row(self) -> None:
        result = section_availability(
            _thermal_requirement(), {_THERMALS: None}, {_THERMALS: _FULL}
        )

        assert result.unavailable == (Unavailable("thermal", None, "baseline"),)

    def test_absent_on_both_sides_is_one_row_for_both(self) -> None:
        result = section_availability(
            _thermal_requirement(), {_THERMALS: None}, {_THERMALS: None}
        )

        assert result.unavailable == (Unavailable("thermal", None, BOTH_SIDES),)
        assert len(result.unavailable) == 1

    def test_a_key_missing_from_the_mapping_is_absent_like_none(self) -> None:
        result = section_availability(_thermal_requirement(), {}, {_THERMALS: _FULL})

        assert result.unavailable == (Unavailable("thermal", None, "baseline"),)

    def test_a_different_required_key_absent_on_each_side_is_one_row_for_both(
        self,
    ) -> None:
        requirement = SectionRequirement(
            "network",
            {_BUSES: frozenset({"bus_id"}), _THERMALS: frozenset({"thermal_id"})},
        )
        baseline = {_BUSES: None, _THERMALS: frozenset({"thermal_id"})}
        candidate = {_BUSES: frozenset({"bus_id"}), _THERMALS: None}

        result = section_availability(requirement, baseline, candidate)

        assert result.unavailable == (Unavailable("network", None, BOTH_SIDES),)

    def test_variables_of_an_unavailable_section_report_no_rows(self) -> None:
        result = section_availability(
            _bus_requirement(), {_BUSES: None}, {_BUSES: frozenset({"bus_id"})}
        )

        assert result.variables == ()
        assert result.unavailable == (Unavailable("bus", None, "baseline"),)


class TestRequiredColumns:
    def test_a_required_column_missing_on_one_side_is_one_row_for_that_side(
        self,
    ) -> None:
        baseline = {_THERMALS: _FULL - {"stage_id"}}
        candidate = {_THERMALS: _FULL}

        result = section_availability(_thermal_requirement(), baseline, candidate)

        assert result.available is False
        assert result.variables == ()
        assert result.unavailable == (Unavailable("thermal", "stage_id", "baseline"),)

    def test_a_required_column_missing_on_both_sides_is_one_row_for_both(self) -> None:
        side = {_THERMALS: _FULL - {"stage_id"}}

        result = section_availability(_thermal_requirement(), side, side)

        assert result.unavailable == (Unavailable("thermal", "stage_id", BOTH_SIDES),)

    def test_the_same_column_missing_from_two_partitions_is_one_row(self) -> None:
        requirement = SectionRequirement(
            "network",
            {_BUSES: frozenset({"stage_id"}), _THERMALS: frozenset({"stage_id"})},
        )
        baseline = {_BUSES: frozenset(), _THERMALS: frozenset({"stage_id"})}
        candidate = {_BUSES: frozenset({"stage_id"}), _THERMALS: frozenset()}

        result = section_availability(requirement, baseline, candidate)

        assert result.unavailable == (Unavailable("network", "stage_id", BOTH_SIDES),)

    def test_the_section_row_comes_first_then_column_rows_by_column_name(self) -> None:
        requirement = SectionRequirement(
            "network",
            {
                _BUSES: frozenset({"bus_id"}),
                _THERMALS: frozenset({"thermal_id", "stage_id"}),
            },
        )
        baseline = {_BUSES: None, _THERMALS: frozenset()}
        candidate = {_BUSES: frozenset({"bus_id"}), _THERMALS: frozenset({"stage_id"})}

        result = section_availability(requirement, baseline, candidate)

        assert result.unavailable == (
            Unavailable("network", None, "baseline"),
            Unavailable("network", "stage_id", "baseline"),
            Unavailable("network", "thermal_id", BOTH_SIDES),
        )

    def test_a_column_of_a_key_absent_on_a_side_is_not_missing_on_that_side(
        self,
    ) -> None:
        baseline = {_THERMALS: None}
        candidate = {_THERMALS: _FULL - {"stage_id"}}

        result = section_availability(_thermal_requirement(), baseline, candidate)

        assert result.unavailable == (
            Unavailable("thermal", None, "baseline"),
            Unavailable("thermal", "stage_id", "candidate"),
        )


class TestVariables:
    def test_a_variable_missing_on_one_side_is_left_out_with_one_row(self) -> None:
        baseline = {_BUSES: frozenset({"bus_id", "deficit_mw"})}
        candidate = {_BUSES: frozenset({"bus_id", "deficit_mw", "spot_price"})}

        result = section_availability(_bus_requirement(), baseline, candidate)

        assert result.available is True
        assert result.variables == ("deficit_mw",)
        assert result.unavailable == (Unavailable("bus", "spot_price", "baseline"),)

    def test_a_variable_missing_on_both_sides_is_one_row_for_both(self) -> None:
        side = {_BUSES: frozenset({"bus_id", "deficit_mw"})}

        result = section_availability(_bus_requirement(), side, side)

        assert result.variables == ("deficit_mw",)
        assert result.unavailable == (Unavailable("bus", "spot_price", BOTH_SIDES),)

    def test_different_variables_missing_on_each_side_give_one_row_per_side(
        self,
    ) -> None:
        baseline = {_BUSES: frozenset({"bus_id", "deficit_mw"})}
        candidate = {_BUSES: frozenset({"bus_id", "spot_price"})}

        result = section_availability(_bus_requirement(), baseline, candidate)

        assert result.available is True
        assert result.variables == ()
        assert result.unavailable == (
            Unavailable("bus", "deficit_mw", "candidate"),
            Unavailable("bus", "spot_price", "baseline"),
        )

    def test_two_variables_sharing_a_missing_source_column_give_one_row(self) -> None:
        requirement = SectionRequirement(
            "bus",
            {_BUSES: frozenset({"bus_id"})},
            {
                "price": (_BUSES, frozenset({"spot_price", "bus_id"})),
                "price_gap": (_BUSES, frozenset({"spot_price", "deficit_mw"})),
            },
        )
        baseline = {_BUSES: frozenset({"bus_id", "deficit_mw"})}
        candidate = {_BUSES: frozenset({"bus_id", "deficit_mw", "spot_price"})}

        result = section_availability(requirement, baseline, candidate)

        assert result.variables == ()
        assert result.unavailable == (Unavailable("bus", "spot_price", "baseline"),)

    def test_a_variable_with_two_missing_columns_gives_one_row_per_column(self) -> None:
        requirement = SectionRequirement(
            "bus",
            {_BUSES: frozenset()},
            {"net": (_BUSES, frozenset({"load_mw", "deficit_mw"}))},
        )
        side = {_BUSES: frozenset()}

        result = section_availability(requirement, side, side)

        assert result.unavailable == (
            Unavailable("bus", "deficit_mw", BOTH_SIDES),
            Unavailable("bus", "load_mw", BOTH_SIDES),
        )

    def test_a_column_missing_from_one_partition_does_not_drop_a_variable_of_another(
        self,
    ) -> None:
        requirement = SectionRequirement(
            "network",
            {_BUSES: frozenset(), _THERMALS: frozenset()},
            {
                "bus_value": (_BUSES, frozenset({"value"})),
                "thermal_value": (_THERMALS, frozenset({"value"})),
            },
        )
        side = {_BUSES: frozenset({"value"}), _THERMALS: frozenset()}

        result = section_availability(requirement, side, side)

        assert result.variables == ("bus_value",)
        assert result.unavailable == (Unavailable("network", "value", BOTH_SIDES),)


class TestEntityLists:
    def test_unusable_list_on_the_baseline_makes_the_section_unavailable_there(
        self,
    ) -> None:
        columns = frozenset({"thermal_id"})
        baseline = with_entity_lists(
            {_THERMALS: columns}, {"thermal": False, "line": True}
        )
        candidate = with_entity_lists(
            {_THERMALS: columns}, {"thermal": True, "line": True}
        )
        requirement = SectionRequirement(
            "thermal",
            {_THERMALS: frozenset({"thermal_id"}), _THERMAL_LIST: frozenset()},
        )

        result = section_availability(requirement, baseline, candidate)

        assert baseline[_THERMAL_LIST] is None
        assert baseline[_LINE_LIST] == frozenset()
        assert result.available is False
        assert result.unavailable == (Unavailable("thermal", None, "baseline"),)

    def test_unusable_list_on_one_side_and_absent_partition_on_the_other_is_both(
        self,
    ) -> None:
        columns = frozenset({"thermal_id"})
        baseline = with_entity_lists({_THERMALS: columns}, {"thermal": False})
        candidate = {_THERMALS: None}
        requirement = SectionRequirement(
            "thermal",
            {_THERMALS: frozenset({"thermal_id"}), _THERMAL_LIST: frozenset()},
        )

        result = section_availability(requirement, baseline, candidate)

        assert result.unavailable == (Unavailable("thermal", None, BOTH_SIDES),)

    def test_unusable_list_on_both_sides_is_one_row_for_both(self) -> None:
        side = with_entity_lists({_THERMALS: _FULL}, {"thermal": False})
        requirement = SectionRequirement(
            "thermal", {_THERMALS: frozenset(), _THERMAL_LIST: frozenset()}
        )

        result = section_availability(requirement, side, side)

        assert result.unavailable == (Unavailable("thermal", None, BOTH_SIDES),)

    def test_a_usable_empty_list_satisfies_the_requirement(self) -> None:
        side = with_entity_lists({_THERMALS: _FULL}, {"thermal": True})
        requirement = SectionRequirement(
            "thermal", {_THERMALS: frozenset(), _THERMAL_LIST: frozenset()}
        )

        assert section_availability(requirement, side, side).available is True

    def test_an_unusable_list_is_a_row_of_every_section_requiring_its_key(
        self,
    ) -> None:
        side = with_entity_lists({_THERMALS: _FULL, _BUSES: _FULL}, {"thermal": False})
        needs_list = SectionRequirement("a", {_THERMAL_LIST: frozenset()})
        also_needs_list = SectionRequirement(
            "b", {_BUSES: frozenset(), _THERMAL_LIST: frozenset()}
        )
        without_list = SectionRequirement("c", {_BUSES: frozenset()})

        results = [
            section_availability(requirement, side, side)
            for requirement in (needs_list, also_needs_list, without_list)
        ]

        assert [result.available for result in results] == [False, False, True]
        assert unavailable_rows(results) == [
            Unavailable("a", None, BOTH_SIDES),
            Unavailable("b", None, BOTH_SIDES),
        ]

    def test_every_entity_type_adds_its_key(self) -> None:
        extended = with_entity_lists(
            {}, {"thermal": True, "line": False, "generic_constraint": True}
        )

        assert extended == {
            "system/thermals.json": frozenset(),
            "system/lines.json": None,
            "constraints/generic_constraints.json": frozenset(),
        }
        assert set(extended) == set(ENTITY_LIST_KEYS.values())

    def test_the_input_mapping_is_left_unmodified(self) -> None:
        partitions = {_THERMALS: _FULL, _THERMAL_LIST: frozenset({"stale"})}

        extended = with_entity_lists(partitions, {"thermal": True})

        assert partitions == {_THERMALS: _FULL, _THERMAL_LIST: frozenset({"stale"})}
        assert extended == {_THERMALS: _FULL, _THERMAL_LIST: frozenset()}
        assert extended is not partitions

    def test_an_entity_type_outside_the_table_raises_naming_it(self) -> None:
        with pytest.raises(ValueError, match="hydro"):
            with_entity_lists({}, {"hydro": True})

    def test_an_unknown_entity_type_raises_even_beside_a_known_one(self) -> None:
        with pytest.raises(ValueError, match="hydro"):
            with_entity_lists({}, {"thermal": True, "hydro": True})


class TestUnavailableRows:
    def test_rows_sort_by_section_then_column_with_none_first(self) -> None:
        availabilities = [
            SectionAvailability(
                "thermal",
                False,
                (),
                (
                    Unavailable("thermal", None, "baseline"),
                    Unavailable("thermal", "stage_id", "both"),
                ),
            ),
            SectionAvailability(
                "bus", True, (), (Unavailable("bus", "spot_price", "candidate"),)
            ),
            SectionAvailability(
                "hydro",
                False,
                (),
                (
                    Unavailable("hydro", "a_col", "both"),
                    Unavailable("hydro", "z_col", "baseline"),
                ),
            ),
            SectionAvailability(
                "line", False, (), (Unavailable("line", None, "both"),)
            ),
        ]

        assert unavailable_rows(availabilities) == [
            Unavailable("bus", "spot_price", "candidate"),
            Unavailable("hydro", "a_col", "both"),
            Unavailable("hydro", "z_col", "baseline"),
            Unavailable("line", None, "both"),
            Unavailable("thermal", None, "baseline"),
            Unavailable("thermal", "stage_id", "both"),
        ]

    def test_a_none_column_precedes_a_column_with_an_empty_name(self) -> None:
        availability = SectionAvailability(
            "s",
            False,
            (),
            (Unavailable("s", "", "both"), Unavailable("s", None, "both")),
        )

        assert unavailable_rows([availability]) == [
            Unavailable("s", None, "both"),
            Unavailable("s", "", "both"),
        ]

    def test_an_available_section_without_rows_adds_none(self) -> None:
        clean = SectionAvailability("bus", True, ("deficit_mw",), ())

        assert unavailable_rows([clean]) == []
        assert unavailable_rows([]) == []

    def test_two_availabilities_of_one_section_and_column_raise_naming_the_section(
        self,
    ) -> None:
        row = Unavailable("thermal", "stage_id", "both")
        twice = [
            SectionAvailability("thermal", False, (), (row,)),
            SectionAvailability("thermal", False, (), (row,)),
        ]

        with pytest.raises(ValueError, match="thermal"):
            unavailable_rows(twice)

    def test_two_availabilities_of_one_section_with_no_column_raise(self) -> None:
        row = Unavailable("thermal", None, "baseline")
        twice = [
            SectionAvailability("thermal", False, (), (row,)),
            SectionAvailability(
                "thermal", False, (), (Unavailable("thermal", None, "both"),)
            ),
        ]

        with pytest.raises(ValueError, match="thermal"):
            unavailable_rows(twice)

    def test_one_section_with_different_columns_across_two_availabilities_is_kept(
        self,
    ) -> None:
        first = SectionAvailability("s", False, (), (Unavailable("s", "a", "both"),))
        second = SectionAvailability("s", False, (), (Unavailable("s", "b", "both"),))

        assert unavailable_rows([first, second]) == [
            Unavailable("s", "a", "both"),
            Unavailable("s", "b", "both"),
        ]


class TestRequirementConstructor:
    def test_a_variable_reading_a_key_outside_the_partitions_raises(self) -> None:
        with pytest.raises(ValueError, match="generation_mw"):
            SectionRequirement(
                "thermal",
                {_THERMALS: frozenset({"thermal_id"})},
                {"generation_mw": (_BUSES, frozenset({"generation_mw"}))},
            )

    def test_the_error_names_the_section_and_the_key(self) -> None:
        with pytest.raises(ValueError, match=r"thermal.*simulation/buses"):
            SectionRequirement(
                "thermal",
                {_THERMALS: frozenset()},
                {"generation_mw": (_BUSES, frozenset({"generation_mw"}))},
            )

    def test_an_empty_variables_mapping_is_allowed(self) -> None:
        requirement = SectionRequirement("thermal", {_THERMALS: frozenset()})

        assert requirement.variables == {}

    def test_a_variable_reading_an_entity_list_key_it_requires_is_allowed(self) -> None:
        requirement = SectionRequirement(
            "thermal",
            {_THERMAL_LIST: frozenset()},
            {"count": (_THERMAL_LIST, frozenset())},
        )

        assert requirement.variables["count"] == (_THERMAL_LIST, frozenset())


def test_the_persisted_constants_keep_their_values() -> None:
    assert ENTITY_LIST_KEYS == {
        "thermal": "system/thermals.json",
        "line": "system/lines.json",
        "generic_constraint": "constraints/generic_constraints.json",
    }
    assert BOTH_SIDES == "both"

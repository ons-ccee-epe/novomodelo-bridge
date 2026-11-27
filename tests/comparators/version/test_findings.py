"""Tests for ``comparators.version.findings``: the Diagnostic each kind of finding
builds from the result types of the alignment, availability and convergence
modules, and the helper that emits them. Every input is a literal."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from novomodelo_bridge.comparators.version import findings
from novomodelo_bridge.comparators.version.alignment import (
    DuplicateName,
    NameAlignment,
    StageAlignment,
)
from novomodelo_bridge.comparators.version.availability import Unavailable
from novomodelo_bridge.comparators.version.constraints import DifferingConstraint
from novomodelo_bridge.comparators.version.convergence import (
    UB_COMPARABILITY,
    BoundsComparison,
    FinalBounds,
)
from novomodelo_bridge.comparators.version.findings import (
    differing_constraints,
    differing_iteration_counts,
    differing_stage_sets,
    duplicated_names,
    emit_findings,
    one_sided_entities,
    same_output_compared,
    unavailable_sections,
    upper_bound_not_comparable,
)
from novomodelo_bridge.core import diagnostics as dx

_RISK_SENTENCE = (
    "When the two cases declare different risk measures, their lower bounds bound "
    "different objectives, so a mismatch in the final lower bound can be genuine."
)
_BOUND_DIFFERENT_OBJECTIVES = "bound different objectives"
_NOT_COMPARABLE = tuple(reason for reason in UB_COMPARABILITY if reason != "comparable")
_REPO_INTERNAL_LEAKS = ("docs/", "plans/", "src/", "~/git", ".py", "ticket-", "epic-")


def _alignment(
    entity_type: str,
    only_in_baseline: tuple[str, ...] = (),
    only_in_candidate: tuple[str, ...] = (),
    duplicates: tuple[DuplicateName, ...] = (),
) -> NameAlignment:
    return NameAlignment(
        entity_type, (), only_in_baseline, only_in_candidate, duplicates
    )


def _final(
    iteration: int = 3, kind: str | None = "statistical", upper: float | None = 1.1e9
) -> FinalBounds:
    return FinalBounds(iteration=iteration, lower=1.0e9, upper=upper, kind=kind)


def _bounds(
    comparability: str = "kind-differs",
    *,
    baseline: FinalBounds | None = None,
    candidate: FinalBounds | None = None,
    risk_measures_differ: bool = False,
) -> BoundsComparison:
    return BoundsComparison(
        baseline=baseline or _final(),
        candidate=candidate or _final(),
        comparability=comparability,
        baseline_gap_percent=None,
        candidate_gap_percent=None,
        gap_delta_pp=None,
        risk_measures_differ=risk_measures_differ,
    )


def _present(diagnostic: dx.Diagnostic | None) -> dx.Diagnostic:
    assert diagnostic is not None
    return diagnostic


def _rows(diagnostic: dx.Diagnostic | None) -> list[list[object]]:
    table = _present(diagnostic).table
    assert table is not None
    return [list(row) for row in table.rows]


def _all_findings() -> list[dx.Diagnostic]:
    """One finding of each kind, each built from content."""
    built = [
        one_sided_entities([_alignment("hydro", only_in_candidate=("TUCURUI",))]),
        duplicated_names(
            [
                _alignment(
                    "thermal", duplicates=(DuplicateName("GAS_A", ("GAS_A",) * 2, ()),)
                )
            ]
        ),
        differing_stage_sets(StageAlignment((1, 2), (0,), (3,))),
        unavailable_sections([Unavailable("fpha", None, "both")]),
        upper_bound_not_comparable(_bounds("kind-differs")),
        differing_iteration_counts(
            _bounds(baseline=_final(iteration=5), candidate=_final(iteration=3))
        ),
        same_output_compared(Path("run"), Path("run")),
    ]
    return [_present(diagnostic) for diagnostic in built]


class TestOneSidedEntities:
    def test_lists_each_one_sided_entity_in_one_diagnostic_sorted_by_cells(
        self,
    ) -> None:
        diagnostic = one_sided_entities(
            [
                _alignment("hydro", only_in_candidate=("TUCURUI",)),
                _alignment("bus", only_in_baseline=("NORTE",)),
            ]
        )

        assert _present(diagnostic).code == "entity-one-sided"
        assert _present(diagnostic).severity is dx.Severity.WARNING
        assert _present(diagnostic).category == "Compare data"
        assert _present(diagnostic).title.endswith(" (2)")
        assert _rows(diagnostic) == [
            ["bus", "NORTE", "baseline"],
            ["hydro", "TUCURUI", "candidate"],
        ]

    def test_table_columns_name_the_type_the_entity_and_the_side(self) -> None:
        diagnostic = one_sided_entities([_alignment("bus", only_in_baseline=("N",))])

        table = _present(diagnostic).table
        assert table is not None
        assert table.columns == ["Entity type", "Name", "Only in"]

    def test_an_entity_on_both_sides_of_two_types_gives_one_row_per_side_and_type(
        self,
    ) -> None:
        diagnostic = one_sided_entities(
            [
                _alignment("thermal", ("B", "A"), ("C",)),
                _alignment("line", only_in_candidate=("A",)),
            ]
        )

        assert _present(diagnostic).title == "Entities found in one output only (4)"
        assert _rows(diagnostic) == [
            ["line", "A", "candidate"],
            ["thermal", "A", "baseline"],
            ["thermal", "B", "baseline"],
            ["thermal", "C", "candidate"],
        ]

    @pytest.mark.parametrize(
        "alignments",
        [
            [],
            [_alignment("hydro"), _alignment("bus")],
            [
                _alignment(
                    "thermal", duplicates=(DuplicateName("GAS_A", ("GAS_A",) * 2, ()),)
                )
            ],
        ],
        ids=["no alignment", "nothing one-sided", "duplicates are not one-sided"],
    )
    def test_returns_none_when_nothing_is_one_sided(
        self, alignments: list[NameAlignment]
    ) -> None:
        assert one_sided_entities(alignments) is None


class TestDuplicatedNames:
    def test_gives_one_row_per_output_that_holds_the_name_more_than_once(
        self,
    ) -> None:
        both = DuplicateName("FURNAS", ("Furnas", "FURNAS"), ("FURNAS", "Furnas"))
        baseline_only = DuplicateName("ITAIPU", ("ITAIPU", "ITAIPU"), ("ITAIPU",))
        candidate_only = DuplicateName("SOBRADINHO", (), ("SOBRADINHO", "SOBRADINHO"))

        diagnostic = duplicated_names(
            [_alignment("hydro", duplicates=(both, baseline_only, candidate_only))]
        )

        assert _present(diagnostic).code == "entity-name-duplicated"
        assert _present(diagnostic).severity is dx.Severity.WARNING
        assert _present(diagnostic).title.endswith(" (4)")
        assert _rows(diagnostic) == [
            ["hydro", "FURNAS, Furnas", "candidate"],
            ["hydro", "Furnas, FURNAS", "baseline"],
            ["hydro", "ITAIPU, ITAIPU", "baseline"],
            ["hydro", "SOBRADINHO, SOBRADINHO", "candidate"],
        ]

    def test_table_columns_name_the_type_the_name_and_the_output(self) -> None:
        duplicate = DuplicateName("A", ("A", "a"), ())

        diagnostic = duplicated_names([_alignment("bus", duplicates=(duplicate,))])

        table = _present(diagnostic).table
        assert table is not None
        assert table.columns == ["Entity type", "Name", "Output"]

    def test_the_summary_says_the_entities_are_left_out_of_the_comparison(
        self,
    ) -> None:
        duplicate = DuplicateName("A", ("A", "a"), ())

        diagnostic = duplicated_names([_alignment("bus", duplicates=(duplicate,))])

        assert _present(diagnostic).summary == (
            "The listed names belong to more than one entity in the baseline "
            "output or in the candidate output, so the comparison cannot tell "
            "which entity of the other output each one matches and leaves those "
            "entities out. Names that differ only in letter case or surrounding "
            "blanks count as one name."
        )

    @pytest.mark.parametrize(
        "alignments",
        [[], [_alignment("hydro", only_in_baseline=("A",))]],
        ids=["no alignment", "no duplicate"],
    )
    def test_returns_none_when_no_name_is_duplicated(
        self, alignments: list[NameAlignment]
    ) -> None:
        assert duplicated_names(alignments) is None


class TestDifferingStageSets:
    def test_lists_each_one_sided_stage_with_the_zero_based_caption(self) -> None:
        diagnostic = differing_stage_sets(StageAlignment((1, 2), (0,), (3,)))

        assert _present(diagnostic).code == "stage-sets-differ"
        assert _present(diagnostic).severity is dx.Severity.WARNING
        assert _present(diagnostic).title == "Stages found in one output only (2)"
        table = _present(diagnostic).table
        assert table is not None
        assert table.columns == ["Stage", "Only in"]
        assert table.caption == "Stage ids are 0-based."
        assert _rows(diagnostic) == [[0, "baseline"], [3, "candidate"]]

    def test_stages_sort_numerically_not_as_text(self) -> None:
        diagnostic = differing_stage_sets(StageAlignment((), (10, 2), (2,)))

        assert _rows(diagnostic) == [
            [2, "baseline"],
            [2, "candidate"],
            [10, "baseline"],
        ]

    def test_returns_none_when_both_outputs_hold_the_same_stages(self) -> None:
        assert differing_stage_sets(StageAlignment((0, 1, 2), (), ())) is None
        assert differing_stage_sets(StageAlignment((), (), ())) is None


_CONSTRAINT_SUMMARY = (
    "The listed generic constraints have the same name in the baseline output and "
    "in the candidate output and different terms or bounds over the stages both "
    "outputs hold. The report draws each one with the terms and bounds of the "
    "candidate output and evaluates each output with its own definition, so the "
    "two values of a listed constraint are not values of the same constraint."
)


def _differing(count: int = 2) -> dx.Diagnostic:
    return _present(
        differing_constraints(
            (
                DifferingConstraint("RE_1", False, True),
                DifferingConstraint("VMINOP_1", True, True),
                DifferingConstraint("AGRINT_1", True, False),
            )[:count]
        )
    )


class TestDifferingConstraints:
    def test_lists_each_constraint_with_where_it_differs(self) -> None:
        diagnostic = _present(
            differing_constraints(
                (
                    DifferingConstraint("RE_1", False, True),
                    DifferingConstraint("VMINOP_1", True, True),
                )
            )
        )

        assert diagnostic.code == "generic-constraint-differs"
        assert diagnostic.severity is dx.Severity.WARNING
        assert diagnostic.category == "Compare data"
        assert diagnostic.title == (
            "Generic constraints defined differently in the two outputs (2)"
        )
        assert _rows(diagnostic) == [
            ["RE_1", "bounds"],
            ["VMINOP_1", "terms and bounds"],
        ]

    def test_the_second_cell_names_terms_bounds_or_both(self) -> None:
        diagnostic = _present(
            differing_constraints(
                (
                    DifferingConstraint("A", True, False),
                    DifferingConstraint("B", False, True),
                    DifferingConstraint("C", True, True),
                )
            )
        )

        assert _rows(diagnostic) == [
            ["A", "terms"],
            ["B", "bounds"],
            ["C", "terms and bounds"],
        ]

    def test_table_columns_name_the_constraint_and_where_it_differs(self) -> None:
        table = _present(_differing()).table

        assert table is not None
        assert table.columns == ["Constraint", "Differs in"]

    def test_the_title_counts_the_rows(self) -> None:
        assert _differing(1).title.endswith("two outputs (1)")
        assert _differing(3).title.endswith("two outputs (3)")

    def test_rows_are_deterministic_whatever_the_input_order(self) -> None:
        constraints = (
            DifferingConstraint("VMINOP_1", True, True),
            DifferingConstraint("RE_1", False, True),
        )

        forward = differing_constraints(constraints)
        backward = differing_constraints(constraints[::-1])

        assert forward == backward
        assert _rows(forward) == [["RE_1", "bounds"], ["VMINOP_1", "terms and bounds"]]

    def test_accepts_any_sequence_of_constraints(self) -> None:
        constraint = DifferingConstraint("RE_1", True, False)

        assert differing_constraints([constraint]) == differing_constraints(
            (constraint,)
        )

    def test_returns_none_for_no_constraints(self) -> None:
        assert differing_constraints(()) is None
        assert differing_constraints([]) is None

    def test_the_summary_says_each_output_is_evaluated_with_its_own_definition(
        self,
    ) -> None:
        assert _differing().summary == _CONSTRAINT_SUMMARY

    def test_the_finding_carries_no_notes_and_no_remediation(self) -> None:
        diagnostic = _differing()

        assert diagnostic.notes == []
        assert diagnostic.remediation is None

    def test_it_is_emitted_once_at_warning_level_by_emit_findings(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        built = _differing()

        with dx.collect() as sink:
            emit_findings([None, built])
        with caplog.at_level(logging.INFO, logger=findings.__name__):
            emit_findings([built])

        assert sink == [built]
        assert [record.levelno for record in caplog.records] == [logging.WARNING]
        assert [record.getMessage() for record in caplog.records] == [built.summary]

    @pytest.mark.parametrize("count", [1, 2, 3])
    def test_no_text_leaks_a_repository_reference(self, count: int) -> None:
        for text in _texts(_differing(count)):
            for leak in _REPO_INTERNAL_LEAKS:
                assert leak not in text, f"{leak!r} in {text!r}"

    @pytest.mark.parametrize("count", [1, 2, 3])
    def test_the_summary_names_both_outputs_in_full_sentences(self, count: int) -> None:
        summary = _differing(count).summary

        assert "the baseline output" in summary.lower()
        assert "the candidate output" in summary.lower()
        assert summary[:1].isupper()
        assert summary.endswith(".")
        assert "(s)" not in summary
        assert re.search(r"\w_\w", summary) is None

    @pytest.mark.parametrize("count", [1, 2, 3])
    def test_the_title_is_sentence_case_without_a_period_and_counts_its_rows(
        self, count: int
    ) -> None:
        diagnostic = _differing(count)

        assert diagnostic.title[:1].isupper()
        assert not diagnostic.title.endswith(".")
        assert "(s)" not in diagnostic.title
        assert diagnostic.title.endswith(f" ({len(_rows(diagnostic))})")


class TestUnavailableSections:
    def test_one_row_per_row_with_a_dash_for_a_whole_section(self) -> None:
        rows = [
            Unavailable("bus", "spot_price", "candidate"),
            Unavailable("fpha", None, "both"),
            Unavailable("thermal", None, "baseline"),
        ]

        diagnostic = unavailable_sections(rows)

        assert _present(diagnostic).code == "section-unavailable"
        assert _present(diagnostic).severity is dx.Severity.WARNING
        assert _present(diagnostic).title == "Results unavailable (3)"
        table = _present(diagnostic).table
        assert table is not None
        assert table.columns == ["Section", "Column", "Missing in"]
        assert _rows(diagnostic) == [
            ["bus", "spot_price", "candidate"],
            ["fpha", "—", "both"],
            ["thermal", "—", "baseline"],
        ]

    def test_summary_is_true_for_one_side_and_for_both(self) -> None:
        diagnostic = unavailable_sections([Unavailable("fpha", None, "both")])

        summary = _present(diagnostic).summary
        assert "one of the outputs" not in summary
        assert "the baseline output, the candidate output or both" in summary
        assert "leaves" in summary

    def test_a_both_row_stays_one_row_and_two_rows_stay_two(self) -> None:
        assert _rows(unavailable_sections([Unavailable("fpha", None, "both")])) == [
            ["fpha", "—", "both"]
        ]
        assert _rows(
            unavailable_sections(
                [
                    Unavailable("fpha", None, "baseline"),
                    Unavailable("fpha", None, "candidate"),
                ]
            )
        ) == [["fpha", "—", "baseline"], ["fpha", "—", "candidate"]]

    def test_a_whole_section_row_comes_before_the_columns_of_its_section(
        self,
    ) -> None:
        diagnostic = unavailable_sections(
            [
                Unavailable("bus", "spot_price", "baseline"),
                Unavailable("bus", None, "candidate"),
                Unavailable("bus", "deficit_mw", "both"),
            ]
        )

        assert _rows(diagnostic) == [
            ["bus", "—", "candidate"],
            ["bus", "deficit_mw", "both"],
            ["bus", "spot_price", "baseline"],
        ]

    def test_rows_are_deterministic_whatever_the_input_order(self) -> None:
        rows = [
            Unavailable("thermal", None, "baseline"),
            Unavailable("bus", "spot_price", "candidate"),
            Unavailable("fpha", None, "both"),
        ]

        assert _rows(unavailable_sections(rows)) == _rows(
            unavailable_sections(rows[::-1])
        )

    def test_returns_none_for_no_rows(self) -> None:
        assert unavailable_sections([]) is None


class TestUpperBoundNotComparable:
    def test_incomparable_pair_is_an_info_finding_with_one_note_per_side(self) -> None:
        bounds = _bounds(
            "kind-differs",
            baseline=_final(kind="statistical"),
            candidate=_final(kind="exact"),
        )

        diagnostic = upper_bound_not_comparable(bounds)

        assert _present(diagnostic).code == "upper-bound-not-comparable"
        assert _present(diagnostic).severity is dx.Severity.INFO
        assert _present(diagnostic).category == "Compare data"
        assert (
            _present(diagnostic).title == "Upper bounds of the two runs not comparable"
        )
        assert _present(diagnostic).table is None
        assert _present(diagnostic).notes == [
            "Baseline output: upper bound kind statistical.",
            "Candidate output: upper bound kind exact.",
        ]

    def test_a_comparable_pair_gives_none(self) -> None:
        assert upper_bound_not_comparable(_bounds("comparable")) is None

    def test_an_unknown_reason_raises_naming_it(self) -> None:
        with pytest.raises(ValueError, match="nonsense"):
            upper_bound_not_comparable(_bounds("nonsense"))

    def test_every_reason_of_the_convergence_module_has_its_own_text(self) -> None:
        summaries = {
            reason: _present(upper_bound_not_comparable(_bounds(reason))).summary
            for reason in _NOT_COMPARABLE
        }

        assert len(_NOT_COMPARABLE) == 6
        assert len(set(summaries.values())) == 6

    def test_bounds_missing_judges_neither_bound_and_never_mentions_the_version(
        self,
    ) -> None:
        bounds = _bounds("bounds-missing", baseline=_final(), candidate=_final())

        summary = _present(upper_bound_not_comparable(bounds)).summary

        assert "judges neither" in summary
        assert "0.14" not in summary

    def test_kind_missing_names_the_novomodelo_version_that_records_the_kind(
        self,
    ) -> None:
        summary = _present(upper_bound_not_comparable(_bounds("kind-missing"))).summary

        assert "novomodelo 0.14" in summary
        assert "final lower bound only" in summary

    def test_kind_differs_says_one_run_simulated_and_the_other_was_exact(self) -> None:
        summary = _present(upper_bound_not_comparable(_bounds("kind-differs"))).summary

        assert "by simulation" in summary
        assert "computed it exactly" in summary

    def test_risk_averse_says_the_simulated_mean_cost_does_not_bound_the_objective(
        self,
    ) -> None:
        summary = _present(upper_bound_not_comparable(_bounds("risk-averse"))).summary

        assert "risk-averse" in summary
        assert "simulated mean cost" in summary

    def test_risk_unknown_says_the_risk_measure_could_not_be_read(self) -> None:
        summary = _present(upper_bound_not_comparable(_bounds("risk-unknown"))).summary

        assert "could not be read" in summary
        assert "simulated mean cost" in summary

    def test_bound_nonpositive_says_a_final_bound_has_no_usable_value(self) -> None:
        summary = _present(
            upper_bound_not_comparable(_bounds("bound-nonpositive"))
        ).summary

        assert "zero, negative" in summary

    @pytest.mark.parametrize("reason", ["risk-averse", "risk-unknown"])
    def test_a_risk_reason_ends_with_the_risk_measures_sentence_without_the_flag(
        self, reason: str
    ) -> None:
        summary = _present(
            upper_bound_not_comparable(_bounds(reason, risk_measures_differ=False))
        ).summary

        assert summary.endswith(_RISK_SENTENCE)
        assert summary.count(_BOUND_DIFFERENT_OBJECTIVES) == 1
        assert "final lower bound can be genuine" in summary

    @pytest.mark.parametrize("reason", _NOT_COMPARABLE)
    def test_every_reason_with_the_flag_carries_the_sentence_exactly_once(
        self, reason: str
    ) -> None:
        summary = _present(
            upper_bound_not_comparable(_bounds(reason, risk_measures_differ=True))
        ).summary

        assert summary.endswith(_RISK_SENTENCE)
        assert summary.count(_BOUND_DIFFERENT_OBJECTIVES) == 1
        assert summary.count(_RISK_SENTENCE) == 1

    @pytest.mark.parametrize(
        "reason",
        ["bounds-missing", "kind-missing", "kind-differs", "bound-nonpositive"],
    )
    def test_other_reasons_without_the_flag_omit_the_sentence(
        self, reason: str
    ) -> None:
        summary = _present(
            upper_bound_not_comparable(_bounds(reason, risk_measures_differ=False))
        ).summary

        assert _BOUND_DIFFERENT_OBJECTIVES not in summary
        assert "genuine" not in summary

    def test_bounds_missing_with_the_flag_adds_the_sentence_after_the_judgement(
        self,
    ) -> None:
        summary = _present(
            upper_bound_not_comparable(
                _bounds("bounds-missing", risk_measures_differ=True)
            )
        ).summary

        assert summary.index("judges neither") < summary.index(_RISK_SENTENCE)

    def test_a_comparable_pair_gives_none_even_when_the_flag_is_true(self) -> None:
        assert (
            upper_bound_not_comparable(_bounds("comparable", risk_measures_differ=True))
            is None
        )

    def test_a_side_without_bounds_says_it_has_no_training_convergence_data(
        self,
    ) -> None:
        bounds = BoundsComparison(
            baseline=None,
            candidate=_final(kind="exact"),
            comparability="bounds-missing",
            baseline_gap_percent=None,
            candidate_gap_percent=None,
            gap_delta_pp=None,
            risk_measures_differ=False,
        )

        notes = _present(upper_bound_not_comparable(bounds)).notes

        assert notes == [
            "Baseline output: no training convergence data.",
            "Candidate output: upper bound kind exact.",
        ]

    def test_a_side_whose_kind_is_null_says_the_kind_is_not_recorded(self) -> None:
        bounds = _bounds("kind-missing", candidate=_final(kind=None))

        notes = _present(upper_bound_not_comparable(bounds)).notes

        assert notes == [
            "Baseline output: upper bound kind statistical.",
            "Candidate output: the kind of its upper bound is not recorded.",
        ]


class TestDifferingIterationCounts:
    def test_different_final_iterations_give_an_info_finding_naming_both(self) -> None:
        bounds = _bounds(baseline=_final(iteration=5), candidate=_final(iteration=3))

        diagnostic = differing_iteration_counts(bounds)

        assert _present(diagnostic).code == "iteration-counts-differ"
        assert _present(diagnostic).severity is dx.Severity.INFO
        assert _present(diagnostic).category == "Compare data"
        assert (
            _present(diagnostic).title
            == "Training runs stopped at different iterations"
        )
        assert "5 in the baseline output" in _present(diagnostic).summary
        assert "3 in the candidate output" in _present(diagnostic).summary

    def test_the_finding_does_not_depend_on_comparability(self) -> None:
        bounds = _bounds(
            "comparable", baseline=_final(iteration=2), candidate=_final(iteration=9)
        )

        assert differing_iteration_counts(bounds) is not None

    def test_equal_final_iterations_give_none(self) -> None:
        bounds = _bounds(
            "comparable", baseline=_final(iteration=4), candidate=_final(iteration=4)
        )

        assert differing_iteration_counts(bounds) is None

    @pytest.mark.parametrize("missing", ["baseline", "candidate"])
    def test_a_side_without_final_bounds_gives_none(self, missing: str) -> None:
        bounds = BoundsComparison(
            baseline=None if missing == "baseline" else _final(iteration=5),
            candidate=None if missing == "candidate" else _final(iteration=3),
            comparability="bounds-missing",
            baseline_gap_percent=None,
            candidate_gap_percent=None,
            gap_delta_pp=None,
            risk_measures_differ=False,
        )

        assert differing_iteration_counts(bounds) is None


class TestSameOutputCompared:
    def test_the_same_directory_twice_gives_an_info_finding(
        self, tmp_path: Path
    ) -> None:
        output = tmp_path / "output"
        output.mkdir()

        diagnostic = same_output_compared(output, output)

        assert _present(diagnostic).code == "same-output-compared"
        assert _present(diagnostic).severity is dx.Severity.INFO
        assert _present(diagnostic).category == "Compare data"
        assert (
            _present(diagnostic).title == "Baseline and candidate are the same output"
        )

    def test_two_spellings_of_one_directory_count_as_the_same(
        self, tmp_path: Path
    ) -> None:
        output = tmp_path / "output"
        elsewhere = tmp_path / "elsewhere"
        output.mkdir()
        elsewhere.mkdir()

        assert same_output_compared(output, elsewhere / ".." / "output") is not None

    def test_two_different_directories_give_none(self, tmp_path: Path) -> None:
        baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
        baseline.mkdir()
        candidate.mkdir()

        assert same_output_compared(baseline, candidate) is None

    def test_the_remediation_names_both_command_arguments(self) -> None:
        diagnostic = same_output_compared(Path("run"), Path("run"))

        remediation = _present(diagnostic).remediation
        assert remediation is not None
        assert "BASELINE_OUTPUT" in remediation
        assert "CANDIDATE_OUTPUT" in remediation


class TestOnlySameOutputSetsARemediation:
    def test_no_other_finding_carries_a_remediation(self) -> None:
        with_remediation = {
            diagnostic.code for diagnostic in _all_findings() if diagnostic.remediation
        }

        assert with_remediation == {"same-output-compared"}


class TestEmitFindings:
    def test_emits_every_finding_in_order_and_skips_none(self) -> None:
        built = _all_findings()

        with dx.collect() as sink:
            emit_findings([None, built[0], None, *built[1:], None])

        assert sink == built

    def test_accepts_any_iterable_and_emits_nothing_for_nothing(self) -> None:
        with dx.collect() as sink:
            emit_findings(iter([]))
            emit_findings([None, None])
            emit_findings(finding for finding in [None])

        assert sink == []

    def test_builder_results_flow_straight_into_emit_findings(self) -> None:
        with dx.collect() as sink:
            emit_findings(
                [
                    one_sided_entities([_alignment("bus", ("A", "B"), ("C",))]),
                    unavailable_sections([Unavailable("fpha", None, "both")]),
                    same_output_compared(Path("a"), Path("b")),
                ]
            )

        assert [diagnostic.code for diagnostic in sink] == [
            "entity-one-sided",
            "section-unavailable",
        ]

    def test_without_a_sink_each_finding_is_one_log_record_at_its_severity(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        built = _all_findings()

        with caplog.at_level(logging.INFO, logger=findings.__name__):
            emit_findings(built)

        assert [record.name for record in caplog.records] == [findings.__name__] * 7
        assert [record.levelno for record in caplog.records] == [
            diagnostic.severity.log_level for diagnostic in built
        ]
        assert [record.getMessage() for record in caplog.records] == [
            diagnostic.summary for diagnostic in built
        ]


def _texts(diagnostic: dx.Diagnostic) -> Iterator[str]:
    yield diagnostic.title
    yield diagnostic.summary
    yield from diagnostic.notes
    if diagnostic.remediation:
        yield diagnostic.remediation


class TestMessageHygiene:
    @pytest.mark.parametrize(
        "diagnostic", _all_findings(), ids=lambda diagnostic: diagnostic.code
    )
    def test_no_text_leaks_a_repository_reference(
        self, diagnostic: dx.Diagnostic
    ) -> None:
        for text in _texts(diagnostic):
            for leak in _REPO_INTERNAL_LEAKS:
                assert leak not in text, f"{leak!r} in {text!r}"

    @pytest.mark.parametrize(
        "diagnostic", _all_findings(), ids=lambda diagnostic: diagnostic.code
    )
    def test_the_summary_names_both_outputs_in_full_sentences(
        self, diagnostic: dx.Diagnostic
    ) -> None:
        summary = diagnostic.summary

        assert "the baseline output" in summary.lower()
        assert "the candidate output" in summary.lower()
        assert summary[:1].isupper()
        assert summary.endswith(".")
        assert "(s)" not in summary

    @pytest.mark.parametrize("reason", _NOT_COMPARABLE)
    @pytest.mark.parametrize("risk_measures_differ", [False, True])
    def test_every_upper_bound_text_reads_as_full_sentences_naming_both_outputs(
        self, reason: str, risk_measures_differ: bool
    ) -> None:
        diagnostic = _present(
            upper_bound_not_comparable(
                _bounds(reason, risk_measures_differ=risk_measures_differ)
            )
        )

        assert "the baseline output" in diagnostic.summary.lower()
        assert "the candidate output" in diagnostic.summary.lower()
        assert diagnostic.summary[:1].isupper()
        assert diagnostic.summary.endswith(".")
        assert re.search(r"\w_\w", diagnostic.summary) is None
        assert "(s)" not in diagnostic.summary
        for leak in _REPO_INTERNAL_LEAKS:
            assert leak not in diagnostic.summary

    def test_every_title_is_sentence_case_without_a_period_and_counts_its_rows(
        self,
    ) -> None:
        for diagnostic in _all_findings():
            assert diagnostic.title[:1].isupper()
            assert not diagnostic.title.endswith(".")
            if diagnostic.table is not None:
                count = len(diagnostic.table.rows)
                assert diagnostic.title.endswith(f" ({count})")
            else:
                assert re.search(r" \(\d+\)$", diagnostic.title) is None

    def test_every_finding_is_in_the_category_compare_data(self) -> None:
        assert {d.category for d in _all_findings()} == {"Compare data"}

    def test_codes_and_severities_are_the_contract(self) -> None:
        assert {d.code: d.severity for d in _all_findings()} == {
            "entity-one-sided": dx.Severity.WARNING,
            "entity-name-duplicated": dx.Severity.WARNING,
            "stage-sets-differ": dx.Severity.WARNING,
            "section-unavailable": dx.Severity.WARNING,
            "upper-bound-not-comparable": dx.Severity.INFO,
            "iteration-counts-differ": dx.Severity.INFO,
            "same-output-compared": dx.Severity.INFO,
        }

    def test_the_four_warnings_carry_a_table_and_the_three_info_findings_do_not(
        self,
    ) -> None:
        for diagnostic in _all_findings():
            has_table = diagnostic.table is not None
            assert has_table == (diagnostic.severity is dx.Severity.WARNING)

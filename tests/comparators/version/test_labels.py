"""Tests for ``comparators.version.labels``: the side labels of a version comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

from novomodelo_bridge.comparators.version.labels import SIDES, side_labels

UNSAFE_CHARACTERS = [
    pytest.param("'", id="apostrophe"),
    pytest.param("\\", id="backslash"),
    pytest.param("<", id="less-than"),
    pytest.param(">", id="greater-than"),
    pytest.param("&", id="ampersand"),
    pytest.param('"', id="double-quote"),
]


def _dirs(tmp_path: Path, baseline: str = "jan-26", candidate: str = "fev-26"):
    return (tmp_path / baseline / "output", tmp_path / candidate / "output")


def test_sides_are_baseline_then_candidate() -> None:
    assert SIDES == ("baseline", "candidate")


class TestDefaults:
    def test_differing_versions_name_the_release(self, tmp_path: Path) -> None:
        labels = side_labels(("0.19.1", "0.20.0"), _dirs(tmp_path))

        assert labels == ("Novomodelo 0.19.1", "Novomodelo 0.20.0")

    def test_versions_are_stripped_before_they_are_compared(
        self, tmp_path: Path
    ) -> None:
        labels = side_labels((" 0.20.0 ", "0.20.0"), _dirs(tmp_path))

        assert labels == ("jan-26", "fev-26")

    def test_equal_versions_name_the_case_directories(self, tmp_path: Path) -> None:
        labels = side_labels(("0.20.0", "0.20.0"), _dirs(tmp_path))

        assert labels == ("jan-26", "fev-26")

    @pytest.mark.parametrize(
        "versions",
        [(None, "0.20.0"), ("0.20.0", None), (None, None), ("  ", "0.20.0")],
        ids=["baseline-unknown", "candidate-unknown", "both-unknown", "blank"],
    )
    def test_an_unknown_version_names_the_case_directories(
        self, tmp_path: Path, versions: tuple[str | None, str | None]
    ) -> None:
        assert side_labels(versions, _dirs(tmp_path)) == ("jan-26", "fev-26")

    def test_a_directory_not_named_output_is_its_own_name(self, tmp_path: Path) -> None:
        dirs = (tmp_path / "runs" / "v1", tmp_path / "runs" / "v2")

        assert side_labels((None, None), dirs) == ("v1", "v2")

    def test_a_relative_directory_is_resolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "jan-26" / "output").mkdir(parents=True)
        monkeypatch.chdir(tmp_path / "jan-26" / "output")
        dirs = (Path("."), tmp_path / "fev-26" / "output")

        assert side_labels((None, None), dirs) == ("jan-26", "fev-26")

    def test_a_directory_with_no_usable_name_is_numbered(self, tmp_path: Path) -> None:
        dirs = (tmp_path / "   " / "output", Path("/"))

        assert side_labels((None, None), dirs) == ("Output 1", "Output 2")

    def test_equal_directory_names_are_numbered(self, tmp_path: Path) -> None:
        dirs = (tmp_path / "a" / "case" / "output", tmp_path / "b" / "case" / "output")

        assert side_labels(("0.20.0", "0.20.0"), dirs) == ("case (1)", "case (2)")


class TestChosenLabels:
    def test_chosen_labels_win_over_the_versions(self, tmp_path: Path) -> None:
        labels = side_labels(("0.19.1", "0.20.0"), _dirs(tmp_path), ("antes", "depois"))

        assert labels == ("antes", "depois")

    def test_one_chosen_label_leaves_the_other_side_its_default(
        self, tmp_path: Path
    ) -> None:
        labels = side_labels(("0.19.1", "0.20.0"), _dirs(tmp_path), (None, "depois"))

        assert labels == ("Novomodelo 0.19.1", "depois")

    def test_a_blank_chosen_label_counts_as_absent(self, tmp_path: Path) -> None:
        labels = side_labels(("0.20.0", "0.20.0"), _dirs(tmp_path), ("  ", None))

        assert labels == ("jan-26", "fev-26")

    def test_equal_chosen_labels_are_numbered(self, tmp_path: Path) -> None:
        labels = side_labels((None, None), _dirs(tmp_path), ("run", "run"))

        assert labels == ("run (1)", "run (2)")

    def test_accented_letters_are_kept(self, tmp_path: Path) -> None:
        labels = side_labels(
            (None, None), _dirs(tmp_path), ("pré-inversão", "LPPs completas")
        )

        assert labels == ("pré-inversão", "LPPs completas")


class TestCleaning:
    @pytest.mark.parametrize("char", UNSAFE_CHARACTERS)
    def test_unsafe_characters_in_a_chosen_label_become_underscores(
        self, tmp_path: Path, char: str
    ) -> None:
        labels = side_labels((None, None), _dirs(tmp_path), (f"a{char}b", None))

        assert labels[0] == "a_b"

    @pytest.mark.parametrize("char", UNSAFE_CHARACTERS)
    def test_unsafe_characters_in_a_version_become_underscores(
        self, tmp_path: Path, char: str
    ) -> None:
        labels = side_labels((f"1{char}2", "0.20.0"), _dirs(tmp_path))

        assert labels == ("Novomodelo 1_2", "Novomodelo 0.20.0")

    @pytest.mark.parametrize("char", UNSAFE_CHARACTERS)
    def test_unsafe_characters_in_a_directory_become_underscores(
        self, tmp_path: Path, char: str
    ) -> None:
        labels = side_labels((None, None), _dirs(tmp_path, f"a{char}b"))

        assert labels[0] == "a_b"

    def test_surrounding_blanks_are_stripped(self, tmp_path: Path) -> None:
        labels = side_labels((None, None), _dirs(tmp_path), ("  antes  ", None))

        assert labels[0] == "antes"

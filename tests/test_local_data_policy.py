"""Local-data policy: no test depends on files outside the repository.

The suite once carried tests guarded on real decks under the gitignored
``example/`` tree and on a sibling novomodelo checkout under the developer's home
directory. Those ran on one machine at best and skipped everywhere else, which
reads as coverage while guarding nothing. The policy now is that every test
runs from the repository alone: real-format data lives as small excerpts under
``tests/fixtures/``, synthetic decks under ``tests/decks/``, and a check that
needs a whole deck is written up in ``docs/real-deck-checks.md`` instead of
being kept as a permanently skipped test.

This module enforces the policy with a plain ``re`` scan over every sibling
test module's source text: never an import, never a filesystem read outside
``tests/``.
"""

from __future__ import annotations

import re
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parent

# A `Path("example/...")` literal or a `/ "example"` path segment; a prose
# mention of the convention inside a docstring is not a path build and does
# not match.
_EXAMPLE_TREE = re.compile(r"""Path\(\s*["']example/|/\s*["']example["']""")
# `Path.home()` / `expanduser()` reach outside the repository by construction.
_HOME_DIRECTORY = re.compile(r"Path\.home\(\)|\.expanduser\(")


def _sibling_sources() -> list[tuple[Path, str]]:
    self_path = Path(__file__).resolve()
    return [
        (path, path.read_text(encoding="utf-8"))
        for path in sorted(_TESTS_ROOT.rglob("*.py"))
        if path.resolve() != self_path and "__pycache__" not in path.parts
    ]


def _offenders(pattern: re.Pattern[str]) -> list[str]:
    return [
        f"{path.relative_to(_TESTS_ROOT)}:{text.count(chr(10), 0, match.start()) + 1}"
        for path, text in _sibling_sources()
        for match in pattern.finditer(text)
    ]


def test_no_test_reads_the_gitignored_example_tree() -> None:
    """No test module builds a path into ``example/``.

    Real-format inputs belong under ``tests/fixtures/`` as excerpts and
    synthetic decks under ``tests/decks/``; a whole-deck check belongs in
    ``docs/real-deck-checks.md``, not in the suite.
    """
    assert _offenders(_EXAMPLE_TREE) == []


def test_no_test_depends_on_the_developer_home_directory() -> None:
    """No test module resolves a path through the user's home directory.

    A sibling checkout or a locally built binary is one developer's layout,
    not a fixture; vendor the file under ``tests/fixtures/`` instead.
    """
    assert _offenders(_HOME_DIRECTORY) == []

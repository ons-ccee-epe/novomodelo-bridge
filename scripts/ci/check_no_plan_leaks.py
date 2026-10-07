#!/usr/bin/env python3
"""check_no_plan_leaks.py — plan-structure leak gate.

Plans live in ``plans/`` (gitignored); shipped artifacts describe behavior,
not how the work was organized (`.claude/rules/comments.md` N4,
`.claude/rules/doc-integrity.md` §5).

Three HARD scopes (any non-empty ⇒ exit 1):

* shipped docs — whole-file scan of README.md, CONTRIBUTING.md, CHANGELOG.md,
  docs/**/*.md.
* src/ prose — comments + docstrings under src/. The pre-existing debt was
  burned down to zero, so this scope is now enforced rather than advisory; a
  new ``ticket-NNN``/``epic-NN`` in shipped source fails the build. On failure
  the src hits are grouped by file so the offender is obvious.
* tests/ prose — comments + docstrings under tests/, same rule and burndown.
  Test *file names* and *test-function names* are never scanned (only comments
  and docstrings are), and identifiers join plan ids with ``_`` (a word
  character), so ``test_ticket_016_regression`` never matches while the prose
  "ticket 016" in a comment or docstring does.

``--all`` prints every src hit (for triage); otherwise the top offender files.

Exit codes: 0 = clean; 1 = leaks in either scope (details printed).
"""

from __future__ import annotations

import re
import sys
from collections import Counter

from _scan import REPO_ROOT, SRC_ROOT, TESTS_ROOT, iter_prose, iter_py_files, rel

PATTERN = re.compile(
    r"\b[Ee]pics?\b|\b[Tt]ickets?\b|\b[Ss]prints?\b|\bT0[0-9][0-9]\b"
    r"|\bAC: |\bAC-[0-9]\b|\b[Dd]ecision [A-Z][0-9]+\b"
    # Roadmap finding-ids (audit registry lives under the gitignored plans/):
    # CONV-08, CMP-06, CLI-12, TST-04, GHA-2, ... a pip reader cannot resolve them.
    r"|\b(?:CONV|CMP|CLI|TST|GHA)-[0-9]+\b"
    # All-caps hyphenated plan-token form (TICKET-012, EPIC-08, SPRINT-3, ...).
    r"|\b(?:TICKET|EPIC|SPRINT)-[0-9]+\b"
    # Finding-id and no-hyphen acceptance-criteria tag forms (FINDING-5, AC5).
    r"|\bFINDING-[0-9]+\b|\bAC[0-9]+\b|\bAC ?#[0-9]+"
)

HARD_DOCS = ["README.md", "CONTRIBUTING.md", "CHANGELOG.md"]
HARD_GLOBS = ["docs/**/*.md"]


def doc_scope_hits() -> list[str]:
    hits: list[str] = []
    files = [REPO_ROOT / name for name in HARD_DOCS]
    for glob in HARD_GLOBS:
        files.extend(sorted(REPO_ROOT.glob(glob)))
    for path in files:
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{rel(path)}:{lineno}: {line.strip()}")
    return hits


def prose_scope_hits(root) -> list[str]:
    hits: list[str] = []
    for path in iter_py_files(root):
        for lineno, text in iter_prose(path):
            for offset, line in enumerate(text.splitlines()):
                if PATTERN.search(line):
                    hits.append(f"{rel(path)}:{lineno + offset}: {line.strip()}")
    return hits


def _print_prose_scope(label: str, hits: list[str], *, show_all: bool) -> None:
    print()
    by_file = Counter(hit.split(":", 1)[0] for hit in hits)
    print(f"{label} ({len(hits)} line(s); amputate the token, keep the invariant):")
    for name, count in by_file.most_common(None if show_all else 10):
        print(f"  {count:4d}  {name}")
    if show_all:
        print()
        print("\n".join(f"  {hit}" for hit in hits))
    else:
        print("  (--all prints every hit)")


def main() -> int:
    show_all = "--all" in sys.argv[1:]

    docs = doc_scope_hits()
    src = prose_scope_hits(SRC_ROOT)
    tests = prose_scope_hits(TESTS_ROOT)

    if docs or src or tests:
        print("FAIL: plan-structure leaks in shipped artifacts.")
        if docs:
            print()
            print("Shipped docs (rewrite in behavioural terms or move to plans/):")
            print("\n".join(f"  {hit}" for hit in docs))
        if src:
            _print_prose_scope("src/ prose", src, show_all=show_all)
        if tests:
            _print_prose_scope("tests/ prose", tests, show_all=show_all)
        print()
        print("See .claude/rules/comments.md N4 / .claude/rules/doc-integrity.md §5.")
        return 1

    print("OK: no plan-structure leaks in shipped docs, src/, or tests/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

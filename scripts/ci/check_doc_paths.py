#!/usr/bin/env python3
"""check_doc_paths.py — resolvable-path gate for shipped docs.

Every repo-relative path cited in README.md, CONTRIBUTING.md, CHANGELOG.md,
docs/**/*.md and CLAUDE.md must resolve against the live tree
(`.claude/rules/doc-integrity.md` §3.4). Citing a gitignored dir (``plans/``) or a machine-local path
(``~/git/...``) in a shipped doc is a violation outright — no reader can
resolve it. CLAUDE.md is the one file allowed to name those two prefixes,
because it documents the convention that bans them.

Recognized citation shapes: backtick-quoted tokens that start with a known
top-level dir (``src/``, ``docs/``, ``scripts/``, ``tests/``, ``e2e/``,
``.github/``, ``.claude/``, ``plans/``) or name a known root file. Bare command
examples and external URLs are not paths and are ignored.

Exit codes: 0 = all citations resolve; 1 = violations printed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from _scan import REPO_ROOT, rel

TOKEN = re.compile(r"`([^`\s]+)`")
PATH_PREFIXES = ("src/", "docs/", "scripts/", "tests/", "e2e/", ".github/", ".claude/")
ROOT_FILES = {
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "CLAUDE.md",
    "LICENSE",
    "pyproject.toml",
    "ruff.toml",
    "uv.lock",
}
ALWAYS_DEAD_PREFIXES = ("plans/", "~/git/")
DEAD = "gitignored/machine-local; no reader can resolve it"

HARD_FILES = ["README.md", "CONTRIBUTING.md", "CHANGELOG.md", "CLAUDE.md"]
HARD_GLOBS = ["docs/**/*.md"]


def looks_like_repo_path(token: str) -> bool:
    if token in ROOT_FILES:
        return True
    return token.startswith(PATH_PREFIXES) or token.startswith(ALWAYS_DEAD_PREFIXES)


def strip_anchor(token: str) -> str:
    return token.split("#", 1)[0].rstrip("/")


def scan(path: Path, *, allow_dead_prefixes: bool = False) -> list[str]:
    """Return unresolved-citation messages for one doc.

    Args:
        allow_dead_prefixes: CLAUDE.md-only — suppresses the
            ALWAYS_DEAD_PREFIXES violation, since CLAUDE.md is the file that
            documents the plans/~/git/ convention by naming those prefixes,
            not a reader trying to resolve them as real paths.
    """
    problems: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for token in TOKEN.findall(line):
            if not looks_like_repo_path(token):
                continue
            where = f"{rel(path)}:{lineno}: `{token}`"
            if token.startswith(ALWAYS_DEAD_PREFIXES):
                if not allow_dead_prefixes:
                    problems.append(f"{where} — {DEAD}")
                continue
            cleaned = strip_anchor(token)
            if "*" in cleaned:
                if not any(REPO_ROOT.glob(cleaned)):
                    problems.append(f"{where} glob matches nothing")
                continue
            if not (REPO_ROOT / cleaned).exists():
                problems.append(f"{where} does not resolve")
    return problems


def main() -> int:
    files = [REPO_ROOT / name for name in HARD_FILES]
    for glob in HARD_GLOBS:
        files.extend(sorted(REPO_ROOT.glob(glob)))
    hard: list[str] = []
    for path in files:
        if path.is_file():
            hard.extend(scan(path, allow_dead_prefixes=rel(path) == "CLAUDE.md"))

    if hard:
        print("FAIL: unresolvable path citations in shipped docs.")
        print()
        print("\n".join(hard))
        print()
        print(
            "Repoint the citation, or state the invariant instead of the path "
            "— .claude/rules/doc-integrity.md §3."
        )
        return 1

    print("OK: all shipped-doc path citations resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

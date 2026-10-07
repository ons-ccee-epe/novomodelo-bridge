#!/usr/bin/env python3
"""check_doc_paths.py — resolvable-path gate for shipped docs.

Every repo-relative path cited in README.md, CONTRIBUTING.md, docs/**/*.md,
and CLAUDE.md must resolve against the live tree
(`.claude/rules/doc-integrity.md` §3.4). Citing a gitignored dir (``plans/``)
or a machine-local path (``~/git/...``) in a shipped doc is a violation
outright — no reader can resolve it.

CLAUDE.md is hard-gated like README.md and docs/**/*.md, with one narrow
allowance: it documents this checkout, so its own ``example/`` deck-path
mentions are legitimate (and already outside every recognized prefix, so the
scanner never flags them — no allowance needed), and the ``plans/``/
``~/git/...`` mentions where it *documents* the banned-prefix convention
itself (rather than citing a real path) do not count as violations. Every
other CLAUDE.md citation — and every README.md/docs/ citation, unconditionally
— must still resolve.

Recognized citation shapes: backtick-quoted tokens that start with a known
top-level dir (``src/``, ``docs/``, ``scripts/``, ``tests/``, ``.github/``,
``.claude/``, ``plans/``) or name a known root file. Bare command examples and
external URLs are not paths and are ignored.

Exit codes: 0 = all hard-scope citations resolve; 1 = violations printed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from _scan import REPO_ROOT, rel

TOKEN = re.compile(r"`([^`\s]+)`")
PATH_PREFIXES = ("src/", "docs/", "scripts/", "tests/", ".github/", ".claude/")
ROOT_FILES = {
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "CLAUDE.md",
    "pyproject.toml",
    "ruff.toml",
    "uv.lock",
}
# `novomodelo-bridge.toml` and `~/.config/...` are user-created locations the docs
# legitimately name (external contracts, not repo paths) — not checked.
# `plans/` is gitignored and `~/git/` is a developer machine — always dead
# for a reader, except where CLAUDE.md documents the convention by naming
# them (see `scan`'s `allow_dead_prefixes`). `example/` needs no matching
# exemption: it is in neither PATH_PREFIXES nor ROOT_FILES, so
# looks_like_repo_path() already ignores it everywhere, CLAUDE.md included.
ALWAYS_DEAD_PREFIXES = ("plans/", "~/git/")

HARD_FILES = ["README.md", "CONTRIBUTING.md", "CLAUDE.md"]
HARD_GLOBS = ["docs/**/*.md"]
# No file is advisory-only today; CLAUDE.md was promoted to HARD_FILES above.
# Kept as a hook for a future doc that needs the same non-failing treatment.
ADVISORY_FILES: list[str] = []


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
            if token.startswith(ALWAYS_DEAD_PREFIXES):
                if allow_dead_prefixes:
                    continue
                problems.append(
                    f"{rel(path)}:{lineno}: `{token}` — gitignored/machine-local; "
                    "no reader can resolve it"
                )
                continue
            cleaned = strip_anchor(token)
            if "*" in cleaned:
                if not any(REPO_ROOT.glob(cleaned)):
                    problems.append(
                        f"{rel(path)}:{lineno}: `{token}` glob matches nothing"
                    )
                continue
            if not (REPO_ROOT / cleaned).exists():
                problems.append(f"{rel(path)}:{lineno}: `{token}` does not resolve")
    return problems


def collect(names: list[str], globs: list[str] | None = None) -> list[Path]:
    files = [REPO_ROOT / name for name in names]
    for glob in globs or []:
        files.extend(sorted(REPO_ROOT.glob(glob)))
    return [f for f in files if f.is_file()]


def main() -> int:
    hard: list[str] = []
    for path in collect(HARD_FILES, HARD_GLOBS):
        hard.extend(scan(path, allow_dead_prefixes=rel(path) == "CLAUDE.md"))

    advisory: list[str] = []
    for path in collect(ADVISORY_FILES):
        advisory.extend(scan(path))

    if advisory:
        print(f"ADVISORY: {len(advisory)} unresolved citation(s) (never fails):")
        print("\n".join(f"  {hit}" for hit in advisory))

    if hard:
        print()
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

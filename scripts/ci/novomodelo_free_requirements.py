#!/usr/bin/env python3
"""novomodelo_free_requirements.py — the declared dependencies except novomodelo-python.

Prints one requirement per line: every `[project].dependencies` entry whose
distribution name is not `novomodelo-python`, then every
`[project.optional-dependencies].dev` entry, in file order. CI installs the bridge
with `--no-deps` and then this list, so the suite runs without the private
novomodelo-python git dependency.

Exit codes: 0 = list printed; 1 = no core dependency is named `novomodelo-python`.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _is_novomodelo_python(requirement: str) -> bool:
    match = NAME.match(requirement.strip())
    return match is not None and re.sub(r"[-_.]+", "-", match.group()).lower() == (
        "novomodelo-python"
    )


def main() -> int:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    core = project["dependencies"]
    if not any(_is_novomodelo_python(dep) for dep in core):
        sys.stderr.write("no core dependency is named novomodelo-python\n")
        return 1
    for requirement in [
        *(dep for dep in core if not _is_novomodelo_python(dep)),
        *project["optional-dependencies"]["dev"],
    ]:
        sys.stdout.write(f"{requirement}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

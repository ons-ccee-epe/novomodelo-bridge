"""Function length guard for `src/novomodelo_bridge`.

No function or method in the package is longer than `_MAX_LINES` lines, docstring
included. A longer one is split into named steps rather than allowlisted.

Tier-1: pure ``ast`` and ``pathlib`` -- no ``novomodelo`` import -- so this
collects and runs even in a novomodelo-free environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src" / "novomodelo_bridge"
_MAX_LINES = 200


def _long_functions() -> list[str]:
    found: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            length = (node.end_lineno or node.lineno) - node.lineno + 1
            if length > _MAX_LINES:
                rel = path.relative_to(_SRC).as_posix()
                found.append(f"{rel}::{node.name} ({length} lines)")
    return found


def test_no_function_exceeds_the_length_limit() -> None:
    assert _long_functions() == []

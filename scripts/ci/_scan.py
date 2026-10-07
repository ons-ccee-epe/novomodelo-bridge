"""Shared comment/docstring extraction for the scripts/ci quality gates.

Yields (lineno, text) pairs per file so every gate scans the same surfaces:
``#`` comment tokens (via tokenize, so strings containing ``#`` are never
false-matched) and module/class/function docstrings (via ast).
"""

from __future__ import annotations

import ast
import tokenize
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "novomodelo_bridge"
TESTS_ROOT = REPO_ROOT / "tests"


def iter_py_files(root: Path) -> Iterator[Path]:
    yield from sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def iter_comments(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (lineno, text) for every ``#`` comment token in the file."""
    try:
        with tokenize.open(path) as handle:
            for tok in tokenize.generate_tokens(handle.readline):
                if tok.type == tokenize.COMMENT:
                    yield tok.start[0], tok.string
    except (tokenize.TokenError, SyntaxError, UnicodeDecodeError):
        return


def iter_docstrings(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (lineno, text) for every module/class/function docstring."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return
    nodes: list[ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef] = [
        tree
    ]
    nodes += [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for node in nodes:
        text = ast.get_docstring(node, clean=False)
        if text:
            first = node.body[0]
            yield first.lineno, text


def iter_prose(path: Path) -> Iterator[tuple[int, str]]:
    """Comments and docstrings interleaved: every shipped-prose surface."""
    yield from iter_comments(path)
    yield from iter_docstrings(path)


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))

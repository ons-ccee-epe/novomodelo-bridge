"""Wording guard for every diagnostic in `src/novomodelo_bridge`.

Checks the mechanical rules of `.claude/rules/diagnostics.md` on the text of
each `Diagnostic(...)` and `DiagnosticTable(...)` call that the source spells
out: literal strings, f-strings, concatenations and conditional expressions.
A field built at run time (`summary=str(exc)`) is out of reach and skipped.
Log calls are checked too: a finding is a diagnostic, never a log warning,
and no log message carries a `(s)` plural marker.

Tier-1: pure ``ast`` and ``pathlib`` -- no ``novomodelo`` import.
"""

from __future__ import annotations

import ast
import itertools
import re
from collections.abc import Callable, Iterator
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src" / "novomodelo_bridge"

_PLACEHOLDER = "{}"
_FILE_WORD = re.compile(r"^[\w-]+\.(dat|json|csv|parquet|txt|out|rv\d+)\b")
_CODE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_LOG_PHRASING = re.compile(
    r";\s*skipp(ing|ed)\b|\bnovomodelo rule\b|\brule \d+\b", re.I
)
_BOILERPLATE = re.compile(r"no action (needed|required)|informational", re.I)
_UPPER_FILE = re.compile(r"\b[A-Z][A-Z0-9_]*\.(DAT|RV\d|RVN)\b")
_PLURAL_MARKER = re.compile(r"\w\((s|es|ies)\)")
_LOGGERS = frozenset({"_LOG", "logger", "logging"})
_LOG_LEVELS = frozenset({"debug", "info", "warning", "error", "exception", "critical"})
_FINDING_LEVELS = frozenset({"warning", "error", "exception", "critical"})
#: Modules whose log warnings still await conversion to diagnostics. Shrink
#: it as each is converted; the stale-entry test fails on a module left here.
_LOG_WARNING_MODULES = frozenset(
    {
        "comparators/constraints.py",
        "comparators/decomp/results.py",
        "comparators/newave/alignment.py",
        "comparators/newave/readers.py",
        "comparators/newave/results.py",
        "dashboard/data.py",
        "dashboard/tabs/__init__.py",
        "dessem/exporters/fcf/core/cross_check.py",
        "dessem/exporters/fcf/core/mapcut_writer.py",
        "dessem/exporters/fcf/core/mapper.py",
        "dessem/exporters/fcf/core/travel_time.py",
        "dessem/exporters/fcf/eco/audit_raw_cuts.py",
        "dessem/exporters/fcf/pipeline.py",
        "novomodelo/case_io.py",
        "novomodelo/readers.py",
    }
)


def _texts(node: ast.expr) -> list[str]:
    """Every text *node* can evaluate to, with ``{}`` for each interpolation.

    Empty when any part is computed at run time.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        parts = [
            value.value if isinstance(value, ast.Constant) else _PLACEHOLDER
            for value in node.values
        ]
        return ["".join(str(part) for part in parts)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _texts(node.left), _texts(node.right)
        return [a + b for a, b in itertools.product(left, right)]
    if isinstance(node, ast.IfExp):
        return _texts(node.body) + _texts(node.orelse)
    return []


def _calls(name: str) -> Iterator[tuple[str, ast.Call]]:
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(_SRC).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
            if called == name:
                yield f"{rel}:{node.lineno}", node


def _field(call: ast.Call, name: str) -> ast.expr | None:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def _starts_well(text: str) -> bool:
    return (
        text.startswith(_PLACEHOLDER)
        or text[:1].isupper()
        or text[:1].isdigit()
        or bool(_FILE_WORD.match(text))
    )


def _ends_well(text: str) -> bool:
    return text.rstrip().endswith((".", _PLACEHOLDER))


def _title_problems(text: str) -> list[str]:
    problems = []
    if text.rstrip().endswith("."):
        problems.append("title ends with a period")
    if _PLURAL_MARKER.search(text):
        problems.append("title has a '(s)' plural marker; count as a ' (N)' suffix")
    if not _starts_well(text):
        problems.append("title does not start with a capital")
    return problems


def _summary_problems(text: str) -> list[str]:
    problems = []
    if not _starts_well(text):
        problems.append("summary does not start with a capital or the count")
    if not _ends_well(text):
        problems.append("summary does not end with a period")
    if _LOG_PHRASING.search(text):
        problems.append("summary uses log phrasing or a validation-rule number")
    if _PLURAL_MARKER.search(text):
        problems.append("summary has a '(s)' plural marker; write 'the listed ...'")
    return problems


def _remediation_problems(text: str) -> list[str]:
    problems = []
    if _BOILERPLATE.search(text):
        problems.append("remediation is boilerplate; leave the field out")
    if not _ends_well(text):
        problems.append("remediation does not end with a period")
    return problems


def _with_file_case(check: Callable[[str], list[str]]) -> Callable[[str], list[str]]:
    def checked(text: str) -> list[str]:
        problems = check(text)
        if _UPPER_FILE.search(text):
            problems.append("a deck file name in capitals; write it in lowercase")
        return problems

    return checked


_CHECKS = {
    "title": _with_file_case(_title_problems),
    "summary": _with_file_case(_summary_problems),
    "remediation": _with_file_case(_remediation_problems),
}


def _diagnostic_violations() -> list[str]:
    found: list[str] = []
    for where, call in _calls("Diagnostic"):
        code = _field(call, "code")
        if isinstance(code, ast.Constant) and not _CODE.match(str(code.value)):
            found.append(f"{where}: code {code.value!r} is not kebab-case")
        for name, check in _CHECKS.items():
            node = _field(call, name)
            if node is None:
                continue
            for text in _texts(node):
                found.extend(f"{where}: {problem}" for problem in check(text))
    return sorted(set(found))


def _table_violations() -> list[str]:
    found: list[str] = []
    for where, call in _calls("DiagnosticTable"):
        node = _field(call, "columns")
        if not isinstance(node, ast.List):
            continue
        columns = [text for element in node.elts for text in _texts(element)]
        if "Plant" in columns and "Code" in columns:
            found.append(
                f"{where}: columns lead with 'Plant', 'Code'; use 'Code', 'Name'"
            )
        if columns[:1] == ["Plant code"]:
            found.append(f"{where}: a leading 'Plant code' column; use 'Code'")
        found.extend(
            f"{where}: column {column!r} writes m3/s; use m³/s"
            for column in columns
            if "m3/s" in column
        )
    return sorted(found)


def _log_calls() -> Iterator[tuple[str, str, str, ast.Call]]:
    """``(module, where, level, call)`` for each ``_LOG``/``logger`` call."""
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(_SRC).as_posix()
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _LOG_LEVELS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in _LOGGERS
            ):
                continue
            yield rel, f"{rel}:{node.lineno}", node.func.attr, node


def _log_warning_modules() -> set[str]:
    return {module for module, _, level, _ in _log_calls() if level in _FINDING_LEVELS}


def test_no_finding_is_a_log_warning() -> None:
    found = sorted(
        f"{where}: log {level} reports a finding; emit a Diagnostic"
        for module, where, level, _ in _log_calls()
        if level in _FINDING_LEVELS and module not in _LOG_WARNING_MODULES
    )
    assert found == []


def test_log_warning_allowlist_has_no_stale_module() -> None:
    assert sorted(_LOG_WARNING_MODULES - _log_warning_modules()) == []


def test_no_log_message_has_a_plural_marker() -> None:
    found = sorted(
        f"{where}: log message has a '(s)' plural marker"
        for _, where, _, call in _log_calls()
        if call.args
        and any(_PLURAL_MARKER.search(text) for text in _texts(call.args[0]))
    )
    assert found == []


def test_every_diagnostic_follows_the_wording_rules() -> None:
    assert _diagnostic_violations() == []


def test_every_diagnostic_table_leads_with_code_and_name() -> None:
    assert _table_violations() == []

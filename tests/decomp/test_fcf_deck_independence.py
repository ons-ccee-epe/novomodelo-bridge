"""Regression guard: every boundary-FCF test module collects without novomodelo.

Every ``tests/decomp/test_fcf_*.py`` module must collect in a novomodelo-free
environment: a module-top ``import novomodelo`` breaks collection everywhere
regardless of skip markers. Nothing in the test runner enforces that, so this
module scans each FCF test module's own source text for one. It is itself a
plain ``pathlib`` + ``re`` scan with no ``novomodelo`` import.
"""

from __future__ import annotations

import re
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
# Matches any column-0 import of the `novomodelo` package itself — `import
# novomodelo`, `import novomodelo as ...`, `import novomodelo.submodule`, or `from novomodelo
# import ...` — including a trailing comment. The `\b` after `novomodelo`
# excludes an unrelated package sharing the prefix (e.g. this project's own
# `novomodelo_bridge`), and the `^` anchor (MULTILINE) excludes an indented
# call-site import.
_TOP_LEVEL_NOVOMODELO_IMPORT = re.compile(
    r"^(?:import novomodelo\b|from novomodelo\b)", re.MULTILINE
)


def _fcf_test_modules() -> list[Path]:
    """Every ``tests/decomp/test_fcf_*.py`` module, sorted for a stable order."""
    return sorted(_TESTS_DIR.glob("test_fcf_*.py"))


def test_fcf_test_modules_have_no_top_level_novomodelo_import() -> None:
    """No FCF test module blocks novomodelo-free collection with a module-top import.

    A call-site import of ``novomodelo`` (inside a function/test body, always
    indented) is fine — every tier-2/3 test defers it there. Only a
    column-0 import of the ``novomodelo`` package itself — ``import novomodelo``,
    ``import novomodelo as ...``, ``import novomodelo.submodule``, or ``from novomodelo
    import ...`` — which pytest would execute at collection time regardless
    of markers, is disallowed.
    """
    modules = _fcf_test_modules()
    assert modules, f"no test_fcf_*.py modules found under {_TESTS_DIR}"

    offenders = [
        module.name
        for module in modules
        if _TOP_LEVEL_NOVOMODELO_IMPORT.search(module.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "module-top `import novomodelo`/`from novomodelo import ...` blocks novomodelo-free "
        f"collection in: {offenders}; move the import into a call site or "
        "test body"
    )

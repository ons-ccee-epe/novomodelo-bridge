"""Dead-code guard for the ``compare decomp`` render/export/verdict unification.

A plain text scan of ``src/`` and ``tests/`` asserting that none of the
symbols and modules retired by that unification survive anywhere in the tree.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Dead-code guard
# ---------------------------------------------------------------------------

# Symbols/modules retired across the compare-decomp unification:
# DecompComparison/compare_decomp_results/decomp_compare_summary
# (folded onto build_decomp_dataset + the shared verdict/export path),
# decomp_export/decomp_html_report (replaced by the shared
# report_builder/export modules), render_decomp_comparison
# (retired from ui/console.py).
_RETIRED_SYMBOLS = (
    "DecompComparison",
    "compare_decomp_results",
    "decomp_export",
    "decomp_html_report",
    "decomp_compare_summary",
    "render_decomp_comparison",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCAN_DIRS = (_REPO_ROOT / "src", _REPO_ROOT / "tests")


def test_no_retired_decomp_symbols_remain() -> None:
    """None of the retired compare-decomp symbols/modules survive in `src/`/`tests/`.

    This guard's own source line naming the retired symbols above is excluded
    from the scan (it is documentation, not a survivor), so the check stays
    meaningful without ever self-matching.
    """
    this_file = Path(__file__).resolve()
    offenders: list[str] = []
    for scan_dir in _SCAN_DIRS:
        for path in sorted(scan_dir.rglob("*.py")):
            if "__pycache__" in path.parts or path.resolve() == this_file:
                continue
            text = path.read_text(encoding="utf-8")
            for symbol in _RETIRED_SYMBOLS:
                if symbol in text:
                    offenders.append(f"{path}: {symbol}")
    assert not offenders, "retired symbol(s) still present:\n" + "\n".join(offenders)

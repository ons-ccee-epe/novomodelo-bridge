#!/usr/bin/env python3
"""quality_report.py — code-quality HOTSPOT report (advisory, read-only).

The check_*.py gates are a ratchet: they stop new drift but do not rank the
debt that already exists. This report joins the signals the repo already has
into one ranked "where to act first" table.

Signals per production source file (src/novomodelo_bridge/**/*.py):

  churn        — commits touching the file within CHURN_SINCE (default
                 "12 months ago"); the dominant hotspot signal.
  loc          — physical line count.
  long_fns     — functions whose body spans > 100 lines (ast-derived).
  suppressions — ``# noqa`` / ``# type: ignore`` directives.
  plan_tokens  — plan-vocabulary lines in comments/docstrings (the
                 check_no_plan_leaks.py advisory scope).

Composite score (0..100), each signal min-max normalized:

  score = 100 * (0.45*churn + 0.20*loc + 0.15*long_fns
                 + 0.10*suppressions + 0.10*plan_tokens)

Usage:
  python3 scripts/ci/quality_report.py           # top 25
  TOP_N=0 CHURN_SINCE="6 months ago" python3 scripts/ci/quality_report.py

Exit code: always 0 (a report, never a gate).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from _scan import REPO_ROOT, SRC_ROOT, iter_prose, iter_py_files, rel
from check_no_plan_leaks import PATTERN as PLAN_PATTERN

CHURN_SINCE = os.environ.get("CHURN_SINCE", "12 months ago")
TOP_N = int(os.environ.get("TOP_N", "25"))

WEIGHTS = {"churn": 0.45, "loc": 0.20, "long_fns": 0.15, "suppr": 0.10, "plan": 0.10}
LONG_FN_LINES = 100


def churn_map() -> Counter[str]:
    out = subprocess.run(
        [
            "git",
            "log",
            f"--since={CHURN_SINCE}",
            "--name-only",
            "--pretty=format:",
            "--",
            "src/novomodelo_bridge/*.py",
            "src/novomodelo_bridge/**/*.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return Counter(line for line in out.stdout.splitlines() if line)


def long_fn_count(path: Path) -> int:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return 0
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            span = (node.end_lineno or node.lineno) - node.lineno + 1
            if span > LONG_FN_LINES:
                count += 1
    return count


def suppression_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return text.count("# noqa") + text.count("# type: ignore")


def plan_token_count(path: Path) -> int:
    return sum(
        1
        for _lineno, text in iter_prose(path)
        for line in text.splitlines()
        if PLAN_PATTERN.search(line)
    )


def main() -> int:
    churn = churn_map()
    rows: list[tuple[str, int, int, int, int, int]] = []
    for path in iter_py_files(SRC_ROOT):
        relpath = rel(path)
        rows.append(
            (
                relpath,
                churn.get(relpath, 0),
                sum(1 for _ in path.open(encoding="utf-8")),
                long_fn_count(path),
                suppression_count(path),
                plan_token_count(path),
            )
        )

    maxima = [max((row[i] for row in rows), default=0) or 1 for i in range(1, 6)]
    weights = list(WEIGHTS.values())
    scored = sorted(
        (
            (
                100 * sum(w * (row[i + 1] / maxima[i]) for i, w in enumerate(weights)),
                row,
            )
            for row in rows
        ),
        reverse=True,
    )

    print(f"Code-quality hotspots (churn since {CHURN_SINCE}). Higher = act first.\n")
    header = (
        f"{'SCORE':>6} {'churn':>5} {'loc':>5} "
        f"{'>100L':>5} {'suppr':>5} {'plan':>5}  FILE"
    )
    print(header)
    print("-" * len(header))
    shown = 0
    for score, (relpath, c, loc, lf, sup, plan) in scored:
        if TOP_N and shown >= TOP_N:
            break
        if score < 0.5:
            continue
        print(f"{score:6.1f} {c:5d} {loc:5d} {lf:5d} {sup:5d} {plan:5d}  {relpath}")
        shown += 1
    print(
        "\nscore = 100*(0.45*churn + 0.20*loc + 0.15*fns>100L + 0.10*suppressions"
        " + 0.10*plan_tokens), min-max normalized. Advisory only."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Content guards for the shipped docs.

``docs/cli.md`` is produced verbatim by ``scripts/gen-cli-docs.sh``; the checks
assert the documented surface is present without coupling to the exact byte
formatting of Typer/Click output, which drifts across the CI Python matrix.
``README.md`` states one literal by hand, the Python floor, which is pinned to
``pyproject.toml`` here so a bump cannot leave it stale. The contract literals
that ``docs/compare-novomodelo.md`` repeats for ``compare novomodelo`` are
pinned here too, so a renamed code, key or file cannot leave the page stale.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from novomodelo_bridge.cli.compare_version import ARTIFACTS_DIRNAME
from novomodelo_bridge.comparators.version.convergence import UB_COMPARABILITY
from novomodelo_bridge.comparators.version.requirements import REQUIREMENTS

_REPO = Path(__file__).resolve().parents[1]
_CLI_DOCS = _REPO / "docs" / "cli.md"
_README = _REPO / "README.md"
_PYPROJECT = _REPO / "pyproject.toml"
_VERSION_PAGE = _REPO / "docs" / "compare-novomodelo.md"

_REQUIRED_TOKENS: tuple[str, ...] = (
    "convert newave",
    "convert decomp",
    "check newave",
    "check decomp",
    "compare newave",
    "compare decomp",
    "compare novomodelo",
    "export dessem",
    "dashboard",
    "--json",
    "--dry-run",
    "--open",
    "--log-file",
    "--validate",
    "--tolerance",
    "--format",
    "--out-dir",
    "--install-completion",
    "--version",
)

_VERSION_FINDING_CODES: tuple[str, ...] = (
    "entity-one-sided",
    "entity-name-duplicated",
    "stage-sets-differ",
    "section-unavailable",
    "generic-constraint-differs",
    "upper-bound-not-comparable",
    "iteration-counts-differ",
    "same-output-compared",
)

_VERSION_SUMMARY_KEYS: tuple[str, ...] = (
    "tolerance",
    "total",
    "close",
    "moderate",
    "far",
    "far_variables",
    "baseline_version",
    "candidate_version",
    "only_in_baseline",
    "only_in_candidate",
    "upper_bound_comparable",
    "baseline_gap_percent",
    "candidate_gap_percent",
    "gap_delta_pp",
)

_VERSION_WORDS: tuple[str, ...] = (
    "baseline",
    "candidate",
    "both",
    "ok",
    "mismatch",
    "no-comparable-rows",
    "comparison.json",
    "comparison.parquet",
    "summary.parquet",
    "metadata.json",
    "summary.json",
    "top_divergences.json",
    "comparison.csv",
    "summary.csv",
    "report.html",
)


def test_cli_docs_file_exists_and_is_non_empty() -> None:
    assert _CLI_DOCS.is_file(), (
        f"{_CLI_DOCS} missing; regenerate with scripts/gen-cli-docs.sh"
    )
    assert _CLI_DOCS.stat().st_size > 0


def test_cli_docs_cover_every_command_and_new_flag() -> None:
    text = _CLI_DOCS.read_text(encoding="utf-8")
    missing = [tok for tok in _REQUIRED_TOKENS if tok not in text]
    assert not missing, (
        f"docs/cli.md is stale; regenerate with scripts/gen-cli-docs.sh "
        f"(missing: {missing})"
    )


def test_readme_states_the_pyproject_python_floor() -> None:
    """The README's "Python X.Y or newer" must equal ``requires-python``."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    floor = data["project"]["requires-python"].removeprefix(">=").strip()
    assert f"Python {floor} or newer" in _README.read_text(encoding="utf-8"), (
        f"README.md must state 'Python {floor} or newer' (pyproject requires-python)"
    )


def test_version_comparison_page_names_every_contract_literal() -> None:
    text = _VERSION_PAGE.read_text(encoding="utf-8")
    literals = (
        *UB_COMPARABILITY,
        *REQUIREMENTS,
        ARTIFACTS_DIRNAME,
        *_VERSION_FINDING_CODES,
        *_VERSION_SUMMARY_KEYS,
        *_VERSION_WORDS,
    )
    missing = [literal for literal in literals if f"`{literal}`" not in text]
    assert not missing, f"docs/compare-novomodelo.md does not name: {missing}"

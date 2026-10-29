"""Packaging guards: the declared dependencies must let a plain install work.

`convert decomp` imports the deck's boundary FCF by default and needs
`import novomodelo` to succeed, so `novomodelo-python` must be a CORE runtime dependency —
not an optional extra. This module locks that down: a fresh
`pip install novomodelo-bridge` (no extras) must pull a checkpoint-capable novomodelo. It was the
absence of exactly this guard that let a release ship with `novomodelo-python` as an
extra, so a plain install failed `convert decomp` on any real deck. It also keeps
the workflows installing from `uv.lock`. Tier-1: reads `pyproject.toml`, `uv.lock` and
the workflows, and never imports novomodelo.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _core_dependencies() -> list[str]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["dependencies"]


def _novomodelo_python_pin(deps: list[str]) -> str | None:
    for dep in deps:
        if dep.replace(" ", "").startswith("novomodelo-python"):
            return dep
    return None


def test_novomodelo_python_is_a_core_runtime_dependency() -> None:
    """`novomodelo-python` must be in `[project].dependencies`, not an extra, so a
    plain `pip install novomodelo-bridge` gives a working `convert decomp`."""
    core = _core_dependencies()
    assert _novomodelo_python_pin(core) is not None, (
        "novomodelo-python must be a core runtime dependency, not an optional extra; "
        f"found core dependencies: {core}"
    )


def test_novomodelo_python_core_pin_is_exactly_min_novomodelo_version() -> None:
    """The core `novomodelo-python` pin must be exactly `MIN_NOVOMODELO_VERSION`: novomodelo
    loads a policy checkpoint only in the version that wrote it, so the
    boundary `convert decomp` writes must come from the paired release."""
    from novomodelo_bridge.cli import MIN_NOVOMODELO_VERSION

    pin = _novomodelo_python_pin(_core_dependencies())
    assert pin is not None
    assert pin.replace(" ", "") == f"novomodelo-python=={MIN_NOVOMODELO_VERSION}", (
        f"novomodelo-python core pin {pin!r} must be exactly "
        f"novomodelo-python=={MIN_NOVOMODELO_VERSION}"
    )


_UV_LOCK = Path(__file__).resolve().parent.parent / "uv.lock"


def _lock_novomodelo_python_requirement() -> dict[str, str] | None:
    """The `novomodelo-bridge` package's `novomodelo-python` requires-dist entry from
    uv.lock, or None if absent."""
    data = tomllib.loads(_UV_LOCK.read_text(encoding="utf-8"))
    for package in data["package"]:
        if package.get("name") == "novomodelo-bridge":
            for req in package.get("metadata", {}).get("requires-dist", []):
                if req.get("name") == "novomodelo-python":
                    return req
    return None


def test_uv_lock_novomodelo_python_is_a_core_dependency() -> None:
    """uv.lock must record novomodelo-python as a core requirement — not gated
    behind an `extra` — so `uv sync` installs it by default, matching
    pyproject. Guards against the lock drifting back to a `validation` extra."""
    req = _lock_novomodelo_python_requirement()
    assert req is not None, "novomodelo-python missing from uv.lock requires-dist"
    marker = req.get("marker", "")
    assert "extra" not in marker, (
        "novomodelo-python must be a core dependency in uv.lock, not gated behind "
        f"an extra; found marker {marker!r}"
    )


def test_uv_lock_novomodelo_python_is_pinned_exactly_to_min_novomodelo_version() -> (
    None
):
    """The uv.lock novomodelo-python specifier must be exactly MIN_NOVOMODELO_VERSION, so
    a regenerated lock never resolves a release other than the paired one."""
    from novomodelo_bridge.cli import MIN_NOVOMODELO_VERSION

    req = _lock_novomodelo_python_requirement()
    assert req is not None
    specifier = req.get("specifier", "").replace(" ", "")
    assert specifier == f"=={MIN_NOVOMODELO_VERSION}", (
        f"uv.lock novomodelo-python specifier {specifier!r} must be exactly "
        f"=={MIN_NOVOMODELO_VERSION}"
    )


_WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def test_workflows_install_from_the_lock() -> None:
    """CI and release run the tool versions uv.lock pins, so a new ruff or ty
    release cannot turn a workflow red, or green on code the pre-commit hook
    rejects, without a change in the repository: no `pip install`, and every
    `uv sync` is `--locked`."""
    workflows = [_WORKFLOWS / "ci.yml"]
    for workflow in workflows:
        text = workflow.read_text(encoding="utf-8")
        assert not re.search(r"(?<!uv )pip install", text), (
            f"{workflow.name} installs with pip; use `uv sync --locked`"
        )
        for command in re.findall(r"uv sync[^\n]*", text):
            assert "--locked" in command, (
                f"{workflow.name}: `{command.strip()}` must pass --locked"
            )

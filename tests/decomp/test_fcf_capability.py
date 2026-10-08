"""Tests for the boundary-FCF capability probe (``fcf/capability.py``)."""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from novomodelo_bridge.decomp.fcf import capability
from novomodelo_bridge.decomp.fcf.capability import (
    REMEDIATION,
    ensure_boundary_fcf_capability,
)
from tests.conftest import requires_novomodelo_python, requires_writer_binding

#: Every fact the remediation message must name: the dependency at fault, the
#: install/upgrade fix, and the ``--no-fcf`` escape hatch. Shared by every
#: "raises" test below.
_REMEDIATION_MARKERS = (
    "novomodelo-python",
    "pip install",
    "--no-fcf",
)

#: Repo-internal references that must NEVER leak into an end-user-facing
#: message: a pip-installed user has no repo checkout, so doc paths, developer
#: worktrees, and internal planning codes are noise (and, as this message's
#: history proved, actively misleading). This is the regression guard.
_REPO_INTERNAL_LEAKS = (
    "docs/",
    "plans/",
    "~/git",
    "feat/",
    "ticket-",
    "epic-",
)


def test_remediation_names_install_fix_and_escape_hatch() -> None:
    for marker in _REMEDIATION_MARKERS:
        assert marker in REMEDIATION


def test_remediation_has_no_repo_internal_references() -> None:
    """The remediation reaches pip-installed end users, who have no repo — it
    must stay self-contained (no doc paths, worktrees, or planning/spec codes)."""
    for leak in _REPO_INTERNAL_LEAKS:
        assert leak not in REMEDIATION, (
            f"remediation leaks repo-internal reference {leak!r}: {REMEDIATION!r}"
        )


def test_ensure_boundary_fcf_capability_raises_when_wheel_lacks_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 1 -- an installed-but-incapable wheel (no `write_policy_checkpoint`
    attribute, mirroring `test_decomp_fcf_bootstrap.py`'s own
    `ensure_writer_binding` test convention) raises with the remediation
    text, chained from the underlying `AttributeError`.
    """
    stub_novomodelo = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "novomodelo", stub_novomodelo)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_boundary_fcf_capability()

    for marker in _REMEDIATION_MARKERS:
        assert marker in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, AttributeError)


def test_ensure_boundary_fcf_capability_raises_when_novomodelo_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 1 (variant) -- novomodelo entirely absent raises the same remediation,
    chained from `ModuleNotFoundError`. Setting the `sys.modules` entry to
    `None` (rather than deleting it) forces the import to fail even in an
    environment where novomodelo is genuinely installed, the same trick
    `test_decomp_fcf_bootstrap.py` uses.
    """
    monkeypatch.setitem(sys.modules, "novomodelo", None)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_boundary_fcf_capability()

    for marker in _REMEDIATION_MARKERS:
        assert marker in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


def test_ensure_boundary_fcf_capability_raises_when_slot_interval_start_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 3 -- a wheel that writes and reloads fine, and whose reloaded pool
    carries every self-describing field (`priced_state_date` included), but
    whose reloaded terminal manifest slot lacks `interval_start` (a novomodelo that
    predates the dated per-slot schema) still raises -- proving the probe
    checks the slot-level date field, not merely the
    `write_policy_checkpoint` attribute.
    """
    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "cost_scale_factor": 1.0,
                "node_id": 0,
                "graph_stage_id": 0,
                "priced_state_date": 20_261_101,
                "entity_manifest": [
                    {
                        "entity_type": 0,
                        "entity_id": 0,
                        "subindex": 0,
                        "was_active": True,
                    }
                ],
            }
        ]
    }
    stub_novomodelo = SimpleNamespace(
        write_policy_checkpoint=lambda *args, **kwargs: None,
        results=SimpleNamespace(load_policy=lambda *args, **kwargs: fake_policy),
    )
    monkeypatch.setitem(sys.modules, "novomodelo", stub_novomodelo)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_boundary_fcf_capability()

    for marker in _REMEDIATION_MARKERS:
        assert marker in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "interval_start" in str(exc_info.value.__cause__)


def test_ensure_boundary_fcf_capability_raises_when_priced_state_date_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 3 (pool-date variant) -- a wheel whose reloaded terminal pool omits
    `priced_state_date` (a novomodelo that predates the dated self-describing
    checkpoint) still raises -- proving the probe checks the pool's date the
    boundary loader selects a source against, not merely the slot-level fields.
    """
    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "cost_scale_factor": 1.0,
                "node_id": 0,
                "graph_stage_id": 0,
                "entity_manifest": [
                    {
                        "entity_type": 0,
                        "entity_id": 0,
                        "subindex": 0,
                        "was_active": True,
                        "interval_start": 20_261_101,
                    }
                ],
            }
        ]
    }
    stub_novomodelo = SimpleNamespace(
        write_policy_checkpoint=lambda *args, **kwargs: None,
        results=SimpleNamespace(load_policy=lambda *args, **kwargs: fake_policy),
    )
    monkeypatch.setitem(sys.modules, "novomodelo", stub_novomodelo)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_boundary_fcf_capability()

    for marker in _REMEDIATION_MARKERS:
        assert marker in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "priced_state_date" in str(exc_info.value.__cause__)


def test_ensure_boundary_fcf_capability_raises_when_season_manifest_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 3 (metadata variant) -- a wheel whose reloaded pool and slot are fully
    formed but whose reloaded metadata omits `season_manifest` (a novomodelo
    predating the season-manifest round-trip) still raises -- proving the probe
    checks the study-global season descriptor the boundary loader's
    season-compatibility gate requires."""
    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "cost_scale_factor": 1.0,
                "node_id": 0,
                "graph_stage_id": 0,
                "priced_state_date": 20_261_101,
                "entity_manifest": [
                    {
                        "entity_type": 0,
                        "entity_id": 0,
                        "subindex": 0,
                        "was_active": True,
                        "interval_start": 20_261_101,
                    }
                ],
            }
        ],
        "metadata": {},
    }
    stub_novomodelo = SimpleNamespace(
        write_policy_checkpoint=lambda *args, **kwargs: None,
        results=SimpleNamespace(load_policy=lambda *args, **kwargs: fake_policy),
    )
    monkeypatch.setitem(sys.modules, "novomodelo", stub_novomodelo)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_boundary_fcf_capability()

    for marker in _REMEDIATION_MARKERS:
        assert marker in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "season_manifest" in str(exc_info.value.__cause__)


def test_ensure_boundary_fcf_capability_raises_when_cost_scale_factor_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 2 (failure variant) -- a wheel that writes and reloads fine, but
    whose reloaded terminal pool omits `cost_scale_factor` (a wheel that drops
    the self-describing fields on reload) still raises -- proving the probe
    checks the pool-level fields, not merely the slot-level date.
    """
    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "node_id": 0,
                "graph_stage_id": 0,
                "entity_manifest": [
                    {
                        "entity_type": 0,
                        "entity_id": 0,
                        "subindex": 0,
                        "was_active": True,
                    }
                ],
            }
        ]
    }
    stub_novomodelo = SimpleNamespace(
        write_policy_checkpoint=lambda *args, **kwargs: None,
        results=SimpleNamespace(load_policy=lambda *args, **kwargs: fake_policy),
    )
    monkeypatch.setitem(sys.modules, "novomodelo", stub_novomodelo)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_boundary_fcf_capability()

    for marker in _REMEDIATION_MARKERS:
        assert marker in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "cost_scale_factor" in str(exc_info.value.__cause__)


def test_capability_module_imports_with_novomodelo_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 4 -- no module-scope `import novomodelo`: forcing `import novomodelo` to
    raise `ModuleNotFoundError` and reloading the already-imported module
    must not raise.
    """
    monkeypatch.setitem(sys.modules, "novomodelo", None)

    reloaded = importlib.reload(capability)

    assert hasattr(reloaded, "ensure_boundary_fcf_capability")


@requires_novomodelo_python
@requires_writer_binding
def test_ensure_boundary_fcf_capability_passes_against_installed_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC 2 -- against the installed (checkpoint-capable) wheel, the guard
    passes silently and writes only under a temporary directory: forcing
    `tempfile.gettempdir()` to resolve under `tmp_path` and asserting
    nothing is left behind afterward (the probe's own `TemporaryDirectory`
    cleans itself up on exit). With `novomodelo-python` a core dependency this
    runs on every CI job, proving the pinned/released wheel supports the
    boundary-FCF checkpoint format end to end.
    """
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    ensure_boundary_fcf_capability()
    assert list(tmp_path.iterdir()) == []

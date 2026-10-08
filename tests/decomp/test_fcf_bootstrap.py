"""Tests for the terminal-manifest bootstrap (``fcf/bootstrap.py``).

The bootstrap runs novomodelo in-process via ``novomodelo.run.run`` (no subprocess
binary), so the mocked tests below stub ``sys.modules['novomodelo']`` with a
``run.run`` no-op plus a ``results.load_policy`` returning a synthetic
checkpoint — exercising the read-back and error paths without live novomodelo I/O
or a ``--novomodelo-bin`` to resolve.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from novomodelo_bridge.decomp.fcf.bootstrap import (
    bootstrap_terminal_manifest,
    ensure_writer_binding,
)

_MINIMAL_CONFIG = (
    '{"training": {"stopping_rules": [{"type": "iteration_limit", "limit": 500}]}}'
)

#: A synthetic study-global season descriptor, mirroring what
#: ``novomodelo.results.load_policy`` returns under ``metadata["season_manifest"]``.
_SEASON_MANIFEST: dict[str, object] = {
    "cycle_code": 0,
    "n_seasons": 12,
    "hydro_orders": [{"hydro_id": 1, "orders": [1] * 12}],
}

#: A node-native ``stages.json`` with a 2-leaf terminal fan — the same shape
#: ``decomp/temporal.py::build_node_graph`` emits for a 2-scenario deck.
_FANNED_STAGES: dict[str, object] = {
    "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.0,
        "nodes": [
            {"id": 0, "stage_id": 0, "scenario_id": 0, "label": "trunk-0"},
            {"id": 1, "stage_id": 1, "scenario_id": 0, "label": "fan-0"},
            {"id": 2, "stage_id": 1, "scenario_id": 1, "label": "fan-1"},
        ],
        "transitions": [
            {"source_id": 0, "target_id": 1, "probability": 0.5},
            {"source_id": 0, "target_id": 2, "probability": 0.5},
        ],
    },
    "stages": [
        {"id": 0, "start_date": "2026-01-01", "end_date": "2026-01-08"},
        {"id": 1, "start_date": "2026-01-08", "end_date": "2026-01-15"},
    ],
}


def _write_case(case_dir: Path, *, stages: dict[str, object] | None = None) -> None:
    """Write a minimal case: ``config.json`` plus a fanned ``stages.json``.

    Every bootstrap test needs a real ``stages.json`` now that
    ``bootstrap_terminal_manifest`` materialises a flattened variant of it
    before ever reaching the (stubbed) ``novomodelo.run.run`` call.
    """
    (case_dir / "config.json").write_text(_MINIMAL_CONFIG, encoding="utf-8")
    (case_dir / "stages.json").write_text(
        json.dumps(stages if stages is not None else _FANNED_STAGES),
        encoding="utf-8",
    )


def _stub_novomodelo(policy: dict, *, run: object | None = None) -> SimpleNamespace:
    """A ``novomodelo`` stand-in whose ``run.run`` is a no-op (or ``run``) and whose
    ``results.load_policy`` returns ``policy`` verbatim."""

    def _default_run(*_args: object, **_kwargs: object) -> None:
        return None

    return SimpleNamespace(
        run=SimpleNamespace(run=run if run is not None else _default_run),
        results=SimpleNamespace(load_policy=lambda *_a, **_k: policy),
    )


def test_ensure_writer_binding_raises_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_novomodelo = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "novomodelo", stub_novomodelo)
    with pytest.raises(RuntimeError, match="novomodelo") as exc_info:
        ensure_writer_binding()
    message = str(exc_info.value)
    # End-user-facing: actionable pip guidance + the --no-fcf escape hatch,
    # and no repo-internal leaks (worktree paths, maturin build commands).
    assert "pip install" in message
    assert "--no-fcf" in message
    assert "~/git" not in message
    assert "maturin" not in message


def test_ensure_writer_binding_passes_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_novomodelo = SimpleNamespace(write_policy_checkpoint=lambda: None)
    monkeypatch.setitem(sys.modules, "novomodelo", stub_novomodelo)
    ensure_writer_binding()  # must not raise


def test_bootstrap_does_not_mutate_input_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The in-process run applies its 1-iteration cap through
    ``config_overrides`` (in memory) and writes only to ``work_dir``; the
    flattened variant is materialised under ``work_dir`` too
    (:func:`~novomodelo_bridge.decomp.fcf.bootstrap._flatten_terminal_fan`), so
    the input case's ``config.json`` and ``stages.json`` are left
    byte-for-byte untouched — no scratch copy under ``case_dir``, no file
    edit there."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_case(case_dir)
    original_config = (case_dir / "config.json").read_bytes()
    original_stages = (case_dir / "stages.json").read_bytes()

    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "state_dimension": 1,
                "entity_manifest": [{"entity_type": 0, "entity_id": 0, "subindex": 0}],
                "node_id": 0,
                "graph_stage_id": 0,
                "priced_state_date": 20_261_101,
            },
        ],
        "metadata": {"season_manifest": _SEASON_MANIFEST},
    }
    monkeypatch.setitem(sys.modules, "novomodelo", _stub_novomodelo(fake_policy))
    bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")

    assert (case_dir / "config.json").read_bytes() == original_config
    assert (case_dir / "stages.json").read_bytes() == original_stages


def test_bootstrap_passes_single_iteration_run_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bootstrap drives novomodelo in-process for a single iteration: it calls
    ``novomodelo.run.run`` on the flattened variant (never ``case_dir`` itself)
    with a ``config_overrides`` capping training at one iteration and disabling
    simulation, and an ``on_iteration`` stop callback — never a subprocess
    binary."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_case(case_dir)

    args: list[object] = []
    calls: list[dict[str, object]] = []

    def _record_run(*_args: object, **kwargs: object) -> None:
        args.extend(_args)
        calls.append(kwargs)

    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "state_dimension": 1,
                "entity_manifest": [{"entity_type": 0, "entity_id": 0, "subindex": 0}],
                "node_id": 0,
                "graph_stage_id": 0,
                "priced_state_date": 20_261_101,
            },
        ],
        "metadata": {"season_manifest": _SEASON_MANIFEST},
    }
    monkeypatch.setitem(
        sys.modules, "novomodelo", _stub_novomodelo(fake_policy, run=_record_run)
    )
    bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")

    assert len(calls) == 1
    assert args == [str(tmp_path / "work" / "bootstrap_variant")]
    kwargs = calls[0]
    assert "skip_simulation" not in kwargs
    assert kwargs["config_overrides"]["training.stopping_rules"] == [
        {"type": "iteration_limit", "limit": 1}
    ]
    assert kwargs["config_overrides"]["simulation.enabled"] is False
    assert callable(kwargs["on_iteration"])
    assert kwargs["on_iteration"](object()) is True  # stops at the first boundary


def test_bootstrap_flattens_terminal_fan_and_preserves_stage_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC 1 — given a ``stages.json`` policy graph with a 2-leaf terminal
    fan, the bootstrap variant's ``policy_graph.nodes``/``transitions`` are
    empty lists while every stage's own content is unchanged; the only
    addition is the ``num_openings: 1`` companion field novomodelo's chain
    dialect requires (see ``_flatten_terminal_fan``'s docstring). The real
    case's own ``stages.json`` is left untouched."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_case(case_dir)

    run_case_paths: list[str] = []

    def _record_run(case_path: str, **_kwargs: object) -> None:
        run_case_paths.append(case_path)

    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "state_dimension": 1,
                "entity_manifest": [{"entity_type": 0, "entity_id": 0, "subindex": 0}],
                "node_id": 0,
                "graph_stage_id": 0,
                "priced_state_date": 20_261_101,
            },
        ],
        "metadata": {"season_manifest": _SEASON_MANIFEST},
    }
    monkeypatch.setitem(
        sys.modules, "novomodelo", _stub_novomodelo(fake_policy, run=_record_run)
    )

    bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")

    assert len(run_case_paths) == 1
    variant_dir = Path(run_case_paths[0])
    assert variant_dir != case_dir

    variant_doc = json.loads((variant_dir / "stages.json").read_text(encoding="utf-8"))
    assert variant_doc["policy_graph"]["nodes"] == []
    assert variant_doc["policy_graph"]["transitions"] == []
    assert [
        {key: value for key, value in stage.items() if key != "num_openings"}
        for stage in variant_doc["stages"]
    ] == _FANNED_STAGES["stages"]
    assert all(stage["num_openings"] == 1 for stage in variant_doc["stages"])

    real_doc = json.loads((case_dir / "stages.json").read_text(encoding="utf-8"))
    assert real_doc == _FANNED_STAGES


def test_bootstrap_raises_on_novomodelo_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure inside the run propagates as ``novomodelo.run.run``'s exception."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_case(case_dir)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom: bad case")

    monkeypatch.setitem(sys.modules, "novomodelo", _stub_novomodelo({}, run=_boom))

    with pytest.raises(RuntimeError, match="boom: bad case"):
        bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")


def test_bootstrap_raises_on_empty_stage_cuts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint whose ``stage_cuts`` list is empty is a load error, not a
    legitimate empty terminal pool (an empty *terminal* pool still carries an
    ``entity_manifest``; an empty ``stage_cuts`` list means no pool at all)."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_case(case_dir)

    monkeypatch.setitem(sys.modules, "novomodelo", _stub_novomodelo({"stage_cuts": []}))

    with pytest.raises(RuntimeError, match="no stage cuts"):
        bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")


def test_bootstrap_raises_on_empty_terminal_entity_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_case(case_dir)

    fake_policy = {
        "stage_cuts": [
            {"stage_id": 0, "state_dimension": 5, "entity_manifest": []},
        ],
    }
    monkeypatch.setitem(sys.modules, "novomodelo", _stub_novomodelo(fake_policy))

    with pytest.raises(RuntimeError, match="empty terminal entity_manifest"):
        bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")


def test_bootstrap_returns_real_single_node_id_after_flattening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC 2 — a flattened bootstrap whose ``load_policy`` seam reports a
    real single node (``node_id=7``, a non-sentinel, non-zero id — a value
    that would stay hidden by an accidental falsy-``0`` check) threads
    through with no ``RuntimeError``."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_case(case_dir)

    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "state_dimension": 1,
                "entity_manifest": [{"entity_type": 0, "entity_id": 0, "subindex": 0}],
                "node_id": 7,
                "graph_stage_id": 1,
                "priced_state_date": 20_261_101,
            },
        ],
        "metadata": {"season_manifest": _SEASON_MANIFEST},
    }
    monkeypatch.setitem(sys.modules, "novomodelo", _stub_novomodelo(fake_policy))

    manifest = bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")

    assert manifest.node_id == 7
    assert manifest.graph_stage_id == 1


def test_bootstrap_returns_node_and_graph_stage_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_case(case_dir)

    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "state_dimension": 1,
                "entity_manifest": [{"entity_type": 0, "entity_id": 0, "subindex": 0}],
                "node_id": 0,
                "graph_stage_id": 4,
                "priced_state_date": 20_261_101,
            },
        ],
        "metadata": {"season_manifest": _SEASON_MANIFEST},
    }
    monkeypatch.setitem(sys.modules, "novomodelo", _stub_novomodelo(fake_policy))

    manifest = bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")

    assert manifest.node_id == 0
    assert manifest.graph_stage_id == 4
    assert manifest.priced_state_date == 20_261_101
    assert manifest.season_manifest == _SEASON_MANIFEST


def test_bootstrap_raises_on_missing_node_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_case(case_dir)

    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "state_dimension": 1,
                "entity_manifest": [{"entity_type": 0, "entity_id": 0, "subindex": 0}],
                "graph_stage_id": 4,
            },
        ],
    }
    monkeypatch.setitem(sys.modules, "novomodelo", _stub_novomodelo(fake_policy))

    with pytest.raises(RuntimeError, match="node_id"):
        bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")


def test_bootstrap_raises_on_missing_graph_stage_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_case(case_dir)

    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "state_dimension": 1,
                "entity_manifest": [{"entity_type": 0, "entity_id": 0, "subindex": 0}],
                "node_id": 0,
            },
        ],
    }
    monkeypatch.setitem(sys.modules, "novomodelo", _stub_novomodelo(fake_policy))

    with pytest.raises(RuntimeError, match="graph_stage_id"):
        bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")


def test_bootstrap_raises_on_missing_priced_state_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_case(case_dir)

    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "state_dimension": 1,
                "entity_manifest": [{"entity_type": 0, "entity_id": 0, "subindex": 0}],
                "node_id": 0,
                "graph_stage_id": 4,
            },
        ],
    }
    monkeypatch.setitem(sys.modules, "novomodelo", _stub_novomodelo(fake_policy))

    with pytest.raises(RuntimeError, match="priced_state_date"):
        bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")


def test_bootstrap_raises_on_priced_state_date_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal pool whose ``priced_state_date`` is the undated sentinel
    (``i32::MIN``) is rejected: the date-driven boundary loader has nothing to
    select a source against, so an undated pool cannot be authored."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_case(case_dir)

    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "state_dimension": 1,
                "entity_manifest": [{"entity_type": 0, "entity_id": 0, "subindex": 0}],
                "node_id": 0,
                "graph_stage_id": 4,
                "priced_state_date": -(2**31),
            },
        ],
    }
    monkeypatch.setitem(sys.modules, "novomodelo", _stub_novomodelo(fake_policy))

    with pytest.raises(RuntimeError, match="priced_state_date is the undated sentinel"):
        bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")


def test_bootstrap_raises_on_missing_season_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint whose metadata carries no ``season_manifest`` (an installed
    novomodelo predating the season-manifest round-trip) is rejected: the authored
    boundary would have nothing to satisfy novomodelo's season-compatibility gate."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_case(case_dir)

    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "state_dimension": 1,
                "entity_manifest": [{"entity_type": 0, "entity_id": 0, "subindex": 0}],
                "node_id": 0,
                "graph_stage_id": 4,
                "priced_state_date": 20_261_101,
            },
        ],
        "metadata": {},
    }
    monkeypatch.setitem(sys.modules, "novomodelo", _stub_novomodelo(fake_policy))

    with pytest.raises(RuntimeError, match="season_manifest"):
        bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")


def test_bootstrap_raises_on_node_id_shared_pool_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_case(case_dir)

    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "state_dimension": 1,
                "entity_manifest": [{"entity_type": 0, "entity_id": 0, "subindex": 0}],
                "node_id": -1,
                "graph_stage_id": 4,
            },
        ],
    }
    monkeypatch.setitem(sys.modules, "novomodelo", _stub_novomodelo(fake_policy))

    with pytest.raises(RuntimeError, match="shared-pool sentinel"):
        bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")


def test_bootstrap_raises_when_novomodelo_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no ``novomodelo`` importable, the in-process bootstrap fails loudly at
    its ``import novomodelo`` rather than silently. Setting the ``sys.modules``
    entry to ``None`` (rather than deleting it) forces ``ModuleNotFoundError``
    even where novomodelo is genuinely installed, so this test behaves the same in
    the dev venv and in a novomodelo-free venv."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_case(case_dir)

    monkeypatch.setitem(sys.modules, "novomodelo", None)

    with pytest.raises(ModuleNotFoundError):
        bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")

"""Tests for the boundary FCF importer's config-patch orchestration
(``fcf/importer.py::import_boundary_fcf``/``_patch_policy_boundary``).

Every test below exercises the importer's orchestration (config-patch,
empty-storage-manifest guard) with every novomodelo/deck seam monkeypatched — no
binary, no deck, and no installed novomodelo wheel needed.
"""

from __future__ import annotations

import inspect
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from novomodelo_bridge.decomp.fcf.importer import (
    _patch_policy_boundary,
    _seed_recent_observations,
    import_boundary_fcf,
)
from novomodelo_bridge.decomp.fcf.mapper import MappingResult
from novomodelo_bridge.novomodelo.case_writer import CaseWriter
from tests._fcf_fixtures import (
    make_boundary_cuts,
    make_cut_record,
    make_id_map,
    make_manifest,
    make_mapped_cut,
    make_slot,
)
from tests.conftest import make_decomp_case, make_decomp_files

if TYPE_CHECKING:
    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.fcf.cortes import BoundaryCuts


def _mock_deck_and_cut_seams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_cuts: BoundaryCuts
) -> DecompCase:
    """Monkeypatch the cut-reader seams shared by every binary-free
    ``import_boundary_fcf`` orchestration test below, and build the shared
    ``DecompCase`` the importer reads instead of re-discovering/re-parsing
    the deck.

    Stubs ``Cortesh``/``read_cortes`` to minimal stand-ins (their return
    values are opaque placeholders — only ``read_cortes`` needs a shape any
    downstream code inspects), so ``read_cortes`` returns ``fake_cuts``
    verbatim, and stubs the ``sys.modules['novomodelo']`` entry so
    ``import_boundary_fcf``'s own unconditional ``import novomodelo`` (needed for
    ``novomodelo.__version__``) resolves without the novomodelo-python wheel
    installed — the `"stub sys.modules['novomodelo']"` pattern
    (``tests/decomp/test_fcf_bootstrap.py``), not a module attribute patch.

    Returns a ``DecompCase`` with ``dadger``/``id_map``/``hidr``/``calendar``
    pre-filled (opaque placeholders — the importer never re-parses them) and
    ``files.cortesh``/``files.cortes``/``files.vazoes`` set under
    ``tmp_path``. No real deck and no ``mlt.dat``, so the inflow-lag mean
    fold and recent-observation seed both stay on their ``None`` no-op branch.
    """
    monkeypatch.setitem(
        sys.modules, "novomodelo", SimpleNamespace(__version__="0.13.0")
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.Cortesh",
        SimpleNamespace(read=lambda _path: object()),
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.read_cortes",
        lambda *_args, **_kwargs: fake_cuts,
    )
    # The coupling-stage per-block-hours read (case_dir/stages.json) is a
    # case-reading seam like the deck ones above; these binary-free
    # orchestration cases carry only a minimal config.json, so stub it to a
    # single 648 h block rather than author a full stages.json. `import_
    # boundary_fcf` derives its scalar `cost_unit_hours` as the sum of these,
    # so a one-element `[648.0]` preserves the prior 648 h; these
    # storage-only cases place no live GNL ring, so the per-block length is
    # never validated against `n_patamares`.
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer._final_stage_block_hours",
        lambda _case_dir: [648.0],
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.ensure_writer_binding", lambda: None
    )

    files = make_decomp_files(
        tmp_path,
        cortesh=tmp_path / "deck" / "cortesh.dat",
        cortes=tmp_path / "deck" / "cortes-010.dat",
    )
    return make_decomp_case(
        files,
        dadger=object(),
        id_map=make_id_map(()),
        hidr=object(),
        calendar=[],
    )


def test_patch_policy_boundary_preserves_other_sections(tmp_path: Path) -> None:
    """AC 1 — `_patch_policy_boundary` sets `policy.boundary` and leaves
    every other top-level section byte-for-byte unchanged, writing the file
    in `decomp/pipeline.py`'s `_write_json` style (`indent=2`,
    `ensure_ascii=False`, single trailing newline).

    Guards the raw serialized text against an independently constructed
    expected string, not just `json.loads`-parsed equality — parsed
    equality is indent-agnostic, so it would pass under any indentation and
    never actually exercise `indent=2`. `simulation.scenario_label` carries
    a raw non-ASCII character (not a `\\uXXXX` escape) so `ensure_ascii=False`
    is genuinely distinguished: under `ensure_ascii=True` that character
    would still round-trip through `json.loads` equality, but the raw text
    on disk would differ.
    """
    config_path = tmp_path / "config.json"
    other_sections = {
        "state_space": {"hydro_storage": {"n_plants": 3}},
        "training": {"stopping_rules": [{"type": "iteration_limit", "limit": 500}]},
        "simulation": {"num_openings": 10, "scenario_label": "cenário"},
    }
    # The pipeline's own in-memory dict — no
    # config.json exists on disk yet; `_patch_policy_boundary` mutates and
    # writes *this* object, never re-reading the file.
    config = dict(other_sections)

    _patch_policy_boundary(CaseWriter(tmp_path), config)

    assert config["policy"]["boundary"] == {"path": "boundary"}
    for key, value in other_sections.items():
        assert config[key] == value

    patched_text = config_path.read_text(encoding="utf-8")
    patched = json.loads(patched_text)

    assert patched["policy"]["boundary"] == {"path": "boundary"}
    for key, value in other_sections.items():
        assert patched[key] == value

    expected_config = {
        **other_sections,
        "policy": {"boundary": {"path": "boundary"}},
    }
    expected_text = json.dumps(expected_config, indent=2, ensure_ascii=False) + "\n"
    assert patched_text == expected_text, (
        "patched config.json's raw serialized text diverges from the "
        "expected indent=2 / ensure_ascii=False / single-trailing-newline "
        "style"
    )
    assert "cenário" in patched_text, (
        "raw non-ASCII character was escaped on write — ensure_ascii=False not honored"
    )


def test_import_boundary_fcf_patches_config_and_emits_no_output_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The orchestration's return value and config patch are correct, and the
    boundary loads on a plain `novomodelo run <case>` — no run-with-`--output`
    warning, because novomodelo now resolves `policy.boundary.path` against the case
    dir (the former C8 gap is closed).

    Needs no novomodelo binary, no real deck, and no installed novomodelo wheel: every
    seam that would touch any of the three (`bootstrap_terminal_manifest`,
    the cut readers, the checkpoint writer, and `import_boundary_fcf`'s own
    internal `import novomodelo`) is monkeypatched to a minimal stand-in — a
    "monkeypatched-shape unit path" — isolating the orchestration under
    test: that `import_boundary_fcf` returns `case_dir / "boundary"` and
    patches `config.json`'s `policy.boundary` to the date-driven
    `{"path": "boundary"}` block (no stage/pool index — novomodelo selects the
    source by calendar date). The regression guards below pin that neither the
    retired `--output` WARNING nor the `TRACKED NOVOMODELO-GAP (C8)` code comment
    comes back.
    """
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    # A real (if minimal) `BoundaryCuts`/`MappingResult` pair, not a bare
    # `SimpleNamespace` — `_emit_import_diagnostics` (called between
    # `map_boundary_cuts` and `write_boundary_checkpoint`) reads
    # `cuts.records` (via `summarize_cut_families`) and `mapping.dropped`, so
    # the fakes must carry those attributes with a shape that survives the
    # real triage pass, not just the fields this test's own assertions read.
    fake_cuts = make_boundary_cuts(
        plant_codes=(1,),
        records=(make_cut_record(pi_varm=(1.5,), rhs=10.0, forward_pass_index=0),),
        boundary_stage=10,
    )
    case = _mock_deck_and_cut_seams(monkeypatch, tmp_path, fake_cuts)
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.bootstrap_terminal_manifest",
        lambda *_args, **_kwargs: make_manifest([make_slot(0, 0, 0)]),
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.map_boundary_cuts",
        lambda *_args, **_kwargs: MappingResult(
            cuts=(make_mapped_cut(coefficients=(1.5,)),), dropped=()
        ),
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.build_stage_cuts_payload",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.write_boundary_checkpoint",
        lambda *_args, **_kwargs: None,
    )

    config: dict = {}
    with caplog.at_level(logging.WARNING):
        boundary_dir = import_boundary_fcf(
            case_dir,
            case,
            work_dir=tmp_path / "work",
            cost_scale_factor=1.0,
            config=config,
            initial_conditions={},
        )

    assert boundary_dir == case_dir / "boundary"

    patched_config = json.loads((case_dir / "config.json").read_text(encoding="utf-8"))
    assert patched_config["policy"]["boundary"] == {"path": "boundary"}
    # The passed dict is mutated in place, never re-read from disk.
    assert config["policy"]["boundary"] == {"path": "boundary"}

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    # The retired C8 workaround emitted a run-with-`--output` WARNING; the gap
    # is closed, so no such warning may surface.
    assert not any("--output" in r.message for r in warnings), (
        f"a retired run-with-output WARNING resurfaced: {caplog.text}"
    )

    source_path = inspect.getsourcefile(import_boundary_fcf)
    assert source_path is not None
    source_text = Path(source_path).read_text(encoding="utf-8")
    assert "TRACKED NOVOMODELO-GAP WORKAROUND (C8" not in source_text


def test_import_boundary_fcf_rejects_storageless_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 4 — a bootstrapped manifest with no `HydroStorage` slot at all is
    a terminal-manifest read bug, not a legitimate empty-storage case.

    `map_boundary_cuts` is deliberately left un-mocked here (unlike the
    config-patch test above) — its own read-bug guard must fire for real, and
    `import_boundary_fcf` must let the resulting `ValueError` propagate
    verbatim rather than swallow it.
    """
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    fake_cuts = make_boundary_cuts(
        plant_codes=(1,),
        records=(make_cut_record(pi_varm=(1.5,), rhs=10.0, forward_pass_index=0),),
        boundary_stage=10,
    )
    case = _mock_deck_and_cut_seams(monkeypatch, tmp_path, fake_cuts)
    # Only a `HydroInflowLag` (entity_type 1) slot — no `HydroStorage`
    # (entity_type 0) slot at all — triggers `map_boundary_cuts`'s own
    # read-bug guard.
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.bootstrap_terminal_manifest",
        lambda *_args, **_kwargs: make_manifest([make_slot(1, 0, 0)]),
    )

    with pytest.raises(ValueError, match="HydroStorage"):
        import_boundary_fcf(
            case_dir,
            case,
            work_dir=tmp_path / "work",
            cost_scale_factor=1.0,
            config={},
            initial_conditions={},
        )


def test_import_boundary_fcf_raises_when_config_missing_with_cut_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC — once cut files are present, ``config``/``initial_conditions``
    left at their ``None`` default (rather than the pipeline's own in-memory
    dicts) is a caller bug, not a legitimate no-op: `import_boundary_fcf`
    raises `ValueError` before touching the cut readers or the writer.

    `config` is left at its default `None` here; the guard's `or` makes
    either missing dict raise the same message, so this covers the pairing.
    """
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    fake_cuts = make_boundary_cuts(
        plant_codes=(1,),
        records=(make_cut_record(pi_varm=(1.5,), rhs=10.0, forward_pass_index=0),),
        boundary_stage=10,
    )
    case = _mock_deck_and_cut_seams(monkeypatch, tmp_path, fake_cuts)

    with pytest.raises(
        ValueError,
        match="requires the converted case's config and initial_conditions",
    ):
        import_boundary_fcf(
            case_dir,
            case,
            work_dir=tmp_path / "work",
            cost_scale_factor=1.0,
            initial_conditions={},
        )


def test_import_boundary_fcf_reuses_shared_case_no_reparse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 2 — the importer reads the shared case's already-parsed ``dadger``/
    ``id_map`` rather than re-discovering and re-parsing the deck itself: a
    spy on ``idecomp.decomp.Dadger.read`` is never called when
    ``import_boundary_fcf`` is handed a pre-parsed shared case."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    fake_cuts = make_boundary_cuts(
        plant_codes=(1,),
        records=(make_cut_record(pi_varm=(1.5,), rhs=10.0, forward_pass_index=0),),
        boundary_stage=10,
    )
    case = _mock_deck_and_cut_seams(monkeypatch, tmp_path, fake_cuts)
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.bootstrap_terminal_manifest",
        lambda *_args, **_kwargs: make_manifest([make_slot(0, 0, 0)]),
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.map_boundary_cuts",
        lambda *_args, **_kwargs: MappingResult(
            cuts=(make_mapped_cut(coefficients=(1.5,)),), dropped=()
        ),
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.build_stage_cuts_payload",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.write_boundary_checkpoint",
        lambda *_args, **_kwargs: None,
    )
    read_spy = MagicMock(name="Dadger.read")
    monkeypatch.setattr("idecomp.decomp.Dadger.read", read_spy)

    boundary_dir = import_boundary_fcf(
        case_dir,
        case,
        work_dir=tmp_path / "work",
        cost_scale_factor=1.0,
        config={},
        initial_conditions={},
    )

    assert boundary_dir == case_dir / "boundary"
    read_spy.assert_not_called()


def test_seed_recent_observations_mutates_passed_dict_no_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC 5 (byte-identity) — `_seed_recent_observations` folds
    `recent_observations` into the passed `initial_conditions` dict and
    writes it through `CaseWriter` once, never re-reading
    `initial_conditions.json` off disk: the written file is byte-identical
    to what a prior read-modify-write produced from the same starting
    document (indent=2 / ensure_ascii=False / single trailing newline, the
    `_write_json` style), and every pre-existing key survives untouched.
    """
    windows = [{"hydro_id": 1, "values": [10.0, 20.0]}]
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.Vazoes",
        SimpleNamespace(read=lambda _path: object()),
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.convert_recent_observation_windows",
        lambda *_args, **_kwargs: windows,
    )

    ic_path = tmp_path / "initial_conditions.json"
    other_sections = {
        "$schema": "https://example.invalid/initial_conditions.json",
        "storage": [{"hydro_id": 1, "volume_hm3": 100.0}],
        "filling_storage": [],
    }
    # The pipeline's own in-memory dict — no initial_conditions.json exists
    # on disk yet; `_seed_recent_observations` mutates and writes *this*
    # object, never re-reading the file.
    initial_conditions = dict(other_sections)
    case = make_decomp_case(
        make_decomp_files(tmp_path),
        dadger=object(),
        id_map=make_id_map(()),
        hidr=object(),
        calendar=[],
    )

    n_windows = _seed_recent_observations(
        CaseWriter(tmp_path),
        case,
        effective=object(),
        calendar=[],
        initial_conditions=initial_conditions,
    )

    assert n_windows == 1
    assert initial_conditions["recent_observations"] == windows
    for key, value in other_sections.items():
        assert initial_conditions[key] == value

    expected = {**other_sections, "recent_observations": windows}
    expected_text = json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
    written_text = ic_path.read_text(encoding="utf-8")
    assert written_text == expected_text, (
        "patched initial_conditions.json's raw serialized text diverges from "
        "the expected indent=2 / ensure_ascii=False / single-trailing-newline "
        "style"
    )

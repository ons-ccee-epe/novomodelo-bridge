"""Tests for the boundary FCF importer orchestration
(``fcf/importer.py::import_boundary_fcf``).

Synthetic fixtures only: no deck, no novomodelo binary.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.units import MONTH_HOURS
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.anticipated import (
    GnlCommitmentModel,
    GnlThermal,
)
from novomodelo_bridge.decomp.fcf.cortes import BoundaryCuts, summarize_cut_families
from novomodelo_bridge.decomp.fcf.importer import (
    _coupling_stage_hours,
    _emit_import_diagnostics,
    _gnl_targets_from,
    _post_horizon_start,
    import_boundary_fcf,
)
from novomodelo_bridge.decomp.fcf.mapper import (
    DroppedTerm,
    GnlRingPlan,
    GnlThermalTarget,
    MappingResult,
    map_boundary_cuts,
)
from tests._fcf_fixtures import (
    make_boundary_cuts,
    make_cortes_header,
    make_cut_record,
    make_id_map,
    make_manifest,
    make_mapped_cut,
    make_slot,
)
from tests.conftest import make_decomp_case, make_decomp_files


def test_emit_import_diagnostics_ac1_ac2_from_synthetic() -> None:
    """Both diagnostics fire against fully synthetic inputs: no deck, no novomodelo
    binary. ``dropped`` uses codes 20/30 (not a real deck's plant codes), so
    this test carries no hidden dependency on ``example/``.
    """
    cuts = make_boundary_cuts((1,), (make_cut_record(pi_varm=(1.5,), rhs=10.0),))
    mapping = MappingResult(
        cuts=(make_mapped_cut(coefficients=(1.5,), intercept=10.0),),
        dropped=(
            DroppedTerm(plant_code=20, beta=0.4),
            DroppedTerm(plant_code=30, beta=0.7),
        ),
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping)

    assert {diagnostic.code for diagnostic in sink} == {
        "boundary-fcf-cut-family-summary",
        "boundary-fcf-source-only-plants-dropped",
    }
    assert all(diagnostic.severity is dx.Severity.INFO for diagnostic in sink)

    by_code = {diagnostic.code: diagnostic for diagnostic in sink}
    dropped_diagnostic = by_code["boundary-fcf-source-only-plants-dropped"]
    assert dropped_diagnostic.table is not None
    assert {row[0] for row in dropped_diagnostic.table.rows} == {20, 30}

    summary_diagnostic = by_code["boundary-fcf-cut-family-summary"]
    summary = summarize_cut_families(cuts)
    # The figures live in `summary`, not a separate restating `notes` bullet
    # (dropped as duplicate bloat).
    assert summary_diagnostic.notes == []
    assert str(summary.n_active_cuts) in summary_diagnostic.summary
    assert str(summary.storage_nonzero_plants) in summary_diagnostic.summary
    assert str(summary.lag_nonzero_by_depth) in summary_diagnostic.summary


def test_emit_import_diagnostics_no_dropped_gates_dropped_diagnostic_off() -> None:
    """``mapping.dropped == ()`` gates off the dropped-plant diagnostic while
    the cut-family-summary diagnostic still fires. No novomodelo binary or real deck
    needed: both payloads are hand-built from the shared
    ``tests/_fcf_fixtures.py`` builders.
    """
    cuts = make_boundary_cuts((1,), (make_cut_record(pi_varm=(1.5,), rhs=10.0),))
    mapping = MappingResult(
        cuts=(make_mapped_cut(coefficients=(1.5,), intercept=10.0),), dropped=()
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping)

    assert [diagnostic.code for diagnostic in sink] == [
        "boundary-fcf-cut-family-summary"
    ]
    assert sink[0].severity is dx.Severity.INFO


def test_import_boundary_fcf_no_cut_files_is_noop(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """AC 4 — ``case.files.cortesh``/``case.files.cortes`` both ``None`` is an
    unconditional no-op: no ``boundary/`` directory, no binary invoked."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    case = make_decomp_case(tmp_path)

    with caplog.at_level(logging.INFO):
        result = import_boundary_fcf(
            case_dir,
            case,
            work_dir=tmp_path / "work",
            cost_scale_factor=1.0,
        )

    assert result is None
    assert not (case_dir / "boundary").exists()
    assert "boundary FCF skipped" in caplog.text


# ---------------------------------------------------------------------------
# Thread node_id/graph_stage_id/priced_state_date from the bootstrap manifest
# into `build_stage_cuts_payload`, while `config.json`'s `policy.boundary`
# stays the date-driven `{"path": "boundary"}` block (the cut file's calendar
# `boundary_stage` flows only into the payload's `stage_id`). Every
# cut-reader/novomodelo-import seam is monkeypatched (mirrors
# `test_fcf_injection.py`'s seam-stubbing convention, kept local per the
# one-home-per-source-module test convention) — no real deck, novomodelo binary, or
# installed novomodelo wheel needed.
# ---------------------------------------------------------------------------


def _stub_import_seams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_cuts: BoundaryCuts
) -> DecompCase:
    """Monkeypatch the deck/cut-reader/novomodelo-import seams shared by the
    node_id/graph_stage_id threading tests below, returning the
    ``DecompCase`` the importer reads instead of re-parsing a deck."""
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


def test_import_boundary_fcf_threads_node_and_graph_stage_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC 1 — given a bootstrap manifest with node_id=0/graph_stage_id=4/
    priced_state_date=20261101 and a cuts file whose boundary_stage=4,
    `build_stage_cuts_payload` is called with `cost_scale_factor=1.0`,
    `node_id=0`, `graph_stage_id=4`, `priced_state_date=20261101`."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    fake_cuts = make_boundary_cuts(
        plant_codes=(1,),
        records=(make_cut_record(pi_varm=(1.5,), rhs=10.0, forward_pass_index=0),),
        boundary_stage=4,
    )
    case = _stub_import_seams(monkeypatch, tmp_path, fake_cuts)
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer._final_stage_block_hours",
        lambda _case_dir: [648.0],
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.bootstrap_terminal_manifest",
        lambda *_args, **_kwargs: make_manifest(
            [make_slot(0, 0, 0)],
            node_id=0,
            graph_stage_id=4,
            priced_state_date=20_261_101,
        ),
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.map_boundary_cuts",
        lambda *_args, **_kwargs: MappingResult(
            cuts=(make_mapped_cut(coefficients=(1.5,)),), dropped=()
        ),
    )
    calls: list[dict[str, object]] = []

    def _spy_build_stage_cuts_payload(
        *_args: object, **kwargs: object
    ) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.build_stage_cuts_payload",
        _spy_build_stage_cuts_payload,
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.write_boundary_checkpoint",
        lambda *_args, **_kwargs: None,
    )

    import_boundary_fcf(
        case_dir,
        case,
        work_dir=tmp_path / "work",
        cost_scale_factor=1.0,
        config={},
        initial_conditions={},
    )

    assert len(calls) == 1
    assert calls[0] == {
        "stage_id": 4,
        "cost_scale_factor": 1.0,
        "node_id": 0,
        "graph_stage_id": 4,
        "priced_state_date": 20_261_101,
    }


def test_import_boundary_fcf_graph_stage_id_distinct_from_boundary_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bootstrap manifest's `graph_stage_id` disagreeing with the cut
    file's own `boundary_stage` is not an error (they are different axes —
    novomodelo's own 0-based pool identity vs. the source model's 1-based
    calendar month count — that only coincidentally share a value):
    `build_stage_cuts_payload` receives `graph_stage_id=3` (the manifest's)
    and `stage_id=4` (the cut file's `boundary_stage`), and `config.json`'s
    `policy.boundary` carries no stage/pool index at all — the source is
    selected by calendar date."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    fake_cuts = make_boundary_cuts(
        plant_codes=(1,),
        records=(make_cut_record(pi_varm=(1.5,), rhs=10.0, forward_pass_index=0),),
        boundary_stage=4,
    )
    case = _stub_import_seams(monkeypatch, tmp_path, fake_cuts)
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer._final_stage_block_hours",
        lambda _case_dir: [648.0],
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.bootstrap_terminal_manifest",
        lambda *_args, **_kwargs: make_manifest(
            [make_slot(0, 0, 0)], node_id=0, graph_stage_id=3
        ),
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.map_boundary_cuts",
        lambda *_args, **_kwargs: MappingResult(
            cuts=(make_mapped_cut(coefficients=(1.5,)),), dropped=()
        ),
    )
    calls: list[dict[str, object]] = []

    def _spy_build_stage_cuts_payload(
        *_args: object, **kwargs: object
    ) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.build_stage_cuts_payload",
        _spy_build_stage_cuts_payload,
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.write_boundary_checkpoint",
        lambda *_args, **_kwargs: None,
    )

    config: dict[str, object] = {}
    import_boundary_fcf(
        case_dir,
        case,
        work_dir=tmp_path / "work",
        cost_scale_factor=1.0,
        config=config,
        initial_conditions={},
    )

    assert len(calls) == 1
    assert calls[0]["stage_id"] == 4
    assert calls[0]["graph_stage_id"] == 3
    assert config["policy"]["boundary"] == {"path": "boundary"}


# ---------------------------------------------------------------------------
# GnlRingPlan build (`_gnl_targets_from`) + the per-cut GNL deviation
# diagnostic. All tier-1: pure Python, no deck, no novomodelo binary. The
# entity_type code is restated locally rather than importing the mapper
# module's private constant — mirrors ``test_fcf_mapper.py``'s identical
# convention.
# ---------------------------------------------------------------------------

_HYDRO_STORAGE = 0
_ANTICIPATED_THERMAL_STATE = 2

#: Every GNL fixture below builds an `n_patamares=3` header; a uniform split
#: of `MONTH_HOURS` across those 3 coupling blocks is enough to satisfy
#: `map_boundary_cuts`'s `coupling_block_hours` guard — none of
#: these tests assert on the placed GNL coefficient's magnitude, only on the
#: deviation diagnostic, which recomputes straight from the raw `pi_gnl`.
_UNIFORM_GNL_BLOCK_HOURS = (MONTH_HOURS / 3, MONTH_HOURS / 3, MONTH_HOURS / 3)


def _gnl_row(width: int, nonzero: dict[int, float]) -> tuple[float, ...]:
    """A `pi_gnl` flat vector of `width` zeros with `nonzero` columns set."""
    row = [0.0] * width
    for column, value in nonzero.items():
        row[column] = value
    return tuple(row)


def _make_gnl_model(
    thermals: tuple[GnlThermal, ...], nl_lag_months: dict[int, int]
) -> GnlCommitmentModel:
    """A minimal `GnlCommitmentModel` carrying only the ring-plan join surface."""
    return GnlCommitmentModel(
        thermals=thermals,
        commitments={},
        weeks_per_month={},
        nl_lag_months=nl_lag_months,
        tg=pd.DataFrame(),
    )


def _make_gnl_thermal(code: int, submarket_code: int) -> GnlThermal:
    """A minimal `GnlThermal`; cost/bounds are irrelevant to the ring join."""
    return GnlThermal(
        code=code,
        name=f"GNL-{code}",
        submarket_code=submarket_code,
    )


def test_gnl_targets_from_maps_codes_to_ids() -> None:
    """AC — the READBACK join: code 86 -> id 94 (SE, lag 2), code 224 -> id 95
    (NE, lag 1), ascending by code."""
    model = _make_gnl_model(
        (_make_gnl_thermal(86, 1), _make_gnl_thermal(224, 3)),
        {86: 2, 224: 1},
    )
    thermals_doc = {
        "thermals": [
            {"id": 1, "name": "not-gnl"},
            {"id": 94, "name": "GNL-86", "anticipated_config": {}},
            {"id": 95, "name": "GNL-224", "anticipated_config": {}},
        ]
    }

    plan = _gnl_targets_from(model, thermals_doc)

    assert plan == GnlRingPlan(
        (
            GnlThermalTarget(thermal_id=94, submercado=1, nl_lag=2),
            GnlThermalTarget(thermal_id=95, submercado=3, nl_lag=1),
        )
    )


def test_gnl_targets_from_count_mismatch_raises() -> None:
    """AC — a thermals.json GNL-fleet count that disagrees with the dadgnl
    registry count raises loudly, naming both counts, rather than silently
    mis-zipping ids onto the wrong codes."""
    model = _make_gnl_model(
        (_make_gnl_thermal(86, 1), _make_gnl_thermal(224, 3)),
        {86: 2, 224: 1},
    )
    thermals_doc = {
        "thermals": [
            {"id": 94, "anticipated_config": {}},
            {"id": 95, "anticipated_config": {}},
            {"id": 96, "anticipated_config": {}},
        ]
    }

    with pytest.raises(ValueError, match="GNL fleet must match exactly"):
        _gnl_targets_from(model, thermals_doc)


def test_gnl_targets_from_skips_plant_without_nl_lag(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC — a plant absent from ``nl_lag_months`` is skipped (its ring stays
    at coefficient 0.0 via the mapper's own drop path), never raised on, with
    a single INFO note naming it."""
    model = _make_gnl_model(
        (_make_gnl_thermal(86, 1), _make_gnl_thermal(224, 3)),
        {224: 1},  # 86 declares no nl dispatch-anticipation lag
    )
    thermals_doc = {
        "thermals": [
            {"id": 94, "anticipated_config": {}},
            {"id": 95, "anticipated_config": {}},
        ]
    }

    with caplog.at_level(logging.INFO):
        plan = _gnl_targets_from(model, thermals_doc)

    assert plan == GnlRingPlan(
        (GnlThermalTarget(thermal_id=95, submercado=3, nl_lag=1),)
    )
    assert "GNL-86" in caplog.text


def test_emit_import_diagnostics_gnl_deviation_fires() -> None:
    """For (submercado 1, lag 2) with col(1,p,2) = (0.1, 0.2, 0.3), the
    deviation diagnostic's row reports carried sum 0.6 and patamar spread
    (0.3 - 0.1) / 0.6. ``mapping`` is produced by the real
    ``map_boundary_cuts``, never hand-built, so the diagnostic is exercised
    against the same placement it reports on.
    """
    id_map = make_id_map(())
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # unrelated dummy, satisfies the guard
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0, interval_start=20260501),
        ]
    )
    header = make_cortes_header(
        (), lag_maximo_gnl=2, n_patamares=3, submercado_codes=(1,)
    )
    # col(1,p,2) for p=1..3 -> flat indices 1, 3, 5 (P=3, L=2):
    # col(s,p,l) = ((s-1)*3 + (p-1))*2 + (l-1).
    pi_gnl = _gnl_row(6, {1: 0.1, 3: 0.2, 5: 0.3})
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=5.0)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    gnl_plan = GnlRingPlan((GnlThermalTarget(thermal_id=94, submercado=1, nl_lag=2),))

    mapping = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        gnl_plan=gnl_plan,
        coupling_block_hours=_UNIFORM_GNL_BLOCK_HOURS,
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping, gnl_plan)

    codes = [diagnostic.code for diagnostic in sink]
    assert codes == [
        "boundary-fcf-cut-family-summary",
        "boundary-fcf-gnl-anticipated-deviation",
    ]
    deviation = sink[1]
    assert deviation.severity is dx.Severity.INFO
    assert deviation.table is not None
    row = next(r for r in deviation.table.rows if r[0] == 1 and r[1] == 2)
    assert row[2] == pytest.approx(0.6)
    assert row[3] == pytest.approx((0.3 - 0.1) / 0.6)


def test_emit_import_diagnostics_gnl_deviation_gated_off_without_plan() -> None:
    """``gnl_plan=None`` (the default) gates the deviation diagnostic off
    entirely: the sink carries only the baseline diagnostic(s), unchanged."""
    cuts = make_boundary_cuts((1,), (make_cut_record(pi_varm=(1.5,), rhs=10.0),))
    mapping = MappingResult(
        cuts=(make_mapped_cut(coefficients=(1.5,), intercept=10.0),), dropped=()
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping)

    assert [diagnostic.code for diagnostic in sink] == [
        "boundary-fcf-cut-family-summary"
    ]


# ---------------------------------------------------------------------------
# Covered-lane filter. `_post_horizon_start` (tier-1, no deck, no
# novomodelo binary) + the deviation diagnostic's dropped-coverage count reading
# the new uncovered-lane drops from `mapping.gnl_dropped`.
# ---------------------------------------------------------------------------


def test_post_horizon_start_returns_earliest_stage(tmp_path: Path) -> None:
    """AC 3 — the earliest ``stages[i].start_date`` becomes a ``YYYYMM01``
    month-anchor, regardless of the list's own order."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "post_study_stages.json").write_text(
        json.dumps(
            {
                "stages": [
                    {"start_date": "2026-05-09", "duration_hours": 168.0},
                    {"start_date": "2026-05-01", "duration_hours": 192.0},
                ],
                "thermal_bounds": [],
            }
        ),
        encoding="utf-8",
    )

    assert _post_horizon_start(case_dir) == 20260501


def test_post_horizon_start_month_anchors_a_mid_month_stage(tmp_path: Path) -> None:
    """AC 1 — an earliest stage starting mid-month (``2026-05-16``) still
    anchors to the 1st of that month (``20260501``), never the day it
    actually starts on: novomodelo's excised ring dates every surviving slot at
    month-anchor granularity, so the filter must not rely on the reference
    deck's day-01 coincidence."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "post_study_stages.json").write_text(
        json.dumps(
            {
                "stages": [{"start_date": "2026-05-16", "duration_hours": 168.0}],
                "thermal_bounds": [],
            }
        ),
        encoding="utf-8",
    )

    assert _post_horizon_start(case_dir) == 20260501


def test_post_horizon_start_docstring_is_month_anchor_not_full_day() -> None:
    """AC 4 — the docstring says ``YYYYMM01`` / month-anchor and no longer
    describes a ``YYYYMMDD`` full-day value or the retired K=0-lead-lane
    framing (source-text check)."""
    docstring = _post_horizon_start.__doc__
    assert docstring is not None
    assert "YYYYMM01" in docstring or "month-anchor" in docstring
    assert "YYYYMMDD" not in docstring
    assert "K=0" not in docstring


def test_post_horizon_start_none_when_file_absent(tmp_path: Path) -> None:
    """AC 3 — no ``post_study_stages.json`` at all returns ``None``."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    assert _post_horizon_start(case_dir) is None


def test_post_horizon_start_none_when_stages_empty(tmp_path: Path) -> None:
    """AC 3 — an empty ``stages`` list is the same "no horizon" case as an
    absent file."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "post_study_stages.json").write_text(
        json.dumps({"stages": [], "thermal_bounds": []}), encoding="utf-8"
    )

    assert _post_horizon_start(case_dir) is None


def test_post_horizon_start_malformed_start_date_propagates(tmp_path: Path) -> None:
    """Error Handling — a corrupt ``start_date`` is NOT silently downgraded
    to "no horizon"; the ``ValueError`` from ``date.fromisoformat`` propagates
    verbatim."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "post_study_stages.json").write_text(
        json.dumps({"stages": [{"start_date": "not-a-date"}]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Invalid isoformat string"):
        _post_horizon_start(case_dir)


def _write_stages(case_dir: Path, stages: list[dict[str, object]]) -> None:
    """Write a minimal ``stages.json`` (only the ``blocks[].hours`` the
    coupling-hours helper reads) into ``case_dir``."""
    (case_dir / "stages.json").write_text(
        json.dumps({"stages": stages}), encoding="utf-8"
    )


def test_coupling_stage_hours_sums_last_stage_blocks(tmp_path: Path) -> None:
    """The coupling-hours factor is the FINAL stage's summed block hours — the
    648 h April coupling period here, not an earlier weekly stage or a fixed
    730 h month."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_stages(
        case_dir,
        [
            {"blocks": [{"hours": 30.0}, {"hours": 74.0}, {"hours": 64.0}]},  # 168
            {"blocks": [{"hours": 108.0}, {"hours": 261.0}, {"hours": 279.0}]},  # 648
        ],
    )

    assert _coupling_stage_hours(case_dir) == pytest.approx(648.0)


def test_coupling_stage_hours_missing_file_raises(tmp_path: Path) -> None:
    """A boundary import cannot scale coefficients without the coupling stage's
    duration — an absent ``stages.json`` is a hard error, never a silent 0/730."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    with pytest.raises(ValueError, match="stages.json not found"):
        _coupling_stage_hours(case_dir)


def test_coupling_stage_hours_empty_stages_raises(tmp_path: Path) -> None:
    """An empty ``stages`` list has no coupling stage to read hours from."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_stages(case_dir, [])

    with pytest.raises(ValueError, match="carries no stages"):
        _coupling_stage_hours(case_dir)


def test_coupling_stage_hours_nonpositive_raises(tmp_path: Path) -> None:
    """A final stage with no positive block hours (e.g. an empty ``blocks``)
    cannot scale the cut coefficients — a hard error, not a divide-by-zero."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_stages(case_dir, [{"blocks": []}])

    with pytest.raises(ValueError, match="non-positive total block hours"):
        _coupling_stage_hours(case_dir)


def test_emit_import_diagnostics_gnl_deviation_dropped_count_includes_uncovered_lane() -> (
    None
):
    """AC 4 — the ``boundary-fcf-gnl-anticipated-deviation`` diagnostic's
    dropped-coverage count includes a covered-lane drop produced by a
    ``post_horizon_start`` that makes a target's only dated slot uncovered
    -- read straight from ``mapping.gnl_dropped``, never recomputed
    independently in the diagnostic.
    """
    id_map = make_id_map(())
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # unrelated dummy, satisfies the guard
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0, interval_start=20260401),
        ]
    )
    header = make_cortes_header(
        (), lag_maximo_gnl=2, n_patamares=3, submercado_codes=(1,)
    )
    # col(1,p,2) for p=1..3 -> flat indices 1, 3, 5 (P=3, L=2).
    pi_gnl = _gnl_row(6, {1: 0.1, 3: 0.2, 5: 0.3})
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=5.0)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    # thermal 94's only dated slot (20260401) is before this post_horizon_start
    # (20260501) -- non-covered, so the mapper drops it rather than placing it.
    gnl_plan = GnlRingPlan(
        (GnlThermalTarget(thermal_id=94, submercado=1, nl_lag=2),),
        post_horizon_start=20260501,
    )

    mapping = map_boundary_cuts(
        cuts, manifest, id_map, cost_unit_hours=MONTH_HOURS, gnl_plan=gnl_plan
    )
    assert any(
        term.thermal_id == 94 and "post-study horizon" in term.reason
        for term in mapping.gnl_dropped
    )
    assert mapping.cuts[0].coefficients[1] == 0.0  # dropped, stays at 0.0

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping, gnl_plan)

    deviation = next(
        d for d in sink if d.code == "boundary-fcf-gnl-anticipated-deviation"
    )
    assert deviation.table is not None
    assert "1 GNL term(s) dropped" in deviation.summary
    # No per-row `Dropped` column; the count above is the only place it is
    # reported.
    assert deviation.table.columns == [
        "Submercado",
        "Lag",
        "Σ pi_gnl (carried)",
        "Patamar spread",
    ]
    assert all(len(row) == 4 for row in deviation.table.rows)


def test_emit_import_diagnostics_c1_panel1_notes_deduped() -> None:
    """Panel 1's `notes` no longer restate `summary`.

    `lag_nonzero_by_depth` is the one fact `notes` used to carry that
    `summary` did not already state; folded into `summary`, `notes` is now
    empty rather than restating `n_active_cuts`/`storage_nonzero_plants`/
    `rhs_min`/`rhs_max`.
    """
    cuts = make_boundary_cuts((1,), (make_cut_record(pi_varm=(1.5,), rhs=10.0),))
    mapping = MappingResult(
        cuts=(make_mapped_cut(coefficients=(1.5,), intercept=10.0),), dropped=()
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping)

    summary_diagnostic = next(
        d for d in sink if d.code == "boundary-fcf-cut-family-summary"
    )
    summary = summarize_cut_families(cuts)
    assert summary_diagnostic.notes == []
    assert str(summary.lag_nonzero_by_depth) in summary_diagnostic.summary
    for stale_bullet in (
        f"n_active_cuts={summary.n_active_cuts}",
        f"storage_nonzero_plants={summary.storage_nonzero_plants}",
        f"rhs_min={summary.rhs_min!r}",
        f"rhs_max={summary.rhs_max!r}",
    ):
        assert stale_bullet not in summary_diagnostic.notes


def test_emit_import_diagnostics_c2_panel3_no_dropped_column() -> None:
    """Panel 3's table drops the constant `Dropped`
    column; the dropped count stays in `summary` only.
    """
    id_map = make_id_map(())
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # unrelated dummy, satisfies the guard
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0, interval_start=20260501),
        ]
    )
    header = make_cortes_header(
        (), lag_maximo_gnl=2, n_patamares=3, submercado_codes=(1,)
    )
    pi_gnl = _gnl_row(6, {1: 0.1, 3: 0.2, 5: 0.3})
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=5.0)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    gnl_plan = GnlRingPlan((GnlThermalTarget(thermal_id=94, submercado=1, nl_lag=2),))
    mapping = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        gnl_plan=gnl_plan,
        coupling_block_hours=_UNIFORM_GNL_BLOCK_HOURS,
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping, gnl_plan)

    deviation = next(
        d for d in sink if d.code == "boundary-fcf-gnl-anticipated-deviation"
    )
    assert deviation.table is not None
    assert deviation.table.columns == [
        "Submercado",
        "Lag",
        "Σ pi_gnl (carried)",
        "Patamar spread",
    ]
    assert deviation.table.justify == ["right", "right", "right", "right"]
    assert "0 GNL term(s) dropped" in deviation.summary


def test_emit_import_diagnostics_c3_headline_excludes_near_zero_sum_group() -> None:
    """A noisy group's inflated relative spread does not
    dominate the `max_spread` HEADLINE; the weight-carrying group's spread
    does, and an absolute spread is reported alongside.

    Uses the REAL-DECK magnitudes from the review that exposed
    the earlier fixed-magnitude floor's bug (`1e-6` failed to exclude a
    `4e-05` group, since `4e-05 >= 1e-6`): group `(submercado 1, lag 1)`
    carries the real weight (`Σ=-4412.0`, relative spread `392/4412 ≈
    0.0888`, matching the observed `~0.09`); group `(submercado 2, lag 1)`
    is the noisy one (`Σ=-4e-05`, relative spread `0.25`, matching the
    observed `0.25` exactly). `_GNL_DEVIATION_REL_FLOOR` (`1e-3`, a
    *fraction of the panel's own max |Σ|*) excludes the noisy group from
    the headline (`4e-05 < 1e-3 * 4412.0`) even though its own relative
    spread is far larger than the weight-carrying group's; the per-row
    table still carries both groups' own values unfiltered.
    """
    header = make_cortes_header(
        (), lag_maximo_gnl=1, n_patamares=3, submercado_codes=(1, 2)
    )
    # col(s,p,1) = (s-1)*3 + (p-1): submercado 1 -> cols 0,1,2; submercado 2
    # -> cols 3,4,5 (lag_maximo_gnl=1 collapses the lag axis to a single
    # column per patamar).
    # Group 1 (weight-carrying): sum -1340-1340-1732 = -4412.0; abs spread
    # -1340-(-1732) = 392.0; relative 392/4412 ≈ 0.0888 (observed ~0.09).
    # Group 2 (noisy): sum -1e-05-1e-05-2e-05 = -4e-05; abs spread
    # -1e-05-(-2e-05) = 1e-05; relative 1e-05/4e-05 = 0.25 (observed exactly).
    pi_gnl = (-1340.0, -1340.0, -1732.0, -1e-05, -1e-05, -2e-05)
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=0.0)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    gnl_plan = GnlRingPlan(
        (
            GnlThermalTarget(thermal_id=94, submercado=1, nl_lag=1),
            GnlThermalTarget(thermal_id=95, submercado=2, nl_lag=1),
        )
    )
    mapping = MappingResult(
        cuts=(make_mapped_cut(coefficients=(0.0,), intercept=0.0),),
        dropped=(),
        gnl_dropped=(),
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping, gnl_plan)

    deviation = next(
        d for d in sink if d.code == "boundary-fcf-gnl-anticipated-deviation"
    )
    assert deviation.table is not None
    rows_by_group = {(row[0], row[1]): row for row in deviation.table.rows}
    # `round(..., 6)` in the table rendering can perturb the raw ratio by up
    # to 5e-7, so use an absolute tolerance rather than the (much tighter)
    # relative default.
    assert rows_by_group[(1, 1)][3] == pytest.approx(392.0 / 4412.0, abs=1e-6)
    assert rows_by_group[(2, 1)][3] == pytest.approx(0.25, abs=1e-6)

    # The headline is the weight-carrying group's relative spread (~0.0888)
    # and the panel's overall absolute spread (392, from the same group) —
    # never the noisy group's inflated 0.25 relative figure, even though
    # 0.25 > 0.0888 on its own.
    expected_relative = f"{392.0 / 4412.0:.4g}"
    expected_absolute = f"{392.0:.4g}"
    assert (
        f"spread {expected_relative} relative / {expected_absolute} absolute"
        in deviation.summary
    )


def test_emit_import_diagnostics_c4_no_remediation_footer() -> None:
    """Panel 3 no longer carries a `remediation` footer; the C12 ledger row
    in novomodelo's conversion-found-improvements registry is the record now, not
    a runtime paragraph.
    """
    id_map = make_id_map(())
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # unrelated dummy, satisfies the guard
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0, interval_start=20260501),
        ]
    )
    header = make_cortes_header(
        (), lag_maximo_gnl=2, n_patamares=3, submercado_codes=(1,)
    )
    pi_gnl = _gnl_row(6, {1: 0.1, 3: 0.2, 5: 0.3})
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=5.0)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    gnl_plan = GnlRingPlan((GnlThermalTarget(thermal_id=94, submercado=1, nl_lag=2),))
    mapping = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        gnl_plan=gnl_plan,
        coupling_block_hours=_UNIFORM_GNL_BLOCK_HOURS,
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping, gnl_plan)

    deviation = next(
        d for d in sink if d.code == "boundary-fcf-gnl-anticipated-deviation"
    )
    assert deviation.remediation is None


# ---------------------------------------------------------------------------
# The dropped-coverage filter reconciled with the excised ring (the in-study
# committed-window reason string, never the retired post-study-horizon/K=0
# framing). Tier-1: pure Python, no deck, no novomodelo binary.
# ---------------------------------------------------------------------------


def test_emit_import_diagnostics_gnl_dropped_count_sums_source_and_instudy() -> None:
    """AC 1/AC 2 -- the dropped-coverage count sums a source-submercado drop
    (submercado 2 has no live GNL thermal at all) and an in-study
    non-covered drop (thermal 94's only dated slot falls before the
    committed-window horizon) to 2, read straight from
    ``mapping.gnl_dropped``; the summary names both clauses and carries
    neither the retired ``post-study horizon`` substring nor ``K=0``.
    """
    id_map = make_id_map(())
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # unrelated dummy, satisfies the guard
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0, interval_start=20260401),
        ]
    )
    header = make_cortes_header(
        (), lag_maximo_gnl=2, n_patamares=3, submercado_codes=(1, 2)
    )
    # col(1,p,2) -> flat indices 1,3,5 (thermal 94's ring, submercado 1);
    # col(2,p,1) -> flat indices 6,8,10 (submercado 2, no live GNL thermal
    # claims it).
    pi_gnl = _gnl_row(12, {1: 0.1, 3: 0.2, 5: 0.3, 6: 0.5})
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=5.0)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    # thermal 94's only dated slot (20260401) is before this
    # post_horizon_start (20260501) -- an in-study, non-covered drop.
    gnl_plan = GnlRingPlan(
        (GnlThermalTarget(thermal_id=94, submercado=1, nl_lag=2),),
        post_horizon_start=20260501,
    )

    mapping = map_boundary_cuts(
        cuts, manifest, id_map, cost_unit_hours=MONTH_HOURS, gnl_plan=gnl_plan
    )
    assert len(mapping.gnl_dropped) == 2

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping, gnl_plan)

    deviation = next(
        d for d in sink if d.code == "boundary-fcf-gnl-anticipated-deviation"
    )
    assert "2 GNL term(s) dropped" in deviation.summary
    assert "no live thermal in that submercado" in deviation.summary
    assert "in-study delivery priced by the committed window" in deviation.summary
    assert "post-study horizon" not in deviation.summary
    assert "K=0" not in deviation.summary


def test_emit_import_diagnostics_gnl_deviation_class4_absent_no_drop() -> None:
    """AC 3 -- a já-comandada (class-4) slot never reaches the terminal
    manifest at all (novomodelo excises it from the ring entirely), so a target
    carrying only a covered (class-3 signaled) dated slot contributes zero
    drops; the per-``(submercado, lag)`` spread table still renders its row.
    """
    id_map = make_id_map(())
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # unrelated dummy, satisfies the guard
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0, interval_start=20260501),
        ]
    )
    header = make_cortes_header(
        (), lag_maximo_gnl=2, n_patamares=3, submercado_codes=(1,)
    )
    pi_gnl = _gnl_row(6, {1: 0.1, 3: 0.2, 5: 0.3})
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=5.0)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    gnl_plan = GnlRingPlan(
        (GnlThermalTarget(thermal_id=94, submercado=1, nl_lag=2),),
        post_horizon_start=20260501,
    )

    mapping = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        gnl_plan=gnl_plan,
        coupling_block_hours=_UNIFORM_GNL_BLOCK_HOURS,
    )
    assert mapping.gnl_dropped == ()

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping, gnl_plan)

    deviation = next(
        d for d in sink if d.code == "boundary-fcf-gnl-anticipated-deviation"
    )
    assert "0 GNL term(s) dropped" in deviation.summary
    assert deviation.table is not None
    assert len(deviation.table.rows) == 1

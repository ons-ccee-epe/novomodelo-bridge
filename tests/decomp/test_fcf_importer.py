"""Tests for the boundary FCF importer orchestration
(``fcf/importer.py::import_boundary_fcf``).

Synthetic fixtures only: no deck, no novomodelo binary.
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.anticipated import (
    GnlCommitment,
    GnlCommitmentModel,
    GnlStageCommitment,
    GnlThermal,
)
from novomodelo_bridge.decomp.fcf.cortes import BoundaryCuts, summarize_cut_families
from novomodelo_bridge.decomp.fcf.coupling import (
    BorrowedOwner,
    CouplingResolution,
    TerminalShare,
)
from novomodelo_bridge.decomp.fcf.importer import (
    _DeckCoupling,
    _emit_coupling_diagnostics,
    _emit_import_diagnostics,
    _gnl_targets_from,
    _plan_gnl_ring,
    _read_complexo_map,
    import_boundary_fcf,
)
from novomodelo_bridge.decomp.fcf.mapper import (
    DroppedTerm,
    GnlCommandedDelivery,
    GnlDeliveryMonth,
    GnlDroppedTerm,
    GnlRingPlan,
    GnlThermalTarget,
    HydroCoupling,
    MappingResult,
    month_key,
)
from novomodelo_bridge.decomp.temporal import OperativeStage
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

_MAY = month_key(2026, 5)
_JUNE = month_key(2026, 6)


def _no_coupling() -> _DeckCoupling:
    return _DeckCoupling(
        cut=(2026, 4),
        resolution=CouplingResolution((), (), ()),
        inflow_lag_offsets={},
        source_only_inflows={},
        terminal_shares=(),
        carried_share=0.0,
        gnl_plan=None,
        recent_observations=[],
    )


def _storage_only_mapping() -> MappingResult:
    return MappingResult(
        cuts=(make_mapped_cut(coefficients=(1.5,), intercept=10.0),), dropped=()
    )


def test_emit_import_diagnostics_reports_family_summary_and_dropped_plants() -> None:
    cuts = make_boundary_cuts((1,), (make_cut_record(pi_varm=(1.5,), rhs=10.0),))
    mapping = MappingResult(
        cuts=(make_mapped_cut(coefficients=(1.5,), intercept=10.0),),
        dropped=(
            DroppedTerm(plant_code=20, beta=0.4, inflow_folded=True),
            DroppedTerm(plant_code=30, beta=0.7),
        ),
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping)

    by_code = {diagnostic.code: diagnostic for diagnostic in sink}
    assert set(by_code) == {
        "boundary-fcf-cut-family-summary",
        "boundary-fcf-source-only-plants-dropped",
    }
    assert all(diagnostic.severity is dx.Severity.INFO for diagnostic in sink)
    dropped = by_code["boundary-fcf-source-only-plants-dropped"]
    assert dropped.table is not None
    assert dropped.table.rows == [[20, 0.4, "yes"], [30, 0.7, "no"]]

    family = by_code["boundary-fcf-cut-family-summary"]
    summary = summarize_cut_families(cuts)
    assert family.notes == []
    assert str(summary.n_active_cuts) in family.summary
    assert str(summary.lag_nonzero_by_depth) in family.summary


def test_emit_import_diagnostics_without_drops_or_plan_reports_only_the_summary() -> (
    None
):
    cuts = make_boundary_cuts((1,), (make_cut_record(pi_varm=(1.5,), rhs=10.0),))

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, _storage_only_mapping())

    assert [diagnostic.code for diagnostic in sink] == [
        "boundary-fcf-cut-family-summary"
    ]


def test_import_boundary_fcf_no_cut_files_is_noop(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
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
# Payload threading: the bootstrap manifest's node/graph-stage/priced-date
# identity reaches `build_stage_cuts_payload`, while `config.json`'s
# `policy.boundary` stays the date-driven `{"path": "boundary"}` block.
# ---------------------------------------------------------------------------


def _stub_import_seams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_cuts: BoundaryCuts,
    *,
    graph_stage_id: int,
) -> tuple[DecompCase, list[dict[str, object]]]:
    """Stub every deck, cut-reader and novomodelo seam; return the case and a spy
    on the payload builder's keyword arguments."""
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
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer._read_deck_coupling",
        lambda *_args: _no_coupling(),
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.bootstrap_terminal_manifest",
        lambda *_args, **_kwargs: make_manifest(
            [make_slot(0, 0, 0)],
            node_id=0,
            graph_stage_id=graph_stage_id,
            priced_state_date=20_261_101,
        ),
    )
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.fcf.importer.map_boundary_cuts",
        lambda *_args, **_kwargs: _storage_only_mapping(),
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

    files = make_decomp_files(
        tmp_path,
        cortesh=tmp_path / "deck" / "cortesh.dat",
        cortes=tmp_path / "deck" / "cortes-010.dat",
    )
    case = make_decomp_case(
        files, dadger=object(), id_map=make_id_map(()), hidr=object(), calendar=[]
    )
    return case, calls


def _boundary_stage_4_cuts() -> BoundaryCuts:
    return make_boundary_cuts(
        plant_codes=(1,),
        records=(make_cut_record(pi_varm=(1.5,), rhs=10.0, forward_pass_index=0),),
        boundary_stage=4,
    )


def test_import_boundary_fcf_threads_node_and_graph_stage_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    case, calls = _stub_import_seams(
        monkeypatch, tmp_path, _boundary_stage_4_cuts(), graph_stage_id=4
    )

    import_boundary_fcf(
        case_dir,
        case,
        work_dir=tmp_path / "work",
        cost_scale_factor=1.0,
        config={},
        initial_conditions={},
    )

    assert calls == [
        {
            "stage_id": 4,
            "cost_scale_factor": 1.0,
            "node_id": 0,
            "graph_stage_id": 4,
            "priced_state_date": 20_261_101,
        }
    ]


def test_import_boundary_fcf_graph_stage_id_distinct_from_boundary_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest's 0-based pool identity and the cut file's calendar stage
    are different axes; the config block carries neither."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    case, calls = _stub_import_seams(
        monkeypatch, tmp_path, _boundary_stage_4_cuts(), graph_stage_id=3
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

    assert calls[0]["stage_id"] == 4
    assert calls[0]["graph_stage_id"] == 3
    assert config["policy"]["boundary"] == {"path": "boundary"}


# ---------------------------------------------------------------------------
# The deck's ``CX`` register.
# ---------------------------------------------------------------------------


def test_read_complexo_map_groups_components_by_complex(tmp_path: Path) -> None:
    dadger = tmp_path / "dadger.rv2"
    dadger.write_text(
        "& CX  complexo componente\n"
        "CX  132  147\n"
        "CX  132  146\n"
        "CX  176  175\n"
        "UH  147  10\n",
        encoding="latin-1",
    )

    assert _read_complexo_map(dadger) == {132: [147, 146], 176: [175]}


def test_read_complexo_map_is_empty_without_a_dadger(tmp_path: Path) -> None:
    assert _read_complexo_map(tmp_path / "absent.rv2") == {}


# ---------------------------------------------------------------------------
# GNL ring plan.
# ---------------------------------------------------------------------------


def _make_gnl_thermal(code: int, submarket_code: int) -> GnlThermal:
    return GnlThermal(code=code, name=f"GNL-{code}", submarket_code=submarket_code)


def _make_gnl_model(
    thermals: tuple[GnlThermal, ...],
    *,
    commitments: dict[int, GnlCommitment] | None = None,
    weeks_per_month: dict[int, int] | None = None,
) -> GnlCommitmentModel:
    return GnlCommitmentModel(
        thermals=thermals,
        commitments=commitments or {},
        weeks_per_month=weeks_per_month or {},
        nl_lag_months={},
        tg=pd.DataFrame(),
    )


def test_gnl_targets_from_maps_codes_to_ids() -> None:
    """Ids ascend with code: code 86 -> 94 (SE), code 224 -> 95 (NE)."""
    model = _make_gnl_model((_make_gnl_thermal(86, 1), _make_gnl_thermal(224, 3)))
    thermals_doc = {
        "thermals": [
            {"id": 1, "name": "not-gnl"},
            {"id": 94, "name": "GNL-86", "anticipated_config": {}},
            {"id": 95, "name": "GNL-224", "anticipated_config": {}},
        ]
    }

    assert _gnl_targets_from(model, thermals_doc) == (
        GnlThermalTarget(thermal_id=94, submercado=1),
        GnlThermalTarget(thermal_id=95, submercado=3),
    )


def test_gnl_targets_from_count_mismatch_raises() -> None:
    model = _make_gnl_model((_make_gnl_thermal(86, 1), _make_gnl_thermal(224, 3)))
    thermals_doc = {
        "thermals": [
            {"id": 94, "anticipated_config": {}},
            {"id": 95, "anticipated_config": {}},
            {"id": 96, "anticipated_config": {}},
        ]
    }

    with pytest.raises(ValueError, match="GNL fleet must match exactly"):
        _gnl_targets_from(model, thermals_doc)


def _stage(index: int, start: date, end: date) -> OperativeStage:
    return OperativeStage(
        index=index,
        start_date=start,
        end_date=end,
        season_id=start.month - 1,
        block_hours=((end - start).days * 24.0,),
    )


#: The ``mar-26-rv2`` calendar shape: three weeks, then the rest of April.
_CALENDAR = (
    _stage(0, date(2026, 3, 14), date(2026, 3, 21)),
    _stage(1, date(2026, 3, 21), date(2026, 3, 28)),
    _stage(2, date(2026, 3, 28), date(2026, 4, 4)),
    _stage(3, date(2026, 4, 4), date(2026, 5, 1)),
)


def _gl_week(
    estagio: int, start: date, block_mw: tuple[float, ...]
) -> GnlStageCommitment:
    return GnlStageCommitment(
        estagio=estagio,
        start_date=start,
        committed_mw=0.0,
        hours=0.0,
        block_mw=block_mw,
    )


def test_plan_gnl_ring_prices_the_months_after_the_cut() -> None:
    model = _make_gnl_model(
        (_make_gnl_thermal(86, 1),), weeks_per_month={1: 3, 2: 4, 3: 5}
    )
    fractions = {(2026, 5): (0.2, 0.3, 0.5), (2026, 6): (0.25, 0.25, 0.5)}

    plan = _plan_gnl_ring(
        model, (GnlThermalTarget(94, 1),), fractions, (2026, 4), 2, _CALENDAR
    )

    assert plan.months == {
        _MAY: GnlDeliveryMonth(1, (0.2, 0.3, 0.5)),
        _JUNE: GnlDeliveryMonth(2, (0.25, 0.25, 0.5)),
    }
    # `gs` books five weeks to May, the last one starting 2026-05-30.
    assert plan.week_months[20260530] == _MAY
    assert plan.week_months[20260502] == _MAY
    assert 20260606 not in plan.week_months


def test_plan_gnl_ring_leaves_out_a_month_without_patamar_shares() -> None:
    model = _make_gnl_model((_make_gnl_thermal(86, 1),))

    plan = _plan_gnl_ring(
        model,
        (GnlThermalTarget(94, 1),),
        {(2026, 5): (1.0, 0.0, 0.0)},
        (2026, 4),
        2,
        _CALENDAR,
    )

    assert set(plan.months) == {_MAY}


def test_plan_gnl_ring_folds_only_nonzero_commanded_weeks() -> None:
    """The horizon stub and a 0 MW week carry nothing; a dispatched week does."""
    commitment = GnlCommitment(
        code=86,
        stages=(
            _gl_week(1, date(2026, 3, 14), (100.0, 100.0, 100.0)),
            _gl_week(8, date(2026, 5, 2), (0.0, 0.0, 0.0)),
            _gl_week(9, date(2026, 5, 9), (500.0, 500.0, 0.0)),
        ),
    )
    model = _make_gnl_model(
        (_make_gnl_thermal(86, 1),),
        commitments={86: commitment},
        weeks_per_month={1: 3, 2: 4, 3: 5},
    )

    plan = _plan_gnl_ring(
        model,
        (GnlThermalTarget(94, 1),),
        {(2026, 5): (0.2, 0.3, 0.5)},
        (2026, 4),
        2,
        _CALENDAR,
    )

    assert plan.commanded == (
        GnlCommandedDelivery(1, _MAY, (500.0, 500.0, 0.0), 168.0),
    )


# ---------------------------------------------------------------------------
# GNL deviation diagnostic.
# ---------------------------------------------------------------------------


def _gnl_row(width: int, nonzero: dict[int, float]) -> tuple[float, ...]:
    row = [0.0] * width
    for column, value in nonzero.items():
        row[column] = value
    return tuple(row)


def _june_lag_2_plan(*targets: GnlThermalTarget) -> GnlRingPlan:
    return GnlRingPlan(
        targets=targets or (GnlThermalTarget(94, 1),),
        months={_JUNE: GnlDeliveryMonth(2, (0.25, 0.25, 0.5))},
    )


def _one_submercado_cuts(pi_gnl: tuple[float, ...]) -> BoundaryCuts:
    header = make_cortes_header(
        (), lag_maximo_gnl=2, n_patamares=3, submercado_codes=(1,)
    )
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=5.0)
    return BoundaryCuts(header=header, boundary_stage=10, records=(record,))


def test_emit_import_diagnostics_gnl_deviation_reports_the_patamar_spread() -> None:
    # col(1, p, 2) = (p - 1) * 2 + 1 -> 1, 3, 5 with P = 3, L = 2.
    cuts = _one_submercado_cuts(_gnl_row(6, {1: 0.1, 3: 0.2, 5: 0.3}))

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, _storage_only_mapping(), _june_lag_2_plan())

    deviation = next(
        d for d in sink if d.code == "boundary-fcf-gnl-anticipated-deviation"
    )
    assert deviation.severity is dx.Severity.INFO
    assert deviation.remediation is None
    assert deviation.table is not None
    assert deviation.table.columns == [
        "Submercado",
        "Lag",
        "Σ pi_gnl (carried)",
        "Patamar spread",
    ]
    (row,) = deviation.table.rows
    assert row[:2] == [1, 2]
    assert row[2] == pytest.approx(0.6)
    assert row[3] == pytest.approx((0.3 - 0.1) / 0.6)
    assert "0 GNL term(s) reach no ring slot" in deviation.summary


def test_emit_import_diagnostics_gnl_deviation_gated_off_without_plan() -> None:
    cuts = _one_submercado_cuts(_gnl_row(6, {1: 0.1}))

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, _storage_only_mapping())

    assert [d.code for d in sink] == ["boundary-fcf-cut-family-summary"]


def test_emit_import_diagnostics_gnl_deviation_lists_every_dropped_term() -> None:
    cuts = _one_submercado_cuts(_gnl_row(6, {1: 0.1, 3: 0.2, 5: 0.3}))
    mapping = MappingResult(
        cuts=(make_mapped_cut(coefficients=(0.0,), intercept=0.0),),
        dropped=(),
        gnl_dropped=(
            GnlDroppedTerm(None, 2, 1, 0.75, "no GNL thermal in submercado"),
            GnlDroppedTerm(94, 1, None, 0.0, "no dated ring slot for thermal"),
        ),
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping, _june_lag_2_plan())

    deviation = next(
        d for d in sink if d.code == "boundary-fcf-gnl-anticipated-deviation"
    )
    assert "2 GNL term(s) reach no ring slot" in deviation.summary
    assert deviation.notes == [
        "submercado 2, lag 1: no GNL thermal in submercado (Σ pi_gnl 0.75)",
        "thermal 94, submercado 1: no dated ring slot for thermal",
    ]


def test_emit_import_diagnostics_headline_excludes_near_zero_sum_group() -> None:
    """A group carrying a negligible share of the largest |Σ| stays in the
    table but out of the relative headline (deck magnitudes: Σ = -4412 with
    spread 392, against a Σ = -4e-05 group whose own ratio is 0.25)."""
    header = make_cortes_header(
        (), lag_maximo_gnl=1, n_patamares=3, submercado_codes=(1, 2)
    )
    # lag_maximo_gnl = 1: col(s, p, 1) = (s - 1) * 3 + (p - 1).
    pi_gnl = (-1340.0, -1340.0, -1732.0, -1e-05, -1e-05, -2e-05)
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=0.0)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    plan = GnlRingPlan(
        targets=(GnlThermalTarget(94, 1), GnlThermalTarget(95, 2)),
        months={_MAY: GnlDeliveryMonth(1, (0.2, 0.3, 0.5))},
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, _storage_only_mapping(), plan)

    deviation = next(
        d for d in sink if d.code == "boundary-fcf-gnl-anticipated-deviation"
    )
    assert deviation.table is not None
    rows_by_group = {(row[0], row[1]): row for row in deviation.table.rows}
    assert rows_by_group[(1, 1)][3] == pytest.approx(392.0 / 4412.0, abs=1e-6)
    assert rows_by_group[(2, 1)][3] == pytest.approx(0.25, abs=1e-6)
    assert (
        f"spread {392.0 / 4412.0:.4g} relative / {392.0:.4g} absolute"
        in deviation.summary
    )


# ---------------------------------------------------------------------------
# Coupling diagnostics.
# ---------------------------------------------------------------------------


def test_emit_coupling_diagnostics_is_silent_when_every_plant_owns_its_row() -> None:
    resolution = CouplingResolution((HydroCoupling(0, 0, 0.0),), (), ())

    with dx.collect() as sink:
        _emit_coupling_diagnostics(resolution, (), 0.0)

    assert sink == []


def test_emit_coupling_diagnostics_reports_borrowed_owners() -> None:
    resolution = CouplingResolution(
        (HydroCoupling(0, 0, 0.0), HydroCoupling(1, 0)),
        (BorrowedOwner(147, 132, via_complex=True), BorrowedOwner(109, 118, False)),
        (),
    )

    with dx.collect() as sink:
        _emit_coupling_diagnostics(resolution, (), 0.0)

    (owners,) = sink
    assert owners.code == "boundary-fcf-coefficient-owners"
    assert owners.severity is dx.Severity.INFO
    assert owners.table is not None
    assert owners.table.rows == [[147, 132, "complex (CX)"], [109, 118, "downstream"]]


def test_emit_coupling_diagnostics_unowned_plant_is_an_error() -> None:
    resolution = CouplingResolution((), (), (305,))

    with dx.collect() as sink:
        _emit_coupling_diagnostics(resolution, (), 0.0)

    (unowned,) = sink
    assert unowned.code == "boundary-fcf-unowned-plants"
    assert unowned.severity is dx.Severity.ERROR
    assert unowned.table is not None
    assert unowned.table.rows == [[305]]


def test_emit_coupling_diagnostics_warns_on_a_carried_lag_share() -> None:
    with dx.collect() as sink:
        _emit_coupling_diagnostics(CouplingResolution((), (), ()), (), 0.1)

    (carried,) = sink
    assert carried.code == "boundary-fcf-coupling-lag-carried-share"
    assert carried.severity is dx.Severity.WARNING
    assert "10.0%" in carried.summary


def test_emit_coupling_diagnostics_lists_unrepresented_coupling_inflows() -> None:
    shares = (TerminalShare(1, 1, 301, 2.5), TerminalShare(260, 260, None, 40.0))

    with dx.collect() as sink:
        _emit_coupling_diagnostics(CouplingResolution((), (), ()), shares, 0.0)

    (terminal,) = sink
    assert terminal.code == "boundary-fcf-coupling-inflow-unrepresented"
    assert terminal.severity is dx.Severity.WARNING
    assert terminal.table is not None
    assert terminal.table.rows == [[1, 1, 301, 2.5], [260, 260, "-", 40.0]]

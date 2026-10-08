"""Unit tests for the source model's GNL reader (``decomp/anticipated.py``).

CI-tier and deck-independent: every test drives a synthetic ``_StubDadgnl`` with
hand-authored ``gl``/``tg``/``gs`` data (mirroring the real register shapes), so
the suite needs no deck and no novomodelo binary. ``gl`` is a list of register stubs
(the real ``gl(df=True)`` is unusable on ragged weekly blocks); ``tg``/``gs`` are
DataFrames (their fixed shapes make ``df=True`` well-formed).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from datetime import date, timedelta

import pandas as pd
import pytest

from novomodelo_bridge.decomp.converters import anticipated
from novomodelo_bridge.decomp.converters.anticipated import (
    GnlClassification,
    GnlCommitment,
    GnlCommitmentModel,
    GnlEmission,
    GnlPlantClassification,
    GnlStageCommitment,
    GnlThermal,
    _build_post_study_calendar,
    _month_end_duration_hours,
    _study_lead_hours,
    classify_gnl_windows,
    convert_gnl,
    is_gnl_enabled,
    read_gnl_model,
)
from novomodelo_bridge.decomp.temporal import OperativeStage


class _GlReg:
    """A stand-in for one ``gl`` register (attribute access, like idecomp's)."""

    def __init__(
        self,
        codigo_usina: int,
        estagio: int,
        data_inicio: str | int,
        duracao: list[float],
        geracao: list[float],
        codigo_submercado: int = 1,
    ) -> None:
        self.codigo_usina = codigo_usina
        self.estagio = estagio
        self.data_inicio = data_inicio
        self.duracao = duracao
        self.geracao = geracao
        self.codigo_submercado = codigo_submercado


class _StubDadgnl:
    """Minimal ``Dadgnl`` stub: ``gl()`` → register list, ``tg``/``gs`` → frames."""

    def __init__(
        self,
        gl_registers: list[_GlReg],
        tg_rows: list[dict],
        gs_rows: list[dict],
        nl_rows: list[dict] | None = None,
    ) -> None:
        self._gl = gl_registers
        self._tg = pd.DataFrame(tg_rows)
        self._gs = pd.DataFrame(gs_rows)
        self._nl = pd.DataFrame(nl_rows or [])

    def gl(self, df: bool = False) -> list[_GlReg]:
        return list(self._gl)

    def tg(self, df: bool = False) -> pd.DataFrame:
        return self._tg

    def gs(self, df: bool = False) -> pd.DataFrame:
        return self._gs

    def nl(self, df: bool = False) -> pd.DataFrame:
        return self._nl


def _tg_row(
    code: int,
    name: str,
    *,
    sub: int = 1,
    cvu: float = 199.22,
    disp: float = 500.0,
    inflex: float = 0.0,
) -> dict:
    """A ``tg`` registry row with equal block values (the common cadastro shape)."""
    return {
        "codigo_submercado": sub,
        "codigo_usina": code,
        "estagio": 1,
        "nome": name,
        "cvu_1": cvu,
        "cvu_2": cvu,
        "cvu_3": cvu,
        "disponibilidade_1": disp,
        "disponibilidade_2": disp,
        "disponibilidade_3": disp,
        "inflexibilidade_1": inflex,
        "inflexibilidade_2": inflex,
        "inflexibilidade_3": inflex,
    }


def test_committed_mw_is_block_duration_weighted() -> None:
    gl = [_GlReg(86, 1, "14032026", [40.0, 48.0, 80.0], [107.0, 131.0, 97.0])]
    model = read_gnl_model(_StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], []))

    assert model is not None
    got = model.commitments[86].stages[0].committed_mw
    expected = (40 * 107 + 48 * 131 + 80 * 97) / 168
    assert abs(got - expected) < 1e-9
    assert model.commitments[86].stages[0].hours == 168.0  # sum of duracao


def test_gate_off_when_gl_absent() -> None:
    stub = _StubDadgnl([], [_tg_row(86, "SANTA CRUZ")], [])

    assert is_gnl_enabled(stub) is False
    assert read_gnl_model(stub) is None


def test_gate_off_when_all_geracao_zero() -> None:
    gl = [_GlReg(86, 1, "14032026", [40.0, 48.0, 80.0], [0.0, 0.0, 0.0])]
    stub = _StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], [])

    assert is_gnl_enabled(stub) is False
    assert read_gnl_model(stub) is None


def test_gate_off_for_none() -> None:
    assert is_gnl_enabled(None) is False


def test_thermals_ascending_by_code() -> None:
    # One nonzero gl register turns the gate on; the registry lists all tg plants.
    gl = [_GlReg(86, 9, "09052026", [500.0], [500.0])]
    tg = [
        _tg_row(224, "PSERGIPE I", sub=3),
        _tg_row(15, "X"),
        _tg_row(86, "SANTA CRUZ"),
    ]
    model = read_gnl_model(_StubDadgnl(gl, tg, []))

    assert model is not None
    assert [t.code for t in model.thermals] == [15, 86, 224]


def test_gl_code_absent_from_tg_raises_naming_code() -> None:
    gl = [_GlReg(999, 1, "14032026", [168.0], [100.0])]
    with pytest.raises(ValueError, match="999"):
        read_gnl_model(_StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], []))


def test_registry_only_plant_has_empty_commitment() -> None:
    # code 86 carries the (enabling) gl dispatch; 224 is registry-only.
    gl = [_GlReg(86, 9, "09052026", [500.0], [500.0])]
    tg = [_tg_row(86, "SANTA CRUZ"), _tg_row(224, "PSERGIPE I", sub=3)]
    model = read_gnl_model(_StubDadgnl(gl, tg, []))

    assert model is not None
    assert model.commitments[224].stages == ()
    assert model.commitments[86].stages != ()


def test_post_horizon_delivery_date_and_mw_captured() -> None:
    gl = [_GlReg(86, 9, "09052026", [500.0, 500.0, 500.0], [500.0, 500.0, 500.0])]
    model = read_gnl_model(_StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], []))

    assert model is not None
    stage = model.commitments[86].stages[0]
    assert stage.estagio == 9
    assert stage.start_date == date(2026, 5, 9)
    assert stage.committed_mw == 500.0


def test_data_inicio_integer_with_dropped_leading_zero() -> None:
    # A ddmmyyyy stored as an int loses its leading-zero day (04.. -> 4042026).
    gl = [_GlReg(86, 4, 4042026, [30.0, 69.0, 69.0], [10.0, 10.0, 10.0])]
    model = read_gnl_model(_StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], []))

    assert model is not None
    assert model.commitments[86].stages[0].start_date == date(2026, 4, 4)


def test_registry_keeps_every_declared_tg_stage() -> None:
    gl = [_GlReg(86, 1, "14032026", [30.0, 74.0, 64.0], [100.0, 100.0, 100.0])]
    tg = [
        _tg_row(86, "SANTA CRUZ", inflex=50.0),
        {**_tg_row(86, "SANTA CRUZ", inflex=0.0), "estagio": 2},
    ]
    model = read_gnl_model(_StubDadgnl(gl, tg, []))

    assert model is not None
    assert model.thermals == (GnlThermal(86, "SANTA CRUZ", 1),)
    assert model.tg["estagio"].tolist() == [1, 2]


def test_weeks_per_month_from_gs() -> None:
    gl = [_GlReg(86, 1, "14032026", [168.0], [100.0])]
    gs = [{"mes": 1, "semanas": 3}, {"mes": 2, "semanas": 4}, {"mes": 3, "semanas": 5}]
    model = read_gnl_model(_StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], gs))

    assert model is not None
    assert model.weeks_per_month == {1: 3, 2: 4, 3: 5}
    assert isinstance(model, GnlCommitmentModel)


def test_nl_lags_read_per_plant() -> None:
    gl = [_GlReg(86, 9, "09052026", [500.0], [500.0])]
    tg = [_tg_row(86, "SANTA CRUZ"), _tg_row(224, "PSERGIPE I", sub=3)]
    nl = [
        {"codigo_usina": 86, "codigo_submercado": 1, "lag": 2},
        {"codigo_usina": 224, "codigo_submercado": 3, "lag": 2},
    ]
    model = read_gnl_model(_StubDadgnl(gl, tg, [], nl))

    assert model is not None
    assert model.nl_lag_months == {86: 2, 224: 2}


def test_nl_lags_empty_when_block_absent() -> None:
    gl = [_GlReg(86, 9, "09052026", [500.0], [500.0])]
    model = read_gnl_model(_StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], []))

    assert model is not None
    assert model.nl_lag_months == {}


# --------------------------------------------------------------------------
# classify_gnl_windows: já-comandada (class-4) vs signaled (class-3)
# --------------------------------------------------------------------------


def _stage_model(
    code: int, stages: tuple[GnlStageCommitment, ...]
) -> GnlCommitmentModel:
    return GnlCommitmentModel(
        tg=_TG,
        thermals=(),
        commitments={code: GnlCommitment(code, stages)},
        weeks_per_month={},
        nl_lag_months={},
    )


def test_classify_gnl_windows_happy_path() -> None:
    # Reference-shaped: SANTA CRUZ's two post-horizon já-comandada weeks
    # (168 h each), horizon ending mid-week (2026-05-01, a Friday).
    stages = (
        GnlStageCommitment(8, date(2026, 5, 2), 0.0, 168.0),
        GnlStageCommitment(9, date(2026, 5, 9), 500.0, 168.0),
    )
    result = classify_gnl_windows(
        _stage_model(86, stages), horizon_end=date(2026, 5, 1)
    )

    santa = result.plants[86]
    assert santa.class4_windows == (
        (date(2026, 5, 1), date(2026, 5, 2), 0.0),
        (date(2026, 5, 2), date(2026, 5, 9), 0.0),
        (date(2026, 5, 9), date(2026, 5, 16), 500.0),
    )
    assert santa.class4_end == date(2026, 5, 16)
    assert isinstance(result, GnlClassification)


def test_classify_gnl_windows_no_post_horizon_weeks_is_free() -> None:
    stages = (GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),)
    result = classify_gnl_windows(
        _stage_model(86, stages), horizon_end=date(2026, 5, 1)
    )

    santa = result.plants[86]
    assert santa.class4_windows == ()
    assert santa.class4_end == date(2026, 5, 1)


def test_classify_gnl_windows_ragged_run_raises_ambiguous() -> None:
    # 05-02 then 05-16 skips 05-09: not one operative week apart.
    stages = (
        GnlStageCommitment(8, date(2026, 5, 2), 0.0, 0.0),
        GnlStageCommitment(10, date(2026, 5, 16), 500.0, 0.0),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        classify_gnl_windows(_stage_model(86, stages), horizon_end=date(2026, 5, 1))


# --------------------------------------------------------------------------
# Study lead H (the class-2 tile-sizing lead, unrelated to the per-plant
# emitted anticipated_config lead) + the grid-anchored post-study calendar
# (independent of H)
# --------------------------------------------------------------------------

# decomp-mar-26-rv2's operative study calendar: 3 weekly March stages (168 h
# each) + 1 monthly April stage (648 h); horizon ends 2026-05-01 (Friday),
# cumulative 1152 h. All weekly stage boundaries are Saturdays.
_STUDY_SPANS = [
    (date(2026, 3, 14), date(2026, 3, 21)),
    (date(2026, 3, 21), date(2026, 3, 28)),
    (date(2026, 3, 28), date(2026, 4, 4)),
    (date(2026, 4, 4), date(2026, 5, 1)),
]


def test_study_lead_hours_sums_study_months() -> None:
    # The study spans March + April (2 distinct months); GS[1]+GS[2] = 3+4 = 7.
    assert _study_lead_hours(_STUDY_SPANS, {1: 3, 2: 4, 3: 5}) == 1176.0
    # Empty GS -> no lead (caller emits no post-study calendar).
    assert _study_lead_hours(_STUDY_SPANS, {}) == 0.0


def test_month_end_duration_hours_uses_month_last_day_not_next_month_start() -> None:
    # 2026-06 has 30 days: 30 - 6 = 24 d = 576 h, not the 25 d/600 h a
    # next-month-start boundary would give.
    assert _month_end_duration_hours(date(2026, 6, 6)) == 576.0
    # 2026 is not a leap year: February has 28 days.
    assert _month_end_duration_hours(date(2026, 2, 1)) == 648.0


def test_build_post_study_calendar_matches_novomodelo_e2e_oracle() -> None:
    # class4_end = 2026-05-16 (both plants' shared já-comandada cutoff): a
    # 24 h stub + 2 operative weeks fill class-4 up to it, then one
    # class-3 stage per study stage (3 weekly mirrors + the trailing
    # monthly mirror, spanning to the end of ITS OWN calendar month, June)
    # -- the exact 7-stage shape novomodelo's own post-study e2e fixture pins.
    stages = _build_post_study_calendar(_STUDY_SPANS, date(2026, 5, 16))

    assert [(s["start_date"], s["duration_hours"]) for s in stages] == [
        ("2026-05-01", 24.0),
        ("2026-05-02", 168.0),
        ("2026-05-09", 168.0),
        ("2026-05-16", 168.0),
        ("2026-05-23", 168.0),
        ("2026-05-30", 168.0),
        ("2026-06-06", 576.0),
    ]
    # One class-3 mirror stage per study stage, plus the class-4 fill.
    assert len(stages) == 3 + len(_STUDY_SPANS)


def test_build_post_study_calendar_is_contiguous() -> None:
    stages = _build_post_study_calendar(_STUDY_SPANS, date(2026, 5, 16))

    assert stages[0]["start_date"] == "2026-05-01"  # the study horizon end
    for cur, nxt in zip(stages, stages[1:]):
        start = date.fromisoformat(cur["start_date"])
        end = start + timedelta(hours=cur["duration_hours"])
        assert end == date.fromisoformat(nxt["start_date"])
    # Every stage after the grid-alignment stub starts on a Saturday --
    # both the class-4 fill and the class-3 mirror tile in exact 168 h
    # steps from a Saturday-aligned point.
    for stage in stages[1:]:
        assert date.fromisoformat(stage["start_date"]).weekday() == 5


def test_build_post_study_calendar_no_stub_when_horizon_end_already_saturday() -> None:
    # An all-weekly study whose last stage ends on a Saturday is already on
    # the grid -- the class-4 fill's one já-comandada week needs no stub,
    # and the class-3 mirror that follows starts exactly on the grid too.
    spans = [
        (date(2026, 3, 14), date(2026, 3, 21)),
        (date(2026, 3, 21), date(2026, 3, 28)),
    ]
    assert spans[-1][1].weekday() == 5  # 2026-03-28 is a Saturday
    stages = _build_post_study_calendar(spans, date(2026, 4, 4))

    assert [(s["start_date"], s["duration_hours"]) for s in stages] == [
        ("2026-03-28", 168.0),  # class-4 fill: one já-comandada week
        ("2026-04-04", 168.0),  # class-3 mirror starts at class4_end
        ("2026-04-11", 168.0),
    ]


def test_build_post_study_calendar_no_class4_fill_when_class4_end_is_horizon_end() -> (
    None
):
    # No já-comandada run at all (class4_end == horizon_end): the class-4
    # fill is empty and the class-3 mirror starts immediately at
    # horizon_end, even though 2026-05-01 (Friday) is off the Saturday grid
    # -- there is nothing left to align to.
    stages = _build_post_study_calendar(_STUDY_SPANS, date(2026, 5, 1))

    assert [(s["start_date"], s["duration_hours"]) for s in stages] == [
        ("2026-05-01", 168.0),
        ("2026-05-08", 168.0),
        ("2026-05-15", 168.0),
        ("2026-05-22", 216.0),  # May has 31 days: 31 - 22 = 9 d = 216 h
    ]
    assert len(stages) == len(_STUDY_SPANS)


def test_build_post_study_calendar_empty_when_no_stage_spans() -> None:
    assert _build_post_study_calendar([], date(2026, 5, 16)) == []


# --------------------------------------------------------------------------
# Emission (convert_gnl) — single global lead, free-only, index-direct
# --------------------------------------------------------------------------

_EMIT_STAGES = [
    {
        "start_date": "2026-03-14",
        "end_date": "2026-03-21",
        "blocks": [{"hours": 168.0}],
    },
    {
        "start_date": "2026-03-21",
        "end_date": "2026-03-28",
        "blocks": [{"hours": 168.0}],
    },
    {
        "start_date": "2026-03-28",
        "end_date": "2026-04-04",
        "blocks": [{"hours": 168.0}],
    },
    {
        "start_date": "2026-04-04",
        "end_date": "2026-05-01",
        "blocks": [{"hours": 648.0}],
    },
]
_EMIT_WEEKS_PER_MONTH = {1: 3, 2: 4, 3: 5}
_BUS_OF = {1: 0, 3: 2}.get
_EMIT_CALENDAR = [
    OperativeStage(
        index=i,
        start_date=date.fromisoformat(s["start_date"]),
        end_date=date.fromisoformat(s["end_date"]),
        season_id=date.fromisoformat(s["start_date"]).month - 1,
        block_hours=tuple(b["hours"] for b in s["blocks"]),
    )
    for i, s in enumerate(_EMIT_STAGES)
]
_SANTA = GnlThermal(86, "SANTA CRUZ", 1)
_PSERG = GnlThermal(224, "PSERGIPE I", 3)
_TG = pd.DataFrame(
    [
        _tg_row(86, "SANTA CRUZ", cvu=199.22, disp=500.0),
        _tg_row(224, "PSERGIPE I", sub=3, cvu=321.26, disp=1593.0),
    ]
)


def _emit_model() -> GnlCommitmentModel:
    """SANTA CRUZ (86) and PSERGIPE I (224) each declare a post-horizon
    já-comandada run (class-4): SANTA CRUZ's is a 0 MW stub then 500 MW;
    PSERGIPE's is all-zero. Both share the 2-month anticipation lag and both
    get a ``thermal_bounds`` row for every mirror-calendar stage."""
    santa = _SANTA
    pserg = _PSERG
    return GnlCommitmentModel(
        tg=_TG,
        thermals=(santa, pserg),
        commitments={
            86: GnlCommitment(
                86,
                (
                    GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),
                    GnlStageCommitment(8, date(2026, 5, 2), 0.0, 0.0),
                    GnlStageCommitment(9, date(2026, 5, 9), 500.0, 0.0),
                ),
            ),
            224: GnlCommitment(
                224,
                (
                    GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),
                    GnlStageCommitment(8, date(2026, 5, 2), 0.0, 0.0),
                    GnlStageCommitment(9, date(2026, 5, 9), 0.0, 0.0),
                ),
            ),
        },
        weeks_per_month=_EMIT_WEEKS_PER_MONTH,
        nl_lag_months={86: 2, 224: 2},
    )


def test_gnl_emission_has_no_future_anticipated_deliveries_field() -> None:
    field_names = {f.name for f in dataclasses.fields(GnlEmission)}
    assert "future_anticipated_deliveries" not in field_names


def test_convert_gnl_emitted_lead_reaches_each_plants_class4_end() -> None:
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_EMIT_CALENDAR
    )
    assert [(t["id"], t["name"]) for t in e.thermals] == [
        (94, "SANTA CRUZ"),
        (95, "PSERGIPE I"),
    ]
    # The emitted lead is per plant -- (class4_end - horizon_start).days * 24 --
    # not the tile-sizing global H: both plants share the same já-comandada run
    # here (class4_end = 2026-05-16), so both land on the same 1512 h (63 days),
    # but the value tracks each plant's own class4_end, not a shared constant.
    assert all(
        t["anticipated_config"] == {"lead_time_hours": 1512.0} for t in e.thermals
    )
    santa = e.thermals[0]
    assert santa["bus_id"] == 0
    assert santa["cost_per_mwh"] == 199.22
    assert santa["generation"] == {"min_mw": 0.0, "max_mw": 500.0}


def test_convert_gnl_left_boundary_tiles_the_h_derived_leading_stages() -> None:
    # The uncapped lead 1176 h exceeds the horizon (1152 h), so
    # lead_delivery_stage_count = every study stage — novomodelo's
    # check_commitment_coverage requires exactly these. Both plants tile all 4
    # (0 MW: no in-horizon gl commitment folds in), followed by each plant's
    # class-4 já-comandada run (asserted separately).
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_EMIT_CALENDAR
    )
    for tid in (94, 95):
        past = [p for p in e.past_anticipated_commitments if p["thermal_id"] == tid]
        class2 = past[:4]
        assert [p["start_date"] for p in class2] == [
            "2026-03-14",
            "2026-03-21",
            "2026-03-28",
            "2026-04-04",
        ]
        assert all(p["value_mw"] == 0.0 for p in class2)


def test_convert_gnl_post_study_stages_carry_grid_calendar() -> None:
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_EMIT_CALENDAR
    )
    pss = e.post_study_stages
    assert pss is not None
    # post_study_stages carries the calendar verbatim (cross-checked against
    # the helper directly, anchored at both plants' shared class4_end).
    assert pss["stages"] == _build_post_study_calendar(_STUDY_SPANS, date(2026, 5, 16))


def _calendar_stage_span(stage: Mapping) -> tuple[date, date]:
    """A calendar stage's ``[start, end)`` as parsed dates."""
    start = date.fromisoformat(stage["start_date"])
    return start, start + timedelta(hours=float(stage["duration_hours"]))


def test_convert_gnl_thermal_bounds_cover_only_class3_stages() -> None:
    # _emit_model()'s shared class4_end (2026-05-16) is exactly where the
    # class-3 study-mirror starts on the 7-stage calendar, so stages 0-2
    # (the class-4 já-comandada fill) get no thermal_bounds row -- their
    # delivery is already fixed by past_anticipated_commitments -- while
    # stages 3-6 (one per study stage) do, matching novomodelo's own post-study
    # e2e fixture.
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_EMIT_CALENDAR
    )
    pss = e.post_study_stages
    assert pss is not None
    stage_spans = {_calendar_stage_span(s) for s in pss["stages"]}
    assert len(stage_spans) == len(pss["stages"])  # every stage distinct

    for tid in (94, 95):
        indices = sorted(
            b["post_study_stage_index"]
            for b in pss["thermal_bounds"]
            if b["thermal_id"] == tid
        )
        assert indices == [3, 4, 5, 6]


def test_convert_gnl_thermal_bounds_cover_every_stage_when_no_class4_run() -> None:
    # No já-comandada commitment at all -> class4_end defaults to
    # horizon_end, so the class-4 fill is empty and the class-3 mirror (one
    # stage per study stage) starts immediately: every calendar stage is
    # class-3 and gets a thermal_bounds row. The complementary boundary to
    # test_convert_gnl_thermal_bounds_cover_only_class3_stages -- the class-3
    # mirror always starts exactly at class4_end, so a calendar with every
    # stage class-4 is not constructible.
    santa = _SANTA
    model = GnlCommitmentModel(
        tg=_TG,
        thermals=(santa,),
        commitments={
            86: GnlCommitment(
                86, (GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),)
            )
        },
        weeks_per_month=_EMIT_WEEKS_PER_MONTH,
        nl_lag_months={86: 2},
    )
    e = convert_gnl(
        model, first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_EMIT_CALENDAR
    )

    pss = e.post_study_stages
    assert pss is not None
    assert [(s["start_date"], s["duration_hours"]) for s in pss["stages"]] == [
        ("2026-05-01", 168.0),
        ("2026-05-08", 168.0),
        ("2026-05-15", 168.0),
        ("2026-05-22", 216.0),
    ]
    indices = sorted(b["post_study_stage_index"] for b in pss["thermal_bounds"])
    assert indices == [0, 1, 2, 3]


def test_convert_gnl_raises_on_differing_class4_end_across_plants() -> None:
    # post_study_stages is emitted once, globally: SANTA CRUZ's já-comandada
    # run through 05-16 and PSERGIPE's total absence of one (class4_end
    # defaults to horizon_end) can't share a single calendar boundary.
    santa = _SANTA
    pserg = _PSERG
    model = GnlCommitmentModel(
        tg=_TG,
        thermals=(santa, pserg),
        commitments={
            86: GnlCommitment(
                86,
                (
                    GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),
                    GnlStageCommitment(8, date(2026, 5, 2), 0.0, 0.0),
                    GnlStageCommitment(9, date(2026, 5, 9), 500.0, 0.0),
                ),
            ),
            224: GnlCommitment(
                224, (GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),)
            ),
        },
        weeks_per_month=_EMIT_WEEKS_PER_MONTH,
        nl_lag_months={86: 2, 224: 2},
    )
    with pytest.raises(ValueError, match="differing já-comandada cutoffs"):
        convert_gnl(
            model, first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_EMIT_CALENDAR
        )


def test_convert_gnl_thermal_bounds_match_capability_and_cost() -> None:
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_EMIT_CALENDAR
    )
    pss = e.post_study_stages
    assert pss is not None
    capability = {94: (199.22, 0.0, 500.0), 95: (321.26, 0.0, 1593.0)}

    assert pss["thermal_bounds"]
    for bound in pss["thermal_bounds"]:
        cost, lo, hi = capability[bound["thermal_id"]]
        assert bound["cost_per_mwh"] == cost
        assert bound["min_mw"] == lo
        assert bound["max_mw"] == hi


def test_convert_gnl_warns_on_nonuniform_nl_lag(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A deck mixing NL lags is surfaced (the single global H tile-sizing the
    # class-2 boundary assumes a uniform lag); the per-plant emitted lead is
    # unaffected by nl_lag_months either way.
    santa = _SANTA
    pserg = _PSERG
    model = GnlCommitmentModel(
        tg=_TG,
        thermals=(santa, pserg),
        commitments={
            86: GnlCommitment(
                86, (GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),)
            ),
            224: GnlCommitment(
                224, (GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),)
            ),
        },
        weeks_per_month=_EMIT_WEEKS_PER_MONTH,
        nl_lag_months={86: 2, 224: 3},  # differing lags
    )
    with caplog.at_level(
        logging.WARNING, logger="novomodelo_bridge.decomp.converters.anticipated"
    ):
        e = convert_gnl(
            model, first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_EMIT_CALENDAR
        )

    assert any("differing anticipation lags" in r.message for r in caplog.records)
    # Neither plant declares a já-comandada window, so class4_end defaults to
    # horizon_end: the emitted lead reaches exactly the study horizon
    # (1152 h = 48 days), not the tile-sizing global H (1176 h).
    assert all(
        t["anticipated_config"] == {"lead_time_hours": 1152.0} for t in e.thermals
    )


def test_convert_gnl_empty_gs_yields_no_post_study() -> None:
    # No GS calendar (degenerate deck): no post-study horizon to price, so
    # post_study_stages stays None — but the plant stays a valid anticipated
    # thermal with the mandatory single leading commitment.
    pserg = _PSERG
    model = GnlCommitmentModel(
        tg=_TG,
        thermals=(pserg,),
        commitments={
            224: GnlCommitment(
                224, (GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),)
            )
        },
        weeks_per_month={},
        nl_lag_months={224: 2},
    )
    e = convert_gnl(
        model, first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_EMIT_CALENDAR
    )

    assert e.post_study_stages is None
    assert len(e.past_anticipated_commitments) == 1  # left boundary still mandatory
    assert e.thermals[0]["anticipated_config"] == {"lead_time_hours": 168.0}


def test_convert_gnl_clamps_past_commitment_above_capability(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # gl geracao and tg disponibilidade are independent; an in-study commitment
    # above the plant's max_mw is clamped into [min_mw, max_mw] (+ warned) so
    # novomodelo's semantic validator (value_mw in [min, max]) never rejects it.
    santa = _SANTA
    model = GnlCommitmentModel(
        tg=_TG,
        thermals=(santa,),
        commitments={
            86: GnlCommitment(
                86, (GnlStageCommitment(1, date(2026, 3, 14), 900.0, 168.0),)
            )
        },
        weeks_per_month={},
        nl_lag_months={86: 2},
    )
    with caplog.at_level(
        logging.WARNING, logger="novomodelo_bridge.decomp.converters.anticipated"
    ):
        e = convert_gnl(
            model, first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_EMIT_CALENDAR
        )

    assert e.past_anticipated_commitments[0]["value_mw"] == 500.0  # 900 -> max 500
    assert any("clamped" in r.message for r in caplog.records)


def test_convert_gnl_reads_tg_per_stage_like_ct(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # TG declares stages 1, 2 and 4; stage 3 inherits stage 2. A 0 MW
    # commitment is clamped only where its own stage requires a minimum.
    tg = pd.DataFrame(
        [
            _tg_row(86, "SANTA CRUZ", cvu=181.0, disp=350.0, inflex=50.0),
            {**_tg_row(86, "SANTA CRUZ", cvu=181.0, disp=350.0), "estagio": 2},
            {**_tg_row(86, "SANTA CRUZ", cvu=200.0, disp=400.0), "estagio": 4},
        ]
    )
    model = GnlCommitmentModel(
        tg=tg,
        thermals=(_SANTA,),
        commitments={
            86: GnlCommitment(
                86, (GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),)
            )
        },
        weeks_per_month=_EMIT_WEEKS_PER_MONTH,
        nl_lag_months={86: 2},
    )
    with caplog.at_level(
        logging.WARNING, logger="novomodelo_bridge.decomp.converters.anticipated"
    ):
        e = convert_gnl(
            model, first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_EMIT_CALENDAR
        )

    tiles = [p["value_mw"] for p in e.past_anticipated_commitments][:4]
    assert tiles == [50.0, 0.0, 0.0, 0.0]
    clamps = [r.message for r in caplog.records if "clamped" in r.message]
    assert clamps == [
        "GNL SANTA CRUZ: committed 0 MW (in-horizon study stage 0) outside "
        "[50, 350]; clamped to 50"
    ]
    [thermal] = e.thermals
    assert thermal["cost_per_mwh"] == 181.0
    assert thermal["generation"] == {"min_mw": 0.0, "max_mw": 400.0}
    assert e.post_study_stages is not None
    assert {
        (b["cost_per_mwh"], b["min_mw"], b["max_mw"])
        for b in e.post_study_stages["thermal_bounds"]
    } == {(200.0, 0.0, 400.0)}


# --------------------------------------------------------------------------
# Class-4 já-comandada past_anticipated_commitments (the horizon-split)
# --------------------------------------------------------------------------

_HORIZON_END = date.fromisoformat(_EMIT_STAGES[-1]["end_date"])


def test_convert_gnl_emits_class4_past_commitments() -> None:
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_EMIT_CALENDAR
    )
    for tid, terminal_mw in ((94, 500.0), (95, 0.0)):
        past = [p for p in e.past_anticipated_commitments if p["thermal_id"] == tid]
        class4 = past[4:]
        assert [(p["start_date"], p["end_date"], p["value_mw"]) for p in class4] == [
            ("2026-05-01", "2026-05-02", 0.0),
            ("2026-05-02", "2026-05-09", 0.0),
            ("2026-05-09", "2026-05-16", terminal_mw),
        ]


def test_convert_gnl_past_commitments_never_straddle_horizon() -> None:
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_EMIT_CALENDAR
    )
    for p in e.past_anticipated_commitments:
        start = date.fromisoformat(p["start_date"])
        end = date.fromisoformat(p["end_date"])
        assert not (start < _HORIZON_END < end)


def test_convert_gnl_straddling_class4_window_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A synthetic classify_gnl_windows result whose window crosses the horizon
    # (never produced by the real classifier, which starts class-4 windows at
    # or after horizon_end) must still be rejected by convert_gnl's own guard.
    straddling = GnlPlantClassification(
        class4_windows=(
            (
                _HORIZON_END - timedelta(days=1),
                _HORIZON_END + timedelta(days=1),
                0.0,
            ),
        ),
        class4_end=_HORIZON_END + timedelta(days=1),
    )
    monkeypatch.setattr(
        anticipated,
        "classify_gnl_windows",
        lambda model, *, horizon_end: GnlClassification(
            plants={t.code: straddling for t in model.thermals}
        ),
    )

    with pytest.raises(ValueError, match="straddle"):
        convert_gnl(
            _emit_model(),
            first_thermal_id=94,
            bus_id_of=_BUS_OF,
            calendar=_EMIT_CALENDAR,
        )


def test_convert_gnl_clamps_class4_commitment_above_capability(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # gl geracao and tg disponibilidade are independent for a class-4 week too;
    # a já-comandada commitment above max_mw is clamped the same as class-2.
    santa = _SANTA
    model = GnlCommitmentModel(
        tg=_TG,
        thermals=(santa,),
        commitments={
            86: GnlCommitment(
                86,
                (
                    GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),
                    GnlStageCommitment(9, date(2026, 5, 9), 900.0, 0.0),
                ),
            )
        },
        weeks_per_month=_EMIT_WEEKS_PER_MONTH,
        nl_lag_months={86: 2},
    )
    with caplog.at_level(
        logging.WARNING, logger="novomodelo_bridge.decomp.converters.anticipated"
    ):
        e = convert_gnl(
            model, first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_EMIT_CALENDAR
        )

    class4 = [
        p
        for p in e.past_anticipated_commitments
        if p["thermal_id"] == 94 and date.fromisoformat(p["start_date"]) >= _HORIZON_END
    ]
    assert any(p["value_mw"] == 500.0 for p in class4)  # 900 -> max 500
    assert any("clamped" in r.message for r in caplog.records)

"""Unit tests for the source model's GNL reader (``decomp/anticipated.py``).

CI-tier and deck-independent: the tests drive a synthetic ``_StubDadgnl`` with
hand-authored ``gl``/``tg``/``gs`` data (mirroring the real register shapes), so
the suite needs no deck and no novomodelo binary; the ones that read a whole
``dadgnl`` write a two-register file to ``tmp_path``. ``gl`` is a list of
register stubs (the real ``gl(df=True)`` is unusable on ragged weekly blocks);
``tg``/``gs`` are DataFrames (their fixed shapes make ``df=True`` well-formed).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from idecomp.decomp import Dadgnl

from novomodelo_bridge.decomp.converters import anticipated
from novomodelo_bridge.decomp.converters.anticipated import (
    GnlClass4Window,
    GnlClassification,
    GnlCommitment,
    GnlCommitmentModel,
    GnlEmission,
    GnlPlantClassification,
    GnlStageCommitment,
    GnlThermal,
    _build_post_study_calendar,
    _month_end_duration_hours,
    classify_gnl_windows,
    convert_gnl,
    fill_undeclared_block_hours,
    gnl_block_commitments,
    is_gnl_enabled,
    operative_week_months,
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
    assert model.commitments[86].stages[0].block_mw == (107.0, 131.0, 97.0)


def _two_stage_calendar() -> list[OperativeStage]:
    """A weekly stage, then a two-week stage standing in for a monthly one."""
    return [
        OperativeStage(
            index=0,
            start_date=date(2026, 3, 14),
            end_date=date(2026, 3, 21),
            season_id=2,
            block_hours=(40.0, 48.0, 80.0),
        ),
        OperativeStage(
            index=1,
            start_date=date(2026, 3, 21),
            end_date=date(2026, 4, 4),
            season_id=2,
            block_hours=(50.0, 96.0, 190.0),
        ),
    ]


def test_gnl_block_commitments_weights_each_block_by_its_duracao() -> None:
    gl = [
        _GlReg(86, 1, "14032026", [40.0, 48.0, 80.0], [100.0, 200.0, 300.0]),
        _GlReg(86, 2, "21032026", [40.0, 48.0, 80.0], [100.0, 100.0, 100.0]),
        _GlReg(86, 3, "28032026", [10.0, 48.0, 110.0], [400.0, 300.0, 100.0]),
    ]
    model = read_gnl_model(_StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], []))
    assert model is not None

    got = gnl_block_commitments(model, _two_stage_calendar())

    assert got[86][0] == (100.0, 200.0, 300.0)
    assert got[86][1] == pytest.approx(((40 * 100 + 10 * 400) / 50, 200.0, 100.0))


def test_gnl_block_commitments_stage_without_gl_week_is_uncommanded() -> None:
    gl = [_GlReg(86, 1, "14032026", [40.0, 48.0, 80.0], [500.0, 500.0, 500.0])]
    model = read_gnl_model(_StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], []))
    assert model is not None

    assert gnl_block_commitments(model, _two_stage_calendar())[86][1] is None


def test_gnl_block_commitments_block_count_mismatch_raises() -> None:
    gl = [_GlReg(86, 1, "14032026", [84.0, 84.0], [500.0, 500.0])]
    model = read_gnl_model(_StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], []))
    assert model is not None

    with pytest.raises(ValueError, match="declares 2 block"):
        gnl_block_commitments(model, _two_stage_calendar())


def test_gate_on_for_a_tg_plant_without_gl() -> None:
    stub = _StubDadgnl([], [_tg_row(86, "SANTA CRUZ")], [])

    assert is_gnl_enabled(stub) is True
    assert read_gnl_model(stub) is not None


def test_gate_on_when_all_geracao_zero() -> None:
    gl = [_GlReg(86, 1, "14032026", [40.0, 48.0, 80.0], [0.0, 0.0, 0.0])]
    stub = _StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], [])

    assert is_gnl_enabled(stub) is True
    model = read_gnl_model(stub)
    assert model is not None
    assert model.commitments[86].stages[0].committed_mw == 0.0


def test_gate_off_without_tg_plants() -> None:
    assert is_gnl_enabled(_StubDadgnl([], [], [])) is False


def test_gate_off_for_none() -> None:
    assert is_gnl_enabled(None) is False


def test_thermals_ascending_by_code() -> None:
    # The registry lists every tg plant.
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
    # code 86 carries a gl dispatch; 224 is registry-only.
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


_ONE_GL_DADGNL = (
    "TG  224   3   P. SERGIPE 1       01593.       299    01593.       299"
    "    01593.       299\n"
    "GL  224   3    1      1517.14  126   1395.11  322   1194.75  296"
    " 01012026\n"
)


def _read_dadgnl_text(tmp_path: Path, text: str) -> Dadgnl:
    path = tmp_path / "dadgnl.rv0"
    path.write_text(text)
    return Dadgnl.read(str(path))


def test_tg_cost_without_decimal_point_reads_with_the_implied_decimals(
    tmp_path: Path,
) -> None:
    # TG's cvu is F10.2: written without a point, the source model reads 299
    # as 2.99, not 299.0.
    model = read_gnl_model(_read_dadgnl_text(tmp_path, _ONE_GL_DADGNL))

    assert model is not None
    assert model.tg[["cvu_1", "cvu_2", "cvu_3"]].iloc[0].tolist() == [2.99] * 3
    assert model.tg["disponibilidade_1"].iloc[0] == 1593.0


def test_read_gnl_model_reads_a_deck_with_a_single_gl_register(
    tmp_path: Path,
) -> None:
    model = read_gnl_model(_read_dadgnl_text(tmp_path, _ONE_GL_DADGNL))

    assert model is not None
    [stage] = model.commitments[224].stages
    assert stage.estagio == 1
    assert stage.start_date == date(2026, 1, 1)
    assert stage.hours == 744.0
    assert stage.committed_mw == pytest.approx(
        (1517.14 * 126 + 1395.11 * 322 + 1194.75 * 296) / 744
    )


def test_weeks_per_month_from_gs() -> None:
    gl = [_GlReg(86, 1, "14032026", [168.0], [100.0])]
    gs = [{"mes": 1, "semanas": 3}, {"mes": 2, "semanas": 4}, {"mes": 3, "semanas": 5}]
    model = read_gnl_model(_StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], gs))

    assert model is not None
    assert model.weeks_per_month == {1: 3, 2: 4, 3: 5}
    assert isinstance(model, GnlCommitmentModel)


def test_operative_week_months_books_weeks_by_gs_counts() -> None:
    weeks = operative_week_months({1: 3, 2: 4, 3: 5}, _EMIT_CALENDAR)

    assert len(weeks) == 12
    assert weeks[date(2026, 3, 28)] == (2026, 3)
    assert weeks[date(2026, 4, 4)] == (2026, 4)
    assert weeks[date(2026, 4, 25)] == (2026, 4)
    # Starts in May, ends in June: booked to May by the count.
    assert weeks[date(2026, 5, 30)] == (2026, 5)


def test_operative_week_months_empty_without_gs() -> None:
    assert operative_week_months({}, _EMIT_CALENDAR) == {}


def test_operative_week_months_books_one_interval_per_month_for_a_monthly_study() -> (
    None
):
    weeks = operative_week_months({1: 1, 2: 1, 3: 1}, _MONTHLY_CALENDAR)

    assert weeks == {
        date(2026, 1, 1): (2026, 1),
        date(2026, 2, 1): (2026, 2),
        date(2026, 3, 1): (2026, 3),
    }


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
    result = classify_gnl_windows(_stage_model(86, stages), _EMIT_CALENDAR)

    santa = result.plants[86]
    assert santa.class4_windows == (
        GnlClass4Window(date(2026, 5, 1), date(2026, 5, 2), 0.0, None),
        GnlClass4Window(date(2026, 5, 2), date(2026, 5, 9), 0.0, stages[0]),
        GnlClass4Window(date(2026, 5, 9), date(2026, 5, 16), 500.0, stages[1]),
    )
    assert santa.class4_end == date(2026, 5, 16)
    assert isinstance(result, GnlClassification)


def test_classify_gnl_windows_no_post_horizon_weeks_is_free() -> None:
    stages = (GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),)
    result = classify_gnl_windows(_stage_model(86, stages), _EMIT_CALENDAR)

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
        classify_gnl_windows(_stage_model(86, stages), _EMIT_CALENDAR)


# --------------------------------------------------------------------------
# The grid-anchored post-study calendar
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
    # The emitted lead runs to the shared já-comandada cutoff:
    # (2026-05-16 - 2026-03-14).days * 24 = 1512 h.
    assert all(
        t["anticipated_config"] == {"lead_time_hours": 1512.0} for t in e.thermals
    )
    santa = e.thermals[0]
    assert santa["bus_id"] == 0
    assert santa["cost_per_mwh"] == 199.22
    assert santa["generation"] == {"min_mw": 0.0, "max_mw": 500.0}


def test_convert_gnl_left_boundary_tiles_the_lead_derived_leading_stages() -> None:
    # The emitted lead 1512 h exceeds the horizon (1152 h), so
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
    # A deck mixing NL lags is surfaced; the emitted lead is unaffected by
    # nl_lag_months either way.
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
    # (1152 h = 48 days).
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


def _tg_with_post_study_months(*rows: dict) -> pd.DataFrame:
    """SANTA CRUZ's stage-1 ``tg`` record plus *rows* past the 4-stage study."""
    return pd.DataFrame([_tg_row(86, "SANTA CRUZ", cvu=199.22, disp=500.0), *rows])


def _santa_model(
    tg: pd.DataFrame, *post_horizon: GnlStageCommitment
) -> GnlCommitmentModel:
    return GnlCommitmentModel(
        tg=tg,
        thermals=(_SANTA,),
        commitments={
            86: GnlCommitment(
                86,
                (GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0), *post_horizon),
            )
        },
        weeks_per_month=_EMIT_WEEKS_PER_MONTH,
        nl_lag_months={86: 2},
    )


def test_convert_gnl_prices_post_study_stages_from_the_month_they_start_in() -> None:
    # Já-comandada weeks to 2026-05-16, then the mirror: weeks starting 16, 23
    # and 30 May (the last ending in June) take May's tg (stage 5), the month
    # starting 6 June takes June's (stage 6).
    tg = _tg_with_post_study_months(
        {**_tg_row(86, "SANTA CRUZ", cvu=210.0, disp=350.0), "estagio": 5},
        {**_tg_row(86, "SANTA CRUZ", cvu=0.0, disp=0.0), "estagio": 6},
    )
    model = _santa_model(
        tg,
        GnlStageCommitment(8, date(2026, 5, 2), 0.0, 0.0),
        GnlStageCommitment(9, date(2026, 5, 9), 0.0, 0.0),
    )
    e = convert_gnl(
        model, first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_EMIT_CALENDAR
    )

    assert e.post_study_stages is not None
    assert [
        (s["start_date"], b["cost_per_mwh"], b["max_mw"])
        for b in e.post_study_stages["thermal_bounds"]
        for s in [e.post_study_stages["stages"][b["post_study_stage_index"]]]
    ] == [
        ("2026-05-16", 210.0, 350.0),
        ("2026-05-23", 210.0, 350.0),
        ("2026-05-30", 210.0, 350.0),
        ("2026-06-06", 0.0, 0.0),
    ]
    assert e.thermals[0]["generation"] == {"min_mw": 0.0, "max_mw": 500.0}


def test_convert_gnl_post_study_month_inherits_the_last_declared_tg() -> None:
    tg = _tg_with_post_study_months(
        {**_tg_row(86, "SANTA CRUZ", cvu=210.0, disp=350.0), "estagio": 5}
    )
    e = convert_gnl(
        _santa_model(tg),
        first_thermal_id=94,
        bus_id_of=_BUS_OF,
        calendar=_EMIT_CALENDAR,
    )

    assert e.post_study_stages is not None
    assert {
        (b["cost_per_mwh"], b["max_mw"]) for b in e.post_study_stages["thermal_bounds"]
    } == {(210.0, 350.0)}


def test_convert_gnl_class4_window_takes_the_last_study_stage_tg() -> None:
    tg = _tg_with_post_study_months(
        {**_tg_row(86, "SANTA CRUZ", cvu=210.0, disp=600.0), "estagio": 5},
        {**_tg_row(86, "SANTA CRUZ", cvu=220.0, disp=300.0), "estagio": 6},
    )
    model = _santa_model(
        tg,
        GnlStageCommitment(8, date(2026, 5, 2), 0.0, 0.0),
        GnlStageCommitment(9, date(2026, 5, 9), 900.0, 0.0),
    )
    e = convert_gnl(
        model, first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_EMIT_CALENDAR
    )

    assert [p["value_mw"] for p in e.past_anticipated_commitments][-1] == 500.0
    assert e.thermals[0]["generation"] == {"min_mw": 0.0, "max_mw": 500.0}
    assert e.post_study_stages is not None
    assert [
        (b["post_study_stage_index"], b["max_mw"])
        for b in e.post_study_stages["thermal_bounds"]
    ] == [(3, 600.0), (4, 600.0), (5, 600.0), (6, 300.0)]


def test_convert_gnl_rejects_tg_past_the_post_study_months() -> None:
    tg = _tg_with_post_study_months(
        {**_tg_row(86, "SANTA CRUZ", disp=0.0), "estagio": 6}
    )

    with pytest.raises(ValueError, match=r"TG stage 6 outside the calendar \(1..5\)"):
        convert_gnl(
            _santa_model(tg),
            first_thermal_id=94,
            bus_id_of=_BUS_OF,
            calendar=_EMIT_CALENDAR,
        )


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
            GnlClass4Window(
                _HORIZON_END - timedelta(days=1),
                _HORIZON_END + timedelta(days=1),
                0.0,
                None,
            ),
        ),
        class4_end=_HORIZON_END + timedelta(days=1),
    )
    monkeypatch.setattr(
        anticipated,
        "classify_gnl_windows",
        lambda model, calendar: GnlClassification(
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


# --------------------------------------------------------------------------
# Emission for a study of monthly stages only (the decomp-mensal shape)
# --------------------------------------------------------------------------

_MONTHLY_CALENDAR = [
    OperativeStage(0, date(2026, 1, 1), date(2026, 2, 1), 0, (126.0, 322.0, 296.0)),
    OperativeStage(1, date(2026, 2, 1), date(2026, 3, 1), 1, (114.0, 291.0, 267.0)),
]


def _monthly_tg(code: int, name: str, *, sub: int, cvus: list[float]) -> list[dict]:
    """One ``tg`` record per stage of the study and its two post-study months."""
    return [
        {**_tg_row(code, name, sub=sub, cvu=cvu, disp=500.0), "estagio": k}
        for k, cvu in enumerate(cvus, start=1)
    ]


def _monthly_model(
    *,
    nl: Mapping[int, int] | None = None,
    weeks_per_month: Mapping[int, int] | None = None,
    extra_gl: tuple[GnlStageCommitment, ...] = (),
) -> GnlCommitmentModel:
    """SANTA CRUZ (86) commits 500 MW in January and a ragged February; PSERGIPE
    (224) commits nothing. ``tg`` prices each month differently."""
    return GnlCommitmentModel(
        tg=pd.DataFrame(
            _monthly_tg(86, "SANTA CRUZ", sub=1, cvus=[296.0, 297.0, 298.0, 299.0])
            + _monthly_tg(224, "PSERGIPE I", sub=3, cvus=[310.0, 311.0, 312.0, 313.0])
        ),
        thermals=(_SANTA, _PSERG),
        commitments={
            86: GnlCommitment(
                86,
                (
                    GnlStageCommitment(1, date(2026, 1, 1), 500.0, 744.0),
                    GnlStageCommitment(2, date(2026, 2, 1), 112.0, 672.0),
                    *extra_gl,
                ),
            ),
            224: GnlCommitment(
                224,
                (
                    GnlStageCommitment(1, date(2026, 1, 1), 0.0, 744.0),
                    GnlStageCommitment(2, date(2026, 2, 1), 0.0, 672.0),
                ),
            ),
        },
        weeks_per_month=dict(weeks_per_month or {1: 1, 2: 1, 3: 1}),
        nl_lag_months=dict(nl or {86: 2, 224: 2}),
    )


def _convert_monthly(model: GnlCommitmentModel) -> GnlEmission:
    return convert_gnl(
        model, first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_MONTHLY_CALENDAR
    )


def test_convert_gnl_monthly_study_leads_by_the_nl_lag_in_stages() -> None:
    e = _convert_monthly(_monthly_model())

    assert [t["anticipated_config"] for t in e.thermals] == [
        {"lead_stages": 2},
        {"lead_stages": 2},
    ]


def test_convert_gnl_monthly_study_pins_the_lagged_study_stages_to_gl() -> None:
    e = _convert_monthly(_monthly_model())

    assert [
        (p["thermal_id"], p["start_date"], p["end_date"], p["value_mw"])
        for p in e.past_anticipated_commitments
    ] == [
        (94, "2026-01-01", "2026-02-01", 500.0),
        (94, "2026-02-01", "2026-03-01", 112.0),
        (95, "2026-01-01", "2026-02-01", 0.0),
        (95, "2026-02-01", "2026-03-01", 0.0),
    ]


def test_convert_gnl_monthly_post_study_is_whole_months_priced_from_tg() -> None:
    e = _convert_monthly(_monthly_model())

    assert e.post_study_stages is not None
    assert e.post_study_stages["stages"] == [
        {"start_date": "2026-03-01", "duration_hours": 744.0},
        {"start_date": "2026-04-01", "duration_hours": 720.0},
    ]
    assert [
        (b["thermal_id"], b["post_study_stage_index"], b["cost_per_mwh"])
        for b in e.post_study_stages["thermal_bounds"]
    ] == [(94, 0, 298.0), (94, 1, 299.0), (95, 0, 312.0), (95, 1, 313.0)]


def test_convert_gnl_monthly_shorter_lag_prices_only_its_reach() -> None:
    e = _convert_monthly(_monthly_model(nl={86: 2, 224: 1}))

    assert e.thermals[1]["anticipated_config"] == {"lead_stages": 1}
    assert [
        p["start_date"] for p in e.past_anticipated_commitments if p["thermal_id"] == 95
    ] == ["2026-01-01"]
    assert e.post_study_stages is not None
    assert len(e.post_study_stages["stages"]) == 2
    assert [
        b["post_study_stage_index"]
        for b in e.post_study_stages["thermal_bounds"]
        if b["thermal_id"] == 95
    ] == [0]


def test_convert_gnl_monthly_rejects_a_gs_month_split_into_weeks() -> None:
    with pytest.raises(ValueError, match=r"one gs interval per month.*\{3: 4\}"):
        _convert_monthly(_monthly_model(weeks_per_month={1: 1, 2: 1, 3: 4}))


def test_convert_gnl_monthly_rejects_a_lag_past_the_study() -> None:
    with pytest.raises(ValueError, match="nl lag of 3 month"):
        _convert_monthly(_monthly_model(nl={86: 3, 224: 2}))


def test_convert_gnl_monthly_rejects_a_plant_without_nl() -> None:
    with pytest.raises(ValueError, match=r"plant 224 \(PSERGIPE I\) declares no nl"):
        _convert_monthly(_monthly_model(nl={86: 2}))


def test_convert_gnl_monthly_rejects_gl_past_the_horizon() -> None:
    late = (GnlStageCommitment(3, date(2026, 3, 1), 500.0, 0.0),)

    with pytest.raises(ValueError, match="past the study horizon .2026-03-01."):
        _convert_monthly(_monthly_model(extra_gl=late))


def test_convert_gnl_monthly_rejects_tg_past_the_post_study_months() -> None:
    model = _monthly_model()
    beyond = {**_tg_row(86, "SANTA CRUZ"), "estagio": 5}
    model = dataclasses.replace(
        model, tg=pd.concat([model.tg, pd.DataFrame([beyond])], ignore_index=True)
    )

    with pytest.raises(ValueError, match=r"TG stage 5 outside the calendar \(1..4\)"):
        _convert_monthly(model)


# A later revision of the same operative month: 4 weekly October stages + a
# monthly November stage (576 h), horizon 2026-12-01. GS counts only the weeks
# left in the study (4 + 3 = 7), while GL still commits 9 weeks: the last two
# are December weeks, one starting before the horizon end (2026-11-28).
_REVISION_CALENDAR = [
    *(
        OperativeStage(
            index=i,
            start_date=date(2026, 10, 10) + timedelta(weeks=i),
            end_date=date(2026, 10, 17) + timedelta(weeks=i),
            season_id=9,
            block_hours=(168.0,),
        )
        for i in range(4)
    ),
    OperativeStage(
        index=4,
        start_date=date(2026, 11, 7),
        end_date=date(2026, 12, 1),
        season_id=10,
        block_hours=(576.0,),
    ),
]


def _revision_week(
    estagio: int, start: date, mw: float, *, declared: bool = True
) -> GnlStageCommitment:
    """A one-block ``gl`` week; past the horizon the source model leaves
    ``duracao`` blank (``declared=False``)."""
    return GnlStageCommitment(
        estagio,
        start,
        mw,
        168.0 if declared else 0.0,
        block_mw=(mw,),
        block_hours=(168.0,) if declared else (),
    )


def _revision_model(week8_mw: float = 0.0) -> GnlCommitmentModel:
    """GL weeks 1-7 inside the study (4 weekly + 3 November), week 8
    (2026-11-28) and week 9 (2026-12-05) the first December weeks."""
    gl = [
        *(
            _revision_week(i + 1, date(2026, 10, 10) + timedelta(weeks=i), 0.0)
            for i in range(4)
        ),
        *(
            _revision_week(i + 5, date(2026, 11, 7) + timedelta(weeks=i), 100.0)
            for i in range(3)
        ),
        _revision_week(8, date(2026, 11, 28), week8_mw, declared=False),
        _revision_week(9, date(2026, 12, 5), 0.0, declared=False),
    ]
    return GnlCommitmentModel(
        tg=_tg_with_post_study_months(),
        thermals=(_SANTA,),
        commitments={86: GnlCommitment(86, tuple(gl))},
        weeks_per_month={1: 4, 2: 3, 3: 6},
        nl_lag_months={86: 2},
    )


def test_convert_gnl_tiles_every_study_stage_when_gl_outruns_the_study_gs_weeks() -> (
    None
):
    e = convert_gnl(
        _revision_model(),
        first_thermal_id=94,
        bus_id_of=_BUS_OF,
        calendar=_REVISION_CALENDAR,
    )

    # 2026-10-10 to the já-comandada cutoff 2026-12-12: 63 days.
    assert e.thermals[0]["anticipated_config"] == {"lead_time_hours": 1512.0}
    class2 = [
        p
        for p in e.past_anticipated_commitments
        if date.fromisoformat(p["end_date"]) <= date(2026, 12, 1)
    ]
    assert [p["start_date"] for p in class2] == [
        "2026-10-10",
        "2026-10-17",
        "2026-10-24",
        "2026-10-31",
        "2026-11-07",
    ]
    # Weeks 5-7 at 100 MW (504 h) plus week 8's 72 h head (2026-11-28 to the
    # horizon end) at 0 MW.
    assert class2[-1]["value_mw"] == pytest.approx(50400.0 / 576.0)


def test_convert_gnl_books_the_week_straddling_the_horizon_to_the_month_after_by_gs() -> (
    None
):
    e = convert_gnl(
        _revision_model(week8_mw=300.0),
        first_thermal_id=94,
        bus_id_of=_BUS_OF,
        calendar=_REVISION_CALENDAR,
    )

    windows = [
        (p["start_date"], p["end_date"], p["value_mw"])
        for p in e.past_anticipated_commitments
    ]
    # Week 8 (2026-11-28) is the first December week, held from the horizon
    # end; its 72 h head joins November's weeks 5-7 (504 h at 100 MW).
    assert ("2026-11-07", "2026-12-01", 125.0) in windows
    assert windows[-2:] == [
        ("2026-12-01", "2026-12-05", 300.0),
        ("2026-12-05", "2026-12-12", 0.0),
    ]
    assert e.post_study_stages is not None
    assert e.post_study_stages["stages"][0] == {
        "start_date": "2026-12-01",
        "duration_hours": 96.0,
    }


def test_classify_gnl_windows_clips_a_week_straddling_the_horizon_to_it() -> None:
    model = _revision_model(week8_mw=300.0)
    week8 = model.commitments[86].stages[7]

    santa = classify_gnl_windows(model, _REVISION_CALENDAR).plants[86]

    assert santa.class4_windows[0] == GnlClass4Window(
        date(2026, 12, 1), date(2026, 12, 5), 300.0, week8
    )
    assert santa.class4_end == date(2026, 12, 12)


def test_classify_gnl_windows_rejects_a_study_week_starting_past_the_horizon() -> None:
    model = _revision_model()
    model = dataclasses.replace(model, weeks_per_month={1: 4, 2: 5, 3: 6})

    with pytest.raises(ValueError, match="to the study, but it starts"):
        classify_gnl_windows(model, _REVISION_CALENDAR)


def test_gnl_block_commitments_weighs_the_straddling_week_by_its_head_hours() -> None:
    model = fill_undeclared_block_hours(
        _revision_model(week8_mw=300.0), _REVISION_CALENDAR, {(2026, 12): (1.0,)}
    )

    got = gnl_block_commitments(model, _REVISION_CALENDAR)

    assert got[86][4] == pytest.approx((125.0,))


def _three_block_revision_model(week8_mw: tuple[float, ...]) -> GnlCommitmentModel:
    model = _revision_model()
    stages = list(model.commitments[86].stages)
    stages[7] = GnlStageCommitment(
        8, date(2026, 11, 28), sum(week8_mw) / 3, 0.0, block_mw=week8_mw
    )
    stages[8] = GnlStageCommitment(
        9, date(2026, 12, 5), 0.0, 0.0, block_mw=(0.0, 0.0, 0.0)
    )
    return dataclasses.replace(
        model, commitments={86: GnlCommitment(86, tuple(stages))}
    )


def _three_block_calendar() -> list[OperativeStage]:
    return [
        *_REVISION_CALENDAR[:-1],
        dataclasses.replace(_REVISION_CALENDAR[-1], block_hours=(96.0, 240.0, 240.0)),
    ]


def test_fill_undeclared_block_hours_weights_by_the_gs_month_patamar_shares() -> None:
    model = _three_block_revision_model((175.0, 30.0, 0.0))

    filled = fill_undeclared_block_hours(
        model, _three_block_calendar(), {(2026, 12): (0.25, 0.25, 0.5)}
    )

    week8 = filled.commitments[86].stages[7]
    assert week8.block_hours == (42.0, 42.0, 84.0)
    assert week8.committed_mw == pytest.approx((175.0 + 30.0) / 4)


def test_fill_undeclared_block_hours_leaves_declared_weeks_untouched() -> None:
    model = _three_block_revision_model((0.0, 0.0, 0.0))

    filled = fill_undeclared_block_hours(
        model, _three_block_calendar(), {(2026, 12): (0.25, 0.25, 0.5)}
    )

    assert filled.commitments[86].stages[:7] == model.commitments[86].stages[:7]


def test_fill_undeclared_block_hours_falls_back_to_the_last_stage_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _three_block_revision_model((175.0, 30.0, 0.0))

    with caplog.at_level(
        logging.WARNING, logger="novomodelo_bridge.decomp.converters.anticipated"
    ):
        filled = fill_undeclared_block_hours(model, _three_block_calendar(), None)

    week8 = filled.commitments[86].stages[7]
    assert week8.committed_mw == pytest.approx((175.0 * 96 + 30.0 * 240) / 576)
    assert any("last study stage" in r.message for r in caplog.records)
    assert any("different MW per patamar" in r.message for r in caplog.records)


def test_fill_undeclared_block_hours_uniform_week_with_shares_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _three_block_revision_model((50.0, 50.0, 50.0))

    with caplog.at_level(
        logging.WARNING, logger="novomodelo_bridge.decomp.converters.anticipated"
    ):
        filled = fill_undeclared_block_hours(
            model, _three_block_calendar(), {(2026, 12): (0.25, 0.25, 0.5)}
        )

    assert filled.commitments[86].stages[7].committed_mw == pytest.approx(50.0)
    assert caplog.records == []


def test_convert_gnl_keeps_a_plant_whose_gl_commits_nothing() -> None:
    model = _revision_model()
    stages = tuple(
        dataclasses.replace(week, committed_mw=0.0, block_mw=(0.0,))
        for week in model.commitments[86].stages
    )
    model = dataclasses.replace(model, commitments={86: GnlCommitment(86, stages)})

    e = convert_gnl(
        model, first_thermal_id=94, bus_id_of=_BUS_OF, calendar=_REVISION_CALENDAR
    )

    assert [t["name"] for t in e.thermals] == ["SANTA CRUZ"]
    assert e.past_anticipated_commitments
    assert all(p["value_mw"] == 0.0 for p in e.past_anticipated_commitments)
    assert e.post_study_stages is not None
    assert e.post_study_stages["thermal_bounds"]

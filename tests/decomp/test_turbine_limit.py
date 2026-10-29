"""Tests for the source model's a-priori turbined-flow limit
(``novomodelo_bridge.decomp.converters.hydro.turbine_limit``) and the ``MP``/``FD``
factors it shares with the availability overlay.

Every plant here has a flat forebay cota of 100 m and ``ρ_esp`` 0.01; over the
flat 20 m tailrace (:data:`_FLAT_TAILRACE`) that is a net head of 80 m and
``ρ_eq`` 0.8 MW per m³/s. The limits read stage 0, a 168 h week. Tier-1:
synthetic inputs only.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from idecomp.decomp.modelos.dadger import ACALTEFE

from novomodelo_bridge.core.diagnostics import collect
from novomodelo_bridge.core.tailrace import TailraceFamily, TailraceSegment
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
from novomodelo_bridge.decomp.converters.hydro import (
    TurbineLimits,
    compute_turbine_limits,
    convert_hydro_group_availability,
    convert_hydros,
    plant_turbine_limit_contributions,
    stage_availability_factors,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import build_operative_calendar
from tests.conftest import make_decomp_case
from tests.decomp.conftest import flat_turbine_limits

_WEEK_SECONDS = 168.0 * 3600.0


def _tailrace(*coefficients: float, reference_m: float = math.nan) -> TailraceFamily:
    padded = (*coefficients, 0.0, 0.0, 0.0, 0.0, 0.0)[:5]
    return TailraceFamily(reference_m, (TailraceSegment(math.inf, padded),))


_FLAT_TAILRACE = (_tailrace(20.0),)
#: The tailrace rises 0.1 m per m³/s of outflow.
_RISING_TAILRACE = (_tailrace(20.0, 0.1),)


def _rising_turbined(outflow_m3s: float) -> float:
    """A 100 m³/s, 80 m rated Francis set's flow at *outflow_m3s* over the
    rising tailrace."""
    return 100.0 * ((80.0 - 0.1 * outflow_m3s) / 80.0) ** 0.5


def _plant_row(
    *,
    name: str = "PLANT",
    jusante: int = 0,
    vmin: float = 100.0,
    vmax: float = 500.0,
    cota: tuple[float, float, float, float, float] = (100.0, 0.0, 0.0, 0.0, 0.0),
    tipo_regulacao: str = "D",
    conjuntos: Sequence[tuple[int, float, float, float]] = ((2, 50.0, 1000.0, 80.0),),
    tipo_turbina: int = 1,
) -> dict:
    """*conjuntos* holds ``(n_machines, q_nom, p_nom, h_nom)`` per machine set."""
    row: dict = {
        "nome_usina": name,
        "submercado": 1,
        "codigo_usina_jusante": jusante,
        "desvio": 0,
        "volume_minimo": vmin,
        "volume_maximo": vmax,
        "numero_conjuntos_maquinas": len(conjuntos),
        "teif": 0.0,
        "ip": 0.0,
        "canal_fuga_medio": 20.0,
        "produtibilidade_especifica": 0.01,
        "tipo_perda": 0,
        "perdas": 0.0,
        "tipo_turbina": tipo_turbina,
        "tipo_regulacao": tipo_regulacao,
    }
    for i, a in enumerate(cota):
        row[f"a{i}_volume_cota"] = a
    for k, (n, q_nom, p_nom, h_nom) in enumerate(conjuntos, start=1):
        row[f"maquinas_conjunto_{k}"] = n
        row[f"vazao_nominal_conjunto_{k}"] = q_nom
        row[f"potencia_nominal_conjunto_{k}"] = p_nom
        row[f"queda_nominal_conjunto_{k}"] = h_nom
    return row


def _hidr_frame(rows: Mapping[int, dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).T
    df.index.name = "codigo_usina"
    return df


def _calendar():
    return build_operative_calendar(
        date(2026, 7, 18), [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    )


def _limits(
    rows: Mapping[int, dict],
    *,
    tailrace: Mapping[int, tuple[TailraceFamily, ...]] | None = None,
    inflows: Mapping[int, float] | None = None,
    initial_volumes: Mapping[int, float] | None = None,
    factors: Mapping[tuple[int, int], float] | None = None,
) -> TurbineLimits:
    """The limits for the plants in *rows*, the same inputs on every stage.

    Every plant defaults to the flat tailrace, zero inflow, its minimum
    volume and an ``MP × FD`` of 1 on group 0 (``(code, group)`` keys in
    *factors* override it).
    """
    hidr = _hidr_frame(rows)
    codes = tuple(rows)
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=codes)
    calendar = _calendar()
    stages = range(len(calendar))
    all_factors = {(code, 0, s): 1.0 for code in codes for s in stages} | {
        (code, group, s): value
        for (code, group), value in (factors or {}).items()
        for s in stages
    }
    return compute_turbine_limits(
        EffectiveCadastro(base=hidr, n_stages=len(calendar), stage_varying={}),
        id_map,
        calendar,
        initial_volumes={
            code: (initial_volumes or {}).get(code, rows[code]["volume_minimo"])
            for code in codes
        },
        incremental_inflows={
            (code, s): (inflows or {}).get(code, 0.0) for code in codes for s in stages
        },
        tailrace={code: (tailrace or {}).get(code, _FLAT_TAILRACE) for code in codes},
        availability_factors=all_factors,
    )


def test_rated_head_turbines_the_rated_flow() -> None:
    limits = _limits({1: _plant_row()})
    assert limits.cap(1, 0, 0) == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("tipo_turbina", "exponent"),
    [pytest.param(1, 0.5, id="francis"), pytest.param(2, 0.2, id="kaplan")],
)
def test_head_below_rated_lowers_the_limit_by_the_affinity_law(
    tipo_turbina: int, exponent: float
) -> None:
    row = _plant_row(conjuntos=((2, 50.0, 1000.0, 320.0),), tipo_turbina=tipo_turbina)
    limits = _limits({1: row})
    assert limits.cap(1, 0, 0) == pytest.approx(100.0 * (80.0 / 320.0) ** exponent)


def test_rated_head_is_the_unit_weighted_mean_of_the_machine_sets() -> None:
    row = _plant_row(conjuntos=((1, 60.0, 600.0, 240.0), (3, 40.0, 400.0, 320.0)))
    limits = _limits({1: row})
    rated_head = (1 * 240.0 + 3 * 320.0) / 4
    assert limits.cap(1, 0, 0) == pytest.approx(180.0 * (80.0 / rated_head) ** 0.5)


def test_missing_rated_head_turbines_the_rated_flow() -> None:
    row = _plant_row(conjuntos=((2, 50.0, 1000.0, math.nan),))
    limits = _limits({1: row})
    assert limits.cap(1, 0, 0) == pytest.approx(100.0)


def test_rising_tailrace_settles_within_a_cubic_metre_of_the_fixed_point() -> None:
    limits = _limits({1: _plant_row()}, tailrace={1: _RISING_TAILRACE})
    cap = limits.cap(1, 0, 0)
    assert cap < 100.0
    assert abs(cap - _rising_turbined(cap)) < 1.0


def test_forebay_sits_at_the_initial_volume() -> None:
    """Cota ``60 + 0.1·V``: the forebay at 200 hm³ is 80 m, a 60 m head."""
    row = _plant_row(cota=(60.0, 0.1, 0.0, 0.0, 0.0), tipo_regulacao="M")
    limits = _limits({1: row}, initial_volumes={1: 200.0})
    assert limits.cap(1, 0, 0) == pytest.approx(100.0 * (60.0 / 80.0) ** 0.5)


def test_inflow_above_the_limit_sets_the_limit_at_that_outflow() -> None:
    limits = _limits(
        {1: _plant_row()}, tailrace={1: _RISING_TAILRACE}, inflows={1: 150.0}
    )
    assert limits.cap(1, 0, 0) == pytest.approx(_rising_turbined(150.0))


@pytest.mark.parametrize(
    ("inflow_m3s", "binds"),
    [
        pytest.param(700.0, False, id="room-absorbs-the-excess"),
        pytest.param(800.0, True, id="excess-beyond-the-room"),
    ],
)
def test_reservoir_room_is_taken_off_the_inflow(
    inflow_m3s: float, *, binds: bool
) -> None:
    """400 hm³ of room over the week absorbs about 661 m³/s."""
    row = _plant_row(tipo_regulacao="M")
    limits = _limits(
        {1: row},
        tailrace={1: _RISING_TAILRACE},
        inflows={1: inflow_m3s},
        initial_volumes={1: 100.0},
    )
    minimum_outflow = inflow_m3s - 400.0e6 / _WEEK_SECONDS
    unbound = _limits({1: row}, tailrace={1: _RISING_TAILRACE}).cap(1, 0, 0)
    expected = _rising_turbined(minimum_outflow) if binds else unbound
    assert limits.cap(1, 0, 0) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("upstream_regulation", "expected_outflow"),
    [
        pytest.param("D", 150.0, id="run-of-river-upstream-adds"),
        pytest.param("M", None, id="reservoir-upstream-holds-its-own"),
    ],
)
def test_upstream_inflow_reaches_the_plant_through_run_of_river_plants(
    upstream_regulation: str, expected_outflow: float | None
) -> None:
    rows = {
        1: _plant_row(name="UP", jusante=2, tipo_regulacao=upstream_regulation),
        2: _plant_row(name="DOWN"),
    }
    limits = _limits(
        rows,
        tailrace={1: _RISING_TAILRACE, 2: _RISING_TAILRACE},
        inflows={1: 100.0, 2: 50.0},
        initial_volumes={1: 500.0},
    )
    unbound = _limits({2: _plant_row()}, tailrace={2: _RISING_TAILRACE}).cap(2, 0, 0)
    expected = unbound if expected_outflow is None else _rising_turbined(150.0)
    assert limits.cap(2, 0, 0) == pytest.approx(expected)


def test_downstream_plant_level_selects_the_tailrace_family() -> None:
    """Families at 10 m and 30 m downstream levels; the plant below sits at
    20 m, halfway, so the tailrace is the families' mean, 25 m."""
    rows = {
        1: _plant_row(name="UP", jusante=2),
        2: _plant_row(name="DOWN", cota=(20.0, 0.0, 0.0, 0.0, 0.0)),
    }
    families = (_tailrace(15.0, reference_m=10.0), _tailrace(35.0, reference_m=30.0))
    limits = _limits(rows, tailrace={1: families})
    assert limits.cap(1, 0, 0) == pytest.approx(100.0 * (75.0 / 80.0) ** 0.5)


def test_generator_limit_binds_below_the_head_limited_flow() -> None:
    row = _plant_row(conjuntos=((1, 100.0, 40.0, 80.0),))
    limits = _limits({1: row})
    assert limits.cap(1, 0, 0) == pytest.approx(40.0 / 0.8)


@pytest.mark.parametrize(
    ("p_nom", "expected"),
    [
        pytest.param(1000.0, 100.0 * 0.5, id="head-limited-flow"),
        pytest.param(40.0, 40.0 * 0.5 / 0.8, id="generator-limit"),
    ],
)
def test_availability_factor_scales_both_limits(p_nom: float, expected: float) -> None:
    row = _plant_row(conjuntos=((1, 100.0, p_nom, 80.0),))
    limits = _limits({1: row}, factors={(1, 0): 0.5})
    assert limits.cap(1, 0, 0) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("p_nom", "expected"),
    [
        pytest.param(1000.0, 100.0, id="head-limited-flow"),
        pytest.param(40.0, 40.0 / 0.8, id="generator-limit"),
    ],
)
def test_fit_maximum_leaves_the_availability_factor_out(
    p_nom: float, expected: float
) -> None:
    row = _plant_row(conjuntos=((1, 100.0, p_nom, 80.0),))
    limits = _limits({1: row}, factors={(1, 0): 0.5})
    assert limits.fit_maxima[1] == pytest.approx((expected,) * 3)


def _itaipu_limits(p_nom_50: float, p_nom_60: float) -> TurbineLimits:
    """Itaipu with a 100 m³/s and an 80 m³/s set at the rated head (a 180 m³/s
    head-limited flow) and an ``MP × FD`` of 0.5 on its 60 Hz group."""
    row = _plant_row(
        name="ITAIPU",
        tipo_regulacao="S",
        conjuntos=((1, 100.0, p_nom_50, 80.0), (1, 80.0, p_nom_60, 80.0)),
    )
    return _limits({66: row}, factors={(66, 1): 0.5})


def test_itaipu_limit_is_plant_wide_and_its_groups_take_their_generator_limits() -> (
    None
):
    """The 60 Hz group's generator limit is 50 × 0.5 / 0.8 = 31.25 m³/s, but
    the plant still turbines its full 180 m³/s through the 50 Hz group."""
    limits = _itaipu_limits(1000.0, 50.0)
    assert limits.group_count(66) == 2
    assert limits.cap(66, 0, 0) == pytest.approx(180.0)
    assert limits.cap(66, 1, 0) == pytest.approx(50.0 * 0.5 / 0.8)
    assert limits.plant_caps[66][0] == pytest.approx(180.0)


def test_itaipu_plant_limit_takes_the_summed_generator_limits() -> None:
    limits = _itaipu_limits(40.0, 40.0)
    assert limits.plant_caps[66][0] == pytest.approx(40.0 / 0.8 + 20.0 / 0.8)


def test_itaipu_fit_maximum_is_its_derated_plant_limit() -> None:
    limits = _itaipu_limits(40.0, 40.0)
    assert limits.fit_maxima[66] == pytest.approx((40.0 / 0.8 + 20.0 / 0.8,) * 3)


def test_plant_limit_below_the_groups_sum_becomes_a_turbined_bound() -> None:
    limits = _itaipu_limits(1000.0, 50.0)
    contributions = plant_turbine_limit_contributions(
        limits, DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(66,))
    )
    assert [(c.entity_id, c.stage_id, c.block_id) for c in contributions] == [
        (0, s, None) for s in range(3)
    ]
    assert {(c.family, c.axis, c.lower) for c in contributions} == {
        ("hydro", "turbined", None)
    }
    assert [c.upper for c in contributions] == pytest.approx([180.0] * 3)


def test_plant_limit_at_the_groups_sum_adds_no_bound() -> None:
    """Both groups generator-bound: their caps already sum to the plant's."""
    limits = _itaipu_limits(40.0, 40.0)
    contributions = plant_turbine_limit_contributions(
        limits, DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(66,))
    )
    assert contributions == []


def test_unconverged_limit_warns_and_keeps_the_last_flow() -> None:
    """A tailrace rising 2 m per m³/s drowns the head at the rated flow, so
    the flow swings between zero and the rated flow."""
    with collect() as sink:
        limits = _limits({1: _plant_row()}, tailrace={1: (_tailrace(20.0, 2.0),)})
    (diagnostic,) = [d for d in sink if d.code == "decomp-turbine-limit-unconverged"]
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [[1, "PLANT", s] for s in (1, 2, 3)]
    assert limits.cap(1, 0, 0) == pytest.approx(100.0)


def test_unconverged_remediation_has_no_repo_internal_references() -> None:
    with collect() as sink:
        _limits({1: _plant_row()}, tailrace={1: (_tailrace(20.0, 2.0),)})
    (diagnostic,) = sink
    message = f"{diagnostic.summary} {diagnostic.remediation}"
    for leak in ("docs/", "plans/", "~/git", ".py", "TRACKED", "_"):
        assert leak not in message


def test_converged_limits_emit_no_diagnostic() -> None:
    with collect() as sink:
        _limits({1: _plant_row()}, tailrace={1: _RISING_TAILRACE})
    assert sink == []


def _gap_rows(row: dict) -> Sequence[Sequence[object]]:
    with collect() as sink:
        _limits({1: row})
    (diagnostic,) = [d for d in sink if d.code == "turbine-limit-data-missing"]
    assert diagnostic.table is not None
    return diagnostic.table.rows


def test_set_without_a_rated_head_warns_on_every_stage() -> None:
    row = _plant_row(conjuntos=((2, 50.0, 1000.0, 80.0), (1, 30.0, 500.0, math.nan)))
    assert _gap_rows(row) == [
        [
            1,
            "PLANT",
            "queda_nominal_conjunto_2",
            "rated flow, not head-corrected",
            "0-2",
        ]
    ]


def test_non_positive_productivity_warns_of_no_generator_limit() -> None:
    row = _plant_row() | {"produtibilidade_especifica": 0.0}
    assert _gap_rows(row) == [
        [1, "PLANT", "positive equivalent productivity", "no generator limit", "0-2"]
    ]


def test_plant_limit_sums_the_groups_unless_the_plant_cap_is_tighter() -> None:
    limits = TurbineLimits(
        {(66, 0): (100.0, 100.0), (66, 1): (80.0, 80.0), (1, 0): (50.0, 0.0)},
        {66: (150.0, 200.0)},
    )
    assert limits.plant_limit(66, 0) == 150.0
    assert limits.plant_limit(66, 1) == 180.0
    assert limits.plant_limit(1, 0) == 50.0
    assert limits.plant_limit(1, 1) == 0.0


def test_envelope_is_the_widest_stage_cap() -> None:
    limits = TurbineLimits({(1, 0): (90.0, 120.0, 110.0)})
    assert limits.envelope(1, 0) == 120.0


class _FakeDadger:
    def __init__(
        self,
        uh: pd.DataFrame,
        altefe: pd.DataFrame | None = None,
        mp: pd.DataFrame | None = None,
        fd: pd.DataFrame | None = None,
    ) -> None:
        self._uh, self._altefe, self._mp, self._fd = uh, altefe, mp, fd

    def uh(self, df: bool = False) -> pd.DataFrame:
        return self._uh

    def mp(self, df: bool = False) -> pd.DataFrame | None:
        return self._mp

    def fd(self, df: bool = False) -> pd.DataFrame | None:
        return self._fd

    def ac(
        self,
        codigo_usina: int | None = None,
        modificacao: type | None = None,
        df: bool = False,
    ) -> pd.DataFrame | None:
        return self._altefe if modificacao is ACALTEFE else None


def _uh_frame(codes: Sequence[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "codigo_usina": code,
                "volume_inicial": 50.0,
                "vazao_defluente_minima": None,
                "volume_morto_inicial": None,
            }
            for code in codes
        ]
    )


def _three_stage_case(dadger: _FakeDadger, hidr: pd.DataFrame) -> DecompCase:
    return make_decomp_case(
        Path("unused"), dadger=dadger, hidr=hidr, calendar=_calendar()
    )


def test_acaltefe_present_warns_and_proceeds(caplog: pytest.LogCaptureFixture) -> None:
    """An ``AC ALTEFE`` override is logged as a tracked gap (idecomp exposes no
    value accessor for it): the rated heads stay the ``hidr.dat`` ones."""
    hidr = _hidr_frame({1: _plant_row()})
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    dadger = _FakeDadger(uh=_uh_frame([1]), altefe=pd.DataFrame([{"codigo_usina": 1}]))

    with caplog.at_level(
        logging.WARNING, logger="novomodelo_bridge.decomp.converters.hydro.entity"
    ):
        convert_hydros(
            _three_stage_case(dadger, hidr),
            id_map,
            effective=EffectiveCadastro(base=hidr, n_stages=3, stage_varying={}),
            turbine_limits=flat_turbine_limits(id_map, 3),
        )

    assert any("ALTEFE" in r.message for r in caplog.records)


def _itaipu_split_case(fd: pd.DataFrame) -> tuple[DecompCase, EffectiveCadastro]:
    """Itaipu with two identical 1000 MW conjuntos, per-frequency ``MP`` rows
    at 1.0 and the given ``FD`` register."""
    hidr = _hidr_frame(
        {
            66: _plant_row(
                name="ITAIPU",
                tipo_regulacao="S",
                conjuntos=((1, 100.0, 1000.0, 80.0), (1, 100.0, 1000.0, 80.0)),
            )
        }
    )
    mp = pd.DataFrame(
        [
            {"codigo_usina": 66, "frequencia": frequency}
            | {f"manutencao_{stage}": 1.0 for stage in (1, 2, 3)}
            for frequency in (50.0, 60.0)
        ]
    )
    dadger = _FakeDadger(uh=_uh_frame([66]), mp=mp, fd=fd)
    effective = EffectiveCadastro(base=hidr, n_stages=3, stage_varying={})
    return _three_stage_case(dadger, hidr), effective


_ITAIPU_ID_MAP = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(66,))


def test_itaipu_fd_row_without_frequency_derates_both_groups() -> None:
    fd = pd.DataFrame(
        [
            {
                "codigo_usina": 66,
                "frequencia": None,
                "fator_1": 1.0,
                "fator_2": 1.0,
                "fator_3": 0.877,
            }
        ]
    )
    case, effective = _itaipu_split_case(fd)

    factors = stage_availability_factors(case, _ITAIPU_ID_MAP, effective=effective)
    values = convert_hydro_group_availability(
        case,
        _ITAIPU_ID_MAP,
        effective=effective,
        turbine_limits=flat_turbine_limits(_ITAIPU_ID_MAP, 3, group_counts={66: 2}),
    )

    for group_id in (0, 1):
        assert factors[(66, group_id, 2)] == pytest.approx(0.877)
        assert values[(0, group_id, 2)].max_generation_mw == pytest.approx(877.0)
        assert (0, group_id, 0) not in values


def test_itaipu_fd_mixing_plant_wide_and_frequency_rows_raises() -> None:
    fd = pd.DataFrame(
        [
            {"codigo_usina": 66, "frequencia": frequency}
            | {f"fator_{stage}": 1.0 for stage in (1, 2, 3)}
            for frequency in (None, 50.0)
        ]
    )
    case, effective = _itaipu_split_case(fd)

    with pytest.raises(ValueError, match="mixes a row without frequencia"):
        stage_availability_factors(case, _ITAIPU_ID_MAP, effective=effective)


def test_single_group_plant_without_mp_or_fd_has_unit_factors() -> None:
    hidr = _hidr_frame({1: _plant_row()})
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    case = _three_stage_case(_FakeDadger(uh=_uh_frame([1])), hidr)
    factors = stage_availability_factors(
        case,
        id_map,
        effective=EffectiveCadastro(base=hidr, n_stages=3, stage_varying={}),
    )
    assert factors == {(1, 0, s): 1.0 for s in range(3)}

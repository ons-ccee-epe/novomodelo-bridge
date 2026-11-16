"""Tests for the boundary-cut coupling resolution (``fcf/coupling.py``).

Tier 1, synthetic: a hand-built cascade and ``vazoes`` stand-in on the
``mar-26-rv2`` calendar shape (three weeks from 2026-03-14, then the rest of
April), coupled at April 2026.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from idecomp.decomp import Vazoes

from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.cadastro import (
    EffectiveCadastro,
    build_effective_cadastro,
)
from novomodelo_bridge.decomp.fcf.coupling import (
    BorrowedOwner,
    coupling_lag_offsets,
    lag_months,
    resolve_couplings,
    source_only_lag_inflows,
    terminal_shares,
)
from novomodelo_bridge.decomp.fcf.mapper import HydroCoupling
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import OperativeStage


def _stage(index: int, start: date, end: date) -> OperativeStage:
    return OperativeStage(
        index=index,
        start_date=start,
        end_date=end,
        season_id=start.month - 1,
        block_hours=((end - start).days * 24.0,),
    )


_CALENDAR = (
    _stage(0, date(2026, 3, 14), date(2026, 3, 21)),
    _stage(1, date(2026, 3, 21), date(2026, 3, 28)),
    _stage(2, date(2026, 3, 28), date(2026, 4, 4)),
    _stage(3, date(2026, 4, 4), date(2026, 5, 1)),
)
_MONTHS = lag_months(2026, 4)

#: code: (tipo_regulacao, posto, codigo_usina_jusante, volume_minimo)
_PLANTS = {
    1: ("M", 1, 2, 100.0),
    2: ("D", 2, 0, 0.0),
    147: ("M", 147, 0, 40.0),
    146: ("D", 146, 0, 0.0),
    109: ("S", 109, 50, 30.0),
    50: ("D", 50, 1, 0.0),  # in hidr.dat only, crossed by 109's walk
    400: ("D", 400, 147, 0.0),  # walks onto a complex component
    305: ("D", 305, 0, 0.0),  # reaches the sink: no owner
    306: ("D", 306, 307, 0.0),
    307: ("D", 307, 306, 0.0),  # 306 <-> 307 is a cycle: no owner
    901: ("D", 901, 0, 0.0),  # a cut plant not converted
    501: ("D", 501, 502, 0.0),  # an NW plant draining to 502
    502: ("D", 502, 0, 0.0),
}


def _effective(
    *,
    stage_varying: Mapping[tuple[int, str], tuple[float, ...]] | None = None,
    coupling_gauges: Mapping[int, tuple[int, ...]] | None = None,
) -> EffectiveCadastro:
    base = pd.DataFrame.from_dict(
        _PLANTS,
        orient="index",
        columns=["tipo_regulacao", "posto", "codigo_usina_jusante", "volume_minimo"],
    )
    base.index.name = "codigo_usina"
    return EffectiveCadastro(
        base=base,
        n_stages=len(_CALENDAR),
        stage_varying={} if stage_varying is None else stage_varying,
        coupling_gauges={} if coupling_gauges is None else coupling_gauges,
    )


def _id_map(*codes: int, dormant: tuple[int, ...] = ()) -> DecompIdMap:
    return DecompIdMap(
        bus_codes=(1,),
        bus_names=("SE",),
        hydro_codes=codes,
        dormant_hydro_codes=dormant,
    )


def test_lag_months_starts_at_the_cut_month_and_crosses_the_year() -> None:
    months = lag_months(2026, 2)

    assert months[:3] == ((2026, 2), (2026, 1), (2025, 12))
    assert len(months) == 12
    assert months[-1] == (2025, 3)


def test_resolve_couplings_follows_own_row_complex_and_downstream_plant() -> None:
    # The last stage's floor (120 hm³) is the one the cut attaches at.
    effective = _effective(
        stage_varying={(1, "volume_minimo"): (100.0, 100.0, 100.0, 120.0)}
    )
    id_map = _id_map(1, 2, 109, 146, 147, 305, 306, 400)

    resolution = resolve_couplings(
        (1, 2, 132, 901), id_map, effective, {132: [147, 146]}
    )

    assert resolution.couplings == (
        HydroCoupling(0, 0, 120.0),
        HydroCoupling(1, 1, None),
        HydroCoupling(2, 0, None),
        HydroCoupling(3, 2, None),
        HydroCoupling(4, 2, 40.0),
        HydroCoupling(7, 2, None),
    )
    assert resolution.borrowed == (
        BorrowedOwner(109, 1, via_complex=False),
        BorrowedOwner(146, 132, via_complex=True),
        BorrowedOwner(147, 132, via_complex=True),
        BorrowedOwner(400, 132, via_complex=False),
    )
    assert resolution.unowned == (305, 306)


def test_resolve_couplings_ignores_a_complex_absent_from_the_cut() -> None:
    resolution = resolve_couplings(
        (1,), _id_map(147), _effective(), {132: [147], 1: [999]}
    )

    assert resolution.couplings == ()
    assert resolution.unowned == (147,)


def test_an_nw_plant_binds_to_its_own_cut_row(decomp_mini_case: DecompCase) -> None:
    id_map = decomp_mini_case.id_map
    effective, _ = build_effective_cadastro(
        decomp_mini_case.dadger, decomp_mini_case.hidr, decomp_mini_case.calendar
    )
    estrela = 260

    resolution = resolve_couplings((estrela,), id_map, effective, {})

    assert estrela in id_map.dormant_hydro_codes
    assert HydroCoupling(id_map.hydro_id(estrela), 0, None) in resolution.couplings
    assert estrela not in resolution.unowned
    assert all(owner.plant_code != estrela for owner in resolution.borrowed)


def _trunk(column: str, values: tuple[float, ...]) -> pd.DataFrame:
    return pd.DataFrame({"estagio": [1, 2, 3], column: list(values)})


def _terminal_fan(column: str, value: float) -> pd.DataFrame:
    """A monthly scenario table holding only the terminal stage, whose rows
    the deterministic trunk never reads."""
    return pd.DataFrame({"estagio": [4], "cenario": [1], column: [value]})


def test_coupling_lag_offsets_month_weights_the_deterministic_stages() -> None:
    """Hydro 0 couples on posto 11 but operates on posto 1; hydro 1's postos
    agree and its offset vanishes."""
    effective = _effective(coupling_gauges={1: (11, 11, 11, 11)})
    vazoes = SimpleNamespace(
        previsoes_com_postos_artificiais=pd.DataFrame(
            {"estagio": [1, 2, 3], "11": [10.0, 20.0, 30.0], "2": [5.0, 5.0, 5.0]}
        ),
        previsoes=pd.DataFrame(
            {"estagio": [1, 2, 3], "1": [4.0, 4.0, 4.0], "2": [5.0, 5.0, 5.0]}
        ),
        cenarios_calculados_com_postos_artificiais=_terminal_fan("11", 900.0),
        cenarios_gerados=_terminal_fan("1", 0.0),
    )
    couplings = (HydroCoupling(0, 0, 0.0), HydroCoupling(1, 1, None))

    offsets = coupling_lag_offsets(
        vazoes, effective, _id_map(1, 2), _CALENDAR, _MONTHS, couplings
    )

    assert set(offsets) == {0}
    april, march, *rest = offsets[0]
    assert april == pytest.approx(3 / 30 * 26.0)
    assert march == pytest.approx((7 * 6.0 + 7 * 16.0 + 4 * 26.0) / 31)
    assert rest == [0.0] * 10


def test_coupling_lag_offsets_rejects_a_missing_posto() -> None:
    effective = _effective(coupling_gauges={1: (11, 11, 11, 11)})
    vazoes = SimpleNamespace(
        previsoes_com_postos_artificiais=_trunk("1", (1.0, 1.0, 1.0)),
        previsoes=_trunk("1", (1.0, 1.0, 1.0)),
        cenarios_calculados_com_postos_artificiais=_terminal_fan("1", 1.0),
        cenarios_gerados=_terminal_fan("1", 1.0),
    )

    with pytest.raises(ValueError, match="no posto 11 column"):
        coupling_lag_offsets(
            vazoes,
            effective,
            _id_map(1),
            _CALENDAR,
            _MONTHS,
            (HydroCoupling(0, 0, 0.0),),
        )


def test_source_only_lag_inflows_join_history_and_trunk_on_the_coupling_posto() -> None:
    vazoes = SimpleNamespace(
        observacoes_mensais=pd.DataFrame({"mes": [1, 2], "901": [100.0, 200.0]}),
        observacoes_semanais=pd.DataFrame({"semana": [1, 2], "901": [300.0, 400.0]}),
        previsoes_com_postos_artificiais=_trunk("901", (10.0, 20.0, 30.0)),
        cenarios_calculados_com_postos_artificiais=_terminal_fan("901", 900.0),
    )

    inflows = source_only_lag_inflows(
        vazoes, _effective(), (1, 901, 999), (1, 2), _CALENDAR, _MONTHS
    )

    assert set(inflows) == {1}  # plant 999 has no hidr.dat row, so no posto
    april, march, february, january, *rest = inflows[1]
    assert april == pytest.approx(3 / 30 * 30.0)
    assert march == pytest.approx(
        (6 * 300.0 + 7 * 400.0 + 7 * 10.0 + 7 * 20.0 + 4 * 30.0) / 31
    )
    # February: 2026-02-01..28 from the monthly row, clipped where the weeks start.
    assert february == pytest.approx((27 * 200.0 + 1 * 300.0) / 28)
    assert january == pytest.approx(100.0)
    assert rest == [0.0] * 8


def test_terminal_shares_lists_differing_and_source_only_plants() -> None:
    effective = _effective(coupling_gauges={1: (11, 11, 11, 11)})
    vazoes = SimpleNamespace(
        cenarios_calculados_com_postos_artificiais=pd.DataFrame(
            {
                "estagio": [4, 4],
                "cenario": [2, 1],
                "11": [60.0, 50.0],
                "2": [8.0, 7.0],
                "901": [3.0, -7.0],
            }
        ),
        cenarios_gerados=pd.DataFrame(
            {"estagio": [4, 4], "cenario": [1, 2], "1": [45.0, 61.0], "2": [7.0, 8.0]}
        ),
    )
    couplings = (HydroCoupling(0, 0, 0.0), HydroCoupling(1, 1, None))

    shares = terminal_shares(
        vazoes, effective, _id_map(1, 2), couplings, (901,), _CALENDAR
    )

    assert [(s.plant_code, s.coupling_posto, s.operation_posto) for s in shares] == [
        (1, 11, 1),
        (901, 901, None),
    ]
    assert [s.max_gap_m3s for s in shares] == pytest.approx([5.0, 7.0])


def _split_group_vazoes(*, terminal_gaps: tuple[float, float]) -> SimpleNamespace:
    """NW plant 501 (posto 501) above plant 502 (posto 502).

    The operation tables give 502 the whole group's flow: its coupling inflow
    plus 501's, and *terminal_gaps* on the two terminal scenarios. 501's own
    operation value is 0.
    """
    c501, c502 = (10.0, 20.0, 30.0), (5.0, 5.0, 5.0)
    fan501, fan502 = (100.0, 150.0), (40.0, 55.0)
    fan = pd.DataFrame({"estagio": [4, 4], "cenario": [1, 2]})
    return SimpleNamespace(
        previsoes_com_postos_artificiais=pd.DataFrame(
            {"estagio": [1, 2, 3], "501": c501, "502": c502}
        ),
        previsoes=pd.DataFrame(
            {
                "estagio": [1, 2, 3],
                "501": [0.0] * 3,
                "502": [a + b for a, b in zip(c501, c502, strict=True)],
            }
        ),
        cenarios_calculados_com_postos_artificiais=fan.assign(
            **{"501": fan501, "502": fan502}
        ),
        cenarios_gerados=fan.assign(
            **{
                "501": [0.0, 0.0],
                "502": [
                    a + b + gap
                    for a, b, gap in zip(fan501, fan502, terminal_gaps, strict=True)
                ],
            }
        ),
    )


_SPLIT_COUPLINGS = (HydroCoupling(0, 0, None), HydroCoupling(1, 1, None))


def test_a_split_group_carries_no_lag_offset() -> None:
    offsets = coupling_lag_offsets(
        _split_group_vazoes(terminal_gaps=(1.0, 0.0)),
        _effective(),
        _id_map(501, 502, dormant=(501,)),
        _CALENDAR,
        _MONTHS,
        _SPLIT_COUPLINGS,
    )

    assert offsets == {}


def test_a_fallback_group_keeps_its_lag_offset() -> None:
    offsets = coupling_lag_offsets(
        _split_group_vazoes(terminal_gaps=(0.0, 1.5)),
        _effective(),
        _id_map(501, 502, dormant=(501,)),
        _CALENDAR,
        _MONTHS,
        _SPLIT_COUPLINGS,
    )

    assert set(offsets) == {0, 1}
    assert offsets[0][0] == pytest.approx(3 / 30 * 30.0)


def test_a_split_group_is_left_out_of_the_terminal_shares() -> None:
    """The terminal gap of 1 m³/s is within the guard, but not within the
    exact-match test the other plants take."""
    shares = terminal_shares(
        _split_group_vazoes(terminal_gaps=(1.0, 0.0)),
        _effective(),
        _id_map(501, 502, dormant=(501,)),
        _SPLIT_COUPLINGS,
        (),
        _CALENDAR,
    )

    assert shares == ()


def test_a_fallback_group_keeps_its_terminal_share() -> None:
    shares = terminal_shares(
        _split_group_vazoes(terminal_gaps=(0.0, 1.5)),
        _effective(),
        _id_map(501, 502, dormant=(501,)),
        _SPLIT_COUPLINGS,
        (),
        _CALENDAR,
    )

    assert [(s.plant_code, s.coupling_posto, s.operation_posto) for s in shares] == [
        (501, 501, 501),
        (502, 502, 502),
    ]
    assert [s.max_gap_m3s for s in shares] == pytest.approx([150.0, 151.5])


def test_decomp_mini_couples_estrela_and_salto_without_offset_or_share(
    decomp_mini_case: DecompCase,
) -> None:
    id_map = decomp_mini_case.id_map
    effective, _ = build_effective_cadastro(
        decomp_mini_case.dadger, decomp_mini_case.hidr, decomp_mini_case.calendar
    )
    vazoes = Vazoes.read(str(decomp_mini_case.files.vazoes))
    calendar = decomp_mini_case.calendar
    resolution = resolve_couplings(id_map.hydro_codes, id_map, effective, {})
    months = lag_months(calendar[-1].start_date.year, calendar[-1].start_date.month)
    estrela, salto = id_map.hydro_id(260), id_map.hydro_id(262)

    offsets = coupling_lag_offsets(
        vazoes, effective, id_map, calendar, months, resolution.couplings
    )
    shares = terminal_shares(
        vazoes, effective, id_map, resolution.couplings, (), calendar
    )

    assert estrela not in offsets
    assert salto not in offsets
    assert {share.plant_code for share in shares}.isdisjoint({260, 262})


#: Monthly stages only: January is the deterministic trunk, February the fan.
_MONTHLY_CALENDAR = (
    _stage(0, date(2026, 1, 1), date(2026, 2, 1)),
    _stage(1, date(2026, 2, 1), date(2026, 3, 1)),
)


def _monthly_vazoes() -> SimpleNamespace:
    """No weekly forecasts; January's single scenario and February's two."""
    empty = pd.DataFrame({"estagio": [], "11": [], "1": []})
    return SimpleNamespace(
        previsoes_com_postos_artificiais=empty,
        previsoes=empty,
        cenarios_calculados_com_postos_artificiais=pd.DataFrame(
            {"estagio": [1, 2, 2], "cenario": [1, 1, 2], "11": [30.0, 60.0, 70.0]}
        ),
        cenarios_gerados=pd.DataFrame(
            {"estagio": [1, 2, 2], "cenario": [1, 1, 2], "1": [10.0, 60.0, 70.0]}
        ),
    )


def test_coupling_lag_offsets_read_a_monthly_trunk_stage() -> None:
    effective = _effective(coupling_gauges={1: (11, 11)})

    offsets = coupling_lag_offsets(
        _monthly_vazoes(),
        effective,
        _id_map(1),
        _MONTHLY_CALENDAR,
        lag_months(2026, 2),
        (HydroCoupling(0, 0, 0.0),),
    )

    february, january, *rest = offsets[0]
    assert february == 0.0
    assert january == pytest.approx(20.0)
    assert rest == [0.0] * 10


def test_terminal_shares_skip_a_monthly_trunk_stage() -> None:
    effective = _effective(coupling_gauges={1: (11, 11)})

    shares = terminal_shares(
        _monthly_vazoes(),
        effective,
        _id_map(1),
        (HydroCoupling(0, 0, 0.0),),
        (),
        _MONTHLY_CALENDAR,
    )

    assert shares == ()

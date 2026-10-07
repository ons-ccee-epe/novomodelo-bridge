"""Tests for the DECOMP-side per-stage storage-bounds emitter.

Pins ``storage_envelope`` (the outer per-stage operating-range helper both
this emitter and the entity ``reservoir`` block share) and
``convert_storage_bounds`` (the sparse per-stage override emitter) against a
synthetic ``EffectiveCadastro`` — no real deck, no dadger.

Rule A (user-confirmed 2026-08-07): a stage's effective ``(volume_minimo,
volume_maximo)`` is compared to the plant's outer envelope
(``storage_envelope``), not to its pre-override base scalar. A stage equal
to the envelope emits no contribution and simply inherits it from the entity
``reservoir`` block; a stage that differs from the envelope — even by being
*narrower* than it — emits an override contribution.

``convert_storage_bounds`` returns ``list[BoundContribution]`` (the E2
accumulator's contribution type) instead of a ``pa.Table`` — the accumulator,
not this emitter, fans contributions into the ``hydro_bounds`` parquet rows.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.bounds import (
    convert_irrigation_withdrawal,
    convert_storage_bounds,
    convert_volume_espera_bounds,
)
from novomodelo_bridge.decomp.converters.cadastro import (
    EffectiveCadastro,
    storage_envelope,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import OperativeStage
from tests.conftest import make_decomp_case


def _hidr_frame() -> pd.DataFrame:
    df = pd.DataFrame({1: {"volume_minimo": 20.0, "volume_maximo": 100.0}}).T
    df.index.name = "codigo_usina"
    return df


@pytest.fixture
def effective() -> EffectiveCadastro:
    """Plant 1: flat ``volume_minimo``; ``volume_maximo`` raised at stage 2."""
    return EffectiveCadastro(
        base=_hidr_frame(),
        n_stages=3,
        stage_varying={(1, "volume_maximo"): (100.0, 100.0, 250.0)},
    )


@pytest.fixture
def id_map() -> DecompIdMap:
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))


@pytest.fixture
def calendar() -> list[OperativeStage]:
    """Three placeholder stages — only their count is used by the emitter."""
    return [
        OperativeStage(
            index=i,
            start_date=date(2026, 7, 4),
            end_date=date(2026, 7, 11),
            season_id=6,
            block_hours=(168.0,),
        )
        for i in range(3)
    ]


def _case(dadger: object, calendar: list[OperativeStage]) -> DecompCase:
    return make_decomp_case(Path("unused"), dadger=dadger, calendar=calendar)


def test_storage_envelope_returns_min_minimo_max_maximo(
    effective: EffectiveCadastro,
) -> None:
    assert storage_envelope(effective, 1) == (20.0, 250.0)


def test_convert_storage_bounds_emits_one_contribution_per_stage_that_differs(
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: list[OperativeStage],
) -> None:
    """Stages 0-1 (ceiling 100.0) differ from the envelope (20.0, 250.0);
    stage 2 equals the envelope exactly and emits no contribution.
    """
    contributions = convert_storage_bounds(
        _case(None, calendar), id_map, effective=effective
    )

    assert [c.stage_id for c in contributions] == [0, 1]
    for contribution in contributions:
        assert contribution.family == "hydro"
        assert contribution.entity_id == 0
        assert contribution.block_id is None
        assert contribution.axis == "storage"
        assert contribution.lower == 20.0
        assert contribution.upper == 100.0


def test_convert_storage_bounds_no_stage_varying_volumes_emits_no_contributions(
    id_map: DecompIdMap,
    calendar: list[OperativeStage],
) -> None:
    no_override = EffectiveCadastro(base=_hidr_frame(), n_stages=3, stage_varying={})
    contributions = convert_storage_bounds(
        _case(None, calendar), id_map, effective=no_override
    )
    assert contributions == []


# ---------------------------------------------------------------------------
# VE (volume de espera) — flood-control hard max-storage ceiling
# ---------------------------------------------------------------------------


def _dadger_with_ve(rows: list[dict] | None):
    """Fake ``dadger`` exposing ``.ve(df=True)`` -> a VE frame (or ``None``)."""
    frame = None if rows is None else pd.DataFrame(rows)
    return SimpleNamespace(ve=lambda df=True: frame)


def test_ve_emits_percent_of_useful_upper_only(
    effective: EffectiveCadastro, id_map: DecompIdMap, calendar: list[OperativeStage]
) -> None:
    # plant 1 envelope = (20, 250) per the fixtures (volume_maximo raised to 250
    # at stage 2), useful = 230. VE 50% at stage 0 -> ceiling 20 + 0.5*230 = 135.
    dadger = _dadger_with_ve(
        [{"codigo_usina": 1, "volume_1": 50.0, "volume_2": 100.0, "volume_3": None}]
    )
    contribs = convert_volume_espera_bounds(
        _case(dadger, calendar), id_map, effective=effective
    )
    # stage 0: 50% tightens (135 < 250) -> emitted; stage 1: 100% is a no-op
    # (ceiling == env_max); stage 2: blank -> skipped.
    assert len(contribs) == 1
    c = contribs[0]
    assert (c.family, c.axis, c.entity_id, c.stage_id, c.block_id) == (
        "hydro",
        "storage",
        0,
        0,
        None,
    )
    assert c.lower is None
    assert c.upper == 20.0 + 0.5 * 230.0
    assert c.contributor == "VE"


def test_ve_full_percent_is_noop(
    effective: EffectiveCadastro, id_map: DecompIdMap, calendar: list[OperativeStage]
) -> None:
    dadger = _dadger_with_ve(
        [{"codigo_usina": 1, "volume_1": 100.0, "volume_2": 100.0, "volume_3": 100.0}]
    )
    assert (
        convert_volume_espera_bounds(
            _case(dadger, calendar), id_map, effective=effective
        )
        == []
    )


def test_ve_absent_register_returns_empty(
    effective: EffectiveCadastro, id_map: DecompIdMap, calendar: list[OperativeStage]
) -> None:
    assert (
        convert_volume_espera_bounds(
            _case(_dadger_with_ve(None), calendar), id_map, effective=effective
        )
        == []
    )


def test_ve_unoperated_plant_skipped(
    effective: EffectiveCadastro, id_map: DecompIdMap, calendar: list[OperativeStage]
) -> None:
    # plant 999 is not in id_map.hydro_codes -> contributes nothing.
    dadger = _dadger_with_ve([{"codigo_usina": 999, "volume_1": 30.0}])
    assert (
        convert_volume_espera_bounds(
            _case(dadger, calendar), id_map, effective=effective
        )
        == []
    )


# ---------------------------------------------------------------------------
# TI (irrigation) — consumptive water withdrawal
# ---------------------------------------------------------------------------


def _dadger_with_ti(rows: list[dict] | None):
    """Fake ``dadger`` exposing ``.ti(df=True)`` -> a TI frame (or ``None``)."""
    frame = None if rows is None else pd.DataFrame(rows)
    return SimpleNamespace(ti=lambda df=True: frame)


def test_ti_emits_per_stage_withdrawal(
    id_map: DecompIdMap, calendar: list[OperativeStage]
) -> None:
    # calendar has 3 stages; taxa_1..3 map to stages 0..2 (plant 1 -> hydro_id 0).
    dadger = _dadger_with_ti(
        [{"codigo_usina": 1, "taxa_1": 5.0, "taxa_2": 7.0, "taxa_3": 9.0}]
    )
    table = convert_irrigation_withdrawal(_case(dadger, calendar), id_map)
    assert table is not None
    got = {
        (h, s): v
        for h, s, v in zip(
            table.column("hydro_id").to_pylist(),
            table.column("stage_id").to_pylist(),
            table.column("water_withdrawal_m3s").to_pylist(),
            strict=True,
        )
    }
    assert got == {(0, 0): 5.0, (0, 1): 7.0, (0, 2): 9.0}


def test_ti_zero_and_blank_skipped(
    id_map: DecompIdMap, calendar: list[OperativeStage]
) -> None:
    # taxa 0.0 (no withdrawal) and NaN both contribute no row.
    dadger = _dadger_with_ti(
        [{"codigo_usina": 1, "taxa_1": 0.0, "taxa_2": None, "taxa_3": 4.0}]
    )
    table = convert_irrigation_withdrawal(_case(dadger, calendar), id_map)
    assert table is not None
    assert table.column("stage_id").to_pylist() == [2]
    assert table.column("water_withdrawal_m3s").to_pylist() == [4.0]


def test_ti_carry_forward_last_rate_for_extra_stages(
    id_map: DecompIdMap, calendar: list[OperativeStage]
) -> None:
    # Only two taxa columns for a 3-stage calendar -> stage 2 repeats taxa_2.
    dadger = _dadger_with_ti([{"codigo_usina": 1, "taxa_1": 5.0, "taxa_2": 7.0}])
    table = convert_irrigation_withdrawal(_case(dadger, calendar), id_map)
    assert table is not None
    assert table.column("water_withdrawal_m3s").to_pylist() == [5.0, 7.0, 7.0]


def test_ti_absent_returns_none(
    id_map: DecompIdMap, calendar: list[OperativeStage]
) -> None:
    assert (
        convert_irrigation_withdrawal(_case(_dadger_with_ti(None), calendar), id_map)
        is None
    )


def test_ti_unoperated_plant_returns_none(
    id_map: DecompIdMap, calendar: list[OperativeStage]
) -> None:
    # plant 999 is not operated -> no rows -> None.
    dadger = _dadger_with_ti([{"codigo_usina": 999, "taxa_1": 5.0}])
    assert convert_irrigation_withdrawal(_case(dadger, calendar), id_map) is None


# EZ (percentual máximo do volume útil para acoplamento) is deliberately NOT
# converted as an operational max-storage cap: it is an FCF-coupling formulation
# detail, not an operational ceiling. Verified against dec_oper_usih on the
# mar-26 deck -- DECOMP holds SERRA MESA at VutilFim% = 64.14 % at the coupling,
# well above its EZ = 55 %, so the register does not bind the operation.

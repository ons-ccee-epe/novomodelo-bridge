"""Tier-1 tests for the RHE stored-energy generic-constraint emitter.

Synthetic ``ConstraintRecord``/``ConstraintTerm``/``StageBounds``/
``EffectiveCadastro``/``DecompIdMap``/``OperativeStage`` only — no deck, no
``example/`` read, no ``import novomodelo``. One test per acceptance criterion:
absolute (``tipo_limite=1``) round-trip, percentage (``tipo_limite=2``) RHS,
two-reservoir cascade override, run-of-river exclusion, weekly-regulating
(``"S"``) reservoir participation with the integrated ρ, no-storage
skip+WARNING (under ``diagnostics.collect()``), weekly-vs-monthly
energy-factor scaling, and the negative-CM sign path.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.constraint_registers import (
    ConstraintRecord,
    ConstraintTerm,
    HeMeta,
    StageBounds,
)
from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
from novomodelo_bridge.decomp.converters.constraints import emit_rhe_generics
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import OperativeStage
from tests.conftest import make_decomp_case

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _stage(index: int, total_hours: float) -> OperativeStage:
    """A single-block stage carrying exactly *total_hours* of duration."""
    return OperativeStage(
        index=index,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 2, 1),
        season_id=0,
        block_hours=(total_hours,),
    )


def _case(calendar: list[OperativeStage]) -> DecompCase:
    return make_decomp_case(Path("unused"), calendar=calendar)


def _census(*records: ConstraintRecord):
    """A synthetic census whose generic slice is exactly *records*."""
    from novomodelo_bridge.decomp.constraint_registers import ConstraintCensus

    return ConstraintCensus(by_family={}, to_bounds=(), to_generic=tuple(records))


def _hydro_row(
    *,
    tipo_regulacao: str,
    a0: float = 100.0,
    canal_fuga_medio: float = 0.0,
    produtibilidade_especifica: float = 0.01,
    volume_minimo: float = 0.0,
    volume_maximo: float = 1000.0,
    volume_referencia: float = 500.0,
    codigo_usina_jusante: int = 0,
) -> dict[str, object]:
    """One synthetic ``hidr``-shaped row, extended with the productivity columns.

    A degenerate (flat) cota polynomial by default (``a0`` only, ``a1..a4 ==
    0``), no tailrace, no hydraulic loss — the plant's productivity then
    reduces to ``produtibilidade_especifica * a0`` for either the
    integrated (``"M"``/``"S"``) or point (``"D"``) productivity branch,
    since the head is constant everywhere.
    """
    row: dict[str, object] = {
        "tipo_regulacao": tipo_regulacao,
        "canal_fuga_medio": canal_fuga_medio,
        "volume_minimo": volume_minimo,
        "volume_maximo": volume_maximo,
        "volume_referencia": volume_referencia,
        "tipo_perda": 0,
        "perdas": 0.0,
        "produtibilidade_especifica": produtibilidade_especifica,
        "codigo_usina_jusante": codigo_usina_jusante,
        "a0_volume_cota": a0,
    }
    for i in range(1, 5):
        row[f"a{i}_volume_cota"] = 0.0
    return row


def _hidr_frame(rows: dict[int, dict[str, object]]) -> pd.DataFrame:
    """A synthetic ``hidr``-shaped frame, indexed by ``codigo_usina``."""
    df = pd.DataFrame(rows).T
    df.index.name = "codigo_usina"
    return df


def _he_record(
    *,
    constraint_id: int,
    ree_code: int,
    coefficient: float,
    bounds: dict[int, StageBounds],
    tipo_limite: int | None,
    valor_penalidade: float | None,
) -> ConstraintRecord:
    return ConstraintRecord(
        family="HE",
        constraint_id=constraint_id,
        stage_start=min(bounds),
        stage_end=max(bounds),
        terms=(
            ConstraintTerm(code=ree_code, coefficient=coefficient, variable="energy"),
        ),
        bounds=bounds,
        per_block=False,
        tipo_limite=tipo_limite,
        he_meta=HeMeta(valor_penalidade=valor_penalidade),
    )


# ---------------------------------------------------------------------------
# Single-REE, single monthly reservoir, tipo_limite=1 (absolute)
# ---------------------------------------------------------------------------


def test_emit_rhe_generics_absolute_round_trip() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5,))
    effective = EffectiveCadastro(
        base=_hidr_frame({5: _hydro_row(tipo_regulacao="M")}),
        n_stages=1,
        stage_varying={},
    )
    record = _he_record(
        constraint_id=100,
        ree_code=1,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(None,))},
        tipo_limite=1,
        valor_penalidade=250.0,
    )
    census = _census(record)
    calendar = [_stage(0, 730.0)]

    result = emit_rhe_generics(
        _case(calendar),
        id_map,
        census=census,
        effective=effective,
        hydro_to_ree={5: 1},
    )

    assert result.result is not None
    assert len(result.result.constraints) == 1
    constraint = result.result.constraints[0]
    assert constraint["expression"] == "@rho_acum_h0 * hydro_storage(0)"
    assert set(constraint) == {"id", "name", "description", "expression", "slack"}
    assert constraint["slack"] == {"enabled": True, "penalty": 250.0}

    rows = result.result.bounds.to_pylist()
    assert len(rows) == 1
    assert rows[0]["bound_lower"] == 50.0
    assert rows[0]["bound_upper"] is None
    assert rows[0]["block_id"] is None
    assert rows[0]["stage_id"] == 0

    expected_rho = 0.01 * 100 / (3600.0 * 730.0 / 1e6)
    assert result.rho_acum_overrides[0][0] == pytest.approx(expected_rho)


# ---------------------------------------------------------------------------
# Same record, tipo_limite=2 (percentage of rho_acum * volume_maximo)
# ---------------------------------------------------------------------------


def test_emit_rhe_generics_percentage_rhs() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5,))
    effective = EffectiveCadastro(
        base=_hidr_frame({5: _hydro_row(tipo_regulacao="M")}),
        n_stages=1,
        stage_varying={},
    )
    record = _he_record(
        constraint_id=101,
        ree_code=1,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(20.0,), upper=(None,))},
        tipo_limite=2,
        valor_penalidade=250.0,
    )
    census = _census(record)
    calendar = [_stage(0, 730.0)]

    result = emit_rhe_generics(
        _case(calendar),
        id_map,
        census=census,
        effective=effective,
        hydro_to_ree={5: 1},
    )

    assert result.result is not None
    rows = result.result.bounds.to_pylist()
    assert len(rows) == 1
    expected_rho = 0.01 * 100 / (3600.0 * 730.0 / 1e6)
    expected_bound = 0.20 * expected_rho * 1000.0
    assert rows[0]["bound_lower"] == pytest.approx(expected_bound)
    assert rows[0]["bound_upper"] is None


# ---------------------------------------------------------------------------
# Two-reservoir cascade (code 5 -> code 6 -> sink), both "M"
# ---------------------------------------------------------------------------


def test_emit_rhe_generics_two_reservoir_cascade_override() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5, 6))
    # produtibilidade_especifica=1.0, a0=1.0, no tailrace/loss -> own rho == 1.0
    # MW/(m^3/s) for both plants (constant head, either regulation branch).
    effective = EffectiveCadastro(
        base=_hidr_frame(
            {
                5: _hydro_row(
                    tipo_regulacao="M",
                    a0=1.0,
                    produtibilidade_especifica=1.0,
                    codigo_usina_jusante=6,
                ),
                6: _hydro_row(
                    tipo_regulacao="M",
                    a0=1.0,
                    produtibilidade_especifica=1.0,
                    codigo_usina_jusante=0,
                ),
            }
        ),
        n_stages=1,
        stage_varying={},
    )
    record = _he_record(
        constraint_id=102,
        ree_code=1,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(None,))},
        tipo_limite=1,
        valor_penalidade=250.0,
    )
    census = _census(record)
    calendar = [_stage(0, 730.0)]

    result = emit_rhe_generics(
        _case(calendar),
        id_map,
        census=census,
        effective=effective,
        hydro_to_ree={5: 1, 6: 1},
    )

    assert result.result is not None
    constraint = result.result.constraints[0]
    assert (
        constraint["expression"]
        == "@rho_acum_h0 * hydro_storage(0) + @rho_acum_h1 * hydro_storage(1)"
    )

    factor = 3600.0 * 730.0 / 1e6
    assert result.rho_acum_overrides[0][0] == pytest.approx(2.0 / factor)  # 5 acc. 6
    assert result.rho_acum_overrides[1][0] == pytest.approx(1.0 / factor)


# ---------------------------------------------------------------------------
# Run-of-river ("D") plant excluded from the stored-energy sum
# ---------------------------------------------------------------------------


def test_emit_rhe_generics_run_of_river_excluded() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5, 7))
    effective = EffectiveCadastro(
        base=_hidr_frame(
            {
                5: _hydro_row(tipo_regulacao="M"),
                7: _hydro_row(tipo_regulacao="D"),
            }
        ),
        n_stages=1,
        stage_varying={},
    )
    record = _he_record(
        constraint_id=103,
        ree_code=1,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(None,))},
        tipo_limite=1,
        valor_penalidade=250.0,
    )
    census = _census(record)
    calendar = [_stage(0, 730.0)]

    result = emit_rhe_generics(
        _case(calendar),
        id_map,
        census=census,
        effective=effective,
        hydro_to_ree={5: 1, 7: 1},
    )

    assert result.result is not None
    constraint = result.result.constraints[0]
    assert constraint["expression"] == "@rho_acum_h0 * hydro_storage(0)"
    assert "hydro_storage(1)" not in constraint["expression"]
    assert 1 not in result.rho_acum_overrides


# ---------------------------------------------------------------------------
# Weekly-regulating ("S") plant is a reservoir: participates, integrated rho
# ---------------------------------------------------------------------------


def test_emit_rhe_generics_weekly_regulating_reservoir_uses_integrated_rho() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5,))
    row = _hydro_row(tipo_regulacao="S", volume_referencia=200.0)
    # A linear cota polynomial separates the integrated mean head over
    # [0, 1000] (100 + 0.01 * 500 = 105 m) from the point head at
    # volume_referencia (100 + 0.01 * 200 = 102 m).
    row["a1_volume_cota"] = 0.01
    effective = EffectiveCadastro(
        base=_hidr_frame({5: row}), n_stages=1, stage_varying={}
    )
    record = _he_record(
        constraint_id=105,
        ree_code=1,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(None,))},
        tipo_limite=1,
        valor_penalidade=250.0,
    )
    calendar = [_stage(0, 730.0)]

    result = emit_rhe_generics(
        _case(calendar),
        id_map,
        census=_census(record),
        effective=effective,
        hydro_to_ree={5: 1},
    )

    assert result.result is not None
    assert result.result.constraints[0]["expression"] == (
        "@rho_acum_h0 * hydro_storage(0)"
    )
    expected_rho = 0.01 * 105.0 / (3600.0 * 730.0 / 1e6)
    assert result.rho_acum_overrides[0][0] == pytest.approx(expected_rho)


# ---------------------------------------------------------------------------
# No cadastred stored-energy reservoir -> skip + WARNING
# ---------------------------------------------------------------------------


def test_emit_rhe_generics_no_storage_plants_skips_and_warns() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(7,))
    effective = EffectiveCadastro(
        base=_hidr_frame({7: _hydro_row(tipo_regulacao="D")}),
        n_stages=1,
        stage_varying={},
    )
    record = _he_record(
        constraint_id=104,
        ree_code=1,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(None,))},
        tipo_limite=1,
        valor_penalidade=250.0,
    )
    census = _census(record)
    calendar = [_stage(0, 730.0)]

    with dx.collect() as collected:
        result = emit_rhe_generics(
            _case(calendar),
            id_map,
            census=census,
            effective=effective,
            hydro_to_ree={7: 1},
        )

    assert result.result is None
    assert result.rho_acum_overrides == {}
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhe-no-storage-plants"


def test_emit_rhe_generics_unmapped_ree_skips_and_warns() -> None:
    """A REE with no members at all in ``hydro_to_ree`` is the other
    no-storage-plants path (never resolves to any participating code)."""
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    effective = EffectiveCadastro(base=_hidr_frame({}), n_stages=1, stage_varying={})
    record = _he_record(
        constraint_id=105,
        ree_code=9,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(None,))},
        tipo_limite=1,
        valor_penalidade=250.0,
    )
    census = _census(record)
    calendar = [_stage(0, 730.0)]

    with dx.collect() as collected:
        result = emit_rhe_generics(
            _case(calendar),
            id_map,
            census=census,
            effective=effective,
            hydro_to_ree={},
        )

    assert result.result is None
    assert result.rho_acum_overrides == {}
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhe-no-storage-plants"


# ---------------------------------------------------------------------------
# Weekly vs monthly energy-factor scaling
# ---------------------------------------------------------------------------


def test_emit_rhe_generics_energy_factor_scales_with_stage_hours() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5,))
    effective = EffectiveCadastro(
        base=_hidr_frame({5: _hydro_row(tipo_regulacao="M")}),
        n_stages=1,
        stage_varying={},
    )
    record = _he_record(
        constraint_id=106,
        ree_code=1,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(None,))},
        tipo_limite=1,
        valor_penalidade=250.0,
    )
    census = _census(record)

    weekly_result = emit_rhe_generics(
        _case([_stage(0, 168.0)]),
        id_map,
        census=_census(record),
        effective=effective,
        hydro_to_ree={5: 1},
    )
    monthly_result = emit_rhe_generics(
        _case([_stage(0, 730.0)]),
        id_map,
        census=census,
        effective=effective,
        hydro_to_ree={5: 1},
    )

    override_weekly = weekly_result.rho_acum_overrides[0][0]
    override_monthly = monthly_result.rho_acum_overrides[0][0]
    assert override_weekly == pytest.approx(override_monthly * 730.0 / 168.0)


# ---------------------------------------------------------------------------
# Negative CM sign on the second REE term -> subtraction join
# ---------------------------------------------------------------------------


def test_emit_rhe_generics_negative_cm_sign_subtracts() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5, 6))
    effective = EffectiveCadastro(
        base=_hidr_frame(
            {
                5: _hydro_row(tipo_regulacao="M"),
                6: _hydro_row(tipo_regulacao="M"),
            }
        ),
        n_stages=1,
        stage_varying={},
    )
    record = ConstraintRecord(
        family="HE",
        constraint_id=107,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(code=1, coefficient=1.0, variable="energy"),
            ConstraintTerm(code=2, coefficient=-1.0, variable="energy"),
        ),
        bounds={0: StageBounds(lower=(50.0,), upper=(None,))},
        per_block=False,
        tipo_limite=1,
        he_meta=HeMeta(valor_penalidade=250.0),
    )
    census = _census(record)
    calendar = [_stage(0, 730.0)]

    result = emit_rhe_generics(
        _case(calendar),
        id_map,
        census=census,
        effective=effective,
        hydro_to_ree={5: 1, 6: 2},
    )

    assert result.result is not None
    constraint = result.result.constraints[0]
    assert (
        constraint["expression"]
        == "@rho_acum_h0 * hydro_storage(0) - @rho_acum_h1 * hydro_storage(1)"
    )


# ---------------------------------------------------------------------------
# Additional coverage: default-penalty and unknown-tipo_limite fallbacks
# ---------------------------------------------------------------------------


def test_emit_rhe_generics_non_positive_penalty_falls_back_and_warns() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5,))
    effective = EffectiveCadastro(
        base=_hidr_frame({5: _hydro_row(tipo_regulacao="M")}),
        n_stages=1,
        stage_varying={},
    )
    record = _he_record(
        constraint_id=108,
        ree_code=1,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(None,))},
        tipo_limite=1,
        valor_penalidade=0.0,
    )
    census = _census(record)
    calendar = [_stage(0, 730.0)]

    with dx.collect() as collected:
        result = emit_rhe_generics(
            _case(calendar),
            id_map,
            census=census,
            effective=effective,
            hydro_to_ree={5: 1},
        )

    assert result.result is not None
    assert result.result.constraints[0]["slack"] == {"enabled": True, "penalty": 1000.0}
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhe-default-penalty"


def test_emit_rhe_generics_unknown_tipo_limite_treated_as_absolute_and_warns() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5,))
    effective = EffectiveCadastro(
        base=_hidr_frame({5: _hydro_row(tipo_regulacao="M")}),
        n_stages=1,
        stage_varying={},
    )
    record = _he_record(
        constraint_id=109,
        ree_code=1,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(None,))},
        tipo_limite=3,
        valor_penalidade=250.0,
    )
    census = _census(record)
    calendar = [_stage(0, 730.0)]

    with dx.collect() as collected:
        result = emit_rhe_generics(
            _case(calendar),
            id_map,
            census=census,
            effective=effective,
            hydro_to_ree={5: 1},
        )

    assert result.result is not None
    rows = result.result.bounds.to_pylist()
    assert rows[0]["bound_lower"] == 50.0
    assert rows[0]["bound_upper"] is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhe-unknown-tipo-limite"


def test_rhe_result_is_named_tuple_with_expected_fields() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    effective = EffectiveCadastro(base=_hidr_frame({}), n_stages=1, stage_varying={})

    with dx.collect():
        result = emit_rhe_generics(
            _case([_stage(0, 730.0)]),
            id_map,
            census=_census(),
            effective=effective,
            hydro_to_ree={},
        )

    assert result.result is None
    assert result.rho_acum_overrides == {}

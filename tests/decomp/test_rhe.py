"""Tier-1 tests for the RHE stored-energy generic-constraint emitter.

Synthetic ``ConstraintRecord``/``ConstraintTerm``/``StageBounds``/
``EffectiveCadastro``/``DecompIdMap``/``OperativeStage`` only — no deck, no
``example/`` read, no ``import novomodelo``. Covers the absolute (``tipo_limite=1``)
MWh bound, the percentage (``tipo_limite=2``) useful-energy RHS, the reservoir
plant set (cascade, run-of-river exclusion, weekly-regulating ``"S"``
participation), the no-storage skip+WARNING, the single constraint per
record with its per_unit penalty, the coupled-cascade expression parameters,
and the negative-CM sign path.
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
    """A synthetic ``hidr``-shaped frame, indexed by ``codigo_usina``.

    A row without a ``posto`` gets its own code, so no two plants share one.
    """
    rows = {code: {"posto": code, **row} for code, row in rows.items()}
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


def _single_reservoir(
    tipo_regulacao: str = "M",
) -> tuple[DecompIdMap, EffectiveCadastro]:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5,))
    effective = EffectiveCadastro(
        base=_hidr_frame({5: _hydro_row(tipo_regulacao=tipo_regulacao)}),
        n_stages=1,
        stage_varying={},
    )
    return id_map, effective


# ---------------------------------------------------------------------------
# Single-REE, single monthly reservoir, tipo_limite=1 (absolute)
# ---------------------------------------------------------------------------


def test_emit_rhe_generics_absolute_limit_becomes_mwh_bound() -> None:
    id_map, effective = _single_reservoir()
    record = _he_record(
        constraint_id=100,
        ree_code=1,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(None,))},
        tipo_limite=1,
        valor_penalidade=250.0,
    )

    result = emit_rhe_generics(
        _case([_stage(0, 730.0)]),
        id_map,
        census=_census(record),
        effective=effective,
        hydro_to_ree={5: 1},
    )

    assert result is not None
    [constraint] = result.constraints
    assert constraint["name"] == "RHE_100"
    assert constraint["expression"] == "@rho_acum_mwh_h0 * hydro_useful_volume_final(0)"
    assert set(constraint) == {"id", "name", "description", "expression", "slack"}
    assert constraint["slack"] == {
        "enabled": True,
        "penalty": 250.0,
        "penalty_basis": "per_unit",
    }
    assert result.per_stage_parameters == {}
    assert result.expression_parameters == {}

    rows = result.bounds.to_pylist()
    assert len(rows) == 1
    # 50 MWmes (stage-average MW) over a 730 h stage.
    assert rows[0]["bound_lower"] == pytest.approx(50.0 * 730.0)
    assert rows[0]["bound_upper"] is None
    assert rows[0]["block_id"] is None
    assert rows[0]["stage_id"] == 0


# ---------------------------------------------------------------------------
# Same record, tipo_limite=2 (percentage of the maximum useful stored energy)
# ---------------------------------------------------------------------------


def test_emit_rhe_generics_percentage_rhs_is_the_maximum_useful_energy() -> None:
    id_map, effective = _single_reservoir()
    record = _he_record(
        constraint_id=101,
        ree_code=1,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(20.0,), upper=(None,))},
        tipo_limite=2,
        valor_penalidade=250.0,
    )

    result = emit_rhe_generics(
        _case([_stage(0, 730.0)]),
        id_map,
        census=_census(record),
        effective=effective,
        hydro_to_ree={5: 1},
    )

    assert result is not None
    assert result.constraints[0]["expression"] == (
        "@rho_acum_mwh_h0 * hydro_useful_volume_final(0)"
        " >= @pct_rhe101 * @earm_max_mwh_h0"
    )
    # 20 % of novomodelo's maximum useful stored energy.
    assert result.per_stage_parameters == {"pct_rhe101": [pytest.approx(0.2)]}
    rows = result.bounds.to_pylist()
    assert rows[0]["bound_lower"] == 0.0
    assert rows[0]["bound_upper"] is None


# ---------------------------------------------------------------------------
# Reservoir plant set
# ---------------------------------------------------------------------------


def test_emit_rhe_generics_two_reservoir_cascade() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5, 6))
    effective = EffectiveCadastro(
        base=_hidr_frame(
            {
                5: _hydro_row(tipo_regulacao="M", codigo_usina_jusante=6),
                6: _hydro_row(tipo_regulacao="M", codigo_usina_jusante=0),
            }
        ),
        n_stages=1,
        stage_varying={},
    )
    record = _he_record(
        constraint_id=102,
        ree_code=1,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(10.0,), upper=(None,))},
        tipo_limite=2,
        valor_penalidade=250.0,
    )

    result = emit_rhe_generics(
        _case([_stage(0, 730.0)]),
        id_map,
        census=_census(record),
        effective=effective,
        hydro_to_ree={5: 1, 6: 1},
    )

    assert result is not None
    assert result.constraints[0]["expression"] == (
        "@rho_acum_mwh_h0 * hydro_useful_volume_final(0)"
        " + @rho_acum_mwh_h1 * hydro_useful_volume_final(1)"
        " >= @pct_rhe102 * (@earm_max_mwh_h0 + @earm_max_mwh_h1)"
    )


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

    result = emit_rhe_generics(
        _case([_stage(0, 730.0)]),
        id_map,
        census=_census(record),
        effective=effective,
        hydro_to_ree={5: 1, 7: 1},
    )

    assert result is not None
    assert result.constraints[0]["expression"] == (
        "@rho_acum_mwh_h0 * hydro_useful_volume_final(0)"
    )


def test_emit_rhe_generics_weekly_regulating_reservoir_participates() -> None:
    id_map, effective = _single_reservoir(tipo_regulacao="S")
    record = _he_record(
        constraint_id=105,
        ree_code=1,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(None,))},
        tipo_limite=1,
        valor_penalidade=250.0,
    )

    result = emit_rhe_generics(
        _case([_stage(0, 730.0)]),
        id_map,
        census=_census(record),
        effective=effective,
        hydro_to_ree={5: 1},
    )

    assert result is not None
    assert result.constraints[0]["expression"] == (
        "@rho_acum_mwh_h0 * hydro_useful_volume_final(0)"
    )


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

    with dx.collect() as collected:
        result = emit_rhe_generics(
            _case([_stage(0, 730.0)]),
            id_map,
            census=_census(record),
            effective=effective,
            hydro_to_ree={7: 1},
        )

    assert result is None
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

    with dx.collect() as collected:
        result = emit_rhe_generics(
            _case([_stage(0, 730.0)]),
            id_map,
            census=_census(record),
            effective=effective,
            hydro_to_ree={},
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhe-no-storage-plants"


# ---------------------------------------------------------------------------
# One constraint per record over its weekly and monthly stages
# ---------------------------------------------------------------------------


def test_emit_rhe_generics_one_constraint_spans_weekly_and_monthly_stages() -> None:
    id_map, effective = _single_reservoir()
    record = _he_record(
        constraint_id=106,
        ree_code=1,
        coefficient=1.0,
        bounds={
            0: StageBounds(lower=(50.0,), upper=(None,)),
            1: StageBounds(lower=(60.0,), upper=(None,)),
            2: StageBounds(lower=(None,), upper=(None,)),
        },
        tipo_limite=1,
        valor_penalidade=250.0,
    )

    result = emit_rhe_generics(
        _case([_stage(0, 168.0), _stage(1, 730.0), _stage(2, 730.0)]),
        id_map,
        census=_census(record),
        effective=effective,
        hydro_to_ree={5: 1},
    )

    assert result is not None
    [constraint] = result.constraints
    assert constraint["name"] == "RHE_106"
    rows = result.bounds.to_pylist()
    assert [(r["constraint_id"], r["stage_id"]) for r in rows] == [
        (constraint["id"], 0),
        (constraint["id"], 1),
    ]
    # Stage-average MW over each stage's own hours.
    assert [r["bound_lower"] for r in rows] == pytest.approx(
        [50.0 * 168.0, 60.0 * 730.0]
    )


def test_emit_rhe_generics_percentage_fraction_is_zero_off_the_record_stages() -> None:
    id_map, effective = _single_reservoir()
    record = _he_record(
        constraint_id=110,
        ree_code=1,
        coefficient=1.0,
        bounds={1: StageBounds(lower=(30.0,), upper=(None,))},
        tipo_limite=2,
        valor_penalidade=250.0,
    )

    result = emit_rhe_generics(
        _case([_stage(0, 168.0), _stage(1, 168.0), _stage(2, 730.0)]),
        id_map,
        census=_census(record),
        effective=effective,
        hydro_to_ree={5: 1},
    )

    assert result is not None
    assert result.per_stage_parameters == {"pct_rhe110": pytest.approx([0.0, 0.3, 0.0])}
    assert [r["stage_id"] for r in result.bounds.to_pylist()] == [1]


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

    result = emit_rhe_generics(
        _case([_stage(0, 730.0)]),
        id_map,
        census=_census(record),
        effective=effective,
        hydro_to_ree={5: 1, 6: 2},
    )

    assert result is not None
    assert result.constraints[0]["expression"] == (
        "@rho_acum_mwh_h0 * hydro_useful_volume_final(0)"
        " - @rho_acum_mwh_h1 * hydro_useful_volume_final(1)"
    )


# ---------------------------------------------------------------------------
# Additional coverage: default-penalty and unknown-tipo_limite fallbacks
# ---------------------------------------------------------------------------


def test_emit_rhe_generics_non_positive_penalty_falls_back_and_warns() -> None:
    id_map, effective = _single_reservoir()
    record = _he_record(
        constraint_id=108,
        ree_code=1,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(None,))},
        tipo_limite=1,
        valor_penalidade=0.0,
    )

    with dx.collect() as collected:
        result = emit_rhe_generics(
            _case([_stage(0, 730.0)]),
            id_map,
            census=_census(record),
            effective=effective,
            hydro_to_ree={5: 1},
        )

    assert result is not None
    assert result.constraints[0]["slack"]["penalty"] == 1000.0
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhe-default-penalty"


def test_emit_rhe_generics_unknown_tipo_limite_treated_as_absolute_and_warns() -> None:
    id_map, effective = _single_reservoir()
    record = _he_record(
        constraint_id=109,
        ree_code=1,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(None,))},
        tipo_limite=3,
        valor_penalidade=250.0,
    )

    with dx.collect() as collected:
        result = emit_rhe_generics(
            _case([_stage(0, 730.0)]),
            id_map,
            census=_census(record),
            effective=effective,
            hydro_to_ree={5: 1},
        )

    assert result is not None
    rows = result.bounds.to_pylist()
    assert rows[0]["bound_lower"] == pytest.approx(50.0 * 730.0)
    assert rows[0]["bound_upper"] is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhe-unknown-tipo-limite"


def test_emit_rhe_generics_empty_census_returns_none() -> None:
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

    assert result is None


def _coupled_cascade() -> tuple[DecompIdMap, EffectiveCadastro, dict[int, int]]:
    """REE 1 (UP -> CUT) drains into REE 2 (DOWN); UP and CUT have twins.

    FICT.UP and FICT.CUT are the non-generating, non-operated ``hidr`` plants
    on UP's and CUT's postos, chained to DOWN. FICT.UP stores less than UP.
    SIDE crosses into REE 2 too, but with no twin, so its cascade is not cut.
    """
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1, 2, 5, 7))
    effective = EffectiveCadastro(
        base=_hidr_frame(
            {
                1: _hydro_row(tipo_regulacao="M", codigo_usina_jusante=2),
                2: _hydro_row(
                    tipo_regulacao="M", codigo_usina_jusante=5, volume_maximo=2000.0
                ),
                3: {
                    **_hydro_row(
                        tipo_regulacao="M",
                        codigo_usina_jusante=4,
                        produtibilidade_especifica=0.0,
                        volume_maximo=500.0,
                    ),
                    "posto": 1,
                },
                4: {
                    **_hydro_row(
                        tipo_regulacao="M",
                        codigo_usina_jusante=5,
                        produtibilidade_especifica=0.0,
                        volume_maximo=2000.0,
                    ),
                    "posto": 2,
                },
                5: _hydro_row(tipo_regulacao="M", codigo_usina_jusante=0),
                7: _hydro_row(tipo_regulacao="M", codigo_usina_jusante=5),
            }
        ),
        n_stages=1,
        stage_varying={},
    )
    return id_map, effective, {1: 1, 2: 1, 5: 2, 7: 1}


def _emit_coupled(tipo_limite: int) -> tuple[dict[str, dict], dict[str, str]]:
    id_map, effective, hydro_to_ree = _coupled_cascade()
    records = [
        _he_record(
            constraint_id=200 + ree_code,
            ree_code=ree_code,
            coefficient=1.0,
            bounds={0: StageBounds(lower=(10.0,), upper=(None,))},
            tipo_limite=tipo_limite,
            valor_penalidade=250.0,
        )
        for ree_code in (1, 2)
    ]
    result = emit_rhe_generics(
        _case([_stage(0, 730.0)]),
        id_map,
        census=_census(*records),
        effective=effective,
        hydro_to_ree=hydro_to_ree,
    )
    assert result is not None
    by_name = {c["name"]: c for c in result.constraints}
    return by_name, result.expression_parameters


def test_emit_rhe_generics_coupled_own_ree_stops_at_the_cut() -> None:
    by_name, params = _emit_coupled(tipo_limite=1)

    assert by_name["RHE_201"]["expression"] == (
        "@rho_cut_mwh_h0 * hydro_useful_volume_final(0)"
        " + @rho_cut_mwh_h1 * hydro_useful_volume_final(1)"
        " + @rho_acum_mwh_h3 * hydro_useful_volume_final(3)"
    )
    assert params["rho_cut_mwh_h0"] == "@rho_eq_mwh_h0 + @rho_eq_mwh_h1"
    assert params["rho_cut_mwh_h1"] == "@rho_eq_mwh_h1"


def test_emit_rhe_generics_coupled_downstream_ree_counts_the_twins() -> None:
    by_name, _ = _emit_coupled(tipo_limite=1)

    assert by_name["RHE_202"]["expression"] == (
        "@rho_acum_mwh_h2 * hydro_useful_volume_final(0)"
        " + @rho_acum_mwh_h2 * hydro_useful_volume_final(1)"
        " + @rho_acum_mwh_h2 * hydro_useful_volume_final(2)"
    )


def test_emit_rhe_generics_coupled_percentage_uses_the_physical_volume() -> None:
    by_name, params = _emit_coupled(tipo_limite=2)

    assert by_name["RHE_202"]["expression"].split(" >= ")[1] == (
        "@pct_rhe202 * (@earm_max_fict_mwh_h0 + @earm_max_fict_mwh_h1"
        " + @earm_max_mwh_h2)"
    )
    # The shares use UP's own 1000 hm³ (not FICT.UP's 500 hm³) and CUT's
    # 2000 hm³; DOWN, uncoupled, keeps novomodelo's maximum useful stored energy.
    assert params["earm_max_fict_mwh_h0"] == "1000.0 * @rho_acum_mwh_h2"
    assert params["earm_max_fict_mwh_h1"] == "2000.0 * @rho_acum_mwh_h2"
    assert params["earm_max_cut_mwh_h0"] == "1000.0 * @rho_cut_mwh_h0"


def test_emit_rhe_generics_coupling_is_reported() -> None:
    with dx.collect() as collected:
        _emit_coupled(tipo_limite=1)

    [diag] = [d for d in collected if d.code == "decomp-rhe-ree-hydraulic-coupling"]
    assert diag.severity is Severity.INFO
    assert diag.table is not None
    assert diag.table.rows == [[1, 1, 2, 3, 2, 5], [2, 1, 2, 4, 2, 5]]

"""Tier-1 tests for the generic-constraint emitter foundation.

Synthetic ``ConstraintTerm``/``ConstraintRecord``/``StageBounds``/
``DecompIdMap``/``OperativeStage``/``EffectiveCadastro`` only — no deck, no
``example/`` read, no ``import novomodelo``. Covers every ``_variable_token``
branch (including the ``bus=`` selector and the fail-loud raises), the
``FI``-interchange line resolver (``build_fi_line_map``/``resolve_fi_term``,
direct/reverse/no-line/unknown-name), the ``_format_expression``
coefficient-formatting rules, ``big_m_penalty``, the ``slots_from_record``
slot-enumeration helper, ``emit_re_generics``'s per-record
resolve-or-skip-whole-constraint behaviour (mixed hydro+interchange,
GNL/unknown-thermal, frequency-split, no-line ``FI``, and an all-skipped
census), and ``emit_rhq_rhv_generics``'s RHQ flow-mix/``QBOM``-station and
RHV multi-``VARM`` additive-floor/volume-tipo-deferral behaviour. The
``GenericConstraintBuilder`` itself is covered by
``tests/core/test_generic_constraint_builder.py``.
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
    ConstraintCensus,
    ConstraintRecord,
    ConstraintTerm,
    StageBounds,
)
from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
from novomodelo_bridge.decomp.converters.constraints import (
    _format_expression,
    _hydro_generation_token,
    _variable_token,
    big_m_penalty,
    build_fi_line_map,
    emit_re_generics,
    emit_rhq_rhv_generics,
    resolve_fi_term,
    slots_from_record,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import OperativeStage
from tests.conftest import make_decomp_case


def _stage(index: int, n_blocks: int) -> OperativeStage:
    return OperativeStage(
        index=index,
        start_date=date(2026, 7, 4),
        end_date=date(2026, 7, 11),
        season_id=6,
        block_hours=tuple(168.0 / n_blocks for _ in range(n_blocks)),
    )


def _case(calendar: list[OperativeStage]) -> DecompCase:
    return make_decomp_case(Path("unused"), calendar=calendar)


@pytest.fixture
def id_map() -> DecompIdMap:
    return DecompIdMap(
        bus_codes=(1,),
        bus_names=("SE",),
        hydro_codes=(5,),
        thermal_codes=(3,),
    )


# ---------------------------------------------------------------------------
# _hydro_generation_token / _variable_token
# ---------------------------------------------------------------------------


def test_hydro_generation_token_bare() -> None:
    assert _hydro_generation_token(0) == "hydro_generation(0)"


def test_hydro_generation_token_with_bus() -> None:
    assert _hydro_generation_token(0, bus_id=5) == "hydro_generation(0, bus=5)"


def test_variable_token_generation(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=5, coefficient=1.0, variable="generation")
    assert _variable_token(term, id_map, {}) == "hydro_generation(0)"


def test_variable_token_thermal_generation(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=3, coefficient=1.0, variable="thermal_generation")
    assert _variable_token(term, id_map, {}) == "thermal_generation(0)"


def test_variable_token_qdef(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=5, coefficient=1.0, variable="QDEF")
    assert _variable_token(term, id_map, {}) == "hydro_outflow(0)"


def test_variable_token_qtur(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=5, coefficient=1.0, variable="QTUR")
    assert _variable_token(term, id_map, {}) == "hydro_turbined(0)"


def test_variable_token_qver(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=5, coefficient=1.0, variable="QVER")
    assert _variable_token(term, id_map, {}) == "hydro_spillage(0)"


def test_variable_token_qdes(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=5, coefficient=1.0, variable="QDES")
    assert _variable_token(term, id_map, {}) == "hydro_diversion(0)"


def test_variable_token_qbom(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=42, coefficient=1.0, variable="QBOM")
    assert _variable_token(term, id_map, {42: 2}) == "pumping_flow(2)"


def test_variable_token_varm(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=5, coefficient=1.0, variable="VARM")
    assert _variable_token(term, id_map, {}) == "hydro_storage(0)"


def test_variable_token_interchange_raises(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=0, coefficient=1.0, variable="interchange")
    with pytest.raises(ValueError, match="interchange"):
        _variable_token(term, id_map, {})


def test_variable_token_vdef_raises(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=5, coefficient=1.0, variable="VDEF")
    with pytest.raises(ValueError, match="VDEF"):
        _variable_token(term, id_map, {})


def test_variable_token_vdes_raises(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=5, coefficient=1.0, variable="VDES")
    with pytest.raises(ValueError, match="VDES"):
        _variable_token(term, id_map, {})


def test_variable_token_vbom_raises(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=5, coefficient=1.0, variable="VBOM")
    with pytest.raises(ValueError, match="VBOM"):
        _variable_token(term, id_map, {})


def test_variable_token_unknown_raises(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=5, coefficient=1.0, variable="FROBNICATE")
    with pytest.raises(ValueError, match="FROBNICATE"):
        _variable_token(term, id_map, {})


# ---------------------------------------------------------------------------
# build_fi_line_map / resolve_fi_term
# ---------------------------------------------------------------------------


def test_build_fi_line_map_round_trip() -> None:
    lines = [
        {"id": 0, "source_bus_id": 5, "target_bus_id": 1},
        {"id": 1, "source_bus_id": 0, "target_bus_id": 5},
    ]
    assert build_fi_line_map(lines) == {(5, 1): 0, (0, 5): 1}


@pytest.fixture
def fi_id_map() -> DecompIdMap:
    """SE -> 0, S -> 1, IV (transhipment) -> 2."""
    return DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "S"))


def test_resolve_fi_term_direct_match(fi_id_map: DecompIdMap) -> None:
    line_map = {(0, 2): 1}
    term = ConstraintTerm(
        code=0,
        coefficient=1.0,
        variable="interchange",
        submarket_de="SE",
        submarket_para="IV",
    )
    assert resolve_fi_term(term, fi_id_map, line_map) == "line_direct(1)"


def test_resolve_fi_term_reverse_match(fi_id_map: DecompIdMap) -> None:
    line_map = {(0, 2): 1}
    term = ConstraintTerm(
        code=0,
        coefficient=1.0,
        variable="interchange",
        submarket_de="IV",
        submarket_para="SE",
    )
    assert resolve_fi_term(term, fi_id_map, line_map) == "line_reverse(1)"


def test_resolve_fi_term_no_line_warns_and_skips(fi_id_map: DecompIdMap) -> None:
    line_map = {(0, 2): 1}
    term = ConstraintTerm(
        code=0,
        coefficient=1.0,
        variable="interchange",
        submarket_de="SE",
        submarket_para="S",
    )
    with dx.collect() as collected:
        result = resolve_fi_term(term, fi_id_map, line_map)

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-fi-no-line"


def test_resolve_fi_term_unknown_name_warns_and_skips(fi_id_map: DecompIdMap) -> None:
    line_map = {(0, 2): 1}
    term = ConstraintTerm(
        code=0,
        coefficient=1.0,
        variable="interchange",
        submarket_de="SE",
        submarket_para="XX",
    )
    with dx.collect() as collected:
        result = resolve_fi_term(term, fi_id_map, line_map)

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-fi-no-line"


# ---------------------------------------------------------------------------
# _format_expression
# ---------------------------------------------------------------------------


def test_format_expression_acceptance_string() -> None:
    terms = [
        (1.0, "hydro_generation(0)"),
        (-1.0, "thermal_generation(2)"),
        (2.0, "line_direct(1)"),
    ]
    assert (
        _format_expression(terms)
        == "hydro_generation(0) - thermal_generation(2) + 2.0 * line_direct(1)"
    )


def test_format_expression_unit_positive() -> None:
    assert _format_expression([(1.0, "hydro_outflow(0)")]) == "hydro_outflow(0)"


def test_format_expression_unit_negative() -> None:
    assert _format_expression([(-1.0, "hydro_outflow(0)")]) == "- hydro_outflow(0)"


def test_format_expression_fractional_positive() -> None:
    assert _format_expression([(0.5, "hydro_outflow(0)")]) == "0.5 * hydro_outflow(0)"


def test_format_expression_fractional_negative_first_term() -> None:
    assert (
        _format_expression([(-0.5, "hydro_outflow(0)")]) == "- 0.5 * hydro_outflow(0)"
    )


def test_format_expression_multi_term_join() -> None:
    terms = [
        (1.0, "hydro_outflow(0)"),
        (0.5, "hydro_outflow(1)"),
        (-1.0, "hydro_outflow(2)"),
    ]
    assert (
        _format_expression(terms)
        == "hydro_outflow(0) + 0.5 * hydro_outflow(1) - hydro_outflow(2)"
    )


def test_format_expression_empty_returns_none() -> None:
    assert _format_expression([]) is None


# ---------------------------------------------------------------------------
# big_m_penalty
# ---------------------------------------------------------------------------


def test_big_m_penalty() -> None:
    assert big_m_penalty(4500.0) == 45000.0


# ---------------------------------------------------------------------------
# slots_from_record
# ---------------------------------------------------------------------------


def _hq_record(
    *, constraint_id: int, bounds: dict[int, StageBounds], per_block: bool = True
) -> ConstraintRecord:
    return ConstraintRecord(
        family="HQ",
        constraint_id=constraint_id,
        stage_start=min(bounds),
        stage_end=max(bounds),
        terms=(ConstraintTerm(code=182, coefficient=1.0, variable="QDEF"),),
        bounds=bounds,
        per_block=per_block,
    )


def test_slots_from_record_two_sided_round_trip() -> None:
    """A genuine band (both sides bounded) yields one slot carrying both
    endpoints."""
    record = _hq_record(
        constraint_id=3,
        bounds={0: StageBounds(lower=(10.0,), upper=(90.0,))},
    )
    calendar = [_stage(0, 1)]

    assert slots_from_record(record, calendar) == [(0, 0, 10.0, 90.0)]


def test_slots_from_record_per_block_clamp_to_calendar() -> None:
    """A 5-slot LU-style bounds row on a 1-block stage only emits block 0 —
    the per-block clamp is to ``calendar[stage].block_hours``, not the
    (up-to-5-wide) ``StageBounds`` tuple width."""
    record = _hq_record(
        constraint_id=3,
        bounds={
            0: StageBounds(
                lower=(1.0, 2.0, 3.0, 4.0, 5.0),
                upper=(10.0, 20.0, 30.0, 40.0, 50.0),
            )
        },
    )
    calendar = [_stage(0, 1)]

    assert slots_from_record(record, calendar) == [(0, 0, 1.0, 10.0)]


def test_slots_from_record_bounded_value_beyond_block_count_dropped() -> None:
    """A bounded value lying only beyond the stage's real block count is
    clamped away entirely — no slot carries it, so a caller can never build
    an orphan constraint (one with no companion bound row) from it."""
    record = _hq_record(
        constraint_id=9,
        # block 0 is unbounded; the only bounded value (500.0) sits in slot 1,
        # but the stage has just 1 real block, so slot 1 is clamped away.
        bounds={0: StageBounds(lower=(None, None), upper=(None, 500.0))},
    )
    calendar = [_stage(0, 1)]

    assert slots_from_record(record, calendar) == [(0, 0, None, None)]


def test_slots_from_record_stage_level_emits_null_block_id() -> None:
    record = ConstraintRecord(
        family="HV",
        constraint_id=7,
        stage_start=0,
        stage_end=0,
        terms=(ConstraintTerm(code=1, coefficient=1.0, variable="VARM"),),
        bounds={0: StageBounds(lower=(25.0,), upper=(50.0,))},
        per_block=False,
    )
    calendar = [_stage(0, 1)]

    assert slots_from_record(record, calendar) == [(0, None, 25.0, 50.0)]


def test_slots_from_record_multi_stage_preserves_bounds_order() -> None:
    """Slots are enumerated one stage at a time, in ``record.bounds``
    iteration order."""
    record = _hq_record(
        constraint_id=3,
        bounds={
            0: StageBounds(lower=(10.0,), upper=(90.0,)),
            1: StageBounds(lower=(20.0,), upper=(80.0,)),
        },
    )
    calendar = [_stage(0, 1), _stage(1, 1)]

    assert slots_from_record(record, calendar) == [
        (0, 0, 10.0, 90.0),
        (1, 0, 20.0, 80.0),
    ]


# ---------------------------------------------------------------------------
# emit_re_generics / _resolve_re_terms
# ---------------------------------------------------------------------------


def _census(*records: ConstraintRecord) -> ConstraintCensus:
    """A synthetic census whose generic slice is exactly *records*."""
    return ConstraintCensus(by_family={}, to_bounds=(), to_generic=tuple(records))


def test_emit_re_generics_mixed_hydro_interchange_round_trip(
    id_map: DecompIdMap,
) -> None:
    record = ConstraintRecord(
        family="RE",
        constraint_id=12,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(code=5, coefficient=1.0, variable="generation"),
            ConstraintTerm(
                code=0,
                coefficient=-1.0,
                variable="interchange",
                submarket_de="SE",
                submarket_para="IV",
            ),
        ),
        bounds={0: StageBounds(lower=(100.0,), upper=(500.0,))},
        per_block=True,
    )
    census = _census(record)
    line_map = {(0, 1): 3}
    calendar = [_stage(0, 1)]

    result = emit_re_generics(
        _case(calendar),
        id_map,
        census=census,
        line_map=line_map,
        big_m=45000.0,
    )

    assert result is not None
    assert len(result.constraints) == 1
    constraint = result.constraints[0]
    assert constraint["expression"] == "hydro_generation(0) - line_direct(3)"
    assert constraint["slack"] == {"enabled": True, "penalty": 45000.0}
    assert set(constraint) == {"id", "name", "description", "expression", "slack"}

    rows = result.bounds.to_pylist()
    assert len(rows) == 1
    assert rows[0]["bound_lower"] == 100.0
    assert rows[0]["bound_upper"] == 500.0


def test_emit_re_generics_gnl_thermal_skips_and_warns(id_map: DecompIdMap) -> None:
    record = ConstraintRecord(
        family="RE",
        constraint_id=20,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(code=99, coefficient=1.0, variable="thermal_generation"),
        ),
        bounds={0: StageBounds(lower=(0.0,), upper=(50.0,))},
        per_block=True,
    )
    census = _census(record)
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        result = emit_re_generics(
            _case(calendar),
            id_map,
            census=census,
            line_map={},
            big_m=45000.0,
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-re-thermal-not-normal"


def test_emit_re_generics_frequency_split_skips_and_warns(
    id_map: DecompIdMap,
) -> None:
    record = ConstraintRecord(
        family="RE",
        constraint_id=21,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(
                code=5, coefficient=1.0, variable="generation", frequency=50.0
            ),
        ),
        bounds={0: StageBounds(lower=(0.0,), upper=(50.0,))},
        per_block=True,
    )
    census = _census(record)
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        result = emit_re_generics(
            _case(calendar),
            id_map,
            census=census,
            line_map={},
            big_m=45000.0,
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-re-frequency-split-deferred"


def test_emit_re_generics_fi_no_line_skips(id_map: DecompIdMap) -> None:
    record = ConstraintRecord(
        family="RE",
        constraint_id=22,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(
                code=0,
                coefficient=1.0,
                variable="interchange",
                submarket_de="SE",
                submarket_para="IV",
            ),
        ),
        bounds={0: StageBounds(lower=(0.0,), upper=(50.0,))},
        per_block=True,
    )
    census = _census(record)
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        result = emit_re_generics(
            _case(calendar),
            id_map,
            census=census,
            line_map={},
            big_m=45000.0,
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-fi-no-line"


def test_emit_re_generics_all_records_skipped_returns_none(
    id_map: DecompIdMap,
) -> None:
    frequency_split = ConstraintRecord(
        family="RE",
        constraint_id=30,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(
                code=5, coefficient=1.0, variable="generation", frequency=60.0
            ),
        ),
        bounds={0: StageBounds(lower=(0.0,), upper=(50.0,))},
        per_block=True,
    )
    gnl_thermal = ConstraintRecord(
        family="RE",
        constraint_id=31,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(code=99, coefficient=1.0, variable="thermal_generation"),
        ),
        bounds={0: StageBounds(lower=(0.0,), upper=(50.0,))},
        per_block=True,
    )
    fi_no_line = ConstraintRecord(
        family="RE",
        constraint_id=32,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(
                code=0,
                coefficient=1.0,
                variable="interchange",
                submarket_de="SE",
                submarket_para="IV",
            ),
        ),
        bounds={0: StageBounds(lower=(0.0,), upper=(50.0,))},
        per_block=True,
    )
    census = _census(frequency_split, gnl_thermal, fi_no_line)
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        result = emit_re_generics(
            _case(calendar),
            id_map,
            census=census,
            line_map={},
            big_m=45000.0,
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 3


# ---------------------------------------------------------------------------
# emit_rhq_rhv_generics
# ---------------------------------------------------------------------------


@pytest.fixture
def unused_effective() -> EffectiveCadastro:
    """Placeholder for RHQ-only cases — the RHQ path never reads ``effective``."""
    df = pd.DataFrame({182: {}}).T
    df.index.name = "codigo_usina"
    return EffectiveCadastro(base=df, n_stages=1, stage_varying={})


def _hv_floor_frame(floors: dict[int, float]) -> pd.DataFrame:
    """A synthetic ``hidr``-shaped frame whose ``volume_minimo`` is each floor."""
    df = pd.DataFrame(
        {
            code: {"volume_minimo": floor, "volume_maximo": floor + 1000.0}
            for code, floor in floors.items()
        }
    ).T
    df.index.name = "codigo_usina"
    return df


def test_emit_rhq_rhv_generics_rhq_multi_term_round_trip(
    unused_effective: EffectiveCadastro,
) -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(182,))
    record = ConstraintRecord(
        family="HQ",
        constraint_id=164,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(code=182, coefficient=1.0, variable="QDEF"),
            ConstraintTerm(code=182, coefficient=1.0, variable="QDES"),
        ),
        bounds={0: StageBounds(lower=(50.0,), upper=(200.0,))},
        per_block=True,
    )
    census = _census(record)
    calendar = [_stage(0, 1)]

    result = emit_rhq_rhv_generics(
        _case(calendar),
        id_map,
        census=census,
        pumping_station_ids={},
        effective=unused_effective,
        big_m=45000.0,
    )

    assert result is not None
    assert len(result.constraints) == 1
    constraint = result.constraints[0]
    assert constraint["expression"] == "hydro_outflow(0) + hydro_diversion(0)"
    assert constraint["slack"] == {"enabled": True, "penalty": 45000.0}

    rows = result.bounds.to_pylist()
    assert len(rows) == 1
    assert rows[0]["bound_lower"] == 50.0
    assert rows[0]["bound_upper"] == 200.0


def test_emit_rhq_rhv_generics_rhq_single_qbom_via_pumping_map(
    unused_effective: EffectiveCadastro,
) -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    record = ConstraintRecord(
        family="HQ",
        constraint_id=10,
        stage_start=0,
        stage_end=0,
        terms=(ConstraintTerm(code=2, coefficient=1.0, variable="QBOM"),),
        bounds={0: StageBounds(lower=(None,), upper=(80.0,))},
        per_block=True,
    )
    census = _census(record)
    calendar = [_stage(0, 1)]

    result = emit_rhq_rhv_generics(
        _case(calendar),
        id_map,
        census=census,
        pumping_station_ids={2: 1},
        effective=unused_effective,
        big_m=45000.0,
    )

    assert result is not None
    assert len(result.constraints) == 1
    assert result.constraints[0]["expression"] == "pumping_flow(1)"


def test_emit_rhq_rhv_generics_rhq_qbom_no_station_skips_and_warns(
    unused_effective: EffectiveCadastro,
) -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    record = ConstraintRecord(
        family="HQ",
        constraint_id=10,
        stage_start=0,
        stage_end=0,
        terms=(ConstraintTerm(code=2, coefficient=1.0, variable="QBOM"),),
        bounds={0: StageBounds(lower=(None,), upper=(80.0,))},
        per_block=True,
    )
    census = _census(record)
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        result = emit_rhq_rhv_generics(
            _case(calendar),
            id_map,
            census=census,
            pumping_station_ids={},
            effective=unused_effective,
            big_m=45000.0,
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhq-qbom-no-station"


def test_emit_rhq_rhv_generics_rhv_multi_varm_additive_floor() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5, 6))
    effective = EffectiveCadastro(
        base=_hv_floor_frame({5: 100.0, 6: 200.0}), n_stages=1, stage_varying={}
    )
    record = ConstraintRecord(
        family="HV",
        constraint_id=7,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(code=5, coefficient=1.0, variable="VARM"),
            ConstraintTerm(code=6, coefficient=1.0, variable="VARM"),
        ),
        bounds={0: StageBounds(lower=(10.0,), upper=(None,))},
        per_block=False,
    )
    census = _census(record)
    calendar = [_stage(0, 1)]

    result = emit_rhq_rhv_generics(
        _case(calendar),
        id_map,
        census=census,
        pumping_station_ids={},
        effective=effective,
        big_m=45000.0,
    )

    assert result is not None
    assert len(result.constraints) == 1
    constraint = result.constraints[0]
    assert constraint["expression"] == "hydro_storage(0) + hydro_storage(1)"
    assert set(constraint) == {"id", "name", "description", "expression", "slack"}

    rows = result.bounds.to_pylist()
    assert len(rows) == 1
    assert rows[0]["bound_lower"] == 310.0
    assert rows[0]["bound_upper"] is None
    assert rows[0]["block_id"] is None
    assert rows[0]["stage_id"] == 0


def test_emit_rhq_rhv_generics_rhv_varm_uncadastred_skips_and_warns() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5, 6))
    effective = EffectiveCadastro(
        base=_hv_floor_frame({5: 100.0}), n_stages=1, stage_varying={}
    )
    record = ConstraintRecord(
        family="HV",
        constraint_id=7,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(code=5, coefficient=1.0, variable="VARM"),
            ConstraintTerm(code=6, coefficient=1.0, variable="VARM"),
        ),
        bounds={0: StageBounds(lower=(10.0,), upper=(None,))},
        per_block=False,
    )
    census = _census(record)
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        result = emit_rhq_rhv_generics(
            _case(calendar),
            id_map,
            census=census,
            pumping_station_ids={},
            effective=effective,
            big_m=45000.0,
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhv-varm-plant-uncadastred"


def test_emit_rhq_rhv_generics_rhv_volume_tipo_deferred_skips_and_warns(
    unused_effective: EffectiveCadastro,
) -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    record = ConstraintRecord(
        family="HV",
        constraint_id=9,
        stage_start=0,
        stage_end=0,
        terms=(ConstraintTerm(code=182, coefficient=1.0, variable="VDEF"),),
        bounds={0: StageBounds(lower=(5.0,), upper=(None,))},
        per_block=False,
    )
    census = _census(record)
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        result = emit_rhq_rhv_generics(
            _case(calendar),
            id_map,
            census=census,
            pumping_station_ids={},
            effective=unused_effective,
            big_m=45000.0,
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhv-volume-tipo-deferred"

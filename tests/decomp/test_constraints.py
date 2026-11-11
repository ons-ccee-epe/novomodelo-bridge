"""Tier-1 tests for the generic-constraint emitter foundation.

Synthetic ``ConstraintTerm``/``ConstraintRecord``/``StageBounds``/
``DecompIdMap``/``OperativeStage``/``EffectiveCadastro`` only — no deck, no
``example/`` read, no ``import novomodelo``. Covers every ``_variable_token``
branch (including the ``bus=`` selector and the fail-loud raises), the
``FI``-interchange line resolver (``build_fi_line_map``/``resolve_fi_term``,
direct/reverse/no-line/unknown-name), the ``FE``-contract resolver
(``build_contract_token_map``/``resolve_fe_term``, import/export/unknown/
ambiguous), the ``_format_expression``
coefficient-formatting rules, ``big_m_penalty``, the ``slots_from_record``
slot-enumeration helper, ``emit_re_generics``'s per-record
resolve-or-skip-whole-constraint behaviour (mixed hydro+interchange,
hydro+contract, GNL/unknown-thermal, frequency-split, no-line ``FI``, and an all-skipped
census), and ``emit_rhq_rhv_generics``'s RHQ flow-mix/``QBOM``-station and
RHV multi-``VARM`` stage-floor-shift/volume-tipo stage-volume behaviour. The
``GenericConstraintBuilder`` itself is covered by
``tests/core/test_generic_constraint_builder.py``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.generic_constraint_builder import soft_slack
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.constraint_registers import (
    ConstraintCensus,
    ConstraintRecord,
    ConstraintTerm,
    StageBounds,
    read_constraints,
)
from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
from novomodelo_bridge.decomp.converters.constraints import (
    _format_expression,
    _hydro_generation_token,
    _variable_token,
    big_m_penalty,
    build_contract_token_map,
    build_fi_line_map,
    emit_re_generics,
    emit_rhq_rhv_generics,
    resolve_fe_term,
    resolve_fi_term,
    slots_from_record,
)
from novomodelo_bridge.decomp.converters.contracts import Contract
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import OperativeStage
from tests.conftest import _FakeDadger, make_decomp_case


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
    assert _variable_token(term, id_map, {}) == "hydro_useful_volume_final(0)"


def test_variable_token_interchange_raises(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=0, coefficient=1.0, variable="interchange")
    with pytest.raises(ValueError, match="interchange"):
        _variable_token(term, id_map, {})


def test_variable_token_vdef(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=5, coefficient=1.0, variable="VDEF")
    assert _variable_token(term, id_map, {}) == "hydro_outflow_volume(0)"


def test_variable_token_vdes(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=5, coefficient=1.0, variable="VDES")
    assert _variable_token(term, id_map, {}) == "hydro_diversion_volume(0)"


def test_variable_token_vbom(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=42, coefficient=1.0, variable="VBOM")
    assert _variable_token(term, id_map, {42: 2}) == "pumping_volume(2)"


def test_variable_token_vbom_without_station_raises_key_error(
    id_map: DecompIdMap,
) -> None:
    term = ConstraintTerm(code=42, coefficient=1.0, variable="VBOM")
    with pytest.raises(KeyError):
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
# build_contract_token_map / resolve_fe_term
# ---------------------------------------------------------------------------


def _contract(contract_id: int, kind: str, numero: int, bus_code: int) -> Contract:
    return Contract(
        id=contract_id,
        kind=kind,
        numero=numero,
        name=f"{kind} {numero}",
        bus_code=bus_code,
        stages=[],
    )


def _fe_term(numero: int, submarket: int) -> ConstraintTerm:
    return ConstraintTerm(
        code=numero, coefficient=1.0, variable="contract", submarket=submarket
    )


def test_resolve_fe_term_picks_the_contract_kind_by_number_and_submarket() -> None:
    tokens = build_contract_token_map(
        [_contract(0, "import", 1, 1), _contract(1, "export", 1, 2)]
    )

    assert resolve_fe_term(_fe_term(1, 1), tokens) == "contract_import(0)"
    assert resolve_fe_term(_fe_term(1, 2), tokens) == "contract_export(1)"


def test_resolve_fe_term_unknown_contract_warns_and_skips() -> None:
    tokens = build_contract_token_map([_contract(0, "import", 1, 1)])

    with dx.collect() as collected:
        result = resolve_fe_term(_fe_term(2, 1), tokens)

    assert result is None
    assert [d.code for d in collected] == ["decomp-fe-no-contract"]
    assert "no such contract" in collected[0].summary


def test_resolve_fe_term_number_shared_by_ci_and_ce_warns_and_skips() -> None:
    tokens = build_contract_token_map(
        [_contract(0, "import", 1, 1), _contract(1, "export", 1, 1)]
    )

    with dx.collect() as collected:
        result = resolve_fe_term(_fe_term(1, 1), tokens)

    assert result is None
    assert [d.code for d in collected] == ["decomp-fe-no-contract"]
    assert "both a CI and a CE" in collected[0].summary


def test_resolve_fe_term_without_submarket_raises() -> None:
    term = ConstraintTerm(code=1, coefficient=1.0, variable="contract")
    with pytest.raises(ValueError, match="must carry a submarket"):
        resolve_fe_term(term, {})


def test_variable_token_refuses_a_contract_term(id_map: DecompIdMap) -> None:
    with pytest.raises(ValueError, match="FE contract resolution path"):
        _variable_token(_fe_term(1, 1), id_map, {})


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
        contract_tokens={},
        big_m=45000.0,
    )

    assert result is not None
    assert len(result.constraints) == 1
    constraint = result.constraints[0]
    assert constraint["expression"] == "hydro_generation(0) - line_direct(3)"
    assert constraint["slack"] == soft_slack(45000.0, "per_hour")
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
            contract_tokens={},
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
            contract_tokens={},
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
            contract_tokens={},
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
            contract_tokens={},
            big_m=45000.0,
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 3


def test_emit_re_generics_hydro_and_contract_terms(id_map: DecompIdMap) -> None:
    record = ConstraintRecord(
        family="RE",
        constraint_id=40,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(code=5, coefficient=1.0, variable="generation"),
            ConstraintTerm(code=7, coefficient=-0.5, variable="contract", submarket=1),
        ),
        bounds={0: StageBounds(lower=(None,), upper=(800.0,))},
        per_block=True,
    )
    tokens = build_contract_token_map([_contract(2, "export", 7, 1)])

    result = emit_re_generics(
        _case([_stage(0, 1)]),
        id_map,
        census=_census(record),
        line_map={},
        contract_tokens=tokens,
        big_m=45000.0,
    )

    assert result is not None
    assert result.constraints[0]["expression"] == (
        "hydro_generation(0) - 0.5 * contract_export(2)"
    )


def test_emit_re_generics_unresolved_contract_skips_the_constraint(
    id_map: DecompIdMap,
) -> None:
    record = ConstraintRecord(
        family="RE",
        constraint_id=41,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(code=5, coefficient=1.0, variable="generation"),
            _fe_term(7, 1),
        ),
        bounds={0: StageBounds(lower=(None,), upper=(800.0,))},
        per_block=True,
    )

    with dx.collect() as collected:
        result = emit_re_generics(
            _case([_stage(0, 1)]),
            id_map,
            census=_census(record),
            line_map={},
            contract_tokens={},
            big_m=45000.0,
        )

    assert result is None
    assert [d.code for d in collected] == ["decomp-fe-no-contract"]


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
        hv_penalty=123.0,
        diversion_hydro_ids=frozenset(),
    )

    assert result is not None
    assert len(result.constraints) == 1
    constraint = result.constraints[0]
    assert constraint["expression"] == "hydro_outflow(0) + hydro_diversion(0)"
    assert constraint["slack"] == soft_slack(45000.0, "per_hour")

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
        hv_penalty=123.0,
        diversion_hydro_ids=frozenset(),
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
            hv_penalty=123.0,
            diversion_hydro_ids=frozenset(),
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhq-qbom-no-station"
    assert warnings[0].title == "RHQ QBOM term has no matching pumping station"
    assert warnings[0].summary == (
        "HQ constraint 10 carries a QBOM term for plant code 2, which has no "
        "matching pumping station; the constraint is skipped."
    )
    assert warnings[0].remediation == (
        "HQ constraint 10 is skipped; check that plant code 2 is declared as "
        "a pumping station."
    )


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
        hv_penalty=123.0,
        diversion_hydro_ids=frozenset(),
    )

    assert result is not None
    assert len(result.constraints) == 1
    constraint = result.constraints[0]
    assert constraint["expression"] == (
        "hydro_useful_volume_final(0) + hydro_useful_volume_final(1)"
    )
    assert set(constraint) == {"id", "name", "description", "expression", "slack"}

    rows = result.bounds.to_pylist()
    assert len(rows) == 1
    assert rows[0]["bound_lower"] == 10.0
    assert rows[0]["bound_upper"] is None
    assert rows[0]["block_id"] is None
    assert rows[0]["stage_id"] == 0


def test_emit_rhq_rhv_generics_hv_slack_is_per_unit_at_hv_penalty() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5, 6, 182))
    effective = EffectiveCadastro(
        base=_hv_floor_frame({5: 100.0, 6: 200.0}), n_stages=1, stage_varying={}
    )
    hv_record = ConstraintRecord(
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
    hq_record = ConstraintRecord(
        family="HQ",
        constraint_id=164,
        stage_start=0,
        stage_end=0,
        terms=(ConstraintTerm(code=182, coefficient=1.0, variable="QDEF"),),
        bounds={0: StageBounds(lower=(50.0,), upper=(200.0,))},
        per_block=True,
    )

    result = emit_rhq_rhv_generics(
        _case([_stage(0, 1)]),
        id_map,
        census=_census(hv_record, hq_record),
        pumping_station_ids={},
        effective=effective,
        big_m=45000.0,
        hv_penalty=123.0,
        diversion_hydro_ids=frozenset(),
    )

    assert result is not None
    slacks = {c["name"]: c["slack"] for c in result.constraints}
    assert slacks == {
        "HQ_164": {"enabled": True, "penalty": 45000.0, "penalty_basis": "per_hour"},
        "HV_7": {"enabled": True, "penalty": 123.0, "penalty_basis": "per_unit"},
    }


def test_emit_rhq_rhv_generics_rhv_varm_uncadastred_skips_and_warns() -> None:
    record, id_map, effective = _two_varm_hv_scenario(cadastred=(5,), operated=(5, 6))
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
            hv_penalty=123.0,
            diversion_hydro_ids=frozenset(),
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhv-varm-plant-uncadastred"


def _two_varm_hv_scenario(
    *, cadastred: tuple[int, ...], operated: tuple[int, ...]
) -> tuple[ConstraintRecord, DecompIdMap, EffectiveCadastro]:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=operated)
    effective = EffectiveCadastro(
        base=_hv_floor_frame({code: 100.0 for code in cadastred}),
        n_stages=1,
        stage_varying={},
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
    return record, id_map, effective


def _vbom_hv_scenario() -> tuple[ConstraintRecord, DecompIdMap, EffectiveCadastro]:
    """A lone ``VBOM`` HV whose UE code 4 is no pumping station (the map is empty)."""
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5,))
    effective = EffectiveCadastro(
        base=_hv_floor_frame({5: 100.0}), n_stages=1, stage_varying={}
    )
    record = ConstraintRecord(
        family="HV",
        constraint_id=7,
        stage_start=0,
        stage_end=0,
        terms=(ConstraintTerm(code=4, coefficient=1.0, variable="VBOM"),),
        bounds={0: StageBounds(lower=(10.0,), upper=(None,))},
        per_block=False,
    )
    return record, id_map, effective


def _vdes_hv_scenario(
    diversions: Mapping[int, tuple[None, ...]] | None = None,
) -> tuple[ConstraintRecord, DecompIdMap, EffectiveCadastro]:
    """A lone ``VDES`` HV on operated plant 5 (hydro id 1, so code and id differ)."""
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(3, 5))
    effective = EffectiveCadastro(
        base=_hv_floor_frame({3: 50.0, 5: 100.0}),
        n_stages=1,
        stage_varying={},
        diversions=diversions or {},
    )
    record = ConstraintRecord(
        family="HV",
        constraint_id=7,
        stage_start=0,
        stage_end=0,
        terms=(ConstraintTerm(code=5, coefficient=1.0, variable="VDES"),),
        bounds={0: StageBounds(lower=(10.0,), upper=(None,))},
        per_block=False,
    )
    return record, id_map, effective


def _zero_coefficient_hv_scenario() -> tuple[
    ConstraintRecord, DecompIdMap, EffectiveCadastro
]:
    """Two operated ``VARM`` terms, both with coefficient ``0.0``."""
    record, id_map, effective = _two_varm_hv_scenario(cadastred=(5, 6), operated=(5, 6))
    zeroed = tuple(dataclasses.replace(t, coefficient=0.0) for t in record.terms)
    return dataclasses.replace(record, terms=zeroed), id_map, effective


def _zero_coefficient_hq_scenario() -> tuple[
    ConstraintRecord, DecompIdMap, EffectiveCadastro
]:
    """A lone ``QDEF`` term on operated plant 182 with coefficient ``0.0``."""
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(182,))
    effective = EffectiveCadastro(
        base=_hv_floor_frame({182: 100.0}), n_stages=1, stage_varying={}
    )
    record = ConstraintRecord(
        family="HQ",
        constraint_id=164,
        stage_start=0,
        stage_end=0,
        terms=(ConstraintTerm(code=182, coefficient=0.0, variable="QDEF"),),
        bounds={0: StageBounds(lower=(50.0,), upper=(200.0,))},
        per_block=True,
    )
    return record, id_map, effective


def test_emit_rhq_rhv_generics_rhv_non_operated_plant_skips_and_warns() -> None:
    record, id_map, effective = _two_varm_hv_scenario(cadastred=(5, 6), operated=(5,))

    with dx.collect() as collected:
        result = emit_rhq_rhv_generics(
            _case([_stage(0, 1)]),
            id_map,
            census=_census(record),
            pumping_station_ids={},
            effective=effective,
            big_m=45000.0,
            hv_penalty=123.0,
            diversion_hydro_ids=frozenset(),
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhv-varm-plant-uncadastred"
    assert "plant code 6" in warnings[0].summary
    assert "plant code 5" not in warnings[0].summary


_REPO_INTERNAL_LEAKS = (
    "docs/",
    "plans/",
    "~/git",
    "ticket-",
    "epic-",
    "_",
    ".py",
    "TRACKED",
)


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            _two_varm_hv_scenario(cadastred=(5,), operated=(5, 6)), id="uncadastred"
        ),
        pytest.param(
            _two_varm_hv_scenario(cadastred=(5, 6), operated=(5,)), id="not-operated"
        ),
        pytest.param(_vbom_hv_scenario(), id="vbom-no-station"),
        pytest.param(_vdes_hv_scenario(), id="vdes-no-diversion"),
        pytest.param(_zero_coefficient_hv_scenario(), id="hv-zero-coefficients"),
        pytest.param(_zero_coefficient_hq_scenario(), id="hq-zero-coefficients"),
    ],
)
def test_rhq_rhv_refusal_diagnostics_have_no_repo_internal_references(
    scenario: tuple[ConstraintRecord, DecompIdMap, EffectiveCadastro],
) -> None:
    record, id_map, effective = scenario

    with dx.collect() as collected:
        emit_rhq_rhv_generics(
            _case([_stage(0, 1)]),
            id_map,
            census=_census(record),
            pumping_station_ids={},
            effective=effective,
            big_m=45000.0,
            hv_penalty=123.0,
            diversion_hydro_ids=frozenset(),
        )

    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert warnings
    for diagnostic in warnings:
        for text in (
            diagnostic.title,
            diagnostic.summary,
            diagnostic.remediation or "",
        ):
            for leak in _REPO_INTERNAL_LEAKS:
                assert leak not in text, (
                    f"{diagnostic.code} leaks repo-internal reference {leak!r}: {text!r}"
                )


def test_emit_rhq_rhv_generics_all_zero_hv_skips_and_warns() -> None:
    record, id_map, effective = _zero_coefficient_hv_scenario()

    with dx.collect() as collected:
        result = emit_rhq_rhv_generics(
            _case([_stage(0, 1)]),
            id_map,
            census=_census(record),
            pumping_station_ids={},
            effective=effective,
            big_m=45000.0,
            hv_penalty=123.0,
            diversion_hydro_ids=frozenset(),
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhq-rhv-zero-coefficients"
    assert warnings[0].category == "Special constraints"
    assert warnings[0].title == "RHV constraint has only zero coefficients"
    assert warnings[0].summary == (
        "HV constraint 7 has a zero coefficient on every term at every stage, "
        "so it bounds nothing; the constraint is skipped."
    )
    assert warnings[0].remediation == (
        "HV constraint 7 is skipped; give at least one term a non-zero "
        "coefficient to keep it."
    )


def test_emit_rhq_rhv_generics_all_zero_hq_skips_and_warns() -> None:
    record, id_map, effective = _zero_coefficient_hq_scenario()

    with dx.collect() as collected:
        result = emit_rhq_rhv_generics(
            _case([_stage(0, 1)]),
            id_map,
            census=_census(record),
            pumping_station_ids={},
            effective=effective,
            big_m=45000.0,
            hv_penalty=123.0,
            diversion_hydro_ids=frozenset(),
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhq-rhv-zero-coefficients"
    assert warnings[0].title == "RHQ constraint has only zero coefficients"
    assert warnings[0].summary.startswith("HQ constraint 164 ")


def test_emit_rhq_rhv_generics_one_zero_term_among_non_zero_is_not_skipped() -> None:
    record, id_map, effective = _two_varm_hv_scenario(cadastred=(5, 6), operated=(5, 6))
    first, second = record.terms
    record = dataclasses.replace(
        record, terms=(dataclasses.replace(first, coefficient=0.0), second)
    )

    with dx.collect() as collected:
        result = emit_rhq_rhv_generics(
            _case([_stage(0, 1)]),
            id_map,
            census=_census(record),
            pumping_station_ids={},
            effective=effective,
            big_m=45000.0,
            hv_penalty=123.0,
            diversion_hydro_ids=frozenset(),
        )

    assert result is not None
    assert [c["name"] for c in result.constraints] == ["HV_7"]
    assert not [d for d in collected if d.code == "decomp-rhq-rhv-zero-coefficients"]


def test_rhv_explicit_zero_first_stage_coefficient_is_not_skipped() -> None:
    dadger = _FakeDadger(
        hv=pd.DataFrame(
            [(9, 1, 2)],
            columns=["codigo_restricao", "estagio_inicial", "estagio_final"],
        ),
        cv=pd.DataFrame(
            [(9, 40, 0.0, 1, "VARM"), (9, 40, 1.0, 2, "VARM")],
            columns=[
                "codigo_restricao",
                "codigo_usina",
                "coeficiente",
                "estagio",
                "tipo",
            ],
        ),
        lv=pd.DataFrame(
            [(9, 1, 100.0, 900.0)],
            columns=[
                "codigo_restricao",
                "estagio",
                "limite_inferior",
                "limite_superior",
            ],
        ),
    )
    census = read_constraints(dadger)
    [record] = census.to_generic
    [term] = record.terms
    assert term.schedule == ((0, 0.0), (1, 1.0))

    with dx.collect() as collected:
        result = emit_rhq_rhv_generics(
            _case([_stage(0, 1), _stage(1, 1)]),
            DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(40,)),
            census=census,
            pumping_station_ids={},
            effective=EffectiveCadastro(
                base=_hv_floor_frame({40: 100.0}), n_stages=2, stage_varying={}
            ),
            big_m=45000.0,
            hv_penalty=123.0,
            diversion_hydro_ids=frozenset(),
        )

    assert result is not None
    assert [c["name"] for c in result.constraints] == ["HV_9"]
    assert not [d for d in collected if d.code == "decomp-rhq-rhv-zero-coefficients"]


@pytest.mark.parametrize(
    ("variable", "code", "expression", "diversion_ids"),
    [
        pytest.param("VDEF", 5, "hydro_outflow_volume(1)", frozenset(), id="vdef"),
        pytest.param("VDES", 5, "hydro_diversion_volume(1)", frozenset({1}), id="vdes"),
        pytest.param("VBOM", 4, "pumping_volume(2)", frozenset(), id="vbom"),
    ],
)
def test_emit_rhq_rhv_generics_rhv_volume_tipo_is_an_unshifted_stage_row(
    variable: str, code: int, expression: str, diversion_ids: frozenset[int]
) -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(3, 5))
    effective = EffectiveCadastro(
        base=_hv_floor_frame({3: 50.0, 5: 100.0}),
        n_stages=2,
        stage_varying={(5, "volume_minimo"): (100.0, 160.0)},
    )
    record = ConstraintRecord(
        family="HV",
        constraint_id=9,
        stage_start=0,
        stage_end=1,
        terms=(ConstraintTerm(code=code, coefficient=1.0, variable=variable),),
        bounds={s: StageBounds(lower=(10.0,), upper=(40.0,)) for s in range(2)},
        per_block=False,
    )

    result = emit_rhq_rhv_generics(
        _case([_stage(s, 1) for s in range(2)]),
        id_map,
        census=_census(record),
        pumping_station_ids={4: 2},
        effective=effective,
        big_m=45000.0,
        hv_penalty=123.0,
        diversion_hydro_ids=diversion_ids,
    )

    assert result is not None
    assert [c["expression"] for c in result.constraints] == [expression]
    rows = result.bounds.to_pylist()
    assert [(row["bound_lower"], row["bound_upper"]) for row in rows] == [
        (10.0, 40.0),
        (10.0, 40.0),
    ]
    assert all(row["block_id"] is None for row in rows)


def test_emit_rhq_rhv_generics_rhv_mixed_varm_and_volume_tipo_shifts_only_varm() -> (
    None
):
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5, 6))
    effective = EffectiveCadastro(
        base=_hv_floor_frame({5: 100.0, 6: 200.0}),
        n_stages=2,
        stage_varying={
            (5, "volume_minimo"): (100.0, 140.0),
            (6, "volume_minimo"): (200.0, 250.0),
        },
    )
    record = ConstraintRecord(
        family="HV",
        constraint_id=7,
        stage_start=0,
        stage_end=1,
        terms=(
            ConstraintTerm(code=6, coefficient=1.0, variable="VARM"),
            ConstraintTerm(code=5, coefficient=0.5, variable="VDEF"),
        ),
        bounds={s: StageBounds(lower=(10.0,), upper=(None,)) for s in range(2)},
        per_block=False,
    )

    result = emit_rhq_rhv_generics(
        _case([_stage(s, 1) for s in range(2)]),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        effective=effective,
        big_m=45000.0,
        hv_penalty=123.0,
        diversion_hydro_ids=frozenset(),
    )

    assert result is not None
    assert [c["name"] for c in result.constraints] == ["HV_7"]
    assert result.constraints[0]["expression"] == (
        "hydro_useful_volume_final(1) + 0.5 * hydro_outflow_volume(0)"
    )
    assert [row["bound_lower"] for row in result.bounds.to_pylist()] == [10.0, 60.0]


def test_emit_rhq_rhv_generics_rhv_vbom_no_station_skips_and_warns() -> None:
    record, id_map, effective = _vbom_hv_scenario()

    with dx.collect() as collected:
        result = emit_rhq_rhv_generics(
            _case([_stage(0, 1)]),
            id_map,
            census=_census(record),
            pumping_station_ids={},
            effective=effective,
            big_m=45000.0,
            hv_penalty=123.0,
            diversion_hydro_ids=frozenset(),
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhv-vbom-no-station"
    assert warnings[0].title == "RHV VBOM term has no matching pumping station"
    assert warnings[0].summary == (
        "HV constraint 7 carries a VBOM term for plant code 4, which has no "
        "matching pumping station; the constraint is skipped."
    )


@pytest.mark.parametrize("variable", ["VDEF", "VDES"])
def test_emit_rhq_rhv_generics_rhv_hydro_volume_tipo_on_non_operated_plant_skips(
    variable: str,
) -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5,))
    effective = EffectiveCadastro(
        base=_hv_floor_frame({5: 100.0, 6: 200.0}), n_stages=1, stage_varying={}
    )
    record = ConstraintRecord(
        family="HV",
        constraint_id=7,
        stage_start=0,
        stage_end=0,
        terms=(ConstraintTerm(code=6, coefficient=1.0, variable=variable),),
        bounds={0: StageBounds(lower=(10.0,), upper=(None,))},
        per_block=False,
    )

    with dx.collect() as collected:
        result = emit_rhq_rhv_generics(
            _case([_stage(0, 1)]),
            id_map,
            census=_census(record),
            pumping_station_ids={},
            effective=effective,
            big_m=45000.0,
            hv_penalty=123.0,
            diversion_hydro_ids=frozenset(),
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert [d.code for d in warnings] == ["decomp-rhv-varm-plant-uncadastred"]
    assert "plant code 6" in warnings[0].summary


@pytest.mark.parametrize(
    "diversions",
    [
        pytest.param({}, id="no-diversion"),
        pytest.param({5: (None,)}, id="closed-channel"),
    ],
)
def test_emit_rhq_rhv_generics_rhv_vdes_without_a_converted_channel_skips_and_warns(
    diversions: Mapping[int, tuple[None, ...]],
) -> None:
    record, id_map, effective = _vdes_hv_scenario(diversions)

    with dx.collect() as collected:
        result = emit_rhq_rhv_generics(
            _case([_stage(0, 1)]),
            id_map,
            census=_census(record),
            pumping_station_ids={},
            effective=effective,
            big_m=45000.0,
            hv_penalty=123.0,
            diversion_hydro_ids=frozenset(),
        )

    assert result is None
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhv-vdes-no-diversion"
    assert warnings[0].title == "RHV VDES term on a plant without a diversion channel"
    assert warnings[0].summary == (
        "HV constraint 7 carries a VDES term for plant code 5, which has no "
        "diversion channel in the converted case; the constraint is skipped."
    )
    assert warnings[0].remediation == (
        "HV constraint 7 is skipped; give plant code 5 a diversion channel to "
        "an operated plant, through the cadastro or an AC DESVIO record with a "
        "positive limit, to keep it."
    )


def test_emit_rhq_rhv_generics_rhv_vdes_with_a_converted_channel_converts() -> None:
    record, id_map, effective = _vdes_hv_scenario()

    with dx.collect() as collected:
        result = emit_rhq_rhv_generics(
            _case([_stage(0, 1)]),
            id_map,
            census=_census(record),
            pumping_station_ids={},
            effective=effective,
            big_m=45000.0,
            hv_penalty=123.0,
            diversion_hydro_ids=frozenset({id_map.hydro_id(5)}),
        )

    assert result is not None
    assert [(c["name"], c["expression"]) for c in result.constraints] == [
        ("HV_7", "hydro_diversion_volume(1)")
    ]
    assert not [d for d in collected if d.severity is Severity.WARNING]


def test_format_expression_parameter_weight() -> None:
    assert (
        _format_expression([(-1.0, "hydro_outflow(0)"), ("coef_x", "hydro_outflow(1)")])
        == "- hydro_outflow(0) + @coef_x * hydro_outflow(1)"
    )
    assert _format_expression([("coef_x", "hydro_outflow(1)")]) == (
        "@coef_x * hydro_outflow(1)"
    )


def test_emit_rhq_rhv_generics_late_term_reads_a_per_stage_parameter(
    unused_effective: EffectiveCadastro,
) -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("N",), hydro_codes=(275, 277))
    record = ConstraintRecord(
        family="HQ",
        constraint_id=119,
        stage_start=0,
        stage_end=2,
        terms=(
            ConstraintTerm(code=275, coefficient=1.0, variable="QTUR"),
            ConstraintTerm(
                code=277, coefficient=1.0, variable="QTUR", schedule=((1, 1.0),)
            ),
        ),
        bounds={s: StageBounds(lower=(None,), upper=(11000.0,)) for s in range(3)},
        per_block=True,
    )
    calendar = [_stage(s, 1) for s in range(3)]

    result = emit_rhq_rhv_generics(
        _case(calendar),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        effective=unused_effective,
        big_m=45000.0,
        hv_penalty=123.0,
        diversion_hydro_ids=frozenset(),
    )

    assert result is not None
    assert result.constraints[0]["expression"] == (
        "hydro_turbined(0) + @coef_hq119_t1 * hydro_turbined(1)"
    )
    assert result.per_stage_parameters == {"coef_hq119_t1": [0.0, 1.0, 1.0]}


def test_emit_rhq_rhv_generics_rhv_varm_floor_uses_the_stage_coefficient() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5, 6))
    effective = EffectiveCadastro(
        base=_hv_floor_frame({5: 100.0, 6: 200.0}),
        n_stages=2,
        stage_varying={(6, "volume_minimo"): (200.0, 250.0)},
    )
    record = ConstraintRecord(
        family="HV",
        constraint_id=7,
        stage_start=0,
        stage_end=1,
        terms=(
            ConstraintTerm(code=5, coefficient=1.0, variable="VARM"),
            ConstraintTerm(
                code=6, coefficient=1.0, variable="VARM", schedule=((0, 1.0), (1, 2.0))
            ),
        ),
        bounds={s: StageBounds(lower=(10.0,), upper=(None,)) for s in range(2)},
        per_block=False,
    )
    calendar = [_stage(s, 1) for s in range(2)]

    result = emit_rhq_rhv_generics(
        _case(calendar),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        effective=effective,
        big_m=45000.0,
        hv_penalty=123.0,
        diversion_hydro_ids=frozenset(),
    )

    assert result is not None
    assert result.constraints[0]["expression"] == (
        "hydro_useful_volume_final(0) + @coef_hv7_t1 * hydro_useful_volume_final(1)"
    )
    assert result.per_stage_parameters == {"coef_hv7_t1": [1.0, 2.0]}
    lowers = [row["bound_lower"] for row in result.bounds.to_pylist()]
    assert lowers == [10.0, 10.0 + 2.0 * (250.0 - 200.0)]


def _emit_two_stage_varm_bounds(
    plant_six_coefficient: float, bounds: StageBounds
) -> dict[int, tuple[float | None, float | None]]:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5, 6))
    effective = EffectiveCadastro(
        base=_hv_floor_frame({5: 100.0, 6: 200.0}),
        n_stages=2,
        stage_varying={(6, "volume_minimo"): (200.0, 250.0)},
    )
    record = ConstraintRecord(
        family="HV",
        constraint_id=7,
        stage_start=0,
        stage_end=1,
        terms=(
            ConstraintTerm(code=5, coefficient=1.0, variable="VARM"),
            ConstraintTerm(code=6, coefficient=plant_six_coefficient, variable="VARM"),
        ),
        bounds={s: bounds for s in range(2)},
        per_block=False,
    )

    result = emit_rhq_rhv_generics(
        _case([_stage(s, 1) for s in range(2)]),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        effective=effective,
        big_m=45000.0,
        hv_penalty=123.0,
        diversion_hydro_ids=frozenset(),
    )

    assert result is not None
    return {
        row["stage_id"]: (row["bound_lower"], row["bound_upper"])
        for row in result.bounds.to_pylist()
    }


def test_emit_rhq_rhv_generics_rhv_bound_shifts_by_stage_floor_minus_static_floor() -> (
    None
):
    bounds = _emit_two_stage_varm_bounds(2.0, StageBounds(lower=(10.0,), upper=(None,)))

    assert bounds == {0: (10.0, None), 1: (110.0, None)}


def test_emit_rhq_rhv_generics_rhv_negative_coefficient_shifts_both_sides() -> None:
    bounds = _emit_two_stage_varm_bounds(
        -1.0, StageBounds(lower=(10.0,), upper=(40.0,))
    )

    assert bounds == {0: (10.0, 40.0), 1: (-40.0, -10.0)}


def test_emit_rhq_rhv_generics_constant_coefficients_declare_no_parameter(
    unused_effective: EffectiveCadastro,
) -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(182,))
    record = ConstraintRecord(
        family="HQ",
        constraint_id=164,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(code=182, coefficient=0.5, variable="QDEF"),
            ConstraintTerm(code=182, coefficient=1.0, variable="QDES"),
        ),
        bounds={0: StageBounds(lower=(50.0,), upper=(200.0,))},
        per_block=True,
    )

    result = emit_rhq_rhv_generics(
        _case([_stage(0, 1)]),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        effective=unused_effective,
        big_m=45000.0,
        hv_penalty=123.0,
        diversion_hydro_ids=frozenset(),
    )

    assert result is not None
    assert result.per_stage_parameters == {}
    assert "@" not in result.constraints[0]["expression"]

"""Tier-1 tests for the generic-constraint emitter foundation.

Synthetic ``ConstraintTerm``/``ConstraintRecord``/``StageBounds``/
``DecompIdMap``/``OperativeStage``/``EffectiveCadastro`` only — no deck, no
``example/`` read, no ``import novomodelo``. Covers every ``_variable_token``
branch (including the ``bus=`` selector and the fail-loud raises), the
``FI``-interchange line resolver (``build_fi_line_map``/``resolve_fi_term``,
direct/reverse/no-line/unknown-name), the ``FE``-contract resolver
(``build_contract_token_map``/``resolve_fe_term``, import/export/unknown/
ambiguous), ``big_m_penalty``, the ``slots_from_record``
slot-enumeration helper, ``emit_re_generics``'s per-record
resolve-or-skip-whole-constraint behaviour (mixed hydro+interchange,
hydro+contract, GNL/unknown-thermal, frequency-split, no-line ``FI``, and an all-skipped
census), and ``emit_rhq_rhv_generics``'s RHQ flow-mix/``QBOM``-station and
RHV ``VARM`` additive-floor and ``VDEF``/``VBOM`` stage-volume behaviour. The
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
from novomodelo_bridge.core.generic_constraint_builder import soft_slack
from novomodelo_bridge.core.outside_terms import OutsideTerm
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.constraint_registers import (
    ConstraintCensus,
    ConstraintRecord,
    ConstraintTerm,
    StageBounds,
)
from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
from novomodelo_bridge.decomp.converters.constraints import (
    Unresolved,
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


def test_variable_token_vdef(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=5, coefficient=1.0, variable="VDEF")
    assert _variable_token(term, id_map, {}) == "hydro_outflow_volume(0)"


def test_variable_token_vdes(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=5, coefficient=1.0, variable="VDES")
    assert _variable_token(term, id_map, {}) == "hydro_diversion_volume(0)"


def test_variable_token_vbom(id_map: DecompIdMap) -> None:
    term = ConstraintTerm(code=42, coefficient=1.0, variable="VBOM")
    assert _variable_token(term, id_map, {42: 2}) == "pumping_volume(2)"


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


def test_resolve_fi_term_without_a_line_leaves_the_term_out(
    fi_id_map: DecompIdMap,
) -> None:
    line_map = {(0, 2): 1}
    term = ConstraintTerm(
        code=0,
        coefficient=1.0,
        variable="interchange",
        submarket_de="SE",
        submarket_para="S",
    )

    unresolved = resolve_fi_term(term, fi_id_map, line_map)

    assert unresolved == Unresolved(
        "decomp-re-term-interchange-no-line", ("SE", "S"), OutsideTerm.NO_LINE
    )
    assert isinstance(unresolved, Unresolved)
    assert not unresolved.drops_constraint


def test_resolve_fi_term_unknown_name_is_unresolved(fi_id_map: DecompIdMap) -> None:
    line_map = {(0, 2): 1}
    term = ConstraintTerm(
        code=0,
        coefficient=1.0,
        variable="interchange",
        submarket_de="SE",
        submarket_para="XX",
    )

    unresolved = resolve_fi_term(term, fi_id_map, line_map)

    assert unresolved == Unresolved(
        "decomp-fi-no-line",
        ("SE", "XX", "a submarket has no SB record"),
        OutsideTerm.ABSENT,
    )
    assert isinstance(unresolved, Unresolved)
    assert unresolved.drops_constraint


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


def test_resolve_fe_term_unknown_contract_is_unresolved() -> None:
    tokens = build_contract_token_map([_contract(0, "import", 1, 1)])

    assert resolve_fe_term(_fe_term(2, 1), tokens) == Unresolved(
        "decomp-fe-no-contract", (2, 1, "no CI or CE record")
    )


def test_resolve_fe_term_number_shared_by_ci_and_ce_is_unresolved() -> None:
    tokens = build_contract_token_map(
        [_contract(0, "import", 1, 1), _contract(1, "export", 1, 1)]
    )

    assert resolve_fe_term(_fe_term(1, 1), tokens) == Unresolved(
        "decomp-fe-no-contract", (1, 1, "both a CI and a CE record")
    )


def test_resolve_fe_term_without_submarket_raises() -> None:
    term = ConstraintTerm(code=1, coefficient=1.0, variable="contract")
    with pytest.raises(ValueError, match="must carry a submarket"):
        resolve_fe_term(term, {})


def test_variable_token_refuses_a_contract_term(id_map: DecompIdMap) -> None:
    with pytest.raises(ValueError, match="FE contract resolution path"):
        _variable_token(_fe_term(1, 1), id_map, {})


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
    assert warnings[0].table is not None
    assert warnings[0].table.rows == [[21, 5, "50"]]


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


def _fi_term(de: str, para: str) -> ConstraintTerm:
    return ConstraintTerm(
        code=0,
        coefficient=1.0,
        variable="interchange",
        submarket_de=de,
        submarket_para=para,
    )


_TWO_SUBMARKETS = DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "S"), hydro_codes=(5,))


def test_emit_re_generics_leaves_an_fi_term_on_no_line_out() -> None:
    """No IA record joins SE and S, so no flow runs between them in DECOMP
    either: the term is left out with a note and the rest of RE 23 kept."""
    record = ConstraintRecord(
        family="RE",
        constraint_id=23,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(code=5, coefficient=1.0, variable="generation"),
            _fi_term("SE", "S"),
        ),
        bounds={0: StageBounds(lower=(0.0,), upper=(50.0,))},
        per_block=True,
    )
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        result = emit_re_generics(
            _case(calendar),
            _TWO_SUBMARKETS,
            census=_census(record),
            line_map={},
            contract_tokens={},
            big_m=45000.0,
        )

    assert result is not None
    assert [c["expression"] for c in result.constraints] == ["hydro_generation(0)"]
    assert [(d.code, d.severity) for d in collected] == [
        ("decomp-re-term-interchange-no-line", Severity.INFO)
    ]
    assert collected[0].table is not None
    assert collected[0].table.rows == [[23, "SE", "S"]]


def test_emit_re_generics_with_only_fi_terms_on_no_line_skips() -> None:
    record = ConstraintRecord(
        family="RE",
        constraint_id=24,
        stage_start=0,
        stage_end=0,
        terms=(_fi_term("SE", "S"),),
        bounds={0: StageBounds(lower=(0.0,), upper=(50.0,))},
        per_block=True,
    )
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        result = emit_re_generics(
            _case(calendar),
            _TWO_SUBMARKETS,
            census=_census(record),
            line_map={},
            contract_tokens={},
            big_m=45000.0,
        )

    assert result is None
    assert [(d.code, d.severity) for d in collected] == [
        ("decomp-fi-no-line", Severity.WARNING)
    ]
    assert collected[0].table is not None
    assert collected[0].table.rows == [[24, "SE", "S", "no IA record joins them"]]


def test_emit_re_generics_fi_term_on_an_undeclared_submarket_skips() -> None:
    record = ConstraintRecord(
        family="RE",
        constraint_id=25,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(code=5, coefficient=1.0, variable="generation"),
            _fi_term("SE", "XX"),
        ),
        bounds={0: StageBounds(lower=(0.0,), upper=(50.0,))},
        per_block=True,
    )
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        result = emit_re_generics(
            _case(calendar),
            _TWO_SUBMARKETS,
            census=_census(record),
            line_map={},
            contract_tokens={},
            big_m=45000.0,
        )

    assert result is None
    assert [(d.code, d.severity) for d in collected] == [
        ("decomp-fi-no-line", Severity.WARNING)
    ]
    assert collected[0].table is not None
    assert collected[0].table.rows == [[25, "SE", "XX", "a submarket has no SB record"]]


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
    fi_no_line = [
        ConstraintRecord(
            family="RE",
            constraint_id=constraint_id,
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
        for constraint_id in (31, 32)
    ]
    census = _census(frequency_split, *fi_no_line)
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
    assert [(w.code, len(w.table.rows) if w.table else 0) for w in warnings] == [
        ("decomp-fi-no-line", 2),
        ("decomp-re-frequency-split-deferred", 1),
    ]


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
    )

    assert result is not None
    assert len(result.constraints) == 1
    assert result.constraints[0]["expression"] == "pumping_flow(1)"


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


def test_emit_rhq_rhv_generics_rhv_vdef_is_one_stage_volume_row(
    unused_effective: EffectiveCadastro,
) -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(182,))
    record = ConstraintRecord(
        family="HV",
        constraint_id=9,
        stage_start=0,
        stage_end=0,
        terms=(ConstraintTerm(code=182, coefficient=1.0, variable="VDEF"),),
        bounds={0: StageBounds(lower=(5.0,), upper=(None,))},
        per_block=False,
    )

    result = emit_rhq_rhv_generics(
        _case([_stage(0, 3)]),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        effective=unused_effective,
        big_m=45000.0,
    )

    assert result is not None
    constraint = result.constraints[0]
    assert constraint["expression"] == "hydro_outflow_volume(0)"
    assert constraint["slack"] == soft_slack(45000.0, "per_hour")
    rows = result.bounds.to_pylist()
    assert len(rows) == 1
    assert rows[0]["block_id"] is None
    assert rows[0]["bound_lower"] == 5.0
    assert rows[0]["bound_upper"] is None


def test_emit_rhq_rhv_generics_rhv_floor_offsets_only_the_varm_terms() -> None:
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
            ConstraintTerm(code=6, coefficient=-1.0, variable="VDEF"),
        ),
        bounds={0: StageBounds(lower=(10.0,), upper=(50.0,))},
        per_block=False,
    )

    result = emit_rhq_rhv_generics(
        _case([_stage(0, 1)]),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        effective=effective,
        big_m=45000.0,
    )

    assert result is not None
    assert result.constraints[0]["expression"] == (
        "hydro_storage(0) - hydro_outflow_volume(1)"
    )
    rows = result.bounds.to_pylist()
    assert (rows[0]["bound_lower"], rows[0]["bound_upper"]) == (110.0, 150.0)


def test_emit_rhq_rhv_generics_rhv_vbom_via_pumping_map(
    unused_effective: EffectiveCadastro,
) -> None:
    record = ConstraintRecord(
        family="HV",
        constraint_id=11,
        stage_start=0,
        stage_end=0,
        terms=(ConstraintTerm(code=2, coefficient=1.0, variable="VBOM"),),
        bounds={0: StageBounds(lower=(None,), upper=(40.0,))},
        per_block=False,
    )

    result = emit_rhq_rhv_generics(
        _case([_stage(0, 1)]),
        DecompIdMap(bus_codes=(1,), bus_names=("SE",)),
        census=_census(record),
        pumping_station_ids={2: 1},
        effective=unused_effective,
        big_m=45000.0,
    )

    assert result is not None
    assert result.constraints[0]["expression"] == "pumping_volume(1)"


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
    )

    assert result is not None
    assert result.constraints[0]["expression"] == (
        "hydro_turbined(0) + @coef_hq119_t1 * hydro_turbined(1)"
    )
    assert result.per_stage_parameters == {"coef_hq119_t1": [0.0, 1.0, 1.0]}


def test_emit_rhq_rhv_generics_rhv_varm_floor_uses_the_stage_coefficient() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5, 6))
    effective = EffectiveCadastro(
        base=_hv_floor_frame({5: 100.0, 6: 200.0}), n_stages=2, stage_varying={}
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
    )

    assert result is not None
    assert result.constraints[0]["expression"] == (
        "hydro_storage(0) + @coef_hv7_t1 * hydro_storage(1)"
    )
    assert result.per_stage_parameters == {"coef_hv7_t1": [1.0, 2.0]}
    lowers = [row["bound_lower"] for row in result.bounds.to_pylist()]
    assert lowers == [10.0 + 100.0 + 200.0, 10.0 + 100.0 + 2.0 * 200.0]


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
    )

    assert result is not None
    assert result.per_stage_parameters == {}
    assert "@" not in result.constraints[0]["expression"]

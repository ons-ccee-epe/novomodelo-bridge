"""One cause table for constraint terms outside the case, on every entry path.

Each path converts a constraint with two terms the case holds plus one term
of the cause under test: the constraint is kept with only that term left out
when the cause leaves out only the term, and is left out whole otherwise,
and the one diagnostic naming the term carries the cause's severity.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.electrical_expression import ParsedTerm
from novomodelo_bridge.core.outside_terms import OutsideTerm
from novomodelo_bridge.decomp.constraint_registers import (
    ConstraintCensus,
    ConstraintRecord,
    ConstraintTerm,
    StageBounds,
    drop_unknown_entities,
)
from novomodelo_bridge.decomp.converters.constraints import emit_re_generics
from novomodelo_bridge.decomp.converters.libs_electrical import (
    AvailablePower,
    ElectricalRestriction,
    LibsElectricalModel,
)
from novomodelo_bridge.decomp.converters.libs_electrical_emit import (
    emit_libs_electrical_generics,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import OperativeStage
from novomodelo_bridge.newave.converters.constraints import (
    _electric_expression,
    _ElectricExpression,
    _ElectricSources,
    _ElectricTermSkip,
    _emit_electric_term_skips,
    _read_formulas,
)
from novomodelo_bridge.newave.id_map import NewaveIdMap
from tests.conftest import make_decomp_case

if TYPE_CHECKING:
    from collections.abc import Callable


class _Outcome(NamedTuple):
    kept: bool
    severities: list[Severity]


# NEWAVE: plants 10 and 20 in the case, 30 a confhd.dat plant that generates
# nothing, 99 undeclared; subsystems 1 and 2 joined by line 0, 3 by none.
_NW_ID_MAP = NewaveIdMap(
    subsystem_ids=[1, 2, 3], hydro_codes=[10, 20], thermal_codes=[]
)
_NW_LINES = {(1, 2): 0}
_NW_IDLE = frozenset({30})


def _newave(sources: _ElectricSources) -> _Outcome:
    skips: list[_ElectricTermSkip] = []
    with dx.collect() as collected:
        expression = _electric_expression(
            1, sources, _NW_ID_MAP, _NW_LINES, _NW_IDLE, skips
        )
        _emit_electric_term_skips(skips)
    return _Outcome(
        isinstance(expression, _ElectricExpression), [d.severity for d in collected]
    )


_NW_FORMULA_TERM = {
    OutsideTerm.NO_LINE: "ener_interc(1,3)",
    OutsideTerm.NOT_GENERATING: "ger_usih(30)",
    OutsideTerm.ABSENT: "ger_usih(99)",
}


def _newave_formula(cause: OutsideTerm) -> _Outcome:
    formula = f"ger_usih(10) + ger_usih(20) + {_NW_FORMULA_TERM[cause]}"
    read = _read_formulas({1: formula}, Path("restricao-eletrica.csv"))
    return _newave(_ElectricSources(read, {}, {}, {}, {}))


_NW_RE_DAT_PLANT = {OutsideTerm.NOT_GENERATING: 30, OutsideTerm.ABSENT: 99}


def _newave_re_dat(cause: OutsideTerm) -> _Outcome:
    plants = [10, 20, _NW_RE_DAT_PLANT[cause]]
    return _newave(_ElectricSources({}, {}, {}, {1: plants}, {}))


# DECOMP: plants 5 and 6 operated, 46 a UH record with no initial volume, 99
# undeclared; submarkets SE (1) and S (2), which no IA record joins.
_DC_ID_MAP = DecompIdMap(
    bus_codes=(1, 2),
    bus_names=("SE", "S"),
    hydro_codes=(5, 6),
    coupling_only_codes=(46,),
)
_DC_CALENDAR = [
    OperativeStage(
        index=0,
        start_date=date(2026, 7, 4),
        end_date=date(2026, 7, 11),
        season_id=6,
        block_hours=(168.0,),
    )
]


def _fu(code: int) -> ConstraintTerm:
    return ConstraintTerm(code=code, coefficient=1.0, variable="generation")


_DC_RE_TERM = {
    OutsideTerm.NO_LINE: ConstraintTerm(
        code=0,
        coefficient=1.0,
        variable="interchange",
        submarket_de="SE",
        submarket_para="S",
    ),
    OutsideTerm.NOT_GENERATING: _fu(46),
    OutsideTerm.ABSENT: _fu(99),
}


def _decomp_re(cause: OutsideTerm) -> _Outcome:
    record = ConstraintRecord(
        family="RE",
        constraint_id=1,
        stage_start=0,
        stage_end=0,
        terms=(_fu(5), _fu(6), _DC_RE_TERM[cause]),
        bounds={0: StageBounds(lower=(0.0,), upper=(50.0,))},
        per_block=True,
    )
    census, found = drop_unknown_entities(
        ConstraintCensus(by_family={}, to_generic=(record,)),
        hydro_codes=_DC_ID_MAP.hydro_codes,
        coupling_only_codes=_DC_ID_MAP.coupling_only_codes,
        thermal_codes=(),
        pumping_codes=(),
    )
    with dx.collect() as collected:
        result = emit_re_generics(
            make_decomp_case(Path("unused"), calendar=_DC_CALENDAR),
            _DC_ID_MAP,
            census=census,
            line_map={},
            contract_tokens={},
            big_m=1000.0,
        )
    return _Outcome(result is not None, [d.severity for d in (*found, *collected)])


_DC_LIBS_TERM = {
    OutsideTerm.NO_LINE: "ener_interc(1,2)",
    OutsideTerm.NOT_GENERATING: "ger_usih(46)",
    OutsideTerm.ABSENT: "ger_usih(99)",
}


def _constant_context(term: ParsedTerm) -> float:
    assert term.token == "__const__"
    return 1.0


def _decomp_libs(cause: OutsideTerm) -> _Outcome:
    restriction = ElectricalRestriction(
        code=1,
        lhs=f"ger_usih(5) + ger_usih(6) + {_DC_LIBS_TERM[cause]}",
        operator="<=",
        rhs="500",
        is_formula=False,
        limits={},
        overrides=(),
        stage_start=None,
        stage_end=None,
        habilita=None,
        violation=None,
    )
    model = LibsElectricalModel(
        expressions={}, aliases={}, restrictions={1: restriction}, rules={}
    )
    with dx.collect() as collected:
        result = emit_libs_electrical_generics(
            model,
            _DC_ID_MAP,
            lambda _stage, _block: _constant_context,
            AvailablePower(overlay={}),
            _DC_CALENDAR,
            {},
            {},
            {},
            big_m=1000.0,
        )
    return _Outcome(result.converted_codes == (1,), [d.severity for d in collected])


_PATHS: dict[str, tuple[Callable[[OutsideTerm], _Outcome], tuple[OutsideTerm, ...]]] = {
    "newave-formula": (_newave_formula, tuple(_NW_FORMULA_TERM)),
    "newave-re-dat": (_newave_re_dat, tuple(_NW_RE_DAT_PLANT)),
    "decomp-re": (_decomp_re, tuple(_DC_RE_TERM)),
    "decomp-libs": (_decomp_libs, tuple(_DC_LIBS_TERM)),
}


@pytest.mark.parametrize(
    ("path", "cause"),
    [(path, cause) for path, (_, causes) in _PATHS.items() for cause in causes],
    ids=lambda value: value.value if isinstance(value, OutsideTerm) else value,
)
def test_a_term_outside_the_case_is_treated_by_its_cause(
    path: str, cause: OutsideTerm
) -> None:
    convert, _ = _PATHS[path]

    outcome = convert(cause)

    assert outcome == _Outcome(not cause.drops_constraint, [cause.severity])


def test_only_an_absent_code_leaves_its_constraint_out() -> None:
    assert {cause: cause.drops_constraint for cause in OutsideTerm} == {
        OutsideTerm.NO_LINE: False,
        OutsideTerm.NOT_GENERATING: False,
        OutsideTerm.ABSENT: True,
    }
    assert {cause.severity for cause in OutsideTerm if cause.drops_constraint} == {
        Severity.WARNING
    }

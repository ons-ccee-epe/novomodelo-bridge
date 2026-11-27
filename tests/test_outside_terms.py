"""One cause table for constraint terms outside the case, on every entry path.

Each path converts a constraint with two terms the case holds plus one term
of the cause under test: the constraint is kept with only that term left out
when the cause leaves out only the term, and is left out whole otherwise,
and the one diagnostic naming the term carries the cause's severity.

A term on a GNL thermal of the DECOMP ``dadgnl`` file is not outside the
case: the ``RE`` path folds its commanded generation into the bound, and
stops where no ``GL`` week commands the thermal, as DECOMP does.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.errors import FieldParseError
from novomodelo_bridge.core.outside_terms import OutsideTerm
from novomodelo_bridge.decomp.constraint_registers import (
    ConstraintCensus,
    ConstraintRecord,
    ConstraintTerm,
    StageBounds,
    drop_unknown_entities,
)
from novomodelo_bridge.decomp.converters.constraints import (
    DecompGenericResult,
    emit_re_generics,
    fold_commanded_gnl_terms,
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


#: GNL thermal 86 is commanded 200 MW at the stage; no GL week commands 87.
_DC_GNL_GENERATION: dict[int, list[tuple[float, ...] | None]] = {
    86: [(200.0,)],
    87: [None],
}


def _re_generics(
    third: ConstraintTerm,
) -> tuple[DecompGenericResult | None, list[dx.Diagnostic]]:
    """``FU 5 + FU 6 + third <= 50`` through the census screens and the RE
    emitter, as ``convert decomp`` runs them."""
    record = ConstraintRecord(
        family="RE",
        constraint_id=1,
        stage_start=0,
        stage_end=0,
        terms=(_fu(5), _fu(6), third),
        bounds={0: StageBounds(lower=(None,), upper=(50.0,))},
        per_block=True,
    )
    census, folded = fold_commanded_gnl_terms(
        ConstraintCensus(by_family={}, to_generic=(record,)),
        _DC_GNL_GENERATION,
        gnl_submarkets={},
        ct_submarkets={},
        dadger_path="dadger.rv0",
        dadgnl_path="dadgnl.rv0",
    )
    census, found = drop_unknown_entities(
        census,
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
    return result, [*folded, *found, *collected]


def _decomp_re(cause: OutsideTerm) -> _Outcome:
    result, found = _re_generics(_DC_RE_TERM[cause])
    return _Outcome(result is not None, [d.severity for d in found])


_PATHS: dict[str, tuple[Callable[[OutsideTerm], _Outcome], tuple[OutsideTerm, ...]]] = {
    "newave-formula": (_newave_formula, tuple(_NW_FORMULA_TERM)),
    "newave-re-dat": (_newave_re_dat, tuple(_NW_RE_DAT_PLANT)),
    "decomp-re": (_decomp_re, tuple(_DC_RE_TERM)),
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


def _gnl_ft(code: int) -> ConstraintTerm:
    return ConstraintTerm(
        code=code, coefficient=0.5, variable="thermal_generation", submarket=1
    )


def test_a_commanded_gnl_term_folds_into_the_bound() -> None:
    re_result, re_found = _re_generics(_gnl_ft(86))

    assert re_result is not None
    re_bounds = re_result.bounds.select(
        ["stage_id", "block_id", "bound_lower", "bound_upper"]
    ).to_pylist()
    assert [row["bound_upper"] for row in re_bounds] == [-50.0]
    assert re_found == []


def test_an_uncommanded_gnl_term_stops_the_conversion() -> None:
    with pytest.raises(FieldParseError, match="DECOMP stops on such a term too$"):
        _re_generics(_gnl_ft(87))

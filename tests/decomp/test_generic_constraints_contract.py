"""Provider side of the generic-constraints contract with novomodelo (tier 1, no
``novomodelo`` import).

The files under ``tests/fixtures/novomodelo_contracts/`` and the schemas under
``tests/fixtures/novomodelo_schemas/`` are byte-for-byte ``cp`` copies from the
novomodelo repository, never edited here: novomodelo owns the contract and its
own tests prove it loads and resolves the fixtures. The synthetic electrical
cards below convert to them through the pipeline's document assembly: the
document byte for byte, the bounds table column by column (the Parquet bytes
carry the writer's version). A failure here means the two repositories disagree
on the format, never that a fixture needs regenerating from the emitter: settle
the change in whichever repository is wrong.
"""

# DEBT(DEBT-001): nothing compares the vendored copies with their novomodelo
# sources — payoff: a hand-run drift check script

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import jsonschema
import pyarrow as pa
import pyarrow.parquet as pq

from novomodelo_bridge.decomp.converters.constraints import big_m_penalty
from novomodelo_bridge.decomp.converters.electrical_constraints import (
    GenericOutputs,
    build_generic_outputs,
)
from novomodelo_bridge.decomp.converters.electrical_translate import EntityTables
from novomodelo_bridge.decomp.converters.libs_electrical import read_libs_electrical
from novomodelo_bridge.decomp.pipeline import _generic_constraints_document
from novomodelo_bridge.novomodelo.case_writer import CaseWriter
from novomodelo_bridge.novomodelo.scalar_parameters import (
    append_scalar_parameters,
    build_scalar_parameters,
)
from tests.decomp.conftest import (
    alias,
    alias_valor,
    expressao,
    habilita,
    horizonte,
    re_formula,
    re_inequacao,
    re_lim_formula,
    regra,
    spelled,
    tratamento_violacao,
    weekly_calendar,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_CONTRACT_JSON = (
    _FIXTURES / "novomodelo_contracts" / "generic_constraints_contract.json"
)
_CONTRACT_BOUNDS = (
    _FIXTURES / "novomodelo_contracts" / "generic_constraint_bounds_contract.parquet"
)
_GENERIC_SCHEMA = _FIXTURES / "novomodelo_schemas" / "generic_constraints.schema.json"
_PARAMETERS_SCHEMA = _FIXTURES / "novomodelo_schemas" / "generic_parameters.schema.json"

_DOCUMENT_RELPATH = "constraints/generic_constraints.json"
_BOUNDS_RELPATH = "constraints/generic_constraint_bounds.parquet"

_CALENDAR = [replace(stage, block_hours=(84.0, 84.0)) for stage in weekly_calendar(6)]

# No source code equals the id it maps to, so an untranslated code cannot match
# the fixture.
_TABLES = EntityTables(
    code_map={
        "hydros": {6: 0, 251: 1},
        "thermals": {12: 0},
        "ncs_pee": {9: 0},
        "submarkets": {1: 0, 4: 1},
        "contracts": {},
    },
    machine_sets={251: {1: 0}},
    contract_kinds={},
)

_REFERENCE = re.compile(r"@(\w+)")


def _contract_cards() -> dict[str, pd.DataFrame]:
    """The cards of the fixture's case, keyed by long accessor."""
    return {
        "expressao_eletrica": expressao(
            (1, "north_exchange", "ener_interc(4,1)-ener_interc(1,4)"),
            (2, "net_load", "val_demanda(4)-reserve_margin"),
        ),
        "alias_eletrico": alias((1, "reserve_margin")),
        "alias_eletrico_valor_periodo_patamar": alias_valor((1, 1, 6, None, 300.0)),
        "restricao_eletrica_inequacao": re_inequacao(
            (
                101,
                "ger_usih(6)+ger_conjh(251,1)+north_exchange",
                "<=",
                "0.1*val_demanda_sin+reserve_margin",
            ),
            (104, "ger_usih(6)", "<=", "se(val_demanda_sin>90000,800,600)"),
        ),
        "restricao_eletrica_formula": re_formula(
            (102, "ger_usit(12)+ger_pee(9)"), (103, "north_exchange+net_load")
        ),
        "restricao_eletrica_horizonte_periodo": horizonte((101, 1, 3)),
        "restricao_eletrica_limite_formula_periodo_patamar": re_lim_formula(
            (102, 1, 6, None, -1e31, 500.0),
            (103, 1, 1, 1, -100.0, 100.0),
            (103, 1, 1, 2, -100.0, 100.0),
        ),
        "restricao_eletrica_regra_ativacao": regra(
            (201, "82000<demanda_sin & demanda_sin<=95000")
        ),
        "restricao_eletrica_habilita": habilita((104, 201)),
        "restricao_eletrica_tratamento_violacao": tratamento_violacao(
            (101, "soft", 2000.0),
            (102, "hard", float("nan")),
            (104, "hard", float("nan")),
        ),
    }


def _outputs(*, short: bool = False) -> GenericOutputs:
    model = read_libs_electrical(spelled(_contract_cards(), short=short), _CALENDAR)
    assert model is not None
    # The cards cite no data term, so no data term has values.
    return build_generic_outputs(
        model,
        _CALENDAR,
        _TABLES,
        {},
        reserved=frozenset(),
        taken=frozenset(),
        default_penalty=big_m_penalty(5100.0),
    )


def _document(outputs: GenericOutputs) -> dict:
    return _generic_constraints_document(outputs.constraints, outputs.expressions)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _divergent_paths(
    emitted: object, expected: object, path: str = "$"
) -> Iterator[str]:
    """The JSON paths where *emitted* differs from *expected* in a value, its
    type, a list's length or an object's keys in order (the bytes keep it)."""
    match emitted, expected:
        case dict(), dict() if list(emitted) == list(expected):
            for key, value in expected.items():
                yield from _divergent_paths(emitted[key], value, f"{path}.{key}")
        case list(), list() if len(emitted) == len(expected):
            for index, (left, right) in enumerate(zip(emitted, expected, strict=True)):
                yield from _divergent_paths(left, right, f"{path}[{index}]")
        case dict(), dict():
            yield path
        case _ if emitted != expected or type(emitted) is not type(expected):
            yield path


def _assert_emits_the_contract_document(tmp_path: Path, *, short: bool) -> None:
    CaseWriter(tmp_path).write_json(_DOCUMENT_RELPATH, _document(_outputs(short=short)))
    emitted = (tmp_path / _DOCUMENT_RELPATH).read_bytes()
    expected = _CONTRACT_JSON.read_bytes()

    paths = list(_divergent_paths(json.loads(emitted), json.loads(expected)))
    assert emitted == expected, (
        f"the emitted document differs from the contract fixture at {paths}"
        if paths
        else "the emitted document has the fixture's value in other bytes"
    )


def test_the_emitted_document_is_the_contract_fixture_byte_for_byte(
    tmp_path: Path,
) -> None:
    _assert_emits_the_contract_document(tmp_path, short=False)


def test_short_form_cards_emit_the_same_document(tmp_path: Path) -> None:
    _assert_emits_the_contract_document(tmp_path, short=True)


def _columns(table: pa.Table) -> list[tuple[str, pa.DataType, bool]]:
    return [(field.name, field.type, field.nullable) for field in table.schema]


def test_the_emitted_bounds_match_the_contract_fixture_column_by_column(
    tmp_path: Path,
) -> None:
    CaseWriter(tmp_path).write_parquet(_BOUNDS_RELPATH, _outputs().bounds)
    emitted = pq.read_table(tmp_path / _BOUNDS_RELPATH)
    expected = pq.read_table(_CONTRACT_BOUNDS)

    assert _columns(emitted) == _columns(expected)
    assert emitted.to_pylist() == expected.to_pylist()


def test_contract_fixture_validates_against_the_vendored_schema() -> None:
    jsonschema.validate(_load(_CONTRACT_JSON), _load(_GENERIC_SCHEMA))


def test_the_emitted_document_validates_against_the_vendored_schema() -> None:
    jsonschema.validate(_document(_outputs()), _load(_GENERIC_SCHEMA))


def _texts(document: dict) -> Iterator[str]:
    for constraint in document["constraints"]:
        yield constraint["expression"]
        if "active_when" in constraint:
            yield constraint["active_when"]
    yield from (record["expression"] for record in document["expressions"])


def test_every_parameter_reference_is_declared() -> None:
    outputs = _outputs()
    document = _document(outputs)
    expressions = {record["name"] for record in document["expressions"]}
    parameters = {entry["name"] for entry in outputs.parameters}
    referenced = {
        name for text in _texts(document) for name in _REFERENCE.findall(text)
    }

    assert referenced - expressions - parameters == set()
    assert referenced & expressions == {"north_exchange", "net_load"}
    assert referenced & parameters == {"reserve_margin"}
    jsonschema.validate(
        append_scalar_parameters(build_scalar_parameters([]), outputs.parameters),
        _load(_PARAMETERS_SCHEMA),
    )

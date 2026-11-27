"""Translate the source model's electrical texts into novomodelo's neutral
vocabulary, entity ids and names.

A source term cites an entity by its source-model code; the translated call
cites the novomodelo id the conversion gave that entity (:class:`EntityTables`).
:func:`translate_call` is the one place that maps a source term to its neutral
term. Only entity arguments are translated. A block argument is copied as
written, never re-based to 0: the translated text must build the same LP as
the source text read verbatim.

novomodelo reads the load only as data, ``bus_load_value``. The system load is
written in place as the sum of the load of every bus of the case. The load as
an LP term, ``demanda`` and ``demanda_sin``, reads its value inside a
condition, which novomodelo reads as data alone, and is refused anywhere else.

Every other name a text cites is read as ``@name`` (:class:`NameTable`): a
declared expression or alias, a data term, and the formula of the restriction
an ``re(R)`` cites, which is an expression of its own. Nothing is inlined,
folded or evaluated: a ``se`` conditional is an ``if`` with its condition and
both branches translated.
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from novomodelo_bridge.core.electrical_expression import MalformedElectricalExpression
from novomodelo_bridge.core.errors import UnsupportedSourceError
from novomodelo_bridge.decomp.converters.libs_electrical_parser import (
    Binary,
    Call,
    Group,
    Keyword,
    Name,
    NameKind,
    Number,
    Param,
    Unary,
    classify_name,
    is_data_term,
    iter_nodes,
    parse_condition,
    parse_expression,
    render,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence, Set

    from novomodelo_bridge.decomp.converters.contracts import Contract
    from novomodelo_bridge.decomp.converters.electrical_constraints import (
        CodeMap,
        MachineSets,
    )
    from novomodelo_bridge.decomp.converters.libs_electrical import (
        LibsElectricalModel,
    )
    from novomodelo_bridge.decomp.converters.libs_electrical_parser import Node

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntityTables:
    """The ids a translated call cites.

    ``code_map`` and ``machine_sets`` are keyed by source-model codes, as
    :func:`~novomodelo_bridge.decomp.converters.electrical_constraints.build_code_map`
    and
    :func:`~novomodelo_bridge.decomp.converters.electrical_constraints.machine_set_table`
    derive them; ``contract_kinds`` maps a contract id to its declared kind,
    ``"import"`` or ``"export"``.
    """

    code_map: CodeMap
    machine_sets: MachineSets
    contract_kinds: Mapping[int, str]


def entity_tables(
    code_map: CodeMap, machine_sets: MachineSets, contracts: Sequence[Contract]
) -> EntityTables:
    """Bundle the id tables with the declared kind of each of ``contracts``."""
    return EntityTables(
        code_map, machine_sets, {contract.id: contract.kind for contract in contracts}
    )


class TermShape(Enum):
    """The argument list of a source term, spelled as a conversion error cites it;
    :func:`translate_call` builds the neutral call by the shape."""

    ENTITY = "code[, block]"
    NO_ENTITY = "[block]"
    CONTRACT = "contract[, block]"
    LINE_FLOW = "from, to"
    UNIT_GROUP = "plant, machine_set[, block]"


@dataclass(frozen=True)
class TermRule:
    """The neutral term of a source term, and the ``code_map`` table its entity
    codes are looked up in (``None`` when it cites no entity)."""

    neutral: str
    shape: TermShape
    table: str | None = None


_TERMS: Mapping[str, TermRule] = {
    "ger_usih": TermRule("hydro_generation", TermShape.ENTITY, "hydros"),
    "ger_usit": TermRule("thermal_generation", TermShape.ENTITY, "thermals"),
    "ger_pee": TermRule("non_controllable_generation", TermShape.ENTITY, "ncs_pee"),
    "disp_usih": TermRule("hydro_availability", TermShape.ENTITY, "hydros"),
    "val_demanda": TermRule("bus_load_value", TermShape.ENTITY, "submarkets"),
    "ener_comerc": TermRule("contract_{kind}", TermShape.CONTRACT, "contracts"),
    "ener_interc": TermRule("line_flow", TermShape.LINE_FLOW, "submarkets"),
    "ger_conjh": TermRule("hydro_generation", TermShape.UNIT_GROUP, "hydros"),
}

_NOT_IN_CASE: Mapping[str, str] = {
    "hydros": "hydro code {code} is not in the converted case",
    "thermals": "thermal code {code} is not in the converted case",
    "ncs_pee": "renewable park code {code} is not in the converted case",
    "submarkets": "submarket code {code} is not in the converted case",
    "contracts": (
        "contract number {code} is not in the converted case; a number declared "
        "by more than one contract cannot be translated"
    ),
}

#: Each load term that is an LP term in the source model, with the term that
#: reads the same load as data.
_LOAD_AS_DATA: Mapping[str, str] = {
    "demanda": "val_demanda",
    "demanda_sin": "val_demanda_sin",
}

_SYSTEM_LOAD = "val_demanda_sin"

_CODE_RE = re.compile(r"[0-9]+")


def translate_call(
    call: Call, tables: EntityTables, *, context: str, in_condition: bool
) -> Node:
    """Translate a source term call into its neutral term: a call, or the sum
    of the bus loads for the system load.

    ``context`` names where the call is written (``"restriction 405 formula"``);
    every error message starts with it and then cites the call as rendered
    from the source. ``in_condition`` is whether the call is written inside a
    condition: the text of an activation rule, or the first argument of a
    ``se`` and everything inside it.

    Raises
    ------
    MalformedElectricalExpression
        ``call`` names no source term (conditionals, references and data
        terms are not term calls and never reach here), its argument list
        does not fit the term's shape, an entity argument is not a
        non-negative integer literal, or a block argument is not a number.
    UnsupportedSourceError
        ``call`` is a load as an LP term outside a condition, or the converted
        case has no entity for a code, or no unit group for a machine set.
    """
    name = _term_name(call, context, in_condition=in_condition)
    if name == _SYSTEM_LOAD:
        term = _TermCall(call, TermShape.NO_ENTITY, tables, context)
        if len(call.args) > 1:
            raise term.malformed("wrong number of arguments")
        return _bus_load_sum(tables, tuple(map(term.block, call.args)))
    rule = _TERMS.get(name)
    if rule is None:
        raise MalformedElectricalExpression(
            f"{context}: {render(call)}: unknown function {call.name!r}"
        )
    term = _TermCall(call, rule.shape, tables, context)
    match rule.shape, rule.table, call.args:
        case TermShape.ENTITY, str(table), (entity, *block) if len(block) <= 1:
            return Call(
                rule.neutral,
                (_number(term.entity_id(table, entity)), *map(term.block, block)),
            )
        case TermShape.CONTRACT, str(table), (contract, *block) if len(block) <= 1:
            contract_id = term.entity_id(table, contract)
            return Call(
                rule.neutral.format(kind=tables.contract_kinds[contract_id]),
                (_number(contract_id), *map(term.block, block)),
            )
        case TermShape.LINE_FLOW, str(table), (source, target):
            return Call(
                rule.neutral,
                (
                    Keyword("source_bus", _number(term.entity_id(table, source))),
                    Keyword("target_bus", _number(term.entity_id(table, target))),
                ),
            )
        case TermShape.UNIT_GROUP, str(table), (plant, number, *block) if (
            len(block) <= 1
        ):
            return Call(
                rule.neutral,
                (
                    _number(term.entity_id(table, plant)),
                    *map(term.block, block),
                    Keyword("unit_group", _number(term.unit_group(plant, number))),
                ),
            )
    raise term.malformed("wrong number of arguments")


def _term_name(written: Name | Call, context: str, *, in_condition: bool) -> str:
    """The source term *written* translates as: inside a condition, a load as
    an LP term reads its value; outside one, it is refused."""
    if written.name not in _LOAD_AS_DATA:
        return written.name
    if in_condition:
        return _LOAD_AS_DATA[written.name]
    # novomodelo reads the load only as data, by design: the load as an LP term
    # has no neutral term and is refused for good, not tracked as a gap.
    raise UnsupportedSourceError(
        f"{context}: {render(written)}: the load as an LP term is not supported; "
        "write val_demanda or val_demanda_sin to read the load as data"
    )


def _bus_load_sum(tables: EntityTables, block: tuple[Number, ...]) -> Node:
    """The load of every bus of the case, each bus once however many
    submarket codes name it, summed in ascending id order."""
    loads: list[Node] = [
        Call("bus_load_value", (_number(bus), *block))
        for bus in sorted(set(tables.code_map["submarkets"].values()))
    ]
    if not loads:
        raise ValueError("the case maps no bus, so its load cannot be summed")
    total = functools.reduce(lambda left, right: Binary("+", left, right), loads)
    return total if len(loads) == 1 else Group(total)


def _number(entity_id: int) -> Number:
    return Number(str(entity_id))


@dataclass(frozen=True)
class _TermCall:
    call: Call
    shape: TermShape
    tables: EntityTables
    context: str

    def unsupported(self, message: str) -> UnsupportedSourceError:
        return UnsupportedSourceError(f"{self.context}: {render(self.call)}: {message}")

    def malformed(self, problem: str) -> MalformedElectricalExpression:
        return MalformedElectricalExpression(
            f"{self.context}: {render(self.call)}: {problem}; expected "
            f"{self.call.name}({self.shape.value})"
        )

    def code(self, node: Node) -> int:
        if isinstance(node, Number) and _CODE_RE.fullmatch(node.text):
            return int(node.text)
        raise self.malformed(f"{render(node)} is not a non-negative integer code")

    def block(self, node: Node) -> Number:
        if isinstance(node, Number):
            return node
        raise self.malformed(f"block {render(node)} is not a number")

    def entity_id(self, table: str, node: Node) -> int:
        code = self.code(node)
        ids = self.tables.code_map[table]
        if code not in ids:
            raise self.unsupported(_NOT_IN_CASE[table].format(code=code))
        return ids[code]

    def unit_group(self, plant: Node, number: Node) -> int:
        plant_code, set_number = self.code(plant), self.code(number)
        sets = self.tables.machine_sets.get(plant_code, {})
        if set_number not in sets:
            raise self.unsupported(
                f"hydro plant {plant_code} has no machine set {set_number} in the "
                "converted case"
            )
        return sets[set_number]


@dataclass(frozen=True)
class NameTable:
    """The name novomodelo reads each name of the model by, as ``@name``
    (:func:`build_name_table`).

    Each lookup raises ``KeyError`` for a name the table does not hold.
    """

    _expressions: Mapping[str, str]
    _aliases: Mapping[str, str]
    _data: Mapping[str, str]
    _references: Mapping[int, str]

    def expression(self, name: str) -> str:
        """The name of the declared expression *name*."""
        return self._expressions[name]

    def alias(self, name: str) -> str:
        """The name of the declared alias *name*, unless an expression shadows it."""
        return self._aliases[name]

    def data(self, name: str) -> str:
        """The name of the parameter of the cited data term *name*, unless a
        declaration shadows it."""
        return self._data[name]

    def reference(self, code: int) -> str:
        """The name of the expression an ``re(code)`` reads: the formula of
        restriction *code*."""
        return self._references[code]


def build_name_table(
    model: LibsElectricalModel, *, reserved: frozenset[str]
) -> NameTable:
    """Name the model's expressions and aliases, the data terms its texts cite
    and the restriction formula each ``re(R)`` reads.

    A deck name passes verbatim unless it is not a novomodelo identifier or it
    collides with ``reserved`` (the names other conversions declare as
    parameters) or with a generated ``restriction_<R>_formula``. The names that
    pass are taken first; then each other name, in the order expressions,
    aliases, data terms, each by name, has its invalid characters replaced by
    ``_`` and takes the smallest free ``_<n>`` suffix from 2. A shadowed name
    (an alias by an expression, a data term by either) is not in the table.
    Neither the table nor the warnings depend on the insertion order of the
    model's mappings.

    Raises
    ------
    MalformedElectricalExpression
        A text of the model does not parse.
    ValueError
        A generated name is reserved.
    """
    data_terms: set[str] = set()
    codes: set[int] = set()
    for node in (node for tree in _model_trees(model) for node in iter_nodes(tree)):
        match node:
            case Name(name) | Call(name, ()) if is_data_term(name):
                data_terms.add(name)
            case Call("re") if (code := _reference_code(node)) is not None:
                codes.add(code)
    references = {code: f"restriction_{code}_formula" for code in sorted(codes)}
    if reserved_references := sorted(reserved.intersection(references.values())):
        raise ValueError(
            f"the generated expression names {reserved_references} are already "
            "declared as parameters by another conversion"
        )
    declared_as = {name: "alias" for name in model.aliases} | {
        name: "expression" for name in model.expressions
    }
    _warn_shadowed(
        sorted(model.aliases.keys() & model.expressions.keys()),
        [[name, declared_as[name]] for name in sorted(data_terms & declared_as.keys())],
    )
    emitted = _assign(
        {
            "expression": sorted(model.expressions),
            "alias": sorted(model.aliases.keys() - model.expressions.keys()),
            "data term": sorted(data_terms - declared_as.keys()),
        },
        reserved.union(references.values()),
    )
    return NameTable(
        emitted["expression"], emitted["alias"], emitted["data term"], references
    )


_OPERATORS = ("<=", ">=", "==")


@dataclass(frozen=True)
class Translator:
    """Translate the texts of ``model`` into neutral texts.

    ``names`` is :func:`build_name_table` of the same ``model``. ``context``
    names where a text is written (``"restriction 405 formula"``); every error
    message starts with it, and the translation errors then cite the text.

    Raises
    ------
    MalformedElectricalExpression
        A text does not parse; it cites a name or a function no rule
        translates; an ``re(R)`` cites anything but one integer code of a
        restriction with a formula of its own; a data term is called with
        arguments; a term call is malformed (:func:`translate_call`).
    UnsupportedSourceError
        A load as an LP term is written outside a condition, or a term call
        cites an entity the converted case lacks (:func:`translate_call`).
    """

    model: LibsElectricalModel
    tables: EntityTables
    names: NameTable

    def formula(self, text: str, *, context: str) -> str:
        """Translate an expression without a relation."""
        return render(self._text(parse_expression, text, context, in_condition=False))

    def relation(self, lhs: str, operator: str, rhs: str, *, context: str) -> str:
        """Translate ``lhs operator rhs``.

        Raises
        ------
        MalformedElectricalExpression
            ``operator`` is not ``<=``, ``>=`` or ``==``.
        """
        if operator not in _OPERATORS:
            raise MalformedElectricalExpression(
                f"{context}: operator {operator!r} is not one of "
                f"{', '.join(_OPERATORS)}"
            )
        return render(
            Binary(
                operator,
                self._text(parse_expression, lhs, context, in_condition=False),
                self._text(parse_expression, rhs, context, in_condition=False),
            )
        )

    def condition(self, text: str, *, context: str) -> str:
        """Translate the condition of an activation rule."""
        return render(self._text(parse_condition, text, context, in_condition=True))

    def _text(
        self,
        parse: Callable[[str], Node],
        text: str,
        context: str,
        *,
        in_condition: bool,
    ) -> Node:
        return self._translate(
            _parse(parse, text, context),
            f"{context}: electrical expression {text!r}",
            in_condition=in_condition,
        )

    def _translate(self, node: Node, where: str, *, in_condition: bool) -> Node:
        def translate(child: Node) -> Node:
            return self._translate(child, where, in_condition=in_condition)

        match node:
            case Name(name):
                return self._name(name, where, in_condition=in_condition)
            case Call("se", (condition, *branches)):
                return Call(
                    "if",
                    (
                        self._translate(condition, where, in_condition=True),
                        *map(translate, branches),
                    ),
                )
            case Call("re"):
                return Param(self.names.reference(self._checked_reference(node, where)))
            case Call(name, args) if is_data_term(name):
                if args:
                    raise MalformedElectricalExpression(
                        f"{where}: {render(node)}: data term {name} takes no arguments"
                    )
                return self._name(name, where, in_condition=in_condition)
            case Call():
                return translate_call(
                    node, self.tables, context=where, in_condition=in_condition
                )
            case Unary(op, operand):
                return Unary(op, translate(operand))
            case Binary(op, left, right):
                return Binary(op, translate(left), translate(right))
            case Group(inner):
                return Group(translate(inner))
            case Number() | Param() | Keyword():
                return node

    def _name(self, name: str, where: str, *, in_condition: bool) -> Node:
        match classify_name(name, self.model):
            case NameKind.EXPRESSION:
                return Param(self.names.expression(name))
            case NameKind.ALIAS:
                return Param(self.names.alias(name))
            case NameKind.LOAD:
                term = _term_name(Name(name), where, in_condition=in_condition)
                return translate_call(
                    Call(term, ()),
                    self.tables,
                    context=where,
                    in_condition=in_condition,
                )
            case NameKind.DATA | NameKind.SMALL_PLANT:
                return Param(self.names.data(name))
            case NameKind.UNRECOGNIZED:
                raise MalformedElectricalExpression(
                    f"{where}: unknown name {name!r}; it is not a declared "
                    "expression or alias, a load term or a data term"
                )

    def _checked_reference(self, reference: Call, where: str) -> int:
        code = _reference_code(reference)
        if code is None:
            problem = "expected re(code), one non-negative integer restriction code"
        elif code not in self.model.restrictions:
            problem = f"restriction {code} is not declared"
        elif self.model.restrictions[code].lhs is None:
            problem = (
                f"restriction {code} has no formula of its own; it is declared "
                "only by per-period rows"
            )
        else:
            return code
        raise MalformedElectricalExpression(f"{where}: {render(reference)}: {problem}")


def _reference_code(reference: Call) -> int | None:
    match reference.args:
        case (Number(text),) if _CODE_RE.fullmatch(text):
            return int(text)
    return None


def _parse(parse: Callable[[str], Node], text: str, context: str) -> Node:
    try:
        return parse(text)
    except MalformedElectricalExpression as error:
        raise MalformedElectricalExpression(f"{context}: {error}") from error


def _model_trees(model: LibsElectricalModel) -> Iterator[Node]:
    for name, expression in model.expressions.items():
        yield _parse(parse_expression, expression.formula, f"expression {name}")
    for code, restriction in model.restrictions.items():
        texts = (
            restriction.lhs,
            restriction.rhs,
            *(side for row in restriction.overrides for side in (row.lhs, row.rhs)),
            *(row.formula for row in restriction.formula_overrides),
        )
        for text in texts:
            if text is not None:
                yield _parse(parse_expression, text, f"restriction {code}")
    for code, condition in model.rules.items():
        yield _parse(parse_condition, condition, f"activation rule {code}")


def _is_identifier(name: str) -> bool:
    return (name[:1].isalpha() or name.startswith("_")) and all(
        char.isalnum() or char == "_" for char in name
    )


def _sanitize(name: str, taken: Set[str]) -> str:
    base = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    if not _is_identifier(base):
        base = f"_{base}"
    candidate, suffix = base, 2
    while candidate in taken:
        candidate, suffix = f"{base}_{suffix}", suffix + 1
    return candidate


def _assign(
    groups: Mapping[str, Sequence[str]], fixed: Set[str]
) -> dict[str, dict[str, str]]:
    kept = {
        name
        for names in groups.values()
        for name in names
        if _is_identifier(name) and name not in fixed
    }
    taken = set(fixed) | kept
    emitted: dict[str, dict[str, str]] = {kind: {} for kind in groups}
    changes: list[list[str]] = []
    for kind, names in groups.items():
        for name in names:
            if name in kept:
                emitted[kind][name] = name
                continue
            emitted[kind][name] = sanitized = _sanitize(name, taken)
            taken.add(sanitized)
            reason = (
                f"collides with {name}"
                if _is_identifier(name)
                else "invalid identifier"
            )
            changes.append([name, kind, sanitized, reason])
    if changes:
        emit(
            Diagnostic(
                code="decomp-electrical-name-sanitized",
                severity=Severity.INFO,
                category="Special constraints",
                title=(
                    f"Electrical constraint names changed to fit novomodelo "
                    f"({len(changes)})"
                ),
                summary=(
                    "The listed names of the LIBS electrical file are not "
                    "novomodelo identifiers or collide with a name the "
                    "conversion declares. The case writes each under the name "
                    "in the table, and every formula reads it under that name, "
                    "so the results do not change."
                ),
                table=DiagnosticTable(
                    columns=["Name", "Kind", "Emitted as", "Reason"], rows=changes
                ),
                remediation=(
                    "Rename them in the deck to keep the deck's names in the "
                    "converted case."
                ),
            ),
            logger=_LOG,
        )
    return emitted


def _warn_shadowed(aliases: Sequence[str], data_terms: Sequence[list[str]]) -> None:
    if aliases:
        emit(
            Diagnostic(
                code="decomp-electrical-alias-shadowed",
                severity=Severity.INFO,
                category="Special constraints",
                title=f"Aliases hidden by an expression of the same name ({len(aliases)})",
                summary=(
                    "The LIBS electrical file declares each listed name both as "
                    "an expression and as an alias. Formulas that cite the name "
                    "read the expression, as in DECOMP, and the conversion "
                    "leaves the alias out."
                ),
                table=DiagnosticTable(
                    columns=["Name"], rows=[[name] for name in aliases]
                ),
                remediation="Rename the alias or the expression to read both.",
            ),
            logger=_LOG,
        )
    if data_terms:
        emit(
            Diagnostic(
                code="decomp-electrical-data-alias-shadowed",
                severity=Severity.INFO,
                category="Special constraints",
                title=(
                    "Data parameters left out because the deck declares the name "
                    f"({len(data_terms)})"
                ),
                summary=(
                    "The LIBS electrical file declares each listed name, which "
                    "the conversion would otherwise create as a parameter from "
                    "the ANDE load (RI) or a PQ series. Formulas that cite the "
                    "name read the declared definition, as in DECOMP, and no "
                    "parameter is created from the register values."
                ),
                table=DiagnosticTable(columns=["Name", "Declared as"], rows=data_terms),
                remediation=(
                    "Rename or remove the declaration to read the RI or PQ "
                    "values instead."
                ),
            ),
            logger=_LOG,
        )

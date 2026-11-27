"""Translate the LIBs electrical special-constraint model into novomodelo's
neutral input: the constraints and expressions of
``constraints/generic_constraints.json``, the rows of
``constraints/generic_constraint_bounds.parquet`` and entries of
``constraints/generic_parameters.json``.

``build_code_map`` and ``machine_set_table`` derive the tables that translate
the DECOMP codes the texts cite to novomodelo ids, and ``data_term_aliases``
the values of the data terms the cards cite but no card defines. Stage indices
pass through as stage ids and a 1-based patamar becomes a 0-based block id.
``read_cells`` parses the texts only to find the cells that read each name
they cite; it changes no text. ``split_restriction`` divides a restriction
into segments of one relation and one violation treatment, the form the
generic constraints take, ``emitted_constraints`` gives each segment the id
and name of its generic constraint and ``segment_slack`` the slack of its
treatment. ``build_generic_bounds`` builds the bounds rows of the segments'
cells and ``build_electrical_parameters`` turns the aliases, data terms and
data-only expressions the texts reach into parameter entries, every text
translated to novomodelo's vocabulary and ids. ``build_generic_outputs`` is
the one place that runs the builders.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

import pyarrow as pa

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from novomodelo_bridge.core.electrical_expression import MalformedElectricalExpression
from novomodelo_bridge.core.generic_constraint_builder import (
    GENERIC_BOUNDS_SCHEMA,
    hard_slack,
    soft_slack,
)
from novomodelo_bridge.decomp.converters.electrical_translate import (
    Translator,
    build_name_table,
)
from novomodelo_bridge.decomp.converters.libs_electrical_parser import (
    Binary,
    Call,
    Group,
    Name,
    NameKind,
    Number,
    Unary,
    classify_name,
    is_data_term,
    iter_nodes,
    parse_condition,
    parse_expression,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence, Set

    from novomodelo_bridge.decomp.converters.contracts import Contract
    from novomodelo_bridge.decomp.converters.electrical_translate import (
        EntityTables,
        NameTable,
    )
    from novomodelo_bridge.decomp.converters.libs_electrical import (
        ElectricalAlias,
        ElectricalRestriction,
        LibsElectricalModel,
    )
    from novomodelo_bridge.decomp.converters.libs_electrical_parser import Node
    from novomodelo_bridge.decomp.id_map import DecompIdMap
    from novomodelo_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)

#: Entity table -> {DECOMP code: novomodelo id}.
CodeMap = Mapping[str, Mapping[int, int]]

#: DECOMP plant code -> {machine-set number: novomodelo unit-group id}.
MachineSets = Mapping[int, Mapping[int, int]]

#: Alias name -> {(stage id, 0-based block id): MW}.
DataAliases = Mapping[str, Mapping[tuple[int, int], float]]

#: ``("alias" | "expression" | "data", name as the deck writes it)`` or
#: ``("reference", R)`` for ``re(R)``.
ReadKey = tuple[str, str] | tuple[str, int]

#: ``(stage id, 0-based block id)``.
Cell = tuple[int, int]

#: ``(lhs, operator, rhs)`` of an inequality or ``(formula,)``, the texts as
#: the deck writes them.
Relation = tuple[str, str, str] | tuple[str]

#: ``("hard", None)`` or ``("soft", cost)``, the cost ``None`` where the
#: default applies.
Treatment = tuple[str, float | None]

#: The relation and the violation treatment of the cells of one segment of a
#: restriction (see :func:`split_restriction`).
SegmentKey = tuple[Relation, Treatment]

#: ``(id, name, segment key, cells)`` of the generic constraint one segment
#: emits (see :func:`emitted_constraints`).
EmittedConstraint = tuple[int, str, SegmentKey, tuple[Cell, ...]]

#: Every name the emitted texts reach -> the cells that read it (see
#: :func:`read_cells`).
ReadCells = Mapping[ReadKey, frozenset[Cell]]

_READ_KINDS = {
    NameKind.EXPRESSION: "expression",
    NameKind.ALIAS: "alias",
    NameKind.DATA: "data",
    NameKind.SMALL_PLANT: "data",
}

#: DECOMP submarket code of the transhipment bus (Ivaiporã) in LIBs formulas.
TRANSHIPMENT_SUBMARKET_CODE = 6


class GenericOutputs(NamedTuple):
    """The electrical cards converted to generic constraints: the
    ``constraints`` and ``expressions`` records of ``generic_constraints.json``,
    the ``generic_constraint_bounds.parquet`` table and the
    ``generic_parameters.json`` entries, without ids."""

    constraints: list[dict]
    expressions: list[dict]
    bounds: pa.Table
    parameters: list[dict]


def build_generic_outputs(
    model: LibsElectricalModel,
    calendar: Sequence[OperativeStage],
    tables: EntityTables,
    data_terms: DataAliases,
    *,
    reserved: frozenset[str],
    taken: Set[int],
    default_penalty: float,
) -> GenericOutputs:
    """Translate ``model`` into generic constraints, named expressions and
    parameter entries.

    ``calendar`` is the one ``model`` was read with; ``data_terms`` holds the
    data terms' values (:func:`data_term_aliases`) and ``reserved`` the names
    other conversions declare as parameters, which no emitted name takes;
    ``taken`` holds the ids of the other conversions' generic constraints and
    ``default_penalty`` the cost of a ``soft`` treatment without one. Each
    constraint :func:`emitted_constraints` gives is one record, by ascending
    id, with the keys ``id``, ``name``, ``expression``, ``active_when`` only
    where the restriction is assigned an activation rule, and ``slack``
    (:func:`segment_slack`), in that order; a restriction holding no row has
    none, and :func:`report_split_restrictions` lists the restrictions not
    emitted as one record under their code. The expressions, with ``name``
    and ``expression`` only, are the reached expressions
    :func:`is_parameter_expression` rejects, by (code, name), then the formula
    of each restriction an ``re(R)`` reads, by ``R``; the parameter entries
    are those of :func:`build_electrical_parameters`, in its order. Both are
    kept only where the ``expression`` or ``active_when`` of a record reaches
    them. No record holds ``None``. The bounds are the rows
    :func:`build_generic_bounds` gives the same constraints.

    Raises:
        MalformedElectricalExpression: an emitted restriction is assigned an
            activation rule the file does not declare, the reached expressions
            and the formulas ``re(R)`` reads cite each other in a cycle, or as
            :func:`read_cells`, :func:`emitted_constraints`,
            :func:`segment_slack`, :func:`build_generic_bounds`,
            :func:`build_electrical_parameters` and :class:`Translator`.
        UnsupportedSourceError: a text writes the load as an LP term outside a
            condition, or cites an entity the converted case lacks
            (:class:`Translator`).
        ValueError: a generated name is reserved.
    """
    names = build_name_table(model, reserved=reserved)
    translator = Translator(model, tables, names)
    reads = read_cells(model, calendar)
    _reject_cycles(model, reads, names)
    emitted = emitted_constraints(model, calendar, taken=taken)
    report_split_restrictions(emitted)
    constraints = _generic_constraint_records(
        model, emitted, translator, default_penalty
    )
    report_ignored_hard_costs(model)
    # The records are the relations governing the cells that hold rows, with
    # their rules, so a name is reached from a record exactly when a cell reads it.
    reached = {key: cells for key, cells in reads.items() if cells}
    return GenericOutputs(
        constraints=constraints,
        expressions=_expression_records(model, translator, reached),
        bounds=build_generic_bounds(model, calendar, emitted),
        parameters=build_electrical_parameters(
            model, calendar, translator, names, data_terms, reads, kept=reached.keys()
        ),
    )


def build_code_map(
    id_map: DecompIdMap,
    gnl_thermal_ids: Mapping[int, int],
    ncs_id_by_pee_code: Mapping[int, int],
    contracts: Sequence[Contract],
) -> CodeMap:
    """Derive the tables that translate a cited entity code to its id in the
    converted case.

    Each table lists every entity of its type; the translator fails on a cited
    code its table lacks. ``thermals`` holds the ``CT`` plants and the ``dadgnl`` GNL plants
    (``gnl_thermal_ids``, code -> created thermal id). ``submarkets`` holds the
    declared subsystem codes plus the transhipment bus under code 6, the code
    the LIBs formulas cite it by, unless a declared subsystem already uses 6. A
    contract number declared by more than one contract is omitted and reported
    in one diagnostic.

    Raises:
        ValueError: a GNL plant code is also a ``CT`` code.
    """
    shared = sorted(set(gnl_thermal_ids) & set(id_map.thermal_codes))
    if shared:
        raise ValueError(f"the GNL plant codes {shared} are also CT thermal codes")
    thermals = {code: id_map.thermal_id(code) for code in id_map.thermal_codes}
    thermals.update(gnl_thermal_ids)
    submarkets = {code: id_map.bus_id(code) for code in id_map.bus_codes}
    submarkets.setdefault(TRANSHIPMENT_SUBMARKET_CODE, id_map.transhipment_bus_id)
    return {
        "hydros": {code: id_map.hydro_id(code) for code in id_map.hydro_codes},
        "thermals": dict(sorted(thermals.items())),
        "ncs_pee": dict(ncs_id_by_pee_code),
        "submarkets": submarkets,
        "contracts": _contract_table(contracts),
    }


def machine_set_table(
    hydros: Sequence[Mapping[str, Any]], id_map: DecompIdMap
) -> MachineSets:
    """Derive each plant's machine sets from the ``hydros`` of ``system/hydros.json``.

    Machine set ``n`` of a plant is its ``n``-th unit group (1-based, in
    ascending group id, since the order of ``unit_groups`` is not
    meaningful), so a set with no group has no entry. Every plant
    with unit groups is listed, not only the ones the formulas cite.
    """
    groups_by_code = {
        id_map.hydro_codes[entry["id"]]: entry["unit_groups"]
        for entry in hydros
        if entry["unit_groups"]
    }
    return {
        code: dict(enumerate(sorted(group["id"] for group in groups), start=1))
        for code, groups in sorted(groups_by_code.items())
    }


def data_term_aliases(
    ande: tuple[int, Mapping[int, Sequence[float]]] | None,
    pq: Mapping[tuple[int, str], Sequence[Sequence[float]]],
    id_map: DecompIdMap,
) -> DataAliases:
    """Derive the aliases that give the formulas' data terms their values.

    ``ande`` is the ``RI`` load of ``ande_load``, named ``carga_ande``: the load
    the scenarios already add to the Itaipu bus, so the alias only supplies the
    value to the formulas and is not a second load. ``pq`` is the
    ``pq_block_generation`` output, one ``peq_<name>_<submarket>`` per series,
    ``<submarket>`` being the subsystem's name in the ``SB`` register, not its
    code. Every series is listed, not only the cited ones.

    Raises:
        ValueError: two ``PQ`` series map to the same alias name.
    """
    aliases: dict[str, dict[tuple[int, int], float]] = {}
    if ande is not None:
        aliases["carga_ande"] = _cells(ande[1].items())
    series_of: dict[str, tuple[int, str]] = {}
    for (code, name), per_stage in pq.items():
        alias = f"peq_{name}_{id_map.bus_names[id_map.bus_id(code)]}"
        if alias in series_of:
            raise ValueError(
                f"alias {alias!r} is produced by PQ series {series_of[alias]} "
                f"and {(code, name)}"
            )
        series_of[alias] = (code, name)
        aliases[alias] = _cells(enumerate(per_stage))
    return aliases


def read_cells(
    model: LibsElectricalModel, calendar: Sequence[OperativeStage]
) -> ReadCells:
    """The cells at which novomodelo reads each name the emitted texts reach.

    The keys are every alias, expression, data term and ``re(R)`` reached,
    through expressions and references, from the sides of a restriction or of
    its overrides, a formula override, or the condition of the rule a
    restriction is assigned; a key no cell reads maps to an empty set.

    The cells mirror novomodelo's row setup. A restriction holds rows only in
    the stages of its horizon. A cell is governed by the override for its
    block that covers its stage, else by the one for every block, else by the
    restriction's own relation. An inequality always holds a row; a formula
    only where the limit row of the stage for the block, else the one for
    every block, sets a side. A cell holding a row reads every name its
    governing relation and the rule condition cite, both ``se`` branches
    included, and ``re(R)`` reads the formula of ``R`` there. Nothing is
    evaluated or folded.

    Raises:
        MalformedElectricalExpression: a text does not parse, or two or more
            overrides of the same precedence cover a cell of the horizon.
    """
    reach = _closure(model)
    reads: dict[ReadKey, set[Cell]] = {}
    for restriction in model.restrictions.values():
        texts = (
            restriction.lhs,
            restriction.rhs,
            *(side for row in restriction.overrides for side in (row.lhs, row.rhs)),
            *(row.formula for row in restriction.formula_overrides),
        )
        text_keys = {
            text: reach(parse_expression(text)) for text in texts if text is not None
        }
        rule = (
            None
            if restriction.habilita is None
            else model.rules.get(restriction.habilita)
        )
        rule_keys = frozenset() if rule is None else reach(parse_condition(rule))
        for key in rule_keys.union(*text_keys.values()):
            reads.setdefault(key, set())
        for cell, relation in _governing(restriction, calendar).items():
            for key in rule_keys.union(text_keys[relation[0]], text_keys[relation[-1]]):
                reads[key].add(cell)
    return {key: frozenset(cells) for key, cells in reads.items()}


def _text_keys(node: Node, model: LibsElectricalModel) -> Iterator[ReadKey]:
    """The keys *node* cites itself, not those of the bodies they name."""
    match node:
        case Name(name):
            if (kind := _READ_KINDS.get(classify_name(name, model))) is not None:
                yield (kind, name)
        case Call("re", (Number(text),)) if (
            text.isdigit()
            and int(text) in model.restrictions
            and model.restrictions[int(text)].lhs is not None
        ):
            yield ("reference", int(text))
        case Call("re"):
            pass  # The translator rejects any other re(...).
        case Call(name, ()) if is_data_term(name):
            yield from _text_keys(Name(name), model)
        case Call(_, args):
            for arg in args:
                yield from _text_keys(arg, model)
        case Unary(_, operand) | Group(operand):
            yield from _text_keys(operand, model)
        case Binary(_, left, right):
            yield from _text_keys(left, model)
            yield from _text_keys(right, model)


def _citations(model: LibsElectricalModel) -> dict[ReadKey, tuple[ReadKey, ...]]:
    """The keys each expression body and each formula ``re(R)`` reads cite
    themselves, in text order."""
    bodies: dict[ReadKey, Node] = {
        ("expression", name): parse_expression(expression.formula)
        for name, expression in model.expressions.items()
    }
    bodies.update(
        (("reference", code), parse_expression(restriction.lhs))
        for code, restriction in model.restrictions.items()
        if restriction.lhs is not None
    )
    return {key: tuple(_text_keys(body, model)) for key, body in bodies.items()}


def _closure(model: LibsElectricalModel) -> Callable[[Node], frozenset[ReadKey]]:
    """Map a tree to the keys it reaches through expression bodies and the
    formulas ``re(R)`` reads; a cycle stops at its first repeated key."""
    cited = _citations(model)

    def reach(tree: Node) -> frozenset[ReadKey]:
        reached: set[ReadKey] = set()
        pending = list(_text_keys(tree, model))
        while pending:
            if (key := pending.pop()) not in reached:
                reached.add(key)
                pending.extend(cited.get(key, ()))
        return frozenset(reached)

    return reach


def _governing(
    restriction: ElectricalRestriction, calendar: Sequence[OperativeStage]
) -> dict[Cell, Relation]:
    """The relation governing each cell where *restriction* holds a row, in
    ascending (stage, block) order, by the rules of :func:`read_cells`.

    Raises:
        MalformedElectricalExpression: two or more overrides of the same
            precedence cover a cell of the horizon, whether or not it holds a
            row.
    """
    start, end = restriction.stage_start, restriction.stage_end
    own: Relation | None = None
    match restriction.lhs, restriction.operator, restriction.rhs:
        case str(lhs), str(operator), str(rhs):
            own = (lhs, operator, rhs)
        case str(lhs), _, _:
            own = (lhs,)
    governing: dict[Cell, Relation] = {}
    for stage_id, stage in enumerate(calendar):
        if (start is not None and stage_id < start) or (
            end is not None and end < stage_id
        ):
            continue
        for block_id in range(len(stage.block_hours)):
            for patamar in (block_id + 1, None):
                relations: list[Relation] = [
                    *(
                        (row.lhs, row.operator, row.rhs)
                        for row in restriction.overrides
                        if row.patamar == patamar
                        and row.stage_start <= stage_id <= row.stage_end
                    ),
                    *(
                        (row.formula,)
                        for row in restriction.formula_overrides
                        if row.patamar == patamar
                        and row.stage_start <= stage_id <= row.stage_end
                    ),
                ]
                if relations:
                    break
            else:
                relations = [] if own is None else [own]
            if len(relations) > 1:
                raise MalformedElectricalExpression(
                    f"restriction {restriction.code}: stage {stage_id} block "
                    f"{block_id} is covered by {len(relations)} overrides of the "
                    "same precedence"
                )
            limit = restriction.limits.get(
                (stage_id, block_id + 1), restriction.limits.get((stage_id, None))
            )
            bounded = limit is not None and limit != (None, None)
            # A one-text relation is a formula, the only kind a limit row bounds.
            if relations and (len(relations[0]) > 1 or bounded):
                governing[(stage_id, block_id)] = relations[0]
    return governing


def split_restriction(
    restriction: ElectricalRestriction, calendar: Sequence[OperativeStage]
) -> dict[SegmentKey, tuple[Cell, ...]]:
    """Split *restriction* into segments of one relation and one violation
    treatment, each with the cells where it holds a row.

    ``calendar`` is the one the restriction was read with. A cell takes the
    relation :func:`read_cells` describes and the treatment of its stage: the
    treatment range that covers the stage, else the row for the whole
    horizon, else ``("soft", None)``; a ``hard`` row takes no cost. The cells
    sharing both, contiguous or not, form one segment, in ascending (stage,
    block), and the segments follow their first cell. The texts are keyed as
    the deck writes them, so two spellings of one relation are two segments.
    A restriction holding no row splits into ``{}``.

    Raises:
        MalformedElectricalExpression: an override's patamar lies outside the
            blocks of a stage it covers, the whole horizon has more than one
            treatment, a stage lies in more than one treatment range, or two
            or more overrides of the same precedence cover a cell; the rows
            are checked even when the restriction holds no row.
    """
    for number, row in enumerate(
        (*restriction.overrides, *restriction.formula_overrides), start=1
    ):
        _block_id(
            row.patamar,
            calendar,
            row.stage_start,
            row.stage_end,
            f"restriction {restriction.code} override {number}",
        )
    whole_horizon, by_stage = _stage_treatments(restriction)
    segments: dict[SegmentKey, list[Cell]] = {}
    for cell, relation in _governing(restriction, calendar).items():
        treatment = by_stage.get(cell[0], whole_horizon)
        segments.setdefault((relation, treatment), []).append(cell)
    return {key: tuple(cells) for key, cells in segments.items()}


def _stage_treatments(
    restriction: ElectricalRestriction,
) -> tuple[Treatment, dict[int, Treatment]]:
    """The treatment of the whole horizon, ``("soft", None)`` without a row
    for it, and the one of each stage a treatment range covers.

    Raises:
        MalformedElectricalExpression: the whole horizon has more than one
            row, or a stage lies in more than one range.
    """
    whole_horizon: list[Treatment] = []
    by_stage: dict[int, Treatment] = {}
    overlapped: set[int] = set()
    for row in restriction.violations:
        treatment: Treatment = (row.kind, row.cost if row.kind == "soft" else None)
        match row.stage_start, row.stage_end:
            case int(first), int(last):
                for stage in range(first, last + 1):
                    if stage in by_stage:
                        overlapped.add(stage)
                    by_stage[stage] = treatment
            case _:
                whole_horizon.append(treatment)
    if len(whole_horizon) > 1:
        raise MalformedElectricalExpression(
            f"restriction {restriction.code}: the whole horizon is given more "
            "than one violation treatment"
        )
    if overlapped:
        raise MalformedElectricalExpression(
            f"restriction {restriction.code}: stage {min(overlapped)} is covered "
            "by more than one violation treatment range"
        )
    return (whole_horizon[0] if whole_horizon else ("soft", None)), by_stage


def emitted_constraints(
    model: LibsElectricalModel,
    calendar: Sequence[OperativeStage],
    *,
    taken: Set[int],
) -> dict[int, tuple[EmittedConstraint, ...]]:
    """The generic constraints each restriction of *model* emits, one per
    segment of :func:`split_restriction` in its order, keyed by ascending code.

    ``taken`` holds the ids of the generic constraints the case already has.
    A restriction of one segment whose code is not taken keeps its code as
    id; every other segment takes the next id above ``taken`` and every code
    of *model*, in code then segment order, so the code of a split
    restriction is no segment's id. One segment is named ``constraint_<code>``
    and the ``k``-th of several ``constraint_<code>_<k>``, from 1; a
    restriction holding no row emits ``()``.

    Raises:
        MalformedElectricalExpression: as :func:`split_restriction`.
    """
    segments = {
        restriction.code: split_restriction(restriction, calendar)
        for restriction in _restrictions_by_code(model)
    }
    fresh = itertools.count(max((*taken, *segments), default=-1) + 1)
    emitted: dict[int, tuple[EmittedConstraint, ...]] = {}
    for code, split in segments.items():
        single = len(split) == 1
        emitted[code] = tuple(
            (
                code if single and code not in taken else next(fresh),
                f"constraint_{code}" if single else f"constraint_{code}_{k}",
                key,
                cells,
            )
            for k, (key, cells) in enumerate(split.items(), start=1)
        )
    return emitted


def segment_slack(
    treatment: Treatment, default_penalty: float, *, context: str
) -> dict[str, object]:
    """The generic ``slack`` of a segment of *treatment*: disabled when
    ``hard``, else charging the ``soft`` cost, or *default_penalty* without
    one, on the ``per_hour`` basis of :data:`PenaltyBasis`.

    ``context`` names the restriction in the messages (``"restriction 405"``).

    Raises:
        MalformedElectricalExpression: the ``soft`` cost, or the default
            penalty a ``soft`` treatment without one takes, is not finite and
            above 0.
    """
    kind, cost = treatment
    if kind == "hard":
        return hard_slack()
    if cost is None:
        if not (math.isfinite(default_penalty) and default_penalty > 0):
            raise MalformedElectricalExpression(
                f"{context}: no violation cost is given and the deck declares no "
                "deficit cost to derive the default from"
            )
        return soft_slack(default_penalty, "per_hour")
    if not (math.isfinite(cost) and cost > 0):
        raise MalformedElectricalExpression(
            f"{context}: the cost of a soft violation treatment is finite and "
            f"above 0, got {cost}"
        )
    return soft_slack(cost, "per_hour")


def report_ignored_hard_costs(model: LibsElectricalModel) -> None:
    """Emit one ``decomp-electrical-hard-treatment-cost-ignored`` info
    diagnostic listing the ``hard`` violation treatment rows of *model* that
    give a cost, by code then stage; nothing when no row does."""
    rows = [
        [
            restriction.code,
            "all"
            if row.stage_start is None or row.stage_end is None
            else f"{row.stage_start} to {row.stage_end}",
            row.cost,
        ]
        for restriction in _restrictions_by_code(model)
        for row in sorted(
            restriction.violations,
            key=lambda row: (
                row.stage_start is not None,
                row.stage_start or 0,
                row.stage_end or 0,
            ),
        )
        if row.kind == "hard" and row.cost is not None
    ]
    if rows:
        emit(
            Diagnostic(
                code="decomp-electrical-hard-treatment-cost-ignored",
                severity=Severity.INFO,
                category="Special constraints",
                title=(
                    "Violation costs of hard electrical constraints not used "
                    f"({len(rows)})"
                ),
                summary=(
                    "The listed hard violation treatments of the LIBS electrical "
                    "file give a cost. A hard treatment allows no violation, so "
                    "the cost is not used and the constraint keeps no slack, as "
                    "in DECOMP, and the results do not change."
                ),
                table=DiagnosticTable(
                    columns=["Code", "Stages", "Cost"],
                    rows=rows,
                    caption="Stage ids are 0-based.",
                ),
            ),
            logger=_LOG,
        )


def report_split_restrictions(
    emitted: Mapping[int, Sequence[EmittedConstraint]],
) -> None:
    """Emit one ``decomp-electrical-restrictions-split`` info diagnostic
    listing each restriction of *emitted* (:func:`emitted_constraints`) that
    is not exactly one constraint whose id is its code: one row per
    constraint, by code then segment, with its cells in full, or one ``none``
    row for a restriction holding no row; nothing when no restriction is
    listed."""
    listed = {
        code: constraints
        for code, constraints in sorted(emitted.items())
        if not (len(constraints) == 1 and constraints[0][0] == code)
    }
    if not listed:
        return
    rows = [
        row
        for code, constraints in listed.items()
        for row in (
            [
                [code, constraint_id, name, _cell_list(cells)]
                for constraint_id, name, _, cells in constraints
            ]
            or [[code, "", "", "none"]]
        )
    ]
    emit(
        Diagnostic(
            code="decomp-electrical-restrictions-split",
            severity=Severity.INFO,
            category="Special constraints",
            title=(
                "Electrical restrictions not written under their own code "
                f"({len(listed)})"
            ),
            summary=(
                "The case writes each listed restriction of the LIBS electrical "
                "file as the generic constraints in the table, each holding only "
                "at its listed cells: a generic constraint keeps one relation and "
                "one violation treatment, under an id no other constraint of the "
                "case uses. A restriction with no limit at any cell is left out. "
                "Every cell keeps the relation and the violation treatment it has "
                "in DECOMP, so the results do not change."
            ),
            table=DiagnosticTable(
                columns=["Code", "Constraint ID", "Constraint name", "Cells"],
                rows=rows,
                caption="Stage and block ids are 0-based.",
            ),
        ),
        logger=_LOG,
    )


def _generic_constraint_records(
    model: LibsElectricalModel,
    emitted: Mapping[int, Sequence[EmittedConstraint]],
    translator: Translator,
    default_penalty: float,
) -> list[dict]:
    records: list[dict] = []
    for code, constraints in emitted.items():
        if not constraints:
            continue
        context = f"restriction {code}"
        activation: dict[str, str] = {}
        if (rule := model.restrictions[code].habilita) is not None:
            if rule not in model.rules:
                raise MalformedElectricalExpression(
                    f"restriction {code} is assigned activation rule {rule}, which "
                    "the file does not declare"
                )
            activation["active_when"] = translator.condition(
                model.rules[rule], context=f"activation rule {rule}"
            )
        for constraint_id, name, (relation, treatment), _ in constraints:
            match relation:
                case (lhs, operator, rhs):
                    expression = translator.relation(
                        lhs, operator, rhs, context=context
                    )
                case (formula,):
                    expression = translator.formula(formula, context=context)
            records.append(
                {
                    "id": constraint_id,
                    "name": name,
                    "expression": expression,
                    **activation,
                    "slack": segment_slack(treatment, default_penalty, context=context),
                }
            )
    return sorted(records, key=lambda record: record["id"])


def build_generic_bounds(
    model: LibsElectricalModel,
    calendar: Sequence[OperativeStage],
    emitted: Mapping[int, Sequence[EmittedConstraint]],
) -> pa.Table:
    """Build the ``generic_constraint_bounds.parquet`` rows of the constraints
    *emitted* (:func:`emitted_constraints`), under :data:`GENERIC_BOUNDS_SCHEMA`.

    A constraint has rows in the stages where it holds cells and none
    elsewhere, since the rows are its horizon. An inequality's cell has no
    endpoint, which only activates it; a formula's cell takes the limit row of
    its patamar, else the stage's every-patamar row, as read, not rounded. A
    stage whose cells are all its blocks is one row with no block id when the
    constraint is an inequality or a formula whose limits give that stage no
    patamar row; any other stage is one row per cell, with its 0-based block
    id. Rows are ordered by (id, stage, block), the row without a block id
    first.

    Raises:
        MalformedElectricalExpression: a limit row's patamar lies outside the
            blocks of its stage, checked for every restriction of *model*.
    """
    for restriction in _restrictions_by_code(model):
        for stage, patamar in restriction.limits:
            _block_id(
                patamar,
                calendar,
                stage,
                stage,
                f"restriction {restriction.code} limits stage {stage}",
            )
    rows = sorted(
        (
            row
            for code, constraints in emitted.items()
            for constraint in constraints
            for row in _generic_bound_rows(
                model.restrictions[code].limits, calendar, constraint
            )
        ),
        key=lambda row: (row[0], *_cell_order((row[1], row[2]))),
    )
    return pa.Table.from_pylist(
        [dict(zip(GENERIC_BOUNDS_SCHEMA.names, row, strict=True)) for row in rows],
        schema=GENERIC_BOUNDS_SCHEMA,
    )


def _generic_bound_rows(
    limits: Mapping[tuple[int, int | None], tuple[float | None, float | None]],
    calendar: Sequence[OperativeStage],
    constraint: EmittedConstraint,
) -> Iterator[tuple[int, int, int | None, float | None, float | None]]:
    constraint_id, _, (relation, _), cells = constraint
    is_formula = len(relation) == 1
    blocks_by_stage: dict[int, list[int]] = {}
    for stage, block in cells:
        blocks_by_stage.setdefault(stage, []).append(block)
    for stage, blocks in blocks_by_stage.items():
        every_block = set(blocks) == set(range(len(calendar[stage].block_hours)))
        patamar_rows = is_formula and any(
            row_stage == stage and patamar is not None for row_stage, patamar in limits
        )
        for block in [None] if every_block and not patamar_rows else blocks:
            key = (stage, None if block is None else block + 1)
            sides = (
                limits[key if key in limits else (stage, None)]
                if is_formula
                else (None, None)
            )
            yield (constraint_id, stage, block, *sides)


def build_electrical_parameters(
    model: LibsElectricalModel,
    calendar: Sequence[OperativeStage],
    translator: Translator,
    names: NameTable,
    data_terms: DataAliases,
    reads: ReadCells,
    *,
    kept: Set[ReadKey],
) -> list[dict]:
    """The ``generic_parameters.json`` entries, without ids, of the aliases,
    data terms and data-only expressions the emitted texts reach.

    ``reads`` is :func:`read_cells` of ``model`` and ``calendar``, the one
    source of what is reached and read; ``kept`` holds the reached keys that
    become entries, while the omitted diagnostic still counts every key of
    ``reads`` as reached. ``data_terms`` gives the data terms their values
    (:func:`data_term_aliases`), and a term it lacks has none.
    Each reached alias and data term is a ``per_stage_block`` entry over every
    study cell: an alias's value for a block wins over its value for every
    block of the stage, and a cell without a value that ``reads`` does not read
    is written as ``0.0`` and listed in one
    ``decomp-electrical-unread-cells-filled`` info diagnostic. Each reached
    expression :func:`is_parameter_expression` accepts is an ``expression``
    entry with the translated body. Entries are named by ``names`` and ordered
    aliases by (code, name), data terms by name, then expressions by (code,
    name), so they do not depend on input insertion order. The declared aliases and
    expressions nothing reaches, except an alias an expression of the same
    name hides, and the activation rules no restriction is assigned are listed
    in one ``decomp-electrical-unread-omitted`` info diagnostic.

    Raises:
        MalformedElectricalExpression: a reached alias or data term has no
            value at a cell that reads it, a reached alias has a patamar
            outside the blocks of its stage, or reached expressions form a
            cycle; the translator's errors pass through.
    """
    study = [
        (stage_id, block_id)
        for stage_id, stage in enumerate(calendar)
        for block_id in range(len(stage.block_hours))
    ]
    valued = [
        *(
            (
                "alias",
                alias.name,
                names.alias(alias.name),
                _alias_cells(alias, calendar),
            )
            for alias in sorted(model.aliases.values(), key=lambda a: (a.code, a.name))
            if ("alias", alias.name) in kept
        ),
        *(
            ("data", name, names.data(name), data_terms.get(name, {}))
            for name in sorted(
                name for kind, name in kept if kind == "data" and isinstance(name, str)
            )
        ),
    ]
    entries: list[dict] = []
    filled: list[list[str]] = []
    for kind, name, emitted, values in valued:
        block_values, unread = _cover(name, values, reads[(kind, name)], study)
        entries.append(
            {"name": emitted, "kind": "per_stage_block", "block_values": block_values}
        )
        if unread:
            filled.append([emitted, kind, _cell_list(unread)])
    entries += [
        {
            "name": names.expression(expression.name),
            "kind": "expression",
            "expression": translator.formula(
                expression.formula, context=f"expression {expression.name}"
            ),
        }
        for expression in sorted(
            model.expressions.values(), key=lambda e: (e.code, e.name)
        )
        if ("expression", expression.name) in kept
        and is_parameter_expression(expression.name, model)
    ]
    _report_unread(sorted(filled), _omitted(model, reads))
    return entries


def is_parameter_expression(name: str, model: LibsElectricalModel) -> bool:
    """Whether the declared expression *name* is arithmetic over data only.

    It is when its text holds only numbers, signs, parentheses, ``+ - * /``
    and names (or ``carga_ande()`` and ``peq_…()`` without arguments) of
    aliases, data terms and expressions that are data-only themselves. Any
    other call or a load name keeps it an expression of the electrical file.

    Raises:
        MalformedElectricalExpression: the expressions *name* cites, directly
            or through others, form a cycle (the message lists it from
            *name*'s side), or a body does not parse.
    """
    verdicts: dict[str, bool] = {}
    path: list[str] = []

    def classify(expression: str) -> bool:
        if expression in path:
            cycle = [*path[path.index(expression) :], expression]
            raise MalformedElectricalExpression(
                f"electrical expressions form a cycle: {' -> '.join(cycle)}"
            )
        if expression not in verdicts:
            path.append(expression)
            tree = parse_expression(model.expressions[expression].formula)
            # Every node, not up to the first non-data one: a cycle fails the
            # same way whatever operand comes first.
            verdicts[expression] = all(
                [_is_data_node(node, model, classify) for node in iter_nodes(tree)]
            )
            path.pop()
        return verdicts[expression]

    return classify(name)


def _is_data_node(
    node: Node, model: LibsElectricalModel, is_parameter: Callable[[str], bool]
) -> bool:
    """Whether *node* itself, not its children, keeps its expression data-only."""
    match node:
        # Only + - * / outside a se() condition, and the se call is not data.
        case Number() | Unary() | Group() | Binary():
            return True
        case Name(name):
            match classify_name(name, model):
                case NameKind.EXPRESSION:
                    return is_parameter(name)
                case NameKind.ALIAS | NameKind.DATA | NameKind.SMALL_PLANT:
                    return True
                case NameKind.LOAD | NameKind.UNRECOGNIZED:
                    return False
        case Call(name, ()) if is_data_term(name):
            return _is_data_node(Name(name), model, is_parameter)
    return False


def _alias_cells(
    alias: ElectricalAlias, calendar: Sequence[OperativeStage]
) -> dict[Cell, float]:
    """The value of *alias* at each study cell it gives one, a block's own
    value winning over the stage's every-block value.

    Raises:
        MalformedElectricalExpression: a patamar lies outside the blocks of
            its stage.
    """
    cells: dict[Cell, float] = {}
    # The every-block value of a stage sorts before its block values, which
    # then overwrite it.
    for (stage, patamar), value in sorted(
        alias.values.items(), key=lambda item: _cell_order(item[0])
    ):
        block = _block_id(
            patamar, calendar, stage, stage, f"alias {alias.name} stage {stage}"
        )
        blocks = range(len(calendar[stage].block_hours)) if block is None else (block,)
        cells.update(((stage, block_id), value) for block_id in blocks)
    return cells


def _cover(
    name: str,
    values: Mapping[Cell, float],
    read: frozenset[Cell],
    study: Sequence[Cell],
) -> tuple[list[list[float]], list[Cell]]:
    """The ``block_values`` of every study cell, ``0.0`` where *values* has
    none, and the cells so filled.

    Raises:
        MalformedElectricalExpression: a cell of *read* has no value; the
            message names *name* and only those cells.
    """
    if missing := sorted(read - values.keys()):
        raise MalformedElectricalExpression(
            f"an electrical constraint reads {name} at {_cell_list(missing)}, "
            "where it has no value"
        )
    return (
        [[stage, block, values.get((stage, block), 0.0)] for stage, block in study],
        [cell for cell in study if cell not in values],
    )


def _cell_list(cells: Iterable[Cell]) -> str:
    return ", ".join(f"stage {stage} block {block}" for stage, block in cells)


def _omitted(model: LibsElectricalModel, reads: ReadCells) -> list[list[str]]:
    """``[kind, deck name or code]`` of each declared alias and expression
    nothing reaches, except an alias an expression of the same name hides,
    which is already reported as hidden, and of each activation rule no
    restriction is assigned."""
    assigned = _assigned_rules(model)
    return [
        *(
            ["alias", alias.name]
            for alias in sorted(model.aliases.values(), key=lambda a: (a.code, a.name))
            if alias.name not in model.expressions
            and ("alias", alias.name) not in reads
        ),
        *(
            ["expression", expression.name]
            for expression in sorted(
                model.expressions.values(), key=lambda e: (e.code, e.name)
            )
            if ("expression", expression.name) not in reads
        ),
        *(
            ["activation rule", str(code)]
            for code in sorted(model.rules)
            if code not in assigned
        ),
    ]


def _report_unread(filled: Sequence[list[str]], omitted: Sequence[list[str]]) -> None:
    if filled:
        emit(
            Diagnostic(
                code="decomp-electrical-unread-cells-filled",
                severity=Severity.INFO,
                category="Special constraints",
                title=(
                    "Parameter cells no electrical constraint reads written as 0.0 "
                    f"({len(filled)})"
                ),
                summary=(
                    "The listed aliases and data terms of the LIBS electrical "
                    "file have no value at the listed cells, which no constraint "
                    "reads. novomodelo needs a value at every cell, so the case "
                    "writes 0.0 there, and the results do not change."
                ),
                table=DiagnosticTable(
                    columns=["Parameter", "Kind", "Filled cells"],
                    rows=filled,
                    caption="Stage and block ids are 0-based.",
                ),
                remediation=(
                    "Declare a value for every stage and block to convert the "
                    "values without filled cells."
                ),
            ),
            logger=_LOG,
        )
    if omitted:
        emit(
            Diagnostic(
                code="decomp-electrical-unread-omitted",
                severity=Severity.INFO,
                category="Special constraints",
                title=(
                    "Electrical aliases, expressions and activation rules no "
                    f"constraint reaches ({len(omitted)})"
                ),
                summary=(
                    "No restriction, override or assigned activation rule of the "
                    "LIBS electrical file cites the listed aliases and "
                    "expressions, directly or through other expressions, and no "
                    "restriction is assigned the listed activation rules. The "
                    "conversion leaves them out, so the results do not change."
                ),
                table=DiagnosticTable(columns=["Kind", "Name"], rows=omitted),
            ),
            logger=_LOG,
        )


def _reject_cycles(
    model: LibsElectricalModel, reads: ReadCells, names: NameTable
) -> None:
    """Fail when the reached expressions and the formulas ``re(R)`` reads cite
    each other in a cycle, which novomodelo rejects.

    Raises:
        MalformedElectricalExpression: the cycle, an expression by its deck
            name and a formula by its generated name
            (``E -> restriction_1_formula -> E``).
    """
    cited = _citations(model)
    path: list[ReadKey] = []
    acyclic: set[ReadKey] = set()

    def visit(key: ReadKey) -> None:
        if key in path:
            cycle = [name for _, name in (*path[path.index(key) :], key)]
            raise MalformedElectricalExpression(
                "electrical expressions form a cycle: "
                + " -> ".join(
                    names.reference(name) if isinstance(name, int) else name
                    for name in cycle
                )
            )
        if key not in acyclic:
            path.append(key)
            for successor in cited.get(key, ()):
                visit(successor)
            path.pop()
            acyclic.add(key)

    for key in sorted(cited.keys() & reads.keys()):
        visit(key)


def _expression_records(
    model: LibsElectricalModel, translator: Translator, reads: ReadCells
) -> list[dict]:
    named = [
        *(
            (
                translator.names.expression(expression.name),
                expression.formula,
                f"expression {expression.name}",
            )
            for expression in sorted(
                model.expressions.values(), key=lambda e: (e.code, e.name)
            )
            if ("expression", expression.name) in reads
            and not is_parameter_expression(expression.name, model)
        ),
        *(
            (
                translator.names.reference(restriction.code),
                restriction.lhs,
                f"restriction {restriction.code} formula",
            )
            for restriction in _restrictions_by_code(model)
            if ("reference", restriction.code) in reads and restriction.lhs is not None
        ),
    ]
    return [
        {"name": name, "expression": translator.formula(text, context=context)}
        for name, text, context in named
    ]


def _cells(
    stages: Iterable[tuple[int, Sequence[float]]],
) -> dict[tuple[int, int], float]:
    return {
        (stage, block): value
        for stage, values in stages
        for block, value in enumerate(values)
    }


def _block_id(
    patamar: int | None,
    calendar: Sequence[OperativeStage],
    first_stage: int,
    last_stage: int,
    context: str,
) -> int | None:
    """1-based patamar to 0-based block id; ``None`` (all blocks) stays ``None``.

    The patamar must exist in every stage of ``first_stage..last_stage``.
    """
    if patamar is None:
        return None
    n_blocks = min(
        len(calendar[stage].block_hours) for stage in range(first_stage, last_stage + 1)
    )
    if not 1 <= patamar <= n_blocks:
        raise MalformedElectricalExpression(
            f"{context}: patamar {patamar} outside 1..{n_blocks}"
        )
    return patamar - 1


def _cell_order(cell: tuple[int, int | None]) -> tuple[int, bool, int]:
    stage, patamar = cell
    return (stage, patamar is not None, patamar or 0)


def _restrictions_by_code(model: LibsElectricalModel) -> list[ElectricalRestriction]:
    return sorted(model.restrictions.values(), key=lambda r: r.code)


def _assigned_rules(model: LibsElectricalModel) -> set[int | None]:
    return {r.habilita for r in model.restrictions.values()}


def _contract_table(contracts: Sequence[Contract]) -> dict[int, int]:
    declared = Counter(contract.numero for contract in contracts)
    ambiguous = sorted(numero for numero, count in declared.items() if count > 1)
    if ambiguous:
        emit(
            Diagnostic(
                code="decomp-libs-electrical-contract-code-ambiguous",
                severity=Severity.INFO,
                category="Special constraints",
                title=(
                    "Ambiguous contract numbers left out of the contract table "
                    f"({len(ambiguous)})"
                ),
                summary=(
                    "More than one CI or CE contract declares each listed "
                    "number, so the LIBS electrical constraints cannot cite it "
                    "by that number: a contract term on it stops the "
                    "conversion with source-unsupported."
                ),
                table=DiagnosticTable(
                    columns=["Contract number"], rows=[[n] for n in ambiguous]
                ),
                remediation=(
                    "Give the import and export contracts distinct numbers in "
                    "dadger.rvN to cite them."
                ),
            ),
            logger=_LOG,
        )
    return {
        contract.numero: contract.id
        for contract in contracts
        if declared[contract.numero] == 1
    }

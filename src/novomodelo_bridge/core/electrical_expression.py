"""The source models' electrical linear-expression language: parse one
formula into flat terms, sort each term into the A/B/C bucket trichotomy, and
fold the input-data terms into a numeric bound.

NEWAVE's ``restricao-eletrica.csv`` and DECOMP's
``lib_restricao-eletrica-especial.csv`` write their electrical constraints in
the same language: a signed sum of decision variables (``ger_usih(h)``,
``ener_interc(a, b)``, …) and input data (``demanda(x)``, numeric constants,
…), with coefficients that multiply a single term (``0.5*ger_usih(66)``,
NEWAVE's juxtaposed ``0.5ger_usih(66)``) or a parenthesized sum, and ``re(R)``
references to another constraint's formula. :func:`parse_linear_expression`
distributes every coefficient onto the atomic terms and inlines every
reference; an :class:`ExpressionScope` supplies what a track's file declares
(named expressions, data aliases, other constraints' formulas, the GNL
thermals whose generation is input data).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from novomodelo_bridge.core.diagnostics import Diagnostic, DiagnosticTable, Severity

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Built-in bare (argument-less) input-data tokens. A bare name the scope does
#: not declare, and not in this set, is an unrecognized identifier.
BUILTIN_BARE_TOKENS = frozenset({"demanda_sin", "val_demanda_sin", "carga_ande"})

#: The demand terms whose constraint multiplier the source model adds to the
#: submarket's marginal cost; their ``val_`` forms are the same load without it.
CMO_DEMAND_TOKENS = frozenset({"demanda", "demanda_sin"})

#: Reserved token for a bare ``peq_<PQ name>_<subsystem>`` reference to one
#: ``dadger`` ``PQ`` series' generation. The series name keeps its own
#: underscores (``N_PCHgd``), so only the last ``_`` separates the suffix.
PEQ_TOKEN = "peq"
PEQ_PREFIX = "peq_"

#: Reserved token for a ``ger_usit(c)`` whose ``c`` is a GNL plant
#: (:attr:`ExpressionScope.gnl_thermal_codes`): its generation is commanded,
#: so it is input data, not a decision.
GNL_TOKEN = "__ger_usit_gnl__"

#: Reserved token for a bare numeric constant; its value lives in
#: :attr:`ParsedTerm.coefficient`.
CONST_TOKEN = "__const__"

#: Reserved token for an unresolved data-alias reference; the alias's own name
#: lives in :attr:`ParsedTerm.alias_name`.
ALIAS_TOKEN = "alias"

#: Reserved token for a structural ``se(cond, X, Y)`` conditional.
SE_TOKEN = "se"

#: How many integer arguments each function the conversion indexes takes.
_ARITY = {
    "ger_usih": 1,
    "ger_usit": 1,
    "ger_pee": 1,
    "ger_conjh": 2,
    "ener_interc": 2,
    "disp_usih": 1,
    "demanda": 1,
    "val_demanda": 1,
}

#: The optional exponent covers the source model's ``1E+31`` "no limit" value.
_NUMBER_RE = re.compile(r"\d+\.?\d*(?:[eE][+-]?\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class ParsedTerm:
    """One atomic, sign-and-coefficient-distributed term of a parsed linear
    expression (see :func:`parse_linear_expression`).

    ``token`` identifies the term's kind:

    - a function-call name (``ger_usih``, ``ener_interc``, ``disp_usih``, …)
      with its integer arguments in ``args``;
    - the reserved :data:`CONST_TOKEN` (``"__const__"``) for a bare numeric
      constant — its value is in ``coefficient``, ``args`` is empty;
    - the reserved :data:`ALIAS_TOKEN` (``"alias"``) for an unresolved data
      alias reference — ``args`` is empty and the alias's own name is in
      ``alias_name``; its numeric value is resolved per-(stage,block) later,
      never here;
    - a built-in bare token (``demanda_sin`` / ``val_demanda_sin`` /
      ``carga_ande``), ``args`` empty;
    - the reserved :data:`GNL_TOKEN` for a ``ger_usit(c)`` naming a GNL plant
      (:attr:`ExpressionScope.gnl_thermal_codes`), the code in ``args``;
      its commanded generation is resolved per-(stage,block) later;
    - the reserved :data:`PEQ_TOKEN` (``"peq"``) for a ``peq_<name>_<sub>``
      reference — the ``PQ`` series name is in ``pq_name`` and the raw
      subsystem suffix in ``pq_submarket``; resolved per-(stage,block) later;
    - the reserved :data:`SE_TOKEN` (``"se"``) for a ``se(cond, X, Y)``
      conditional, kept structural: ``condition``, ``branch_true``, and
      ``branch_false`` carry the raw, unparsed sub-expression text (evaluated
      later, never here; ``cond`` is not evaluated by this parser).

    A named-expression or ``re(R)`` reference never survives to a
    ``ParsedTerm`` — the parser inlines it into its own terms, multiplying the
    enclosing coefficient through.
    """

    coefficient: float
    token: str
    args: tuple[int, ...] = ()
    alias_name: str | None = None
    pq_name: str | None = None
    pq_submarket: str | None = None
    condition: str | None = None
    branch_true: str | None = None
    branch_false: str | None = None


@dataclass(frozen=True)
class AssembledBound:
    """The result of folding one restriction's bucket-B/-C terms into a
    per-(stage,block) numeric bound, sign-canonical for novomodelo.

    ``terms`` is the surviving bucket-A (decision) expression, already in
    novomodelo's single-LHS form: ``lower <= Σ terms <= upper``. A formula
    restriction keeps its LHS bucket-A terms verbatim; a fold that moves a
    term across the relation (an inequality's RHS decision terms, a reserve's
    generation terms under the operator flip) has already sign-flipped it, so
    a caller reading ``terms`` never has to know which shape produced it.
    """

    terms: tuple[ParsedTerm, ...]
    lower: float | None
    upper: float | None


def negate(term: ParsedTerm) -> ParsedTerm:
    """*term* with its coefficient sign flipped, every other field verbatim."""
    return replace(term, coefficient=-term.coefficient)


class ExpressionScope(Protocol):
    """What one track's constraint file declares, as the parser reads it."""

    @property
    def gnl_thermal_codes(self) -> frozenset[int]:
        """The thermals whose ``ger_usit`` is commanded input data."""
        ...

    def expression_formula(self, name: str) -> str | None:
        """The formula of the named expression *name*, ``None`` when the file
        declares none by that name."""
        ...

    def is_alias(self, name: str) -> bool:
        """Whether *name* is a declared data alias."""
        ...

    def restriction_formula(self, code: int) -> str | None:
        """The formula ``re(code)`` stands for, ``None`` when the file
        declares no restriction *code*.

        Raises:
            MalformedElectricalExpression: restriction *code* has no formula
                for ``re`` to reference; the message says why.
        """
        ...


class MalformedElectricalExpression(ValueError):
    """Formula text the source model's grammar rejects: unbalanced
    parentheses, a non-numeric coefficient or argument, a function the
    grammar lacks or given the wrong number of arguments, a reference to a
    restriction the file does not declare, or a reference cycle.

    The caller that knows the file and the constraint turns it into a
    :class:`~novomodelo_bridge.core.errors.FieldParseError`.
    """


class UnrecognizedElectricalToken(ValueError):
    """A well-formed but undeclared bare identifier, *name*: neither a
    named expression or data alias the scope declares, nor a member of
    :data:`BUILTIN_BARE_TOKENS`.

    Distinct from :class:`MalformedElectricalExpression` so a caller can leave
    out just the constraint that names it without masking a malformed formula
    as a routine drop.
    """

    def __init__(self, name: str, text: str) -> None:
        super().__init__(
            f"unrecognized identifier {name!r} in electrical expression "
            f"{text!r} (not a declared named expression, data alias, "
            "built-in bare token, or peq_ small-plant reference)"
        )
        self.name = name


class _ExpressionParser:
    """Character-position recursive-descent parser for one DSL expression string.

    Threads a running numeric ``scale`` — the product of every sign and scalar
    coefficient enclosing the current position — down to each atomic term, so a
    coefficient distributed over a parenthesized sum (``0.05*(a + b)``) lands on
    every term inside it, and a named-expression/``re(R)`` reference inlines its
    own terms multiplied by the scale in effect where the reference occurred.
    """

    def __init__(self, text: str, scope: ExpressionScope, seen: frozenset[str]) -> None:
        self._text = text
        self._scope = scope
        self._seen = seen
        self._pos = 0

    def parse(self) -> list[ParsedTerm]:
        terms = self._parse_expr(1.0)
        self._skip_ws()
        if self._pos != len(self._text):
            raise MalformedElectricalExpression(
                f"unexpected trailing content {self._text[self._pos :]!r} in "
                f"electrical expression {self._text!r}"
            )
        return terms

    def _skip_ws(self) -> None:
        while self._pos < len(self._text) and self._text[self._pos].isspace():
            self._pos += 1

    def _peek(self) -> str:
        self._skip_ws()
        return self._text[self._pos] if self._pos < len(self._text) else ""

    def _consume(self, char: str, context: str) -> None:
        self._skip_ws()
        if self._pos >= len(self._text) or self._text[self._pos] != char:
            raise MalformedElectricalExpression(
                f"expected {char!r} {context} in electrical expression {self._text!r}"
            )
        self._pos += 1

    def _parse_expr(self, scale: float) -> list[ParsedTerm]:
        """``['+'|'-'] term (('+'|'-') term)*`` — a signed sum of terms."""
        terms: list[ParsedTerm] = []
        sign = 1.0
        if self._peek() in ("+", "-"):
            sign = -1.0 if self._text[self._pos] == "-" else 1.0
            self._pos += 1
        terms.extend(self._parse_term(scale * sign))
        while self._peek() in ("+", "-"):
            sign = -1.0 if self._text[self._pos] == "-" else 1.0
            self._pos += 1
            terms.extend(self._parse_term(scale * sign))
        return terms

    def _parse_term(self, scale: float) -> list[ParsedTerm]:
        """``[NUMBER ['*']] factor`` — a bare number alone is a constant term.

        NEWAVE writes a coefficient against its term (``0.5ger_usih(66)``);
        a number followed by anything but a name or ``(`` stands alone.
        """
        self._skip_ws()
        number = _NUMBER_RE.match(self._text, self._pos)
        if number is None:
            return self._parse_factor(scale)
        probe = number.end()
        while probe < len(self._text) and self._text[probe].isspace():
            probe += 1
        if probe < len(self._text) and self._text[probe] == "*":
            self._pos = probe + 1
            return self._parse_factor(scale * float(number.group()))
        if probe < len(self._text) and (
            self._text[probe] == "(" or _IDENT_RE.match(self._text, probe)
        ):
            self._pos = probe
            return self._parse_factor(scale * float(number.group()))
        self._pos = number.end()
        return [
            ParsedTerm(coefficient=scale * float(number.group()), token=CONST_TOKEN)
        ]

    def _parse_factor(self, scale: float) -> list[ParsedTerm]:
        """``'(' expr ')' | IDENT '(' args ')' | IDENT`` — a group or an atom."""
        self._skip_ws()
        if self._pos >= len(self._text):
            raise MalformedElectricalExpression(
                "expected a term but reached the end of electrical expression "
                f"{self._text!r}"
            )
        if self._text[self._pos] == "(":
            self._pos += 1
            terms = self._parse_expr(scale)
            self._consume(")", "to close '('")
            return terms
        ident = _IDENT_RE.match(self._text, self._pos)
        if ident is None:
            raise MalformedElectricalExpression(
                f"malformed token at {self._text[self._pos :]!r} in electrical "
                f"expression {self._text!r}"
            )
        name = ident.group()
        self._pos = ident.end()
        if self._peek() == "(":
            return self._parse_function_call(name, scale)
        return self._parse_bare_name(name, scale)

    def _parse_function_call(self, name: str, scale: float) -> list[ParsedTerm]:
        if name == SE_TOKEN:
            return self._parse_se(scale)
        if name == "re":
            return self._parse_re(scale)
        args = self._parse_int_args(name)
        arity = _ARITY.get(name)
        if arity is not None and len(args) != arity:
            expected = "one argument" if arity == 1 else f"{arity} arguments"
            raise MalformedElectricalExpression(
                f"{name}(...) expects {expected}, got {args!r} in electrical "
                f"expression {self._text!r}"
            )
        if name == "ger_usit" and args[0] in self._scope.gnl_thermal_codes:
            return [ParsedTerm(coefficient=scale, token=GNL_TOKEN, args=args)]
        return [ParsedTerm(coefficient=scale, token=name, args=args)]

    def _parse_int_args(self, fn_name: str) -> tuple[int, ...]:
        self._consume("(", f"after {fn_name!r}")
        args: list[int] = []
        if self._peek() == ")":
            self._pos += 1
            return tuple(args)
        while True:
            self._skip_ws()
            number = _NUMBER_RE.match(self._text, self._pos)
            if number is None or "." in number.group():
                raise MalformedElectricalExpression(
                    f"expected an integer argument for {fn_name!r}(...) at "
                    f"{self._text[self._pos :]!r} in electrical expression "
                    f"{self._text!r}"
                )
            args.append(int(number.group()))
            self._pos = number.end()
            if self._peek() == ",":
                self._pos += 1
                continue
            break
        self._consume(")", f"to close {fn_name!r}(...)")
        return tuple(args)

    def _parse_re(self, scale: float) -> list[ParsedTerm]:
        """``re(R)`` — a reference to restriction ``R``'s formula, inlined."""
        args = self._parse_int_args("re")
        if len(args) != 1:
            raise MalformedElectricalExpression(
                f"re(...) expects exactly one restriction id, got {args!r} in "
                f"electrical expression {self._text!r}"
            )
        code = args[0]
        try:
            formula = self._scope.restriction_formula(code)
        except MalformedElectricalExpression as err:
            raise MalformedElectricalExpression(
                f"{err} in electrical expression {self._text!r}"
            ) from err
        if formula is None:
            raise MalformedElectricalExpression(
                f"re({code}) references unknown restriction {code} in electrical "
                f"expression {self._text!r}"
            )
        return self._expand_reference(f"re({code})", formula, scale)

    def _parse_se(self, scale: float) -> list[ParsedTerm]:
        """``se(cond, X, Y)`` — one structural term; ``cond`` is never evaluated
        here, so its raw text (and both branches') is captured by a
        depth-aware scan for the top-level commas rather than by tokenizing it."""
        self._consume("(", "after 'se'")
        args_start = self._pos
        comma_positions: list[int] = []
        depth = 0
        pos = self._pos
        while True:
            if pos >= len(self._text):
                raise MalformedElectricalExpression(
                    "unbalanced parentheses in se(...) in electrical expression "
                    f"{self._text!r}"
                )
            char = self._text[pos]
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    break
                depth -= 1
            elif char == "," and depth == 0:
                comma_positions.append(pos)
            pos += 1
        close_pos = pos
        if len(comma_positions) != 2:
            raise MalformedElectricalExpression(
                "se(...) expects exactly 3 arguments (condition, true-branch, "
                f"false-branch) in electrical expression {self._text!r}"
            )
        first_comma, second_comma = comma_positions
        condition = self._text[args_start:first_comma].strip()
        branch_true = self._text[first_comma + 1 : second_comma].strip()
        branch_false = self._text[second_comma + 1 : close_pos].strip()
        self._pos = close_pos + 1
        return [
            ParsedTerm(
                coefficient=scale,
                token=SE_TOKEN,
                condition=condition,
                branch_true=branch_true,
                branch_false=branch_false,
            )
        ]

    def _parse_bare_name(self, name: str, scale: float) -> list[ParsedTerm]:
        """Disambiguate a bare name: a named expression, else a data alias,
        else a built-in bare token, else a ``peq_`` ``PQ`` reference — else an
        unrecognized identifier."""
        formula = self._scope.expression_formula(name)
        if formula is not None:
            return self._expand_reference(name, formula, scale)
        if self._scope.is_alias(name):
            return [ParsedTerm(coefficient=scale, token=ALIAS_TOKEN, alias_name=name)]
        if name in BUILTIN_BARE_TOKENS:
            return [ParsedTerm(coefficient=scale, token=name)]
        pq_name, _, submarket = name.removeprefix(PEQ_PREFIX).rpartition("_")
        if name.startswith(PEQ_PREFIX) and pq_name and submarket:
            return [
                ParsedTerm(
                    coefficient=scale,
                    token=PEQ_TOKEN,
                    pq_name=pq_name,
                    pq_submarket=submarket,
                )
            ]
        raise UnrecognizedElectricalToken(name, self._text)

    def _expand_reference(
        self, key: str, formula: str, scale: float
    ) -> list[ParsedTerm]:
        """Recursively parse *formula* (a named expression's or a restriction's
        LHS), multiplying every resulting term's coefficient by *scale*, and
        guard the recursion against a cycle through *key*."""
        if key in self._seen:
            cycle = " -> ".join(sorted({*self._seen, key}))
            raise MalformedElectricalExpression(
                f"cyclic electrical expression reference: {cycle}"
            )
        sub_terms = parse_linear_expression(
            formula, self._scope, _seen=self._seen | {key}
        )
        return [
            replace(term, coefficient=scale * term.coefficient) for term in sub_terms
        ]


def parse_linear_expression(
    text: str,
    scope: ExpressionScope,
    *,
    _seen: frozenset[str] = frozenset(),
) -> list[ParsedTerm]:
    """Parse one electrical linear-expression DSL string into flat terms.

    Distributes every sign and scalar coefficient onto each atomic term
    (including across a parenthesized sum) and recursively inlines
    named-expression references and ``re(R)`` restriction references,
    multiplying through the enclosing coefficient — ``_seen`` (the names/
    ``re(R)`` keys expanded on the current recursion path) guards that
    inlining against a cycle. A data-alias
    reference is kept as a single unresolved :data:`ALIAS_TOKEN` term (its
    value is resolved per-(stage,block) later, never here); a
    ``se(cond, X, Y)`` conditional is kept as a single structural term with
    its condition and both branches unevaluated.

    This is the source model's linear-expression grammar, not novomodelo's: the
    parser never normalizes the relation (no RHS→LHS move, no same-token
    merge) — it preserves the author's side and term multiplicity verbatim;
    novomodelo owns relational normalization.

    Raises
    ------
    MalformedElectricalExpression
        On text the grammar rejects, ``re(R)`` to an undeclared restriction
        or to one with no formula, or a reference cycle.
    UnrecognizedElectricalToken
        On a bare identifier that is well-formed but undeclared — neither a
        named expression, a data alias, nor a built-in bare token.
    """
    return _ExpressionParser(text, scope, _seen).parse()


class Bucket(Enum):
    """Which of the source model's A/B/C trichotomy a parsed term belongs to.

    - **A** — a novomodelo decision term: stays a structural LP term.
    - **B** — an input-data term: the bridge evaluates it to a number and
      folds it into the bound.
    - **C** — ``disp_usih``: novomodelo has no available-power ``VariableRef``, so
      the bridge resolves it to a per-stage constant that rewrites the
      constraint form instead of emitting it as a structural LP term.
    """

    A = "A"
    B = "B"
    C = "C"


#: Bucket A — novomodelo decision tokens that stay structural LP terms.
#: ``re(X)`` is not listed here: the parser already inlines a restriction
#: reference into its own terms, so no :class:`ParsedTerm` ever carries a
#: bare ``"re"`` token.
BUCKET_A_TOKENS = frozenset(
    {"ger_usih", "ger_usit", "ger_pee", "ger_conjh", "ener_interc", "ener_comerc"}
)

#: Bucket B — input-data tokens the bridge evaluates to a number and folds
#: into the bound: the built-in demand/load tokens, plus the parser's reserved
#: tokens (a bare numeric constant, an unresolved data alias, a ``peq_``
#: series, a GNL plant's commanded ``ger_usit``, and a ``se(cond, X, Y)``
#: conditional).
BUCKET_B_TOKENS = frozenset(
    {
        "demanda",
        "val_demanda",
        "demanda_sin",
        "val_demanda_sin",
        "carga_ande",
        CONST_TOKEN,
        ALIAS_TOKEN,
        PEQ_TOKEN,
        GNL_TOKEN,
        SE_TOKEN,
    }
)

_TOKEN_BUCKET: dict[str, Bucket] = {
    **dict.fromkeys(BUCKET_A_TOKENS, Bucket.A),
    **dict.fromkeys(BUCKET_B_TOKENS, Bucket.B),
    "disp_usih": Bucket.C,
}

#: The function names a formula may call, as the source model writes them.
_FUNCTION_NAMES = (
    frozenset(_TOKEN_BUCKET)
    - BUILTIN_BARE_TOKENS
    - {CONST_TOKEN, ALIAS_TOKEN, PEQ_TOKEN, GNL_TOKEN}
) | {"re"}


def classify_term(term: ParsedTerm) -> Bucket:
    """Classify one parsed term into its A/B/C bucket, purely from
    ``term.token``: the parser has already resolved every name to a concrete
    token.

    Raises
    ------
    MalformedElectricalExpression
        When ``term.token`` is a function the grammar lacks — fail loud
        rather than silently drop or mis-bucket a term, which would corrupt
        either the LP structure (bucket A/C) or the bound (bucket B).
    """
    bucket = _TOKEN_BUCKET.get(term.token)
    if bucket is None:
        raise MalformedElectricalExpression(
            f"unknown function {term.token}(...); an electrical expression "
            f"calls only {', '.join(sorted(_FUNCTION_NAMES))}"
        )
    return bucket


def classify_terms(terms: Sequence[ParsedTerm]) -> dict[Bucket, list[ParsedTerm]]:
    """Partition a term sequence into its three buckets.

    Every :class:`Bucket` key is present (an empty list when no term
    classified into it), and each bucket's list preserves the input order.

    Raises
    ------
    MalformedElectricalExpression
        Propagated from :func:`classify_term` on the first unknown function.
    """
    partition: dict[Bucket, list[ParsedTerm]] = {bucket: [] for bucket in Bucket}
    for term in terms:
        partition[classify_term(term)].append(term)
    return partition


#: Resolves one input-data term's value at one (stage, block) cell — a
#: built-in bare token (``demanda_sin``), a parameterized one
#: (``demanda(3)``), an unresolved alias reference (``term.alias_name``), or a
#: ``se(cond, X, Y)`` conditional.
#:
#: A bare numeric constant (``term.token == CONST_TOKEN``) is the one
#: exception: its value already lives in the term's own ``coefficient``, so a
#: resolver **must** return ``1.0`` for it — the ``coefficient *
#: context(term)`` product would otherwise double-apply the constant.
type DataContext = Callable[[ParsedTerm], float]


def evaluate_terms(terms: Sequence[ParsedTerm], context: DataContext) -> float:
    """Sum ``coefficient * context(term)`` over *terms*.

    A bare numeric constant is read straight from ``term.coefficient`` rather
    than through *context*: a conforming context already returns ``1.0`` for
    it, so this changes nothing for one, but a non-conforming context can
    never double- or zero-apply a constant.
    """
    return sum(
        term.coefficient
        if term.token == CONST_TOKEN
        else term.coefficient * context(term)
        for term in terms
    )


def fold_formula_bound(
    partition: Mapping[Bucket, Sequence[ParsedTerm]],
    lower: float | None,
    upper: float | None,
    context: DataContext,
) -> AssembledBound:
    """Fold a formula's bucket-B sum into its two-sided limit, keeping its
    bucket-A terms verbatim: ``lower <= lhs_A + lhs_B <= upper`` becomes
    ``lower - lhs_B <= lhs_A <= upper - lhs_B``. A ``None`` side stays
    unbounded."""
    data_sum = evaluate_terms(partition[Bucket.B], context)
    return AssembledBound(
        terms=tuple(partition[Bucket.A]),
        lower=None if lower is None else lower - data_sum,
        upper=None if upper is None else upper - data_sum,
    )


def demand_in_cmo_diagnostic(
    code: str, rows: Sequence[tuple[int, str]], *, source: str
) -> Diagnostic:
    """The warning for converted constraints with a :data:`CMO_DEMAND_TOKENS`
    term, one ``(constraint, term)`` row each; *source* names the file that
    holds the formulas."""
    unique = sorted(set(rows))
    return Diagnostic(
        code=code,
        severity=Severity.WARNING,
        category="Special constraints",
        title=f"Electric constraint demand terms left out of the CMO ({len(unique)})",
        summary=(
            "The listed constraints write their demand as demanda or demanda_sin, "
            "so the source model adds each constraint's multiplier to the "
            "marginal cost (CMO) of every submarket whose load the term reads. "
            "The case folds the demand into the constraint's limits and cannot "
            "do that, so wherever a listed constraint binds, the case's CMO "
            "differs from the source model's."
        ),
        table=DiagnosticTable(
            columns=["Constraint", "Term"],
            rows=[list(row) for row in unique],
            justify=["right", "left"],
        ),
        remediation=(
            "If the constraint's multiplier should not count in the CMO, write "
            f"the term as val_demanda or val_demanda_sin in {source}."
        ),
    )


def term_text(term: ParsedTerm) -> str:
    """*term* as a formula writes it, without its coefficient."""
    if term.token == PEQ_TOKEN:
        return f"{PEQ_PREFIX}{term.pq_name}_{term.pq_submarket}"
    if term.token == SE_TOKEN:
        return f"se({term.condition}, {term.branch_true}, {term.branch_false})"
    if term.token == ALIAS_TOKEN:
        return str(term.alias_name)
    if term.token == CONST_TOKEN:
        return f"{term.coefficient:g}"
    if term.token in BUILTIN_BARE_TOKENS:
        return term.token
    name = "ger_usit" if term.token == GNL_TOKEN else term.token
    return f"{name}({', '.join(str(arg) for arg in term.args)})"

"""Parse the source model's electrical DSL into a tree that keeps what the
author wrote.

Every source token appears once in the tree, in source order: parentheses stay
a :class:`Group`, signs a :class:`Unary`, numbers their literal text, and bare
names, ``re(R)`` references and ``se(c, a, b)`` conditionals stay as written.
Nothing is distributed, inlined, merged, reordered or evaluated, so novomodelo
repeats the author's arithmetic exactly. :func:`render` prints a tree back with
canonical spacing; the translation into novomodelo's vocabulary maps one tree
to another and prints it with the same function.

The grammar is the one novomodelo reads the rendered text with: ``+ - * /``,
unary signs, parentheses, calls, and conditions made of single comparisons
joined by ``&`` (binding tighter) and ``|``, never inside parentheses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from novomodelo_bridge.core.electrical_expression import (
    PEQ_PREFIX,
    MalformedElectricalExpression,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from novomodelo_bridge.decomp.converters.libs_electrical import (
        LibsElectricalModel,
    )


@dataclass(frozen=True)
class Number:
    """A numeric literal; ``text`` is its source spelling, never normalized."""

    text: str


@dataclass(frozen=True)
class Name:
    """A name written without parentheses."""

    name: str


@dataclass(frozen=True)
class Call:
    """``name(args)``; a ``se`` call holds its condition as the first argument."""

    name: str
    args: tuple[Node, ...]


@dataclass(frozen=True)
class Unary:
    """A ``-`` or ``+`` sign applied to its operand."""

    op: str
    operand: Node


@dataclass(frozen=True)
class Binary:
    """``left op right``: arithmetic, a comparison, ``&`` or ``|``."""

    op: str
    left: Node
    right: Node


@dataclass(frozen=True)
class Group:
    """A pair of parentheses the author wrote."""

    inner: Node


@dataclass(frozen=True)
class Param:
    """A ``@name`` reference; only a translated tree carries it."""

    name: str


@dataclass(frozen=True)
class Keyword:
    """A ``key=value`` call argument; only a translated tree carries it."""

    key: str
    value: Node


type Node = Number | Name | Call | Unary | Binary | Group | Param | Keyword


class NameKind(Enum):
    """What a name written without parentheses refers to (:func:`classify_name`)."""

    EXPRESSION = auto()
    ALIAS = auto()
    LOAD = auto()
    DATA = auto()
    SMALL_PLANT = auto()
    UNRECOGNIZED = auto()


_BUILTIN_NAMES = {
    "demanda_sin": NameKind.LOAD,
    "val_demanda_sin": NameKind.LOAD,
    "carga_ande": NameKind.DATA,
}


def classify_name(name: str, model: LibsElectricalModel) -> NameKind:
    """What *name* refers to, in the source model's order: a declared named
    expression, then a declared data alias, then a built-in data term, then a
    ``peq_`` small-plant series."""
    if name in model.expressions:
        return NameKind.EXPRESSION
    if name in model.aliases:
        return NameKind.ALIAS
    if name in _BUILTIN_NAMES:
        return _BUILTIN_NAMES[name]
    if split_small_plant(name) is not None:
        return NameKind.SMALL_PLANT
    return NameKind.UNRECOGNIZED


def split_small_plant(name: str) -> tuple[str, str] | None:
    """Split ``peq_<series>_<submarket>`` into ``(series, submarket)``.

    Only the last ``_`` separates the submarket: the series name keeps its own
    underscores (``peq_N_PCHgd_SE`` is ``("N_PCHgd", "SE")``). ``None`` when
    *name* lacks the ``peq_`` prefix or either part is empty.
    """
    if not name.startswith(PEQ_PREFIX):
        return None
    series, _, submarket = name.removeprefix(PEQ_PREFIX).rpartition("_")
    return (series, submarket) if series and submarket else None


def is_data_term(name: str) -> bool:
    """Whether *name* spells a data term: the built-in data name or a ``peq_``
    small-plant series. Judged by the name alone, so a declared expression or
    alias of the same name does not change the answer."""
    return (
        _BUILTIN_NAMES.get(name) is NameKind.DATA or split_small_plant(name) is not None
    )


def parse_expression(text: str) -> Node:
    """Parse an arithmetic DSL text: a formula, an inequality side, a named
    expression.

    The first argument of a ``se`` call is a condition (see
    :func:`parse_condition`) and ``se`` takes exactly three arguments; every
    other call argument is an expression.

    Raises
    ------
    MalformedElectricalExpression
        On any syntax error, a comparison outside a condition included; the
        message names the text and the 0-based position of the problem.
    """
    return _Parser(text).expression()


def parse_condition(text: str) -> Node:
    """Parse a condition: comparisons (``< <= > >= ==``) of two expressions,
    joined by ``&`` and ``|``.

    Raises
    ------
    MalformedElectricalExpression
        As :func:`parse_expression`; a chained comparison (``a < b < c``) or
        one inside parentheses is a syntax error.
    """
    return _Parser(text).condition()


def render(node: Node) -> str:
    """Print *node* with canonical spacing.

    One space around every binary operator, a sign glued to its operand,
    ``", "`` between arguments, ``key=value`` without spaces, nothing inside
    the parentheses. Parentheses are printed only for a :class:`Group`, so
    rendering a parsed text changes nothing but its whitespace.
    """
    match node:
        case Number(text):
            return text
        case Name(name):
            return name
        case Param(name):
            return f"@{name}"
        case Call(name, args):
            return f"{name}({', '.join(render(arg) for arg in args)})"
        case Unary(op, operand):
            return f"{op}{render(operand)}"
        case Binary(op, left, right):
            return f"{render(left)} {op} {render(right)}"
        case Group(inner):
            return f"({render(inner)})"
        case Keyword(key, value):
            return f"{key}={render(value)}"


def iter_nodes(node: Node) -> Iterator[Node]:
    """Yield *node* and every node below it in pre-order: a node before its
    children, the children in source order."""
    stack: list[Node] = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(_children(current)))


def _children(node: Node) -> tuple[Node, ...]:
    match node:
        case Number() | Name() | Param():
            return ()
        case Call(_, args):
            return args
        case Unary(_, operand) | Group(operand) | Keyword(_, operand):
            return (operand,)
        case Binary(_, left, right):
            return (left, right)


class _Kind(Enum):
    NUMBER = auto()
    NAME = auto()
    SYMBOL = auto()
    END = auto()


@dataclass(frozen=True)
class _Token:
    kind: _Kind
    text: str
    position: int


#: The optional exponent covers the source model's ``1E+31`` "no limit" value.
_NUMBER_RE = re.compile(r"\d+\.?\d*(?:[eE][+-]?\d+)?", re.ASCII)

#: Two-character operators first: ``_tokenize`` takes the first symbol that
#: matches, and ``<=`` must never be read as ``<`` then ``=``.
_COMPARISONS = ("<=", ">=", "==", "<", ">")
_SYMBOLS = (*_COMPARISONS, "+", "-", "*", "/", "(", ")", ",", "&", "|")


def _starts_name(char: str) -> bool:
    return char.isalpha() or char == "_"


def _continues_name(char: str) -> bool:
    return char.isalnum() or char == "_"


def _syntax_error(
    text: str, position: int, message: str
) -> MalformedElectricalExpression:
    return MalformedElectricalExpression(
        f"electrical expression {text!r}, position {position}: {message}"
    )


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    position = 0
    while position < len(text):
        char = text[position]
        if char.isspace():
            position += 1
            continue
        if number := _NUMBER_RE.match(text, position):
            end = number.end()
            if end < len(text) and (text[end] == "." or _continues_name(text[end])):
                raise _syntax_error(text, position, "malformed number")
            tokens.append(_Token(_Kind.NUMBER, number.group(), position))
        elif _starts_name(char):
            end = position + 1
            while end < len(text) and _continues_name(text[end]):
                end += 1
            tokens.append(_Token(_Kind.NAME, text[position:end], position))
        elif symbol := next(
            (s for s in _SYMBOLS if text.startswith(s, position)), None
        ):
            end = position + len(symbol)
            tokens.append(_Token(_Kind.SYMBOL, symbol, position))
        else:
            raise _syntax_error(text, position, f"unexpected character {char!r}")
        position = end
    tokens.append(_Token(_Kind.END, "", len(text)))
    return tokens


class _Parser:
    def __init__(self, text: str) -> None:
        self._text = text
        self._tokens = _tokenize(text)
        self._index = 0

    def expression(self) -> Node:
        node = self._sum()
        self._finish()
        return node

    def condition(self) -> Node:
        node = self._disjunction()
        self._finish()
        return node

    def _peek(self) -> _Token:
        return self._tokens[self._index]

    def _accept(self, *symbols: str) -> str | None:
        token = self._peek()
        if token.kind is _Kind.SYMBOL and token.text in symbols:
            self._index += 1
            return token.text
        return None

    def _disjunction(self) -> Node:
        node = self._conjunction()
        while op := self._accept("|"):
            node = Binary(op, node, self._conjunction())
        return node

    def _conjunction(self) -> Node:
        node = self._comparison()
        while op := self._accept("&"):
            node = Binary(op, node, self._comparison())
        return node

    def _comparison(self) -> Node:
        left = self._sum()
        op = self._accept(*_COMPARISONS)
        if op is None:
            raise self._unexpected("a comparison operator (<, <=, >, >=, ==)")
        node = Binary(op, left, self._sum())
        if self._peek().text in _COMPARISONS:
            raise self._error(
                "a comparison relates exactly two expressions; join comparisons "
                "with '&' or '|'"
            )
        return node

    def _sum(self) -> Node:
        node = self._product()
        while op := self._accept("+", "-"):
            node = Binary(op, node, self._product())
        return node

    def _product(self) -> Node:
        node = self._sign()
        while op := self._accept("*", "/"):
            node = Binary(op, node, self._sign())
        return node

    def _sign(self) -> Node:
        if op := self._accept("+", "-"):
            return Unary(op, self._sign())
        return self._operand()

    def _operand(self) -> Node:
        token = self._peek()
        if token.kind is _Kind.NUMBER:
            self._index += 1
            return Number(token.text)
        if token.kind is _Kind.NAME:
            self._index += 1
            open_paren = self._peek()
            if self._accept("("):
                return self._call(token, open_paren)
            return Name(token.text)
        if self._accept("("):
            inner = self._sum()
            self._close(token, "')'")
            return Group(inner)
        raise self._unexpected("an operand")

    def _call(self, name: _Token, open_paren: _Token) -> Call:
        args: list[Node] = []
        if self._peek().text != ")":
            args.append(self._disjunction() if name.text == "se" else self._sum())
            while self._accept(","):
                args.append(self._sum())
        self._close(open_paren, "',' or ')'")
        if name.text == "se" and len(args) != 3:
            raise _syntax_error(
                self._text,
                name.position,
                "se(...) takes exactly 3 arguments (a condition, the value when "
                f"it holds, the value otherwise), found {len(args)}",
            )
        return Call(name.text, tuple(args))

    def _close(self, open_paren: _Token, expected: str) -> None:
        if self._accept(")"):
            return
        if self._peek().kind is _Kind.END:
            raise _syntax_error(self._text, open_paren.position, "'(' is never closed")
        raise self._after_operand(expected)

    def _finish(self) -> None:
        if self._peek().text == ")":
            raise self._error("')' has no matching '('")
        if self._peek().kind is not _Kind.END:
            raise self._after_operand("the end of the text")

    def _after_operand(self, expected: str) -> MalformedElectricalExpression:
        token = self._peek()
        if token.text in _COMPARISONS:
            return self._error(
                f"comparison {token.text!r} is only allowed at the top level of "
                "a condition"
            )
        return self._unexpected(expected)

    def _unexpected(self, expected: str) -> MalformedElectricalExpression:
        token = self._peek()
        if token.kind is _Kind.END:
            return self._error(f"expected {expected} but the text ends")
        return self._error(f"expected {expected}, found {token.text!r}")

    def _error(self, message: str) -> MalformedElectricalExpression:
        return _syntax_error(self._text, self._peek().position, message)
